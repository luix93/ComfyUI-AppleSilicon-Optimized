"""Generation-scoped exact converted-weight cache for Qwen3-VL 4B on MPS.

ComfyUI deliberately uses floating matrix execution for these text-encoder
layers. Cache its existing INT8 ConvRot dequantization; do not change precision,
dot-product arithmetic, attention, sampling, or diffusion conditioning.
Validated single-token projection grouping and norm/RoPE fusion reduce launches.
Copies die at generation exit (also on cancellation/error). Hooks outside this scope are no-ops.
"""
import functools
import hashlib
import inspect
import logging
import math
import os
from contextvars import ContextVar
from pathlib import Path

import torch

_scope = ContextVar('as_qwen3vl_decode', default=None)
_installed = False
_installed_cast = None
_last_stats = {}
_library = None
_kernel_failed = False


def _budget():
    try:
        gib = float(os.environ.get('ASFP8_QWEN3VL_CACHE_GIB', '8'))
        if not math.isfinite(gib) or gib <= 0:
            return 0
        return int(min(gib * 2**30, torch.mps.recommended_max_memory() * .10))
    except (ValueError, RuntimeError, AttributeError):
        return 0


def _revision(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor) and t.is_inference():
        return id(t), None
    try:
        return id(t), t._version
    except RuntimeError:  # inference tensor; cache never survives this call
        return id(t), None


def _key(module, p):
    w = module.weight
    params = getattr(w, '_params', None)
    x = p['input']
    dtype = p['dtype'] or getattr(x, 'dtype', None)
    device = p['device'] if p['device'] is not None else getattr(x, 'device', None)
    return (id(module), _revision(w), _revision(getattr(w, '_qdata', None)),
            _revision(getattr(params, 'scale', None)), _revision(module.bias),
            dtype, str(device), p['bias_dtype'] or dtype, p['compute_dtype'],
            p['want_requant'], getattr(params, 'convrot', False),
            getattr(params, 'convrot_groupsize', None))


def _cast(original, signature, quantized_type, *args, **kwargs):
    state = _scope.get()
    if state is None or state['budget'] <= 0 or torch.is_grad_enabled():
        return original(*args, **kwargs)
    # Only the fixed, generation-local set of quantized projection modules.
    module = args[0] if args else kwargs.get('s')
    if (id(module) not in state['modules']
            or not (getattr(module, '_full_precision_mm', False)
                    or getattr(module, 'num_embeddings', None) == 151936)
            or getattr(module, 'weight_function', [])
            or getattr(module, 'bias_function', [])
            or getattr(module, 'weight_lowvram_function', None) is not None
            or getattr(module, 'bias_lowvram_function', None) is not None):
        return original(*args, **kwargs)
    names = ('s', 'input', 'dtype', 'device', 'bias_dtype', 'offloadable',
             'compute_dtype', 'want_requant')
    if (1 <= len(args) <= 2 and tuple(signature.parameters) == names
            and not kwargs.keys() - set(names)
            and not any(name in kwargs for name in names[:len(args)])):
        p = dict(input=None, dtype=None, device=None, bias_dtype=None,
                 offloadable=False, compute_dtype=None, want_requant=False)
        p.update(kwargs)
        p.update(zip(names, args))
    else:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
    x = p['input']
    device = p['device'] if p['device'] is not None else getattr(x, 'device', None)
    if device is None or torch.device(device).type != 'mps' or p['want_requant']:
        return original(*args, **kwargs)
    key = _key(module, p)
    entry = state['cache'].get(key)
    if entry is not None:
        state['hits'] += 1
        return (*entry, None) if p['offloadable'] else entry
    result = original(*args, **kwargs)
    weight, bias = result[:2]
    # forward_comfy_cast_weights would perform this exact operation immediately
    # after the cast context returns. Moving it here permits reuse next token.
    if isinstance(weight, quantized_type) and getattr(module, 'out_features', None) is not None:
        weight = weight.dequantize()
        result = (weight, *result[1:])
    if type(weight) is torch.Tensor and weight.is_floating_point():
        size = weight.numel() * weight.element_size()
        if bias is not None:
            size += bias.numel() * bias.element_size()
        if (state['bytes'] + size <= state['budget'] and
                torch.mps.driver_allocated_memory() + size <
                torch.mps.recommended_max_memory() * .65):
            # Copies cannot alias cast/offload scratch storage. Keep the original
            # offload lease on the miss; cached hits need no lease.
            state['cache'][key] = (weight.detach().clone(),
                                  None if bias is None else bias.detach().clone())
            state['bytes'] += size
            state['misses'] += 1
    return result


def _generate(original, model, *args, **kwargs):
    embeds = args[0] if args else kwargs.get('embeds')
    if (os.environ.get('ASFP8_QWEN3VL_DECODE', '1') != '1'
            or torch.is_grad_enabled() or _scope.get() is not None
            or getattr(model, 'model_type', None) != 'qwen3vl_4b'
            or not isinstance(embeds, torch.Tensor) or embeds.device.type != 'mps'):
        return original(model, *args, **kwargs)
    budget = _budget()
    from comfy_kitchen.tensor import QuantizedTensor
    modules = {id(m) for m in model.model.modules()
               if (getattr(m, 'out_features', None) is not None
                   or getattr(m, 'num_embeddings', None) == 151936)
               and isinstance(getattr(m, 'weight', None), QuantizedTensor)
               and getattr(m.weight, '_layout_cls', None) == 'TensorWiseINT8Layout'}
    if not modules:
        return original(model, *args, **kwargs)
    state = dict(modules=modules, cache={}, bytes=0, budget=budget, hits=0, misses=0)
    return _run_scoped(state, original, model, *args, **kwargs)


def _run_scoped(state, original, model, *args, **kwargs):
    global _last_stats
    token = _scope.set(state)
    try:
        _prepare_projection_groups(state, model)
        _prepare_qk_norms(state, model)
        return original(model, *args, **kwargs)
    finally:
        _scope.reset(token)
        _last_stats = {k: state[k] for k in ('bytes', 'budget', 'hits', 'misses')}
        _last_stats['head_calls'] = state.get('head_calls', 0)
        _last_stats['floating_head_calls'] = state.get('floating_head_calls', 0)
        _last_stats['rope_calls'] = state.get('rope_calls', 0)
        _last_stats['grouped_projection_calls'] = state.get('grouped_projection_calls', 0)
        _last_stats['norm_rope_calls'] = state.get('norm_rope_calls', 0)
        for module, wrapper, previous, had_override in reversed(state.pop('forward_overrides', [])):
            if module.forward is wrapper:
                if had_override:
                    module.forward = previous
                else:
                    del module.forward
        state.get('projection_groups', {}).clear()
        state.get('projection_pending', {}).clear()
        state.get('norm_pending', {}).clear()
        state['cache'].clear()
        if state['hits']:
            logging.info('[AppleSilicon-Optimized/Qwen3VL] exact decode weight cache: '
                         '%d conversions reused, %.2f GiB released.',
                         state['hits'], state['bytes'] / 2**30)



def _projection_eligible(module, state):
    return (id(module) in state['modules']
            and getattr(module, '_full_precision_mm', False)
            and getattr(module, 'comfy_force_cast_weights', False)
            and module.bias is None
            and not getattr(module, 'weight_function', [])
            and not getattr(module, 'bias_function', [])
            and getattr(module, 'weight_lowvram_function', None) is None
            and getattr(module, 'bias_lowvram_function', None) is None
            and getattr(module, 'pre_quant_scale', None) is None
            and not getattr(module, '_forward_hooks', {})
            and not getattr(module, '_forward_pre_hooks', {})
            and '_forward' not in module.__dict__
            and 'forward_comfy_cast_weights' not in module.__dict__)


def _grouped_projection(state, modules, x):
    # Reuse only conversions already accepted by the ordinary cast cache. A
    # constrained budget, changed weight or foreign hook retains native calls.
    if not all(_projection_eligible(m, state) for m in modules):
        return None
    p = dict(input=x, dtype=None, device=None, bias_dtype=None,
             compute_dtype=x.dtype, want_requant=False)
    keys = tuple(_key(m, p) for m in modules)
    groups = state.setdefault('projection_groups', {})
    ids = tuple(id(m) for m in modules)
    cached = groups.get(ids)
    if cached is not None and cached[0] == keys:
        # Owned packed tensor metadata is fixed for this generation. Validate
        # source revisions/hooks above, without rechecking every cached view.
        state['grouped_projection_calls'] = state.get('grouped_projection_calls', 0) + 1
        return torch.nn.functional.linear(x, cached[1]).split(cached[2], dim=-1)
    entries = [state['cache'].get(k) for k in keys]
    if any(e is None for e in entries):
        return None
    weights = [e[0] for e in entries]
    if any(type(w) is not torch.Tensor or w.dtype != x.dtype
           or w.device != x.device or not w.is_contiguous()
           or w.ndim != 2 or w.shape[1] != x.shape[-1] or e[1] is not None
           for w, e in zip(weights, entries)):
        return None
    size = sum(w.numel() * w.element_size() for w in weights)
    if torch.mps.driver_allocated_memory() + size >= torch.mps.recommended_max_memory() * .65:
        return None
    packed = torch.cat(weights, dim=0)
    sizes = tuple(w.shape[0] for w in weights)
    # Replace owned separate allocations with views of the owned packed
    # buffer. Retained cache bytes do not grow; one group is copied at a time.
    for key, view in zip(keys, packed.split(sizes, dim=0)):
        state['cache'][key] = (view, None)
    groups[ids] = (keys, packed, sizes)
    state['grouped_projection_calls'] = state.get('grouped_projection_calls', 0) + 1
    return torch.nn.functional.linear(x, packed).split(sizes, dim=-1)



@functools.lru_cache(maxsize=1)
def _projection_source_supported():
    try:
        import comfy.ops
        return hashlib.sha256(Path(comfy.ops.__file__).read_bytes()).hexdigest() == (
            'a0b8c7f32a146b383a795363bef35e480ffa84b5b3e9c9503f323b1446876f45')
    except (ImportError, OSError, TypeError):
        return False


def _prepare_projection_groups(state, model):
    if (os.environ.get('ASFP8_QWEN3VL_GROUPED_PROJ', '1') != '1'
            or state['budget'] <= 0 or model is None):
        return
    from . import torch_mpp_qcache as qc, yue2_rope_mps as yr
    if (not qc._hardware() or not qc._torch_build_supported()
            or not yr._sources_supported() or not _projection_source_supported()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return
    for block in model.model.layers:
        attention, mlp = block.self_attn, block.mlp
        if getattr(attention, 'merged_qkv', False) or getattr(mlp, 'merged_mlp', False):
            continue
        for modules in ((attention.q_proj, attention.k_proj, attention.v_proj),
                        (mlp.gate_proj, mlp.up_proj)):
            if (not all(_projection_eligible(m, state) for m in modules)
                    or any('forward' in m.__dict__ for m in modules)):
                continue
            _wrap_projection_group(state, modules)


def _wrap_projection_group(state, modules):
    ids = tuple(id(m) for m in modules)
    pending = state.setdefault('projection_pending', {})
    for index, module in enumerate(modules):
        original = module.forward
        # Preserve instance overrides and class-bound methods, even on errors.
        had_override = 'forward' in module.__dict__
        @functools.wraps(original)
        def forward(x, *args, _index=index, _original=original, **kwargs):
            if (_scope.get() is not state or args or kwargs or torch.is_grad_enabled()
                    or type(x) is not torch.Tensor or x.device.type != 'mps'
                    or x.dtype != torch.bfloat16 or x.ndim != 3
                    or tuple(x.shape[:2]) != (1, 1) or not x.is_contiguous()):
                return _original(x, *args, **kwargs)
            if _index == 0:
                pending.pop(ids, None)
                parts = _grouped_projection(state, modules, x)
                if parts is None:
                    return _original(x)
                pending[ids] = (x, parts)
            entry = pending.get(ids)
            if entry is None or entry[0] is not x:
                return _original(x)
            # Keep core's interruption/accounting seam for each logical call.
            import comfy.ops as ops
            ops.run_every_op()
            out = entry[1][_index]
            if _index == len(modules) - 1:
                pending.pop(ids, None)
            return out
        module.forward = forward
        state.setdefault('forward_overrides', []).append((module, forward, original, had_override))



@functools.lru_cache(maxsize=1)
def _rms_source_supported():
    try:
        import comfy.rmsnorm
        return hashlib.sha256(Path(comfy.rmsnorm.__file__).read_bytes()).hexdigest() == (
            'e8dfbebe9272587038f504538f72fe7e82261307f2c7fbd6f0b05cce28172fdd')
    except (ImportError, OSError, TypeError):
        return False


def _prepare_qk_norms(state, model):
    if (model is None or _kernel_failed
            or os.environ.get('ASFP8_QWEN3VL_NORM_ROPE', '1') != '1'
            or os.environ.get('ASFP8_QWEN3VL_ROPE', '1') != '1'
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return
    from . import torch_mpp_qcache as qc, yue2_rope_mps as yr
    from comfy.text_encoders import llama
    if (not qc._hardware() or not qc._torch_build_supported()
            or not yr._sources_supported() or not _rms_source_supported()):
        return
    for block in model.model.layers:
        for name, heads in (('q_norm', 32), ('k_norm', 8)):
            norm = getattr(block.self_attn, name, None)
            if (type(norm) is not llama.RMSNorm or norm.add
                    or norm._forward_hooks or norm._forward_pre_hooks
                    or 'forward' in norm.__dict__
                    or tuple(norm.weight.shape) != (128,)):
                continue
            _wrap_qk_norm(state, norm, heads)


def _wrap_qk_norm(state, norm, heads):
    original = norm.forward
    @functools.wraps(original)
    def forward(x):
        if (_scope.get() is not state or torch.is_grad_enabled()
                or type(x) is not torch.Tensor or x.device.type != 'mps'
                or x.dtype != torch.bfloat16 or tuple(x.shape) != (1, heads, 1, 128)
                or not x.is_contiguous() or norm.add
                or norm.weight.device != x.device or norm.weight.dtype != x.dtype
                or not norm.weight.is_contiguous()
                or tuple(norm.weight.shape) != (128,)):
            return original(x)
        state.setdefault('norm_pending', {})[id(x)] = (x, original, norm.weight, norm.eps)
        return x
    norm.forward = forward
    state.setdefault('forward_overrides', []).append((norm, forward, original, False))


def _finish_norm(x, entry):
    return entry[1](x) if entry is not None and entry[0] is x else x


def _gemv(x, weight, row_scale, col_scale):
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
    n, k = weight.shape
    out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
    _library.int8_gemv(x, weight, row_scale, col_scale, out, k, n,
                       int(col_scale.numel() == 1),
                       threads=((n + 3) // 4) * 128, group_size=128)
    return out


def _logits(original, model, x):
    global _kernel_failed
    state = _scope.get()
    if (state is None or _kernel_failed
            or os.environ.get('ASFP8_QWEN3VL_HEAD', '1') != '1'
            or type(x) is not torch.Tensor or x.device.type != 'mps'
            or x.dtype != torch.bfloat16 or x.ndim != 3 or x.shape[0] != 1
            or x.shape[-1] != 2560):
        return original(model, x)
    from . import torch_mpp_qcache as qc
    if not qc._hardware() or not qc._torch_build_supported():
        return original(model, x)
    import comfy.ops as ops
    from comfy_kitchen.tensor import QuantizedTensor
    from . import int8_linear_kernel_mps as ik
    module = getattr(model.model, 'lm_head', model.model.embed_tokens)
    if (not module.comfy_cast_weights or ik._kernel is None
            or getattr(module, 'weight_function', [])
            or getattr(module, 'bias_function', [])):
        return original(model, x)
    last = x[:, -1:]
    with ops.CastBiasWeightContext(module, last, offloadable=True) as (weight, _):
        if type(weight) is torch.Tensor and weight.is_floating_point():
            # The pack's FP8/quantized compatibility wrapper can return the
            # floating tied table. Preserve that exact F.linear instead of
            # changing the head to W8A8 or repeating the cast on fallback.
            state['floating_head_calls'] = state.get('floating_head_calls', 0) + 1
            return torch.nn.functional.linear(last, weight, None)
        params = getattr(weight, '_params', None)
        if (isinstance(weight, QuantizedTensor)
                and weight._layout_cls == 'TensorWiseINT8Layout'
                and weight._qdata.device == x.device
                and weight._qdata.dtype == torch.int8
                and tuple(weight._qdata.shape) == (151936, 2560)
                and weight._qdata.is_contiguous()
                and not getattr(params, 'transposed', False)
                and params.scale.numel() in (1, 151936)
                and params.scale.device == x.device
                and getattr(params, 'convrot_groupsize', 256) == 256):
            try:
                quant, scale = ik._quant_cached(last, params.convrot, 256)
                out = _gemv(quant, weight._qdata, scale.float().reshape(-1),
                            params.scale.float().reshape(-1).contiguous())
                state['head_calls'] = state.get('head_calls', 0) + 1
                return out.reshape(1, 1, -1)
            except Exception as exc:
                _kernel_failed = True
                logging.warning('[AppleSilicon-Optimized/Qwen3VL] head kernel unavailable; '
                                'retaining native projection: %s', exc)
        # We already own the same cast/offload context used by BaseGenerate.
        # Reuse its weight for an unsupported kernel shape instead of recasting.
        return torch.nn.functional.linear(last, weight, None)


def _rope(original, q, k, freqs_cis):
    global _library, _kernel_failed
    state = _scope.get()
    pending = {} if state is None else state.get('norm_pending', {})
    nq, nk = pending.pop(id(q), None), pending.pop(id(k), None)
    def fallback():
        return original(_finish_norm(q, nq), _finish_norm(k, nk), freqs_cis)
    if (state is None or _kernel_failed or os.environ.get('ASFP8_QWEN3VL_ROPE','1') != '1'
            or type(q) is not torch.Tensor or type(k) is not torch.Tensor
            or q.device.type != 'mps' or k.device != q.device
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype
            or tuple(q.shape) != (1,32,1,128) or tuple(k.shape) != (1,8,1,128)
            or not q.is_contiguous() or not k.is_contiguous()):
        return fallback()
    from . import torch_mpp_qcache as qc, yue2_rope_mps as yr
    if not qc._hardware() or not qc._torch_build_supported() or not yr._sources_supported():
        return fallback()
    llama, eager, registry = yr._dependencies()
    if state.get('rope_frequencies') is not freqs_cis:
        matrix = llama.rope_matrix(freqs_cis)
        if matrix.ndim == 5:
            matrix = matrix.unsqueeze(0)
        if (tuple(matrix.shape) != (1,1,1,64,2,2) or matrix.dtype != torch.float32
                or matrix.device != q.device or not matrix.is_contiguous()):
            return fallback()
        selected = registry.get_implementation('apply_rope_split_half', kwargs={
            'xq':q, 'xk':k, 'freqs_cis':matrix})
        if selected is not eager.apply_rope_split_half:
            return fallback()
        state['rope_frequencies'], state['rope_matrix'] = freqs_cis, matrix
    try:
        if _library is None:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        if nq is not None and nk is not None and nq[0] is q and nk[0] is k and nq[3] == nk[3]:
            _library.qk_norm_rope(q, k, nq[2], nk[2], state['rope_matrix'], oq, ok,
                                  float(nq[3]), threads=1280, group_size=32)
            state['norm_rope_calls'] = state.get('norm_rope_calls', 0) + 1
        else:
            qn, kn = _finish_norm(q, nq), _finish_norm(k, nk)
            _library.qwen_rope(qn, kn, state['rope_matrix'], oq, ok, threads=2560, group_size=256)
        state['rope_calls'] = state.get('rope_calls', 0) + 1
        return oq, ok
    except Exception as exc:
        _kernel_failed = True
        logging.warning('[AppleSilicon-Optimized/Qwen3VL] RoPE fusion unavailable: %s', exc)
        return fallback()


def install():
    global _installed, _installed_cast
    if _installed or not torch.backends.mps.is_available():
        return
    import comfy.ops as ops
    from comfy.text_encoders.qwen3vl import Qwen3VL
    from comfy_kitchen.tensor import QuantizedTensor
    cast = ops.cast_bias_weight
    signature = inspect.signature(cast)
    required = {'s','input','dtype','device','bias_dtype','offloadable','compute_dtype','want_requant'}
    if set(signature.parameters) != required:
        return  # changed core API: retain native semantics
    @functools.wraps(cast)
    def cached(*a, **kw):
        return _cast(cast, signature, QuantizedTensor, *a, **kw)
    _installed_cast = cached
    ops.cast_bias_weight = cached
    generate = Qwen3VL.generate
    @functools.wraps(generate)
    def wrapped(self, *a, **kw):
        return _generate(generate, self, *a, **kw)
    Qwen3VL.generate = wrapped
    logits = Qwen3VL.logits
    @functools.wraps(logits)
    def head(self, x):
        return _logits(logits, self, x)
    Qwen3VL.logits = head
    from comfy.text_encoders import llama
    rope = llama.apply_rope
    @functools.wraps(rope)
    def fused_rope(q, k, freqs_cis):
        return _rope(rope, q, k, freqs_cis)
    llama.apply_rope = fused_rope
    _installed = True
