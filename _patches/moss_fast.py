"""Exact MOSS-Image BF16 elementwise execution on the tested MPS stack.

SiTU tables are evaluated by PyTorch for every BF16 input bit pattern, retaining
all intermediate rounding. An opt-in exact FC1/SiTU epilogue can eliminate
the wide FFN intermediate. The default RMS/modulation fusion retains PyTorch's
original FP32 reduction and every BF16 rounding boundary. ASFP8_MOSS_RMS_MODE=
native opts into native weighted RMSNorm and changes numerical rounding;
original retains the preceding implementation. Sampler and VAE stay unchanged.
Patches are active only inside an eligible MOSS sampling call.

Operation semantics follow ComfyUI's MOSS implementation (GPL-3.0) and Comfy
Kitchen's eager RMS/RoPE/modulation routines (Apache-2.0). The query-cache shader
uses this pack's attributed PyTorch source adapter. See THIRD_PARTY_NOTICES.md.
"""
import functools
import hashlib
import importlib
from importlib.metadata import version
import logging
import inspect
import math
from threading import RLock
import psutil
import os
import sys
from contextvars import ContextVar
from pathlib import Path

import torch
from . import torch_mpp_qcache

_TAG = '[AppleSilicon-Optimized/MOSS]'
_installed = False
_library = _tables = None
_attention_library = None
_attention_dead = False
_dead = False
_scope = ContextVar('as_moss_exact', default=None)
_stats = {}
_cast_lock = RLock()
_cast_users = 0
_active_cast = None
_direct_ffn_supported = False
_rope_options = False
_OPS_HASH = '2a5e52fe5176be3ac48becebf1678509867f769c1467fa7c7115d1897c16a429'
_BLOCK_HASH = 'c6f63b080fee07147e43608d17b499f3adf7ada81c600b9d65f6b61f23e56998'
_MODEL_HASH = 'c70010e9e51ef21ddd5396ed971c68d6c0c225525cf3ae7cebb79d1071c7f22f'
_MODEL_HASHES = {
    _MODEL_HASH,
    # ComfyUI 8f04fd6e: RoPE options/signature added; block arithmetic unchanged.
    '5384f7e0d1ed36dc78f76ba7f96a600918627f5317cedcd2c58ba92a83c5bbf8',
}
_ROPE_HASH = '630a3d287bf602046935086f9d43eb6db2ec4971913a1cb3fb35bae6ea4ac9ee'
_ADALN_HASH = '0de0af829d63679300911c8765d12a4ddc05be3755f5dd1f6fb8566bd9c3a160'
_SITU_HASH = '2078a73661794535cbcfe08ac947d9c0866ec51be1630b446c9498191ae14430'


def _matches(module, expected):
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
    except (OSError, AttributeError, TypeError):
        return False


def _function_matches(function, expected):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == expected
    except (OSError, TypeError):
        return False


def _enabled(part):
    return (_scope.get() is not None and not torch.is_grad_enabled()
            and os.environ.get('ASFP8_MOSS_FAST', '1') == '1'
            and os.environ.get('ASFP8_MOSS_' + part, '1') == '1')


def _ready():
    global _library, _tables, _dead
    if _dead:
        return False
    if _library is None:
        try:
            library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
            values = torch.arange(65536, device='mps', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            tables = ((25. * torch.tanh(values / 25.)).contiguous(),
                      (4. * torch.tanh(values / 4.) * torch.sigmoid(values)).contiguous())
            _library, _tables = library, tables
        except Exception as exc:
            _dead = True
            logging.warning('%s keeping upstream operations: %s', _TAG, exc)
    return _library is not None


def _bf16(x):
    return (type(x) is torch.Tensor and x.is_mps and x.dtype == torch.bfloat16
            and 0 < x.numel() < 2**31)


def _situ_tensor(x):
    width = x.shape[-1] // 2
    out = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    _library.situ(x, *_tables, out, width, x.stride(-2), threads=out.numel(), group_size=256)
    return out


def _situ(original, x):
    if (not _enabled('SITU') or not _bf16(x) or x.ndim != 3
            or not x.is_contiguous() or x.shape[-1] != 16384 or not _ready()):
        return original(x)
    _scope.get()['situ_calls'] += 1
    return _situ_tensor(x)


def _rope_tensors(q, k, pe):
    q, k = q.contiguous(), k.contiguous()
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    _library.rope(q, k, pe.contiguous(), qo, ko, q.shape[2], q.shape[1],
                  threads=q.numel(), group_size=256)
    return qo, ko


def _qk_rope_tensors(q, k, qw, kw, pe, eps):
    # Native D=128 weighted RMS reduction, retaining the BF16 store before RoPE.
    # Keep larger batches/shapes and unusual weights on the existing route.
    if (not _enabled('QK_FUSION') or q.ndim != 4
            or q.shape != k.shape or q.shape[0] != 1
            or not 1 <= q.shape[1] <= 6656 or q.shape[2:] != (24, 128)
            or not all(_bf16(t) for t in (q, k))
            or not all(type(t) in (torch.Tensor, torch.nn.Parameter)
                       and t.is_mps and t.dtype == torch.bfloat16 for t in (qw, kw))
            or not all(t.stride(-1) == 1 and t.stride(-2) == 128
                       and 0 < t.stride(1) < 2**31 for t in (q, k))
            or qw.shape != (128,) or kw.shape != (128,)
            or not qw.is_contiguous() or not kw.is_contiguous()
            or pe.dtype != torch.float32 or not pe.is_mps
            or pe.shape != (1, q.shape[1], 1, 64, 2, 2)):
        return None
    qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ko = torch.empty_like(qo)
    _library.rms_rope_split(q, k, qw, kw, pe.contiguous(), qo, ko, 24,
                            q.stride(1), k.stride(1), eps,
                            threads=(q.shape[1] * 32, 24, 2), group_size=(32, 1, 1))
    state = _scope.get()
    state['qk_fusion_calls'] = state.get('qk_fusion_calls', 0) + 1
    return qo, ko


def _attention(original, module, q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options):
    if (not _enabled('ROPE') or not all(_bf16(t) for t in (q, k))
            or q.ndim != 3 or q.shape != k.shape or heads != 24
            or q.shape[-1] != 3072 or pe.dtype != torch.float32 or not pe.is_mps
            or pe.shape != (1, q.shape[1], 1, 64, 2, 2) or not _ready()):
        return original(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options)
    import comfy.ops as ops
    b, n, _ = q.shape
    q, k = q.view(b, n, heads, 128), k.view(b, n, heads, 128)
    qw, _, qs = ops.cast_bias_weight(q_norm, q, offloadable=True)
    try:
        kw, _, ks = ops.cast_bias_weight(k_norm, k, offloadable=True)
        try:
            fused = _qk_rope_tensors(q, k, qw, kw, pe, q_norm.eps)
            if fused is None:
                # Upstream applies q_norm.eps to both tensors.
                q = torch.nn.functional.rms_norm(q, (128,), qw, q_norm.eps)
                k = torch.nn.functional.rms_norm(k, (128,), kw, q_norm.eps)
                q, k = _rope_tensors(q, k, pe)
            else:
                q, k = fused
        finally:
            ops.uncast_bias_weight(k_norm, kw, None, ks)
    finally:
        ops.uncast_bias_weight(q_norm, qw, None, qs)
    _scope.get()['rope_calls'] += 1
    return module.optimized_attention(module.AttentionTensorContainer(q.flatten(2)),
                                      module.AttentionTensorContainer(k.flatten(2)),
                                      module.AttentionTensorContainer(v), heads,
                                      transformer_options=transformer_options, preferred_attention=comfy_attention)


def _mod_tensor(x, scale, shift):
    out = torch.empty_like(x)
    _library.mod(x, scale, shift, out, x.shape[-1], x.shape[-2], threads=x.numel(), group_size=256)
    return out


def _rms_mod_tensor(x, folded, shift, eps):
    # Keep PyTorch's original FP32 reduction/add/rsqrt. Only eliminate full-size
    # cast and normalization intermediates; retain each BF16 rounding boundary.
    squared = torch.empty_like(x, dtype=torch.float32)
    _library.square_bf16(x, squared, threads=x.numel(), group_size=256)
    inv = torch.rsqrt(squared.mean(dim=-1, keepdim=True).add_(eps))
    out = torch.empty_like(x)
    _library.rms_mod_tail(x, inv, folded, shift, out, x.shape[-1], x.shape[-2],
                          threads=x.numel(), group_size=256)
    return out


def _mod(original, x, norm, scale, shift):
    if (not _enabled('MODULATION') or not all(_bf16(t) and t.is_contiguous() for t in (x, scale, shift))
            or x.ndim != 3 or x.shape[-1] != 3072
            or scale.shape != shift.shape or scale.shape != (x.shape[0], 1, 3072)
            or not _ready()):
        return original(x, norm, scale, shift)
    import comfy.ops as ops
    w, _, stream = ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        folded = w * (1 + scale) - 1
        mode = os.environ.get('ASFP8_MOSS_RMS_MODE', 'exact')
        if mode == 'exact':
            result = _rms_mod_tensor(x, folded, shift, norm.eps)
        else:
            # Native weighted RMSNorm changes reduction rounding: opt-in only.
            unit = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device) if mode == 'native' else None
            normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=unit, eps=norm.eps)
            result = _mod_tensor(normalized, folded, shift)
        state = _scope.get()
        key = 'rms_' + (mode if mode in ('exact', 'native') else 'original') + '_calls'
        state[key] = state.get(key, 0) + 1
    finally:
        ops.uncast_bias_weight(norm, w, None, stream)
    _scope.get()['modulation_calls'] += 1
    return result


def _eligible(guider, model_type):
    model = getattr(getattr(guider, 'inner_model', None), 'diffusion_model', None)
    patcher = getattr(guider, 'model_patcher', None)
    options = getattr(guider, 'model_options', {})
    tr = options.get('transformer_options', {})
    if (type(model) is not model_type or patcher is None
            or getattr(patcher.load_device, 'type', None) != 'mps'
            or torch.is_grad_enabled() or _scope.get() is not None
            or os.environ.get('ASFP8_MOSS_FAST', '1') != '1'
            or any(getattr(patcher, k, None) for k in ('hook_patches', 'current_hooks', 'forced_hooks', 'injections'))
            or any(options.get(k) for k in ('model_function_wrapper', 'wrappers', 'multigpu_thread_pool'))
            or any(tr.get(k) for k in ('patches', 'patches_replace', 'wrappers', 'optimized_attention_override'))):
        return False
    from .flux3_fast import _has_hooks
    if _has_hooks(getattr(guider, 'conds', {})):
        return False
    return not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())


def _specialized_attention(moss_attention_mps, mode, q, k, v, tensors, heads, shape):
    qq, kk, vv = [t.unflatten(-1, (heads, 128)).transpose(1, 2) for t in tensors]
    out = moss_attention_mps.maybe_attention(qq, kk, vv)
    if out is None:
        return None
    state = _scope.get()
    key = mode + '_attention_calls'
    if not state.get(key):
        contract = 'exact tile/layout' if mode == 'tile32' else 'approximate dense; all keys retained'
        logging.info('%s %s attention enabled (%s).', _TAG, mode, contract)
    state[key] = state.get(key, 0) + 1
    for container in (q, k, v):
        container.take()
    return out.transpose(1, 2).reshape(shape)


def _optimized_attention(original, q, k, v, heads, **kwargs):
    global _attention_library, _attention_dead
    preferred = kwargs.get('preferred_attention')
    options = kwargs.get('transformer_options', {})
    if (not _enabled('ATTENTION') or _attention_dead or heads != 24
            or os.environ.get('ASFP8_ATTN_BACKEND', 'pytorch') != 'pytorch'
            or kwargs.keys() - {'preferred_attention', 'transformer_options'}
            or getattr(preferred, 'function', None) is not None
            or options.get('optimized_attention_override')
            or not all(hasattr(t, 'peek') for t in (q, k, v))):
        return original(q, k, v, heads, **kwargs)
    tensors = [t.peek() for t in (q, k, v)]
    shape = tensors[0].shape
    if (not all(_bf16(t) and t.shape == shape and t.stride(-1) == 1 for t in tensors)
            or len(shape) != 3 or shape[0] != 1 or shape[2] != 3072
            # Includes the validated 1248-square SFT workflow and prompt-token tails.
            or not 4096 <= shape[1] <= 6656):
        return original(q, k, v, heads, **kwargs)
    import comfy.ldm.modules.attention as attention
    from . import attn_router_mps, torch_attention
    native = inspect.unwrap(original) is inspect.unwrap(attention.attention_pytorch)
    routed = (original is attn_router_mps._router
              and torch_attention.supports(*tensors, heads))
    if not (native or routed):
        # Preserve an explicitly selected CLI backend or a third-party dispatcher.
        return original(q, k, v, heads, **kwargs)
    from . import moss_attention_mps
    mode = moss_attention_mps.selected_mode()
    # Approximate dense INT8 modes are explicit opt-ins and take precedence.
    if mode not in ('exact', 'tile32'):
        out = _specialized_attention(moss_attention_mps, mode, q, k, v, tensors, heads, shape)
        if out is not None:
            return out
    # Experimental and deliberately separate from the exact/default path.
    # Only reviewed MOSS calls with an explicit checkpoint request qualify.
    if os.environ.get('ASFP8_MOSS_KITCHEN_INT8', '0') == '1':
        from . import kitchen_attention_mps
        state = _scope.get()
        if (not state.get('kitchen_int8_disabled')
                and kitchen_attention_mps.requested(getattr(preferred, 'config', None))):
            qq, kk, vv = [t.unflatten(-1, (heads, 128)).transpose(1, 2) for t in tensors]
            try:
                out = kitchen_attention_mps.attention(qq, kk, vv)
            except (ValueError, SyntaxError) as exc:
                state['kitchen_int8_disabled'] = True
                logging.warning('%s experimental INT8 attention unavailable; keeping exact: %s', _TAG, exc)
            else:
                for container in (q, k, v):
                    container.take()
                if not state.get('kitchen_int8_calls'):
                    logging.info('%s experimental Kitchen-style INT8 attention enabled; non-bit-exact, CUDA parity unverified.', _TAG)
                state['kitchen_int8_calls'] = state.get('kitchen_int8_calls', 0) + 1
                return out.transpose(1, 2).reshape(shape)
    # Default: exact 32-row query tiles, byte-identical to the 64-row kernel below.
    if mode == 'tile32':
        out = _specialized_attention(moss_attention_mps, mode, q, k, v, tensors, heads, shape)
        if out is not None:
            return out
    try:
        if _attention_library is None:
            _attention_library = torch.mps.compile_shader(torch_mpp_qcache._source())
        qq, kk, vv = [t.unflatten(-1, (24, 128)).transpose(1, 2) for t in tensors]
        params = torch_mpp_qcache._parameters(tuple(qq.shape), tuple(kk.shape), qq.stride(), kk.stride(), vv.stride(), 128**-.5)
        out = torch.empty(qq.shape, device=qq.device, dtype=qq.dtype)
        _attention_library.custom_attn(qq, kk, vv, out, params, params, params,
                                      threads=(((shape[1]+63)//64)*128, 24, 1), group_size=(128, 1, 1))
    except Exception as exc:
        _attention_dead = True
        logging.warning('%s keeping native SDPA: %s', _TAG, exc)
        return original(q, k, v, heads, **kwargs)
    for container in (q, k, v):
        container.take()
    _scope.get()['attention_calls'] += 1
    return out.transpose(1, 2).reshape(shape)


def _position(original, module, h, w, text_len, device, *args, **kwargs):
    # Newer core passes transformer_options to rope(). Preserve all arguments
    # and leave non-default positional controls uncached, avoiding stale positions.
    options = args[0] if args else kwargs.get('transformer_options', {})
    if (not _enabled('REUSE') or len(args) > 1
            or kwargs.keys() - {'transformer_options'}
            or not isinstance(options, dict) or options.get('rope_options')):
        return original(module, h, w, text_len, device, *args, **kwargs)
    state = _scope.get()
    key = (h, w, text_len, str(device))
    value = state['positions'].get(key)
    if value is None:
        value = original(module, h, w, text_len, device, *args, **kwargs)
        size = value.numel() * value.element_size()
        if len(state['positions']) < 4 and state['reuse_bytes'] + size <= 32 * 2**20:
            state['positions'][key] = value
            state['reuse_bytes'] += size
    else:
        state['position_hits'] += 1
    return value


def _text(original, module, x, pe, transformer_options={}):
    if (not _enabled('REUSE') or transformer_options.get('optimized_attention_override')
            or transformer_options.get('wrappers') or transformer_options.get('patches')):
        return original(module, x, pe, transformer_options)
    state = _scope.get()
    for xx, pp, out in state['text']:
        if (x.shape == xx.shape and x.dtype == xx.dtype and x.device == xx.device
                and pe.shape == pp.shape and torch.equal(x, xx) and torch.equal(pe, pp)):
            state['text_hits'] += 1
            return out
    out = original(module, x, pe, transformer_options)
    size = sum(t.numel() * t.element_size() for t in (x, pe, out))
    if len(state['text']) < 4 and state['reuse_bytes'] + size <= 32 * 2**20:
        state['text'].append((x.detach().clone(), pe.detach().clone(), out.detach().clone()))
        state['reuse_bytes'] += size
    return out


def _cache_budget():
    try:
        gib = float(os.environ.get('ASFP8_MOSS_CACHE_GIB', '4'))
        if not math.isfinite(gib) or gib <= 0:
            return 0
        return int(min(gib * 2**30, 8 * 2**30, torch.mps.recommended_max_memory() * .05,
                       psutil.virtual_memory().available * .1))
    except (ValueError, RuntimeError):
        return 0


def _cached_cast(original, signature, *args, **kwargs):
    state = _scope.get()
    module = args[0] if args else kwargs.get('s')
    if (state is None or id(module) not in state['modules'] or not _enabled('WEIGHT_CACHE')
            or state['budget'] <= 0 or hasattr(module, '_v')
            or getattr(module, 'weight_function', []) or getattr(module, 'bias_function', [])
            or getattr(module, 'weight_lowvram_function', None) is not None
            or getattr(module, 'bias_lowvram_function', None) is not None):
        return original(*args, **kwargs)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    p = bound.arguments
    x = p['input']
    device = p['device'] if p['device'] is not None else getattr(x, 'device', None)
    if device is None or torch.device(device).type != 'mps':
        return original(*args, **kwargs)
    from .flux3_fast import _weight_key
    dtype = p['dtype'] or getattr(x, 'dtype', None)
    key = (id(module), _weight_key(module.weight), _weight_key(module.bias),
           dtype, str(device), p['bias_dtype'] or dtype, p['compute_dtype'], p['want_requant'])
    entry = state['cache'].get(key)
    if entry is not None:
        state['cache_hits'] += 1
        return (*entry[:2], None) if p['offloadable'] else entry[:2]
    result = original(*args, **kwargs)
    w, b = result[:2]
    if type(w) is not torch.Tensor or not w.is_floating_point() or (b is not None and type(b) is not torch.Tensor):
        return result
    size = w.numel() * w.element_size() + (0 if b is None else b.numel() * b.element_size())
    if (state['cache_bytes'] + size <= state['budget']
            and torch.mps.driver_allocated_memory() + size < torch.mps.recommended_max_memory() * .65):
        # Own the data; never retain a borrowed offload buffer as the cached value.
        state['cache'][key] = (w.detach().clone(), None if b is None else b.detach().clone(),
                               module.weight, module.bias)
        state['cache_bytes'] += size
        state['cache_misses'] += 1
    return result


def _owned_dequantizing_cast(cast):
    # Inspect identities, not functools.wraps' copied names. Unknown wrappers
    # may change requantization semantics and must retain the ordinary path.
    from . import ops_bias_fp8, qwen3vl_decode, yue2_fast
    owned = ops_bias_fp8._installed_cast
    if owned is None:
        return False
    allowed = [_active_cast]
    for module in (qwen3vl_decode, yue2_fast):
        if module._scope.get() is None:
            allowed.append(module._installed_cast)
    seen = set()
    while cast is not owned:
        if cast is None or id(cast) in seen or not any(cast is fn for fn in allowed):
            return False
        seen.add(id(cast))
        cast = getattr(cast, '__wrapped__', None)
    return True


def _linear_input_act(original, linear, x, input_act, *args, **kwargs):
    # Our active FP8-compatibility cast also dequantizes INT8 weights. On the
    # reviewed core, the fused quantized helper then
    # returns NotImplemented, and linear_input_act calls Linear.forward anyway.
    # Skip only that provably unused preparation, retaining module-owned input
    # processing and the exact same SiTU / INT8 arithmetic.
    state = _scope.get()
    if (not _direct_ffn_supported or not _enabled('FFN_DISPATCH')
            or args or kwargs or input_act != 'situ' or state is None
            or id(linear) not in state['modules'] or not _bf16(x)
            or x.ndim != 3 or x.shape[-1] != 16384 or not x.is_contiguous()
            or hasattr(linear, '_v') or getattr(linear, 'bias', None) is not None
            or any(getattr(linear, k, None) for k in (
                'weight_function', 'bias_function', 'weight_lowvram_function',
                'bias_lowvram_function', '_forward_hooks', '_forward_pre_hooks'))):
        return original(linear, x, input_act, *args, **kwargs)
    from comfy_kitchen.tensor import QuantizedTensor
    weight = getattr(linear, 'weight', None)
    if (not isinstance(weight, QuantizedTensor)
            or getattr(weight, '_layout_cls', None) != 'TensorWiseINT8Layout'
            or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or weight.device != x.device
            or tuple(weight.shape) != (3072, 8192)):
        return original(linear, x, input_act, *args, **kwargs)
    import comfy.ops as ops
    if not _owned_dequantizing_cast(ops.cast_bias_weight):
        return original(linear, x, input_act, *args, **kwargs)
    ops.run_every_op()
    if os.environ.get('ASFP8_MOSS_SITU_QUANT', '0') == '1' and _ready():
        from . import moss_situ_quant_mps
        result = moss_situ_quant_mps.linear(linear, x, _tables)
        if result is not None:
            if not state.get('situ_quant_calls'):
                logging.info('%s experimental fused SiTU/ConvRot/INT8 enabled; non-bit-exact rotation.', _TAG)
            state['situ_quant_calls'] = state.get('situ_quant_calls', 0) + 1
            return result
    state['direct_ffn_calls'] = state.get('direct_ffn_calls', 0) + 1
    return linear(ops._eager_input_act(x, input_act))


def _block(original, module, layer, x, mod, pe, transformer_options={}):
    if os.environ.get('ASFP8_MOSS_FFN_EPILOGUE', '0') != '1':
        return original(layer, x, mod, pe, transformer_options)
    from . import moss_ffn_mps
    return moss_ffn_mps.block(original, module, sys.modules[__name__], layer,
                              x, mod, pe, transformer_options)


def _joining():
    # The opt-in FFN epilogue replaces Block.forward, which joined rows bypass.
    return _enabled('CFG_JOIN') and os.environ.get('ASFP8_MOSS_FFN_EPILOGUE', '0') != '1'


def _can_join(original, a, b):
    """Let the two CFG branches share a model call despite different prompt lengths.

    MOSS positions depend on each prompt's own token count, so ordinary
    batching (equal conditioning shapes) never pairs positive and negative.
    """
    if not _joining():
        return original(a, b)
    if (set(a.conditioning) != {'c_crossattn'} or set(b.conditioning) != {'c_crossattn'}
            or a.input_x.shape[0] != 1
            or any(v is not None for v in (a.control, b.control, a.patches, b.patches,
                                           getattr(a, 'hooks', None), getattr(b, 'hooks', None),
                                           getattr(a, 'area', None), getattr(b, 'area', None)))):
        return original(a, b)
    ta, tb = a.conditioning['c_crossattn'].cond, b.conditioning['c_crossattn'].cond
    return (ta.shape[0] == tb.shape[0] == 1 and ta.shape[2:] == tb.shape[2:]
            and ta.device == tb.device
            and original(a._replace(conditioning={}), b._replace(conditioning={})))


def _join(original, conds, device=None):
    if not _joining() or len(conds) != 2 or any(set(c) != {'c_crossattn'} for c in conds):
        return original(conds, device)
    texts = [c['c_crossattn'].cond for c in conds]
    lengths = [t.shape[1] for t in texts]
    longest = max(lengths)
    # Transport padding only; every branch is trimmed back to its own length.
    context = torch.cat([torch.nn.functional.pad(t, (0, 0, 0, longest - t.shape[1])) for t in texts])
    if device is not None:
        context = context.to(device)
    return {'c_crossattn': context, 'moss_cfg_lengths': lengths}


def _int8_projections(model):
    # Joined rows are exact because INT8 accumulation is independent of the
    # row count; a floating matmul may round differently for a taller input.
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError:
        return False
    return all(isinstance(getattr(layer.proj[name], 'weight', None), QuantizedTensor)
               and getattr(layer.proj[name].weight, '_layout_cls', None) == 'TensorWiseINT8Layout'
               for layer in model.layers for name in ('qkv', 'o', 'ffn_up_gate', 'ffn_down'))


def _joined_block(m, layer, x, mod, branches, sizes):
    # Block.forward with token rows joined for every projection; attention,
    # positions and options stay per branch.
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = m.modulated_rms_norm(x, layer.norm1, scale_a, shift_a)
    outs = [m.attention(*qkv.chunk(3, dim=-1), layer.q_norm, layer.k_norm, pe, layer.num_heads,
                        layer.comfy_attention, options)
            for qkv, (pe, options) in zip(layer.proj['qkv'](y).split(sizes, 1), branches)]
    x = m.gated_residual(x, layer.proj['o'](torch.cat(outs, 1)), gate_a)
    y = m.modulated_rms_norm(x, layer.norm2, scale_f, shift_f)
    return m.gated_residual(x, m.comfy.ops.linear_input_act(
        layer.proj['ffn_down'], layer.proj['ffn_up_gate'](y), 'situ'), gate_f)


def _joined_forward(original, m, model, x, timestep, context, transformer_options={}, **kwargs):
    lengths = kwargs.pop('moss_cfg_lengths', None)
    if lengths is None:
        return original(model, x, timestep, context, transformer_options, **kwargs)

    def branch_options(j):
        options = transformer_options.copy()
        for key in ('cond_or_uncond', 'uuids'):
            if key in options:
                options[key] = options[key][j:j + 1]
        return options

    # Upstream _forward ignores extra keywords (model_base always passes control).
    if (not _joining() or x.shape[0] != 2 or len(lengths) != 2
            or not all(layer.fused_qkv for layer in model.layers) or not _int8_projections(model)
            or transformer_options.get('patches_replace', {}).get('dit')
            or not torch.equal(timestep[0], timestep[1])):
        # Separate branch calls, exactly as unjoined CFG would run them.
        return torch.cat([original(model, x[j:j + 1], timestep[j:j + 1], context[j:j + 1, :n],
                                   branch_options(j), **kwargs)
                          for j, n in enumerate(lengths)])
    B, _, H, W = x.shape
    device = x.device
    states, branches = [], []
    for j, n in enumerate(lengths):
        options = branch_options(j)
        options.update(total_blocks=len(model.layers), block_type='single')
        pe = model.rope(H, W, n, device, options) if _rope_options else model.rope(H, W, n, device)
        vec = model.time_embedding(m.timestep_embedding(timestep[j:j + 1], 256).to(x.dtype))
        mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head = model.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)
        txt = model.text_embedder(context[j:j + 1, :n], pe[:, H * W:], options)
        states.append(torch.cat((model.video_embedder(x[j:j + 1].flatten(2).transpose(1, 2)), txt), dim=1))
        branches.append((pe, options))
    del txt
    sizes = [s.shape[1] for s in states]
    # Equal timesteps give byte-identical modulation for both branches.
    joined = torch.cat(states, 1)
    del states
    queue = m.comfy.model_prefetch.make_prefetch_queue(list(model.layers), device, transformer_options)
    m.comfy.model_prefetch.malloc_graph_begin(device)
    for i, layer in enumerate(model.layers):
        m.comfy.model_prefetch.prefetch_queue_pop(queue, device, layer, x.dtype, malloc_scope='block')
        for _, options in branches:
            options['block_index'] = i
        joined = _joined_block(m, layer, joined, mod, branches, sizes)
    m.comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope='block')
    m.comfy.model_prefetch.malloc_graph_end()
    _scope.get()['joined_cfg_calls'] = _scope.get().get('joined_cfg_calls', 0) + 1
    head_shift, head_scale = head
    outs = [model.final_linear_video(m.modulated_rms_norm(y[:, :H * W], model.final_norm_video, head_scale, head_shift))
            for y in joined.split(sizes, 1)]
    return torch.cat([out.transpose(1, 2).reshape(1, model.latent_channels, H, W) for out in outs])


def _acquire_cast():
    import comfy.ops as ops
    global _cast_users, _active_cast
    with _cast_lock:
        if _cast_users == 0:
            original = ops.cast_bias_weight
            signature = inspect.signature(original)
            @functools.wraps(original)
            def cast(*a, **k):
                if _scope.get() is None:
                    return original(*a, **k)
                return _cached_cast(original, signature, *a, **k)
            _active_cast = cast
            ops.cast_bias_weight = cast
        _cast_users += 1
    def release():
        global _cast_users, _active_cast
        with _cast_lock:
            _cast_users -= 1
            if _cast_users == 0:
                if ops.cast_bias_weight is _active_cast:
                    ops.cast_bias_weight = _active_cast.__wrapped__
                _active_cast = None
    return release


def _sample(original, model_type, guider, *args, **kwargs):
    if not _eligible(guider, model_type):
        return original(guider, *args, **kwargs)
    model = guider.inner_model.diffusion_model
    state = dict(situ_calls=0, rope_calls=0, modulation_calls=0, attention_calls=0,
                 positions={}, text=[], position_hits=0, text_hits=0, reuse_bytes=0,
                 modules={id(block.proj['ffn_down']) for block in model.layers},
                 cache={}, cache_hits=0, cache_misses=0, cache_bytes=0, budget=_cache_budget())
    token = _scope.set(state)
    from . import int8_linear_kernel_mps
    schedule = int8_linear_kernel_mps.moss_gemm_schedule.set(_enabled('GEMM_SCHEDULE'))
    release = None
    try:
        if _enabled('WEIGHT_CACHE') and state['budget'] > 0:
            release = _acquire_cast()
        return original(guider, *args, **kwargs)
    finally:
        int8_linear_kernel_mps.moss_gemm_schedule.reset(schedule)
        _scope.reset(token)
        if release is not None:
            release()
        _stats.clear()
        _stats.update({k: v for k, v in state.items() if k not in ('modules', 'cache', 'positions', 'text')})
        state['cache'].clear()
        state['positions'].clear()
        state['text'].clear()


def install():
    global _installed, _direct_ffn_supported, _rope_options
    if (_installed or os.environ.get('ASFP8_MOSS_FAST', '1') != '1'
            or os.environ.get('ASFP8_DIT_EXT') != '1'
            or not torch.backends.mps.is_available()
            or torch.__version__.split('+')[0] != '2.14.1'
            or not torch_mpp_qcache._torch_build_supported() or not torch_mpp_qcache._hardware()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return
    try:
        if version('comfy-kitchen') not in ('0.2.38', '0.2.39'):
            return
        import comfy.ldm.moss_image.model as m
        import comfy.ops as ops
        import comfy.samplers as samplers
        import comfy_kitchen.backends.eager.rope as rope
        adaln = importlib.import_module('comfy_kitchen.backends.eager.adaln')
        if not (any(_matches(m, sha) for sha in _MODEL_HASHES) and _matches(rope, _ROPE_HASH)
                and _matches(adaln, _ADALN_HASH)
                and _function_matches(getattr(ops, '_situ_eager', None), _SITU_HASH)
                and ops.INPUT_ACT_EAGER.get('situ') is ops._situ_eager):
            return
    except ImportError:
        return
    _direct_ffn_supported = _matches(ops, _OPS_HASH)
    original_input_act = ops.linear_input_act
    ops.linear_input_act = functools.wraps(original_input_act)(
        lambda *a, **k: _linear_input_act(original_input_act, *a, **k))
    original_sample = samplers.CFGGuider.inner_sample
    @functools.wraps(original_sample)
    def sample(self, *a, **k):
        return _sample(original_sample, m.MOSSImage, self, *a, **k)
    samplers.CFGGuider.inner_sample = sample
    original_situ = ops.INPUT_ACT_EAGER['situ']
    ops.INPUT_ACT_EAGER['situ'] = functools.wraps(original_situ)(lambda x: _situ(original_situ, x))
    original_attention = m.attention
    m.attention = functools.wraps(original_attention)(lambda *a, **k: _attention(original_attention, m, *a, **k))
    original_mod = m.modulated_rms_norm
    m.modulated_rms_norm = functools.wraps(original_mod)(lambda *a, **k: _mod(original_mod, *a, **k))
    original_optimized = m.optimized_attention
    m.optimized_attention = functools.wraps(original_optimized)(lambda *a, **k: _optimized_attention(original_optimized, *a, **k))
    original_position = m.MOSSImage.rope
    m.MOSSImage.rope = functools.wraps(original_position)(lambda *a, **k: _position(original_position, *a, **k))
    original_text = m.TextEmbedder.forward
    m.TextEmbedder.forward = functools.wraps(original_text)(lambda *a, **k: _text(original_text, *a, **k))
    _rope_options = 'transformer_options' in inspect.signature(m.MOSSImage.rope).parameters
    original_can_concat, original_cat = samplers.can_concat_cond, samplers.cond_cat
    samplers.can_concat_cond = functools.wraps(original_can_concat)(lambda a, b: _can_join(original_can_concat, a, b))
    samplers.cond_cat = functools.wraps(original_cat)(lambda c, device=None: _join(original_cat, c, device))
    original_forward = m.MOSSImage._forward
    m.MOSSImage._forward = functools.wraps(original_forward)(
        lambda self, *a, **k: _joined_forward(original_forward, m, self, *a, **k))
    original_block = m.Block.forward
    if _function_matches(original_block, _BLOCK_HASH):
        m.Block.forward = functools.wraps(original_block)(
            lambda *a, **k: _block(original_block, m, *a, **k))
    _installed = True
    logging.info('%s exact SiTU/RoPE/modulation, query reuse, joined CFG rows, INT8 tile schedule and bounded sampling caches enabled.', _TAG)
