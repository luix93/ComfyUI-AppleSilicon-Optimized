"""Exact Fibo 2.0 sampling routes for M5 Ultra / PyTorch 2.14.1 MPS.

Byte-identical to the pack's previous Fibo 2.0 output:

- Joined CFG rows (ASFP8_FIBO2_CFG_JOIN): when the positive and negative
  prompts give the same gist count, both branches run as one batch-2 forward.
  The Perceiver, text cross-attention, BF16 embedder, final layer and the
  per-block modulation stay per branch. The INT8 projections (integer
  accumulation), RMSNorm, RoPE, attention and elementwise ops are row/batch
  independent; this was verified byte for byte on the real model.
- Fused glue (ASFP8_FIBO2_GLUE): RMSNorm+modulate and RMSNorm+gate+residual
  in one pass each, with PyTorch's RMSNorm reduction order and every BF16
  rounding of the upstream chain.
- tile32 attention (ASFP8_FIBO2_ATTN): the hash-checked PyTorch MPP kernel
  with 32-row query tiles; key tiles, MMA arithmetic, softmax and reduction
  order are unchanged.

Inside a Fibo 2.0 sample only; anything unexpected keeps the upstream forward.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import inspect
import logging
import os
from pathlib import Path

import torch

from . import torch_mpp_qcache

_TAG = '[AppleSilicon-Optimized/fibo2]'
_MODEL_HASHES = {'f679cade565b098b9d9a9279a52d4350148423f454e34938ee4993724b223889'}
_LUMINA_HASHES = {
    'modulate': 'b77735f51a16eb0104eb663bfe2c04ce239fe922a5e4523186cb48ded48d6a03',
    'apply_gate': '58d7dc0f55c1734bcf35ba021574649e551576d5dec4f640e6f2f7b1d66fc124',
}
_installed = False
_library = None
_dead = False
_attention_dead = False
_scope = ContextVar('as_fibo2_exact', default=None)
_stats = {}
_f2 = None


def _enabled(part):
    return (_scope.get() is not None and not torch.is_grad_enabled()
            and os.environ.get('ASFP8_FIBO2_FAST', '1') == '1'
            and os.environ.get('ASFP8_FIBO2_' + part, '1') == '1')


def _count(key):
    state = _scope.get()
    state[key] = state.get(key, 0) + 1


def _ready():
    global _library, _dead
    if _library is None and not _dead:
        try:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead = True
            logging.warning('%s keeping upstream operations: %s', _TAG, exc)
    return _library is not None


def _norm_weight(norm, x):
    w = getattr(norm, 'weight', None)
    if (w is None or type(w) is not torch.nn.Parameter and type(w) is not torch.Tensor
            or w.dtype != torch.bfloat16 or w.device != x.device or w.shape != (x.shape[-1],)
            or getattr(norm, 'weight_function', None) or getattr(norm, 'bias_function', None)
            or norm._forward_hooks or norm._forward_pre_hooks):
        return None
    return w


def _threads(width):
    # PyTorch RMSNorm.metal: ceil(D/128) SIMD groups per row up to 4096 columns.
    return ((width + 127) // 128) * 32


def _norm_mod(x, norm, factor, span):
    w = _norm_weight(norm, x)
    if w is None or not _enabled('GLUE') or not _ready():
        return None
    out = torch.empty_like(x)
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    _library.norm_mod(x, w, factor, out, width, x.shape[1], span[0], span[1], float(norm.eps),
                      threads=(rows * _threads(width), 1, 1), group_size=(_threads(width), 1, 1))
    return out


def _norm_gate_residual(x, y, norm, gate, span):
    w = _norm_weight(norm, y)
    if w is None or not _enabled('GLUE') or not _ready() or not y.is_contiguous() or y.shape != x.shape:
        return False
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    _library.norm_gate_residual(x, y, w, gate, width, x.shape[1], span[0], span[1], float(norm.eps),
                                threads=(rows * _threads(width), 1, 1), group_size=(_threads(width), 1, 1))
    return True


def _modulated(x, norm, scale, span):
    # Upstream modulate: (1 + scale) rounded to BF16, then one BF16 multiply per row group.
    factor = (1 + scale).contiguous()
    out = _norm_mod(x, norm, factor, span)
    if out is not None:
        return out
    return _f2.modulate(norm(x), scale, timestep_zero_index=None if span[1] == 0 else [span])


def _gated_residual(x, y, norm, gate, span):
    gate = gate.tanh().contiguous()
    if _norm_gate_residual(x, y, norm, gate, span):
        return x
    return _f2.residual(x, _f2.apply_gate(gate.unsqueeze(1), norm(_f2.clamp_fp16(y)),
                                          timestep_zero_index=None if span[1] == 0 else [span]))


def _attention_core(q, k, v, heads, options):
    global _attention_dead
    b, h, n, d = q.shape
    if (_enabled('ATTN') and not _attention_dead and d == 128 and k.shape == q.shape == v.shape
            and all(t.is_contiguous() for t in (q, k, v)) and 2048 <= n <= 32768
            and not options.get('optimized_attention_override')):
        try:
            from . import moss_attention_mps
            lib, _ = moss_attention_mps._library('tile32')
            p = torch_mpp_qcache._parameters(tuple(q.shape), tuple(k.shape), q.stride(), k.stride(),
                                             v.stride(), 128**-.5, True)
            out = torch.empty((b, n, h, d), dtype=q.dtype, device=q.device)
            lib.custom_attn(q, k, v, out.transpose(1, 2), p, p, p, threads=(((n + 31) // 32) * 64, h, b),
                            group_size=(64, 1, 1))
            _count('tile32_calls')
            return out.view(b, n, h * d)
        except Exception as exc:
            _attention_dead = True
            logging.warning('%s tile32 attention unavailable; keeping PyTorch attention: %s', _TAG, exc)
    from comfy.ldm.modules.attention import optimized_attention_masked
    return optimized_attention_masked(q, k, v, heads, None, skip_reshape=True, transformer_options=options)


def _self_attention(att, x, freqs_rows, options):
    # The pack's Lumina/Z-Image path: INT8 qkv, fused QK norm + RoPE, attention, INT8 out.
    from . import lumina_attn_fast as lumina
    b, s, _ = x.shape
    qkv = att.qkv(x)
    q, k, v = lumina._kernel.qkv_norm_rope(
        qkv.contiguous(), freqs_rows,
        att.q_norm.weight.to(dtype=torch.bfloat16, device=x.device),
        att.k_norm.weight.to(dtype=torch.bfloat16, device=x.device),
        att.n_local_heads, att.n_local_kv_heads, att.head_dim,
        lumina._eps(att.q_norm, x.dtype), lumina._eps(att.k_norm, x.dtype))
    return att.out(_attention_core(q, k, v, att.n_local_heads, options))


def _block(block, x, freqs_rows, adaln, span, texts, options):
    """Block.forward for a batch of CFG branches that share one timestep.

    adaln holds the single-branch rows (zero, noisy) or (noisy,), so every
    modulation vector is the same GEMM a lone branch would compute.
    """
    scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln).chunk(4, dim=1)
    y = _self_attention(block.attention, _modulated(x, block.attention_norm1, scale_msa, span), freqs_rows, options)
    x = _gated_residual(x, y, block.attention_norm2, gate_msa, span)
    if block.cross_attention is not None:
        for j, text in enumerate(texts):
            xj = x[j:j + 1]
            y = block.cross_attention(block.cross_attn_norm1(xj), text, options)
            _f2.residual(xj, block.cross_attn_norm2(y), _f2.comfy.ops.cast_to_input(block.cross_attn_gate, y).tanh())
    y = block.feed_forward(_modulated(x, block.ffn_norm1, scale_mlp, span))
    return _gated_residual(x, y, block.ffn_norm2, gate_mlp, span)


def _int8_projections(model):
    # Joined rows are exact because INT8 accumulation is independent of the
    # row count; a floating or patched weight may round differently.
    try:
        from comfy_kitchen.tensor import QuantizedTensor
    except ImportError:
        return False
    def plain(layer):
        return (isinstance(getattr(layer, 'weight', None), QuantizedTensor)
                and not getattr(layer, 'weight_function', None) and not getattr(layer, 'bias_function', None))
    return all(plain(b.attention.qkv) and plain(b.attention.out) and plain(b.feed_forward.w13)
               and plain(b.feed_forward.w2) and b.feed_forward.fused
               for b in list(model.noise_refiner) + list(model.layers))


def _supported(model, x, transformer_options):
    from . import lumina_attn_fast as lumina
    return (_scope.get() is not None and x.device.type == 'mps' and x.dtype == torch.bfloat16
            and not torch.is_grad_enabled() and lumina._kernel is not None
            and not transformer_options.get('patches_replace', {}).get('dit')
            and not transformer_options.get('patches')
            and all(b.attention.n_local_heads == b.attention.n_local_kv_heads and b.attention.head_dim == 128
                    and b.attention.qk_norm for b in list(model.noise_refiner) + list(model.layers)))


def _fast(model, x, timestep, contexts, transformer_options, ref_latents):
    f2 = _f2
    B, C, H, W = x.shape
    dtype, device = x.dtype, x.device
    options = transformer_options.copy()
    t = timestep[:1]
    adaln_mixed = model.t_embedder(torch.cat((torch.zeros_like(t), t * 1000)), dtype)
    adaln_noisy = adaln_mixed[1:]

    num_gist = f2.gist_count(contexts[0].shape[1], model.max_gist_tokens)
    plane = model.max_gist_tokens + 1
    ids = [torch.zeros(1, num_gist, 3, dtype=torch.float32, device=device), f2.pos_ids_x(plane, H, W, 1, device, transformer_options=options)]
    ids[0][0, :, 0] = torch.arange(1, num_gist + 1, dtype=torch.float32, device=device)
    for k, ref in enumerate(ref_latents):
        ids.append(f2.pos_ids_x(plane + 1 + k, ref.shape[-2], ref.shape[-1], 1, device, transformer_options=options))
    states = []
    for j in range(B):
        tokens = [x[j:j + 1].flatten(2).transpose(1, 2)]
        tokens += [f2.comfy.utils.repeat_to_batch_size(ref, 1).flatten(2).transpose(1, 2) for ref in ref_latents]
        states.append(model.x_embedder(torch.cat(tokens, dim=1)))
    img = states[0] if B == 1 else torch.cat(states)
    del states, tokens
    freqs_cis = model.rope_embedder(torch.cat(ids, dim=1)).movedim(1, 2)

    def rows(freqs):
        return freqs.expand(B, *freqs.shape[1:]).reshape(B * freqs.shape[1], -1).contiguous()

    n_img = H * W
    refiner_adaln, refiner_span = (adaln_mixed, (0, n_img)) if len(ref_latents) > 0 else (adaln_noisy, (0, 0))
    refiner_rows = rows(freqs_cis[:, num_gist:])
    queue = f2.comfy.model_prefetch.make_prefetch_queue(list(model.noise_refiner), device, transformer_options)
    for block in model.noise_refiner:
        f2.comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype)
        img = _block(block, img, refiner_rows, refiner_adaln, refiner_span, (), options)
    f2.comfy.model_prefetch.prefetch_queue_pop(queue, device, None)
    del refiner_rows

    gists, texts = [], []
    for context in contexts:
        bundles = context.chunk(1 + len(model.injection_layers), dim=-1)
        gists.append(model.perceiver(bundles[0], num_gist, options))
        texts.append(dict(zip(model.injection_layers, bundles[1:])))
    x = torch.cat([torch.cat((g, img[j:j + 1]), dim=1) for j, g in enumerate(gists)])
    del gists, img
    span = (num_gist, num_gist + n_img)
    main_rows = rows(freqs_cis)

    options['total_blocks'] = len(model.layers)
    options['block_type'] = 'single'
    queue = f2.comfy.model_prefetch.make_prefetch_queue(list(model.layers), device, transformer_options)
    f2.comfy.model_prefetch.malloc_graph_begin(device)
    for i, block in enumerate(model.layers):
        f2.comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype, malloc_scope='block')
        options['block_index'] = i
        x = _block(block, x, main_rows, adaln_mixed, span,
                   [text[i] for text in texts] if block.cross_attention is not None else (), options)
    f2.comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope='block')
    f2.comfy.model_prefetch.malloc_graph_end()
    _count('joined_cfg_calls' if B == 2 else 'single_calls')
    outs = [model.final_layer(x[j:j + 1, num_gist:num_gist + n_img], adaln_noisy) for j in range(B)]
    return torch.cat([out.transpose(1, 2).reshape(1, C, H, W) for out in outs])


def _forward(original, model, x, timestep, context, transformer_options={}, ref_latents=[], **kwargs):
    lengths = kwargs.pop('fibo2_cfg_lengths', None)

    def branch_options(j):
        options = transformer_options.copy()
        for key in ('cond_or_uncond', 'uuids'):
            if key in options:
                options[key] = options[key][j:j + 1]
        return options

    def separate():
        # Exactly what unjoined CFG would run, one branch per call.
        return torch.cat([_forward(original, model, x[j:j + 1], timestep[j:j + 1], context[j:j + 1, :n],
                                   branch_options(j), ref_latents, **kwargs)
                          for j, n in enumerate(lengths)])

    if lengths is not None:
        if (x.shape[0] != 2 or len(lengths) != 2 or not _enabled('CFG_JOIN') or ref_latents
                or not _supported(model, x, transformer_options) or not _scope.get().get('join_ok')
                or not torch.equal(timestep[0], timestep[1])
                or len({_f2.gist_count(n, model.max_gist_tokens) for n in lengths}) != 1):
            return separate()
        return _fast(model, x, timestep, [context[j:j + 1, :n] for j, n in enumerate(lengths)],
                     transformer_options, ref_latents)
    # Upstream ignores extra keywords (model_base always passes control).
    if (x.shape[0] != 1 or not _supported(model, x, transformer_options)
            or not (_enabled('GLUE') or _enabled('ATTN'))):
        return original(model, x, timestep, context, transformer_options, ref_latents=ref_latents, **kwargs)
    return _fast(model, x, timestep, [context], transformer_options, ref_latents)


def _can_join(original, a, b):
    """Let the two CFG branches share a model call despite different prompt lengths."""
    state = _scope.get()
    if state is None or not _enabled('CFG_JOIN') or not state.get('join_ok'):
        return original(a, b)
    if (set(a.conditioning) != {'c_crossattn'} or set(b.conditioning) != {'c_crossattn'}
            or a.input_x.shape[0] != 1
            or any(v is not None for v in (a.control, b.control, a.patches, b.patches,
                                           getattr(a, 'hooks', None), getattr(b, 'hooks', None),
                                           getattr(a, 'area', None), getattr(b, 'area', None)))):
        return original(a, b)
    ta, tb = a.conditioning['c_crossattn'].cond, b.conditioning['c_crossattn'].cond
    return (ta.shape[0] == tb.shape[0] == 1 and ta.shape[2:] == tb.shape[2:] and ta.device == tb.device
            and _f2.gist_count(ta.shape[1], state['max_gist']) == _f2.gist_count(tb.shape[1], state['max_gist'])
            and original(a._replace(conditioning={}), b._replace(conditioning={})))


def _join(original, conds, device=None):
    state = _scope.get()
    if (state is None or not _enabled('CFG_JOIN') or not state.get('join_ok') or len(conds) != 2
            or any(set(c) != {'c_crossattn'} for c in conds)):
        return original(conds, device)
    texts = [c['c_crossattn'].cond for c in conds]
    lengths = [t.shape[1] for t in texts]
    longest = max(lengths)
    # Transport padding only; every branch is trimmed back to its own length.
    context = torch.cat([torch.nn.functional.pad(t, (0, 0, 0, longest - t.shape[1])) for t in texts])
    if device is not None:
        context = context.to(device)
    return {'c_crossattn': context, 'fibo2_cfg_lengths': lengths}


def _eligible(guider):
    model = getattr(getattr(guider, 'inner_model', None), 'diffusion_model', None)
    patcher = getattr(guider, 'model_patcher', None)
    options = getattr(guider, 'model_options', {})
    tr = options.get('transformer_options', {})
    if (type(model) is not _f2.Fibo2 or patcher is None
            or getattr(patcher.load_device, 'type', None) != 'mps'
            or torch.is_grad_enabled() or _scope.get() is not None
            or os.environ.get('ASFP8_FIBO2_FAST', '1') != '1'
            or any(getattr(patcher, k, None) for k in ('hook_patches', 'current_hooks', 'forced_hooks', 'injections'))
            or any(options.get(k) for k in ('model_function_wrapper', 'wrappers', 'multigpu_thread_pool'))
            or any(tr.get(k) for k in ('patches', 'patches_replace', 'wrappers', 'optimized_attention_override'))):
        return False
    from .flux3_fast import _has_hooks
    return not _has_hooks(getattr(guider, 'conds', {}))


@contextmanager
def scope(model):
    """A Fibo 2.0 exact-sampling scope (also used by the diagnostics)."""
    state = dict(max_gist=model.max_gist_tokens, join_ok=_int8_projections(model))
    token = _scope.set(state)
    try:
        yield state
    finally:
        _scope.reset(token)
        _stats.clear()
        _stats.update(state)


def _sample(original, guider, *args, **kwargs):
    if not _eligible(guider):
        return original(guider, *args, **kwargs)
    with scope(guider.inner_model.diffusion_model):
        return original(guider, *args, **kwargs)


def _function_matches(function, expected):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == expected
    except (OSError, TypeError):
        return False


def install():
    global _installed, _f2
    if (_installed or os.environ.get('ASFP8_FIBO2_FAST', '1') != '1'
            or os.environ.get('ASFP8_DIT_EXT') != '1'
            or not torch.backends.mps.is_available()
            or torch.__version__.split('+')[0] != '2.14.1'
            or not torch_mpp_qcache._torch_build_supported() or not torch_mpp_qcache._hardware()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return
    try:
        import comfy.ldm.fibo2.model as f2
        import comfy.ldm.lumina.model as lumina_model
        import comfy.samplers as samplers
        from . import lumina_attn_fast
    except ImportError:
        return
    try:
        source_ok = hashlib.sha256(Path(f2.__file__).read_bytes()).hexdigest() in _MODEL_HASHES
    except OSError:
        source_ok = False
    if (not source_ok or not lumina_attn_fast._installed
            or not all(_function_matches(getattr(lumina_model, name), sha) for name, sha in _LUMINA_HASHES.items())):
        return
    _f2 = f2
    original_sample = samplers.CFGGuider.inner_sample
    samplers.CFGGuider.inner_sample = functools.wraps(original_sample)(
        lambda self, *a, **k: _sample(original_sample, self, *a, **k))
    original_can_concat, original_cat = samplers.can_concat_cond, samplers.cond_cat
    samplers.can_concat_cond = functools.wraps(original_can_concat)(lambda a, b: _can_join(original_can_concat, a, b))
    samplers.cond_cat = functools.wraps(original_cat)(lambda c, device=None: _join(original_cat, c, device))
    original_forward = f2.Fibo2._forward
    f2.Fibo2._forward = functools.wraps(original_forward)(
        lambda self, *a, **k: _forward(original_forward, self, *a, **k))
    _installed = True
    logging.info('%s exact joined CFG rows, fused norm/modulation/residual and tile32 attention enabled.', _TAG)
