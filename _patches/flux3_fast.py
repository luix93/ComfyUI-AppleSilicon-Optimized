"""Exact Flux.3 MPS glue and phase-local prepared-weight reuse.

Keep ComfyUI's RMS normalization, matrix arithmetic, conditioning, sampling and
RNG unchanged. Cache owned copies of ComfyUI's prepared weights only within a
sampling/generation call. Release them on success, cancellation and failure.
"""
import functools
import inspect
import logging
import math
import os
from contextvars import ContextVar
from threading import RLock

import torch
import psutil

from . import flux2_glue, torch_mpp_qcache

_TAG = '[AppleSilicon-Optimized/flux3]'
_installed = False
_scope = ContextVar('apple_silicon_flux3_casts', default=None)
_library = None
_rope_dead = False
_stats = {}
_prompt_stats = {}
_rope_calls = _activation_calls = 0
_cast_lock = RLock()
_cast_users = 0
_active_cast = None

# Preserve separate FP32 products/addition, including the BF16 store rounding.
# An explicit fma here does NOT match the reviewed MPS addcmul implementation.
_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {
 uint u=as_type<uint>(f);u+=0x7fff+((u>>16)&1);
 return as_type<bfloat>(ushort(u>>16));
}
kernel void rope(device const bfloat* q,device const bfloat* k,
 device const float* pe,device bfloat* qo,device bfloat* ko,
 constant uint& N,constant uint& H,constant uint& Bpe,
 uint i [[thread_position_in_grid]]) {
 uint c=i%128,pair=c/2,token=i/(H*128),j=c%2;
 uint off=(i/128)*128+pair*2;
 uint fp=((Bpe==1?token%N:token)*64+pair)*4+j*2;
 float a=pe[fp],b=pe[fp+1];
 float q0=float(q[off]),q1=float(q[off+1]);
 float k0=float(k[off]),k1=float(k[off+1]);
 float qleft=a*q0,qright=b*q1,kleft=a*k0,kright=b*k1;
 qo[i]=rte(qleft+qright);ko[i]=rte(kleft+kright);
}
'''


def _enabled(part):
    return (os.environ.get('ASFP8_FLUX3_FAST', '1') == '1'
            and os.environ.get('ASFP8_FLUX3_' + part, '1') == '1'
            and not torch.is_grad_enabled())


def _budget(prompt=False):
    name = 'ASFP8_FLUX3_PROMPT_CACHE_GIB' if prompt else 'ASFP8_FLUX3_CACHE_GIB'
    try:
        gib = float(os.environ.get(name, '24' if prompt else '4'))
    except ValueError:
        return 0
    if not math.isfinite(gib) or gib <= 0:
        return 0
    return int(min(gib * 2**30, 32 * 2**30,
                   torch.mps.recommended_max_memory() * .12,
                   psutil.virtual_memory().available * .2))


def _revision(t):
    if t is None:
        return None
    version = None if torch.is_inference(t) else t._version
    return id(t), version, str(t.device), t.dtype


def _weight_key(t):
    if t is None:
        return None
    p = getattr(t, '_params', None)
    return (_revision(t), _revision(getattr(t, '_qdata', None)),
            _revision(getattr(p, 'scale', None)),
            getattr(p, 'convrot', None), getattr(p, 'convrot_groupsize', None),
            getattr(p, 'transposed', None))


def _cast_cached(original, signature, *args, **kwargs):
    state = _scope.get()
    if state is None or not _enabled(state['part']) or state.get('disabled'):
        return original(*args, **kwargs)
    names = ('s', 'input', 'dtype', 'device', 'bias_dtype', 'offloadable',
             'compute_dtype', 'want_requant')
    if (1 <= len(args) <= 2 and tuple(signature.parameters) == names
            and not kwargs.keys() - set(names)
            and not any(n in kwargs for n in names[:len(args)])):
        p = dict(input=None, dtype=None, device=None, bias_dtype=None,
                 offloadable=False, compute_dtype=None, want_requant=False)
        p.update(kwargs)
        p.update(zip(names, args))
    else:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
    module, x = p['s'], p['input']
    device = p['device'] if p['device'] is not None else getattr(x, 'device', None)
    if (id(module) not in state['modules'] or device is None
            or torch.device(device).type != 'mps' or hasattr(module, '_v')
            or getattr(module, 'weight_function', [])
            or getattr(module, 'bias_function', [])
            or getattr(module, 'weight_lowvram_function', None) is not None
            or getattr(module, 'bias_lowvram_function', None) is not None):
        return original(*args, **kwargs)
    dtype = p['dtype'] or getattr(x, 'dtype', None)
    key = (id(module), _weight_key(module.weight), _weight_key(module.bias),
           dtype, str(device), p['bias_dtype'] or dtype,
           p['compute_dtype'], p['want_requant'])
    entry = state['cache'].get(key)
    if entry is not None:
        state['hits'] += 1
        weight, bias = entry[:2]
        return (weight, bias, None) if p['offloadable'] else (weight, bias)
    result = original(*args, **kwargs)
    weight, bias = result[:2]
    if type(weight) is not torch.Tensor or not weight.is_floating_point():
        return result
    if bias is not None and type(bias) is not torch.Tensor:
        return result
    size = weight.numel() * weight.element_size()
    if bias is not None:
        size += bias.numel() * bias.element_size()
    if (state['bytes'] + size > state['budget']
            or torch.mps.driver_allocated_memory() + size
            > torch.mps.recommended_max_memory() * .65):
        return result
    # Own copies: cast buffers may be streamed/reused after the original lease.
    # Retain source objects as well, so replacing an inference parameter cannot
    # recycle its identity into an existing cache key within the same phase.
    wc = weight.detach().clone()
    bc = None if bias is None else bias.detach().clone()
    sources = (module.weight, module.bias, getattr(module.weight, '_qdata', None),
               getattr(getattr(module.weight, '_params', None), 'scale', None))
    state['cache'][key] = wc, bc, sources
    state['bytes'] += size
    state['misses'] += 1
    return result


def _state(modules, prompt=False):
    return dict(modules={id(m) for m in modules}, cache={}, hits=0, misses=0,
                bytes=0, budget=_budget(prompt),
                part='PROMPT_CACHE' if prompt else 'WEIGHT_CACHE')


def _wrap_cast(original):
    signature = inspect.signature(original)
    @functools.wraps(original)
    def cached(*args, **kwargs):
        # All other models take the shortest pass-through path. In particular,
        # avoid a second Python call/argument packing for every YuE token layer.
        if _scope.get() is None:
            return original(*args, **kwargs)
        return _cast_cached(original, signature, *args, **kwargs)
    return cached


def _acquire_cast():
    """Install only during active Flux phases; preserve concurrent/foreign hooks."""
    import comfy.ops as ops
    global _cast_users, _active_cast
    with _cast_lock:
        if _cast_users == 0:
            _active_cast = _wrap_cast(ops.cast_bias_weight)
            ops.cast_bias_weight = _active_cast
        _cast_users += 1
    def release():
        global _cast_users, _active_cast
        with _cast_lock:
            _cast_users -= 1
            if _cast_users == 0:
                # Never remove another extension's replacement. If it wrapped
                # ours, our empty-context pass-through remains safe.
                if ops.cast_bias_weight is _active_cast:
                    ops.cast_bias_weight = _active_cast.__wrapped__
                _active_cast = None
    return release


def _begin_cast(state):
    if state['modules'] and state['budget'] > 0 and _enabled(state['part']):
        state['release_cast'] = _acquire_cast()


def _finish(state, token, target):
    _scope.reset(token)
    release = state.pop('release_cast', None)
    if release is not None:
        release()
    target.clear()
    target.update({k: state[k] for k in ('hits', 'misses', 'bytes', 'budget')})
    target.update(rope_calls=_rope_calls, activation_calls=_activation_calls)
    state['cache'].clear()
    if os.environ.get('ASFP8_FLUX3_VERBOSE', '0') == '1':
        logging.info('%s %s: %s', _TAG, state['part'], target)


def _has_hooks(value):
    if isinstance(value, dict):
        return bool(value.get('hooks')) or any(_has_hooks(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_hooks(x) for x in value)
    return False


def _sample(original, flux_type, block_type, guider, *args, **kwargs):
    model = getattr(getattr(guider, 'inner_model', None), 'diffusion_model', None)
    patcher = getattr(guider, 'model_patcher', None)
    options = getattr(guider, 'model_options', {})
    transformer = options.get('transformer_options', {})
    if (type(model) is not flux_type or patcher is None
            or not (_enabled('WEIGHT_CACHE') or _enabled('SWIGLU'))
            or torch.device(patcher.load_device).type != 'mps'
            or getattr(patcher, 'hook_patches', None)
            or getattr(patcher, 'current_hooks', None)
            or getattr(patcher, 'forced_hooks', None)
            or getattr(patcher, 'injections', None)
            or options.get('model_function_wrapper')
            or options.get('wrappers')
            or transformer.get('patches') or transformer.get('patches_replace')
            or transformer.get('wrappers')
            or transformer.get('optimized_attention_override')
            or options.get('multigpu_thread_pool')
            or _has_hooks(getattr(guider, 'conds', {}))):
        return original(guider, *args, **kwargs)
    state = _state(m.mlp_out for m in model.modules() if type(m) is block_type)
    token = _scope.set(state)
    try:
        _begin_cast(state)
        return original(guider, *args, **kwargs)
    finally:
        _finish(state, token, _stats)


def _generate(original, model, *args, **kwargs):
    embeds = kwargs.get('embeds', args[0] if args else None)
    if (not _enabled('PROMPT_CACHE') or getattr(model, 'model_type', '') != 'qwen35_9b'
            or not isinstance(embeds, torch.Tensor) or not embeds.is_mps
            or (getattr(model, 'mtp', None) is not None and kwargs.get('mtp', True))):
        return original(model, *args, **kwargs)
    state = _state(model.model.modules(), prompt=True)
    token = _scope.set(state)
    try:
        _begin_cast(state)
        return original(model, *args, **kwargs)
    finally:
        _finish(state, token, _prompt_stats)


def _ready():
    global _library, _rope_dead
    if _rope_dead:
        return False
    if _library is None:
        try:
            _library = torch.mps.compile_shader(_SOURCE)
        except Exception as exc:
            _rope_dead = True
            logging.warning('%s retaining original RoPE (%s)', _TAG, exc)
    return _library is not None


def _rope_supported(q, k, pe):
    return (type(q) is torch.Tensor and type(k) is torch.Tensor
            and type(pe) is torch.Tensor and q.is_mps and k.device == q.device
            and pe.device == q.device and q.dtype == k.dtype == torch.bfloat16
            and q.ndim == 4 and q.shape == k.shape and q.shape[-1] == 128
            and 0 < q.numel() < 2**31 and pe.dtype == torch.float32
            and pe.ndim == 6 and pe.shape[0] in (1, q.shape[0])
            and pe.shape[1] == q.shape[1] and pe.shape[2:] == (1, 64, 2, 2))


def _qknorm(original, module, q, k, pe):
    if not _enabled('ROPE') or not _rope_supported(q, k, pe) or not _ready():
        return original(module, q, k, pe)
    import comfy.ops
    qw, _, qs = comfy.ops.cast_bias_weight(module.query_norm, q, offloadable=True)
    kw, _, ks = comfy.ops.cast_bias_weight(module.key_norm, k, offloadable=True)
    try:
        # Retain the original normalization and its intermediate rounding.
        qn = torch.nn.functional.rms_norm(q, (128,), qw, module.query_norm.eps).contiguous()
        kn = torch.nn.functional.rms_norm(k, (128,), kw, module.key_norm.eps).contiguous()
        qo, ko = torch.empty_like(qn), torch.empty_like(kn)
        _library.rope(qn, kn, pe.contiguous(), qo, ko, q.shape[1], q.shape[2], pe.shape[0],
                      threads=qn.numel(), group_size=256)
    finally:
        comfy.ops.uncast_bias_weight(module.query_norm, qw, None, qs)
        comfy.ops.uncast_bias_weight(module.key_norm, kw, None, ks)
    global _rope_calls
    _rope_calls += 1
    return qo, ko


def _activation(original, x):
    state = _scope.get()
    if (state is None or state['part'] != 'WEIGHT_CACHE' or not _enabled('SWIGLU')
            or type(x) is not torch.Tensor or not x.is_mps
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.ndim < 2 or x.shape[-1] != 18432 or not 0 < x.numel() < 2**31
            or not flux2_glue._ready()):
        return original(x)
    out = torch.empty((*x.shape[:-1], 9216), dtype=x.dtype, device=x.device)
    flux2_glue._library.activation(x, flux2_glue._table, out, 9216, 18432,
                                  threads=out.numel(), group_size=256)
    global _activation_calls
    _activation_calls += 1
    return out


def install():
    global _installed
    if (_installed or os.environ.get('ASFP8_FLUX3_FAST', '1') != '1'
            or not torch.backends.mps.is_available()
            or torch.__version__.split('+')[0] != '2.14.1'
            or not torch_mpp_qcache._torch_build_supported()
            or not torch_mpp_qcache._hardware()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return
    # Installation runs with grad enabled; runtime guards apply at execution.
    from importlib.metadata import version
    try:
        import comfy.ops as ops
        import comfy.samplers as samplers
        from comfy.ldm.flux3.model import Flux3, Block, QKNorm
        from comfy.text_encoders.qwen35 import Qwen35
        # Both reviewed builds retain the eager RoPE/SwiGLU arithmetic.
        if version('comfy-kitchen') not in ('0.2.37', '0.2.38'):
            return
    except (ImportError, ModuleNotFoundError):
        return
    sample = samplers.CFGGuider.inner_sample
    @functools.wraps(sample)
    def sampled(self, *args, **kwargs):
        return _sample(sample, Flux3, Block, self, *args, **kwargs)
    samplers.CFGGuider.inner_sample = sampled
    generate = Qwen35.generate
    @functools.wraps(generate)
    def generated(self, *args, **kwargs):
        return _generate(generate, self, *args, **kwargs)
    Qwen35.generate = generated
    norm = QKNorm.forward
    @functools.wraps(norm)
    def normalized(self, q, k, pe):
        return _qknorm(norm, self, q, k, pe)
    QKNorm.forward = normalized
    act = ops.INPUT_ACT_EAGER['swiglu']
    @functools.wraps(act)
    def activated(x):
        return _activation(act, x)
    ops.INPUT_ACT_EAGER['swiglu'] = activated
    _installed = True
    logging.info('%s exact phase-local weight reuse, RoPE and SwiGLU enabled.', _TAG)
