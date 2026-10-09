"""Exact, phase-local YuE2 autoregressive acceleration on MPS.

Cache the floating weights returned by ComfyUI's own cast/offload operation;
keep its conversion, ConvRot, linear arithmetic, sampler and RNG unchanged.
Converted copies live only during a score/music generation phase. Vocabulary
slicing uses the existing floating linear, never switches the encoder to W8A8.
"""
import functools
import inspect
import logging
import math
import os
import sys
from contextvars import ContextVar

import torch

_installed = False
_scope = ContextVar('apple_silicon_yue2_casts', default=None)
_acoustic_scope = ContextVar('apple_silicon_yue2_acoustic', default=False)
_stats = {}
_TAG = '[AppleSilicon-Optimized/yue2]'


def _enabled(part):
    return (os.environ.get('ASFP8_YUE2_FAST', '1') == '1'
            and os.environ.get('ASFP8_YUE2_' + part, '1') == '1'
            and not torch.is_grad_enabled())


def _budget():
    try:
        gib = float(os.environ.get('ASFP8_YUE2_CACHE_GIB', '8'))
    except ValueError:
        return 0
    if not math.isfinite(gib) or gib <= 0:
        return 0
    # Respect small unified-memory machines; this is a temporary cache rather
    # than permission to fill their RAM. Individual misses retain the stock path.
    recommended = torch.mps.recommended_max_memory()
    return int(min(gib * 2**30, 64 * 2**30, recommended * .2))


def _revision(param):
    if param is None:
        return None
    # Inference tensors have no version counter; identity remains phase-local.
    if torch.is_inference(param):
        return id(param), None
    try:
        version = param._version
    except RuntimeError:
        version = None
    return id(param), version


def _cast_cached(original, signature, *args, **kwargs):
    state = _scope.get()
    if state is None or not _enabled('WEIGHT_CACHE'):
        return original(*args, **kwargs)
    # CastBiasWeightContext normally passes just module/input positionally.
    # Avoid rebuilding an inspect.BoundArguments for every layer/token; keep
    # the full signature path for unusual calls and future API extensions.
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
    module, x = p['s'], p['input']
    device = p['device'] if p['device'] is not None else getattr(x, 'device', None)
    if (id(module) not in state['modules'] or device is None
            or torch.device(device).type != 'mps'
            or getattr(module, 'weight_function', [])
            or getattr(module, 'bias_function', [])
            or getattr(module, 'weight_lowvram_function', None) is not None
            or getattr(module, 'bias_lowvram_function', None) is not None):
        return original(*args, **kwargs)
    dtype = p['dtype'] or getattr(x, 'dtype', None)
    bias_dtype = p['bias_dtype'] or dtype
    key = (id(module), _revision(module.weight), _revision(module.bias), dtype,
           str(device), bias_dtype, p['compute_dtype'], p['want_requant'])
    entry = state['cache'].get(key)
    if entry is not None:
        state['hits'] += 1
        weight, bias = entry
        return (weight, bias, None) if p['offloadable'] else (weight, bias)

    result = original(*args, **kwargs)
    weight, bias = result[:2]
    # A quantized wrapper or custom tensor must retain its original lifetime
    # and dispatch. Clone ordinary floating outputs so streamed buffers cannot
    # later overwrite cached weights, and return the original lease on a miss.
    if type(weight) is torch.Tensor and weight.is_floating_point():
        size = weight.numel() * weight.element_size()
        if bias is not None:
            size += bias.numel() * bias.element_size()
        headroom = torch.mps.recommended_max_memory() * .65
        if (state['bytes'] + size <= state['budget']
                and torch.mps.driver_allocated_memory() + size <= headroom):
            wc = weight.detach().clone()
            bc = None if bias is None else bias.detach().clone()
            state['cache'][key] = wc, bc
            state['bytes'] += size
            state['misses'] += 1
    return result


def _head_bounds(phase, vocab):
    # Includes a few masked score rows to retain a contiguous matrix view.
    if vocab != 184704:
        return None
    if phase == 'abc':
        return 0, 151849
    if phase == 'semantic':
        return 151852, 184621
    return None


def _head_supported(head, device, dtype, bounds):
    from . import torch_mpp_qcache
    return (bounds is not None and _enabled('VOCAB_HEAD')
            and torch.device(device).type == 'mps' and dtype == torch.bfloat16
            and torch.__version__.split('+')[0] == '2.14.0'
            and torch.version.git_version == torch_mpp_qcache._TORCH_GIT
            and os.environ.get('PYTORCH_MPS_FAST_MATH', '0') == '0'
            and torch_mpp_qcache._hardware()
            and callable(getattr(head, '_forward', None))
            and not getattr(head, 'weight_function', [])
            and not getattr(head, 'bias_function', [])
            and getattr(head, 'pre_quant_scale', None) is None)


def _slice_head(original, lo, hi, input, weight, bias):
    if (type(weight) is not torch.Tensor or type(input) is not torch.Tensor
            or input.ndim != 2 or input.shape[0] != 1):
        return original(input, weight, bias)
    value = torch.nn.functional.linear(input, weight[lo:hi],
                                      None if bias is None else bias[lo:hi])
    # Preserve the full vocabulary and CPU multinomial dimension/RNG behavior.
    # Unused logits become -inf under the original phase mask. Zero here also
    # prevents inf-inf NaNs if autoregressive classifier-free guidance is used.
    out = torch.zeros((*input.shape[:-1], weight.shape[0]),
                      device=input.device, dtype=value.dtype)
    out[..., lo:hi] = value
    return out


def _generate(original, self, *args, **kwargs):
    if (not _enabled('WEIGHT_CACHE') and not _enabled('VOCAB_HEAD') and not _enabled('ROPE')
            or torch.device(self.execution_device).type != 'mps'):
        return original(self, *args, **kwargs)
    phase = kwargs.get('phase', args[3] if len(args) > 3 else None)
    dtype = kwargs.get('dtype', args[4] if len(args) > 4 else None)
    head = self.model.lm_head
    bounds = _head_bounds(phase, self.config.vocab_size)
    state = dict(modules={id(m) for m in self.model.modules()}, cache={},
                 hits=0, misses=0, bytes=0, budget=_budget())
    from . import yue2_rope_mps
    state['rope_state'] = yue2_rope_mps.prepare(self.config) if _enabled('ROPE') and dtype == torch.bfloat16 else None
    token = _scope.set(state)
    # Restore the instance exactly; do not leave a bound method override behind.
    previous = head.__dict__.get('_forward')
    had_override = '_forward' in head.__dict__
    sliced = _head_supported(head, self.execution_device, dtype, bounds)
    if sliced:
        head._forward = functools.partial(_slice_head, head._forward, *bounds)
    try:
        return original(self, *args, **kwargs)
    finally:
        if sliced:
            if had_override:
                head._forward = previous
            else:
                del head.__dict__['_forward']
        _scope.reset(token)
        _stats.clear()
        _stats.update(phase=phase, hits=state['hits'], misses=state['misses'],
                      bytes=state['bytes'], budget=state['budget'], sliced_head=sliced,
                      peak_driver_gib=torch.mps.driver_allocated_memory() / 2**30,
                      rope_calls=(state['rope_state'] or {}).get('calls', 0),
                      rope_matrices=(state['rope_state'] or {}).get('matrices', 0))
        state['cache'].clear()
        if os.environ.get('ASFP8_YUE2_VERBOSE', '0') == '1':
            logging.info('%s %s', _TAG, _stats)


def install():
    global _installed
    if _installed or sys.platform != 'darwin' or not torch.backends.mps.is_available():
        return
    if os.environ.get('ASFP8_YUE2_FAST', '1') != '1':
        return
    try:
        import comfy.ops as ops
        from comfy.text_encoders.yue2 import YuE2TEModel
        from comfy.ldm.yue2.model import YuE2
    except ImportError:
        return
    if getattr(YuE2TEModel._generate, '_apple_silicon_yue2', False):
        _installed = True
        return
    cast = ops.cast_bias_weight
    cast_signature = inspect.signature(cast)
    @functools.wraps(cast)
    def cached(*args, **kwargs):
        return _cast_cached(cast, cast_signature, *args, **kwargs)
    ops.cast_bias_weight = cached
    generate = YuE2TEModel._generate
    @functools.wraps(generate)
    def wrapped(self, *args, **kwargs):
        return _generate(generate, self, *args, **kwargs)
    wrapped._apple_silicon_yue2 = True
    YuE2TEModel._generate = wrapped
    from comfy.text_encoders import llama
    from . import yue2_rope_mps
    original_rope = llama.apply_rope
    @functools.wraps(original_rope)
    def rope(q, k, freqs_cis):
        out = yue2_rope_mps.maybe_rope(q, k, freqs_cis, _scope.get())
        return original_rope(q, k, freqs_cis) if out is None else out
    llama.apply_rope = rope
    forward = YuE2.forward
    @functools.wraps(forward)
    def acoustic(self, *args, **kwargs):
        token = _acoustic_scope.set(_enabled('ATTN'))
        try:
            return forward(self, *args, **kwargs)
        finally:
            _acoustic_scope.reset(token)
    YuE2.forward = acoustic
    _installed = True
    logging.info('%s exact phase-local weight cache, vocabulary head, single-token RoPE and guarded acoustic attention enabled.', _TAG)
