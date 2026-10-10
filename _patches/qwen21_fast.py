"""Exact BF16 Qwen-Image 2.1 fusions for the validated M5 Ultra / Torch 2.14.

Retain LayerNorm, attention, prefix-cache behavior and INT8 GEMM arithmetic.
Fuse Q/K RMSNorm+RoPE and the surrounding modulation/residual memory passes.
Reschedule channel normalization in the Qwen image VAE without changing its
serial channel reduction or intermediate rounding.
"""
import os
import functools
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.nn.functional as F

from . import torch_mpp_qcache

_installed = False
_library = None
_dead = False
_original_norm = _original_residual = _original_attention = None
_ops = _model = None
_vae_scope = ContextVar('qwen21_vae', default=False)


def _lib():
    global _library, _dead
    if _library is None and not _dead:
        try:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead = True
            print(f'[AppleSilicon-Optimized/qwen21] kernel unavailable ({exc!r}); keeping eager operations.')
    return _library


def _enabled(part, x):
    return (not _dead and os.environ.get('ASFP8_QWEN21_FAST', '1') == '1'
            and os.environ.get('ASFP8_QWEN21_' + part, '1') == '1'
            and not torch.is_grad_enabled() and type(x) is torch.Tensor
            and x.is_mps and x.dtype == torch.bfloat16 and x.ndim == 3
            and x.shape[0] == 1 and x.shape[-1] == 4096 and x.is_contiguous()
            and 0 < x.numel() < 2**31)


def _vector(x):
    return type(x) is torch.Tensor and x.is_mps and x.dtype == torch.bfloat16 and x.numel() == 4096


def _modulate(norm, x, scale, prefix_len, zero):
    if (_enabled('MODULATE', x) and not _model.comfy.model_management.in_training
            and 0 <= prefix_len <= x.shape[1] and all(_vector(t) for t in (*scale, zero))
            and _lib() is not None):
        normalized = F.layer_norm(x, (4096,), eps=norm.eps)
        out = torch.empty_like(x)
        _library.modulate(normalized, scale[0].contiguous(), scale[1].contiguous(),
                          zero.contiguous(), out, 4096, prefix_len,
                          threads=x.numel(), group_size=256)
        return out
    return _original_norm(norm, x, scale, prefix_len, zero)


def _residual(x, y, gate, prefix_len):
    if (_enabled('RESIDUAL', x) and _enabled('RESIDUAL', y) and x.shape == y.shape
            and 0 <= prefix_len <= x.shape[1] and all(_vector(t) for t in gate)
            and _lib() is not None):
        _library.residual(x, y, gate[0].contiguous(), gate[1].contiguous(),
                          4096, prefix_len, threads=x.numel(), group_size=256)
        return x
    return _original_residual(x, y, gate, prefix_len)


def _rope(q, k, pe, qw, kw, eps):
    # Projection and weight-cast hooks have already run. Unsupported outputs
    # use Kitchen directly, without evaluating those modules a second time.
    if (all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.is_mps and t.dtype == torch.bfloat16
            and t.is_contiguous() for t in (q, k, qw, kw))
            and q.shape == k.shape and q.shape[-2:] == (32, 128)
            and qw.numel() == kw.numel() == 128):
        oq, ok = torch.empty_like(q), torch.empty_like(k)
        _library.rms_rope(q, k, qw, kw, pe, oq, ok, 32, float(eps),
                          threads=(q.shape[1] * 32, 32, 2), group_size=(32, 1, 1))
        return oq, ok
    return _model.comfy.quant_ops.ck.rms_rope(q, k, pe, qw, kw, eps)


def _attention(self, x, pe, attn_fn, prefix_len, transformer_options={}):
    if (not _enabled('ROPE', x) or _model.comfy.model_management.in_training
            or transformer_options.get('patches', {}).get('attn1_patch')
            or self.heads != 32 or self.norm_q.eps != self.norm_k.eps
            or type(pe) is not torch.Tensor or not pe.is_mps or pe.dtype != torch.float32
            or not pe.is_contiguous() or pe.shape != (1, x.shape[1], 1, 64, 2, 2)
            or _lib() is None):
        return _original_attention(self, x, pe, attn_fn, prefix_len, transformer_options)
    q = self.to_q(x).view(1, x.shape[1], 32, 128)
    k = self.to_k(x).view(1, x.shape[1], 32, 128)
    v = self.to_v(x).view(1, x.shape[1], 32, 128)
    with _ops.CastBiasWeightContext(self.norm_q, q, offloadable=True) as (qw, _):
        with _ops.CastBiasWeightContext(self.norm_k, k, offloadable=True) as (kw, _):
            q, k = _rope(q, k, pe, qw, kw, self.norm_q.eps)
    container = getattr(_model, 'AttentionTensorContainer', None)
    if container is not None:
        q, k, v = [container(t) for t in (q, k, v)]
    return self.to_out[0](attn_fn(q, k, v, self.heads, preferred_attention=self.comfy_attention))


def _vae_rms(x, gamma, scale, eps, silu):
    # The same serial C reduction as wan_ext, with one spatial value per
    # thread instead of 32. More groups fill the Ultra at these image shapes.
    b, c, s = x.shape
    out = torch.empty_like(x)
    _library.vae_chan_rms(x, gamma, out, c, s, b, float(scale), float(eps),
                          int(silu), threads=b * ((s + 255) // 256) * 256,
                          group_size=256)
    return out


def _install_vae():
    from . import wan_vae_fast
    import comfy.ldm.wan.vae2_2 as vae
    if wan_vae_fast._kernel is None:
        return
    original_rms = wan_vae_fast._kernel.chan_rms

    @functools.wraps(original_rms)
    def rms(x, gamma, scale, eps, silu):
        if (_vae_scope.get() and x.dtype == torch.bfloat16 and x.is_mps
                and x.is_contiguous() and x.ndim == 3 and 0 < x.numel() < 2**31
                and gamma.is_mps and gamma.is_contiguous() and gamma.dtype == x.dtype
                and gamma.numel() == x.shape[1] and not torch.is_grad_enabled()
                and _lib() is not None):
            return _vae_rms(x, gamma, scale, eps, silu)
        return original_rms(x, gamma, scale, eps, silu)

    def wrap(original):
        @functools.wraps(original)
        def run(self, *args, **kwargs):
            x = args[0] if args else kwargs.get('x', kwargs.get('z'))
            enabled = (os.environ.get('ASFP8_QWEN21_FAST', '1') == '1'
                       and os.environ.get('ASFP8_QWEN21_VAE', '1') == '1'
                       and self.patch_size == 1 and self.z_dim == 64
                       and self.decoder.head[2].weight.shape[2] == 1
                       and type(x) is torch.Tensor and x.ndim == 4 and x.shape[0] == 1
                       and x.is_mps and x.dtype == torch.bfloat16)
            token = _vae_scope.set(enabled)
            try:
                return original(self, *args, **kwargs)
            finally:
                _vae_scope.reset(token)
        return run

    wan_vae_fast._kernel.chan_rms = rms
    vae.WanVAE.encode = wrap(vae.WanVAE.encode)
    vae.WanVAE.decode = wrap(vae.WanVAE.decode)


def install():
    global _installed, _original_norm, _original_residual, _original_attention, _ops, _model
    if (_installed or os.environ.get('ASFP8_DIT_EXT') != '1'
            or os.environ.get('ASFP8_QWEN21_FAST', '1') != '1'
            or not torch_mpp_qcache._torch_build_supported()
            or not torch_mpp_qcache._hardware() or not torch.backends.mps.is_available()):
        return
    try:
        import comfy.ops as ops
        import comfy.ldm.qwen_image21.model as model
    except ImportError:
        return
    _ops, _model = ops, model
    _original_norm, _original_residual = model._modulated_norm, model._gated_residual
    _original_attention = model.Attention.forward
    model._modulated_norm, model._gated_residual = _modulate, _residual
    model.Attention.forward = _attention
    _install_vae()
    _installed = True
    print('[AppleSilicon-Optimized/qwen21] exact QK normalization/RoPE, modulation, residual and image-VAE scheduling enabled.')
