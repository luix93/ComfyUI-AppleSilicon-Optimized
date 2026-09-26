"""Exact current-API Ideogram 4 attention preparation on M5 Ultra / 2.14.

Keep projection hooks, weight casting, the selected attention interface and
all BF16 rounding. The D=256 attention kernel retains PyTorch's math and
writes the following projection's layout directly and removes forced Q/K
loop unrolling. Query caching is slower at this width and stays disabled.
"""
from contextvars import ContextVar
from pathlib import Path
import os

import torch

from . import torch_mpp_qcache

_installed = False
_dead = False
_attention_dead = False
_library = _attention_library = None
_original = _model = None
_calls = _prep_calls = _attention_calls = 0
_scope = ContextVar('ideogram4_exact_attention', default=False)


def _runtime():
    return (os.environ.get('ASFP8_DIT_EXT') == '1'
            and os.environ.get('ASFP8_IDEOGRAM_FAST', '1') == '1'
            and os.environ.get('ASFP8_IDEOGRAM_EXACT', '1') == '1'
            and os.environ.get('PYTORCH_MPS_FAST_MATH', '0') == '0'
            and torch.__version__.split('+')[0] == '2.14.0'
            and torch.version.git_version == torch_mpp_qcache._TORCH_GIT
            and torch_mpp_qcache._hardware())


def _lib():
    global _library, _dead
    if _library is None and not _dead:
        try:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead = True
            print(f'[AppleSilicon-Optimized/ideogram4_exact] keeping eager preparation ({exc!r}).')
    return _library


def _prep(qkv, pe, qw, kw, heads, eps):
    # A custom projection/cast may return an unsupported layout or dtype.
    # Reuse its result in the ordinary path instead of invoking it twice.
    global _prep_calls
    if (not _dead and os.environ.get('ASFP8_IDEOGRAM_PREP', '1') == '1'
            and type(qkv) is torch.Tensor and qkv.is_mps
            and qkv.dtype == torch.bfloat16 and qkv.ndim == 3
            and qkv.shape[0] == 1 and heads == 18 and qkv.shape[-1] == 3 * heads * 256
            and qkv.is_contiguous()
            and all(type(w) in (torch.Tensor, torch.nn.Parameter) and w.is_mps
                    and w.dtype == qkv.dtype and w.shape == (256,) and w.is_contiguous()
                    for w in (qw, kw)) and _lib() is not None):
        seq = qkv.shape[1]
        q, k, v = [torch.empty((1, heads, seq, 256), device=qkv.device, dtype=qkv.dtype)
                   for _ in range(3)]
        _library.qkv_prep(qkv, qw, kw, pe, q, k, v, seq, heads, float(eps),
                          threads=(seq * 64, heads, 3), group_size=(64, 1, 1))
        _prep_calls += 1
        return q, k, v
    q, k, v = qkv.view(1, qkv.shape[1], 3, heads, 256).unbind(2)
    q, k = _model.comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, qw, kw, eps)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def _attention(self, x, attn_mask, freqs_cis, transformer_options={}):
    global _calls
    if (not _runtime() or torch.is_grad_enabled() or _model.comfy.model_management.in_training
            or type(x) is not torch.Tensor or not x.is_mps or x.dtype != torch.bfloat16
            or x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 4608
            or not 0 < x.numel() < 2**31 or self.num_heads != 18 or self.head_dim != 256
            or self.norm_q.eps != self.norm_k.eps
            or type(freqs_cis) is not torch.Tensor or not freqs_cis.is_mps
            or freqs_cis.dtype != torch.float32 or not freqs_cis.is_contiguous()
            or freqs_cis.shape != (1, x.shape[1], 1, 128, 2, 2)):
        return _original(self, x, attn_mask, freqs_cis, transformer_options)
    qkv = self.qkv(x)
    q_view, k_view, _ = qkv.view(1, x.shape[1], 3, 18, 256).unbind(2)
    ops = _model.comfy.ops
    with ops.CastBiasWeightContext(self.norm_q, q_view, offloadable=True) as (qw, _):
        with ops.CastBiasWeightContext(self.norm_k, k_view, offloadable=True) as (kw, _):
            q, k, v = _prep(qkv, freqs_cis, qw, kw, 18, self.norm_q.eps)
    token = _scope.set(True)
    try:
        out = _model.optimized_attention_masked(
            q, k, v, 18, attn_mask, skip_reshape=True, transformer_options=transformer_options)
    finally:
        _scope.reset(token)
    _calls += 1
    return self.o(out)


def _attention_source():
    source = torch_mpp_qcache._source(head_dim=256, cache_query=False)
    # D=256 needs sixteen Q/K fragments. Forcing all iterations to unroll
    # slows the Ultra; let the compiler schedule the loop. Dot products,
    # their order, softmax, and output reductions are unchanged.
    needle = 'MPP_PRAGMA_UNROLL\n        for (short id = 0; id < TD; id++)'
    if source.count(needle) != 1:
        raise RuntimeError('Unrecognized PyTorch Q/K loop')
    return source.replace(needle, 'for (short id = 0; id < TD; id++)')


def maybe_attention(q, k, v, *, output_bshd=False, **kwargs):
    global _attention_library, _attention_dead, _attention_calls
    if (not _scope.get() or _attention_dead or not _runtime() or torch.is_grad_enabled()
            or not output_bshd or os.environ.get('ASFP8_IDEOGRAM_ATTN_LAYOUT', '1') != '1'
            or os.environ.get('ASFP8_ATTN_BACKEND') != 'pytorch'
            or any(type(t) is not torch.Tensor or not t.is_mps or t.ndim != 4
                   or t.dtype != torch.bfloat16 or t.stride(-1) != 1 for t in (q, k, v))
            or q.shape != k.shape or q.shape != v.shape or q.shape[:2] != (1, 18)
            or q.shape[-1] != 256 or not 2048 <= q.shape[2] <= 32768
            or any(s < 0 for t in (q, k, v) for s in t.stride())
            or any(t.stride(2) >= 2**31 // 64 for t in (q, k, v))
            or kwargs.keys() - {'attn_mask', 'dropout_p', 'is_causal', 'scale', 'enable_gqa'}
            or kwargs.get('attn_mask') is not None or kwargs.get('dropout_p', 0.) != 0.
            or kwargs.get('is_causal', False) or kwargs.get('scale') is not None):
        return None
    try:
        if _attention_library is None:
            _attention_library = torch.mps.compile_shader(_attention_source())
        params = torch_mpp_qcache._parameters(tuple(q.shape), tuple(k.shape),
                    q.stride(), k.stride(), v.stride(), 256**-.5, True)
        seq = q.shape[2]
        out = torch.empty((1, seq, 18, 256), device=q.device, dtype=q.dtype).transpose(1, 2)
        _attention_library.custom_attn(q, k, v, out, params, params, params,
                                      threads=(((seq + 63) // 64) * 128, 18, 1),
                                      group_size=(128, 1, 1))
    except Exception as exc:
        _attention_dead = True
        print(f'[AppleSilicon-Optimized/ideogram4_exact] keeping standard SDPA ({exc!r}).')
        return None
    _attention_calls += 1
    return out


def install():
    global _installed, _model, _original
    if _installed or not _runtime() or not torch.backends.mps.is_available():
        return
    import comfy.ldm.ideogram4.model as model
    _model = model
    _original = model.Ideogram4Attention.forward
    model.Ideogram4Attention.forward = _attention
    _installed = True
    print('[AppleSilicon-Optimized/ideogram4_exact] current split-half RoPE and exact attention output layout enabled.')
