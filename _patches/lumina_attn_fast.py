"""Patch #21 — fused Lumina/Z-Image attention pre-processing (opt-in ASFP8_DIT_EXT=1).

Lumina-family models (Z-Image, Lumina2) spend ~7.5 ms/layer of eager time
between the qkv Linear and the attention call:

    split -> view -> q_norm -> k_norm -> apply_rope -> (no-op n_rep repeat!) -> 3x movedim copy

F.rms_norm over [S, H, 128]-shaped rows is pathological on MPS (~1.6 ms per
call at Z-Image shapes), eager RoPE is ~3.6 ms, and `if n_rep >= 1:` performs a
full k/v repeat-copy even when n_rep == 1. The dit_ext Metal kernel does the
whole chain in one pass (~0.5 ms) and writes q/k/v directly in [B, H, S, D]
attention layout.

Numerics: norm+rope rounding matches the eager chain (fp32 normalize -> bf16
round -> bf16 weight mul; fp32 rope on the rounded values); only the row-sum
reduction order differs, so outputs are exact-or-1ulp vs eager (bounded in
tests). Guarded per call: any unsupported case (non-MPS, non-bf16, D not in
{64, 96, 128}, GQA with n_rep > 1, unexpected freqs layout) falls back to the
original forward. Kill switch: ASFP8_DIT_EXT unset/0 (opt-in).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/lumina_fast]"

_installed = False
_orig_forward = None
_kernel = None


def _eps(norm, dtype):
    e = getattr(norm, "eps", None)
    return float(e) if e is not None else float(torch.finfo(dtype).eps)


def _fast_forward(self, x, x_mask, freqs_cis, transformer_options={}):
    B, S, _ = x.shape
    Hq, Hkv, D = self.n_local_heads, self.n_local_kv_heads, self.head_dim
    qw = getattr(self.q_norm, "weight", None)
    kw = getattr(self.k_norm, "weight", None)
    if (
        _kernel is None
        or x.device.type != "mps"
        or x.dtype != torch.bfloat16
        or D not in (64, 96, 128)
        or Hq != Hkv                      # n_rep > 1 needs the repeat path
        or qw is None or kw is None
        or freqs_cis is None
        or freqs_cis.dtype != torch.float32
        or freqs_cis.shape[-3:] != (D // 2, 2, 2)
        or freqs_cis.numel() != B * S * (D // 2) * 4   # no head-dim broadcast beyond 1
    ):
        return _orig_forward(self, x, x_mask, freqs_cis,
                             transformer_options=transformer_options)

    qkv = self.qkv(x)
    if qkv.dtype != torch.bfloat16:
        return _orig_forward(self, x, x_mask, freqs_cis,
                             transformer_options=transformer_options)
    xq, xk, xv = _kernel.qkv_norm_rope(
        qkv.contiguous(),
        freqs_cis.reshape(B * S, -1).contiguous(),
        qw.to(dtype=torch.bfloat16, device=x.device),
        kw.to(dtype=torch.bfloat16, device=x.device),
        Hq, Hkv, D,
        _eps(self.q_norm, x.dtype), _eps(self.k_norm, x.dtype),
    )
    from comfy.ldm.modules.attention import optimized_attention_masked
    out = optimized_attention_masked(xq, xk, xv, Hq, x_mask, skip_reshape=True,
                                     transformer_options=transformer_options)
    return self.out(out)


_orig_block_forward = None
_orig_silu_gating = None


def _norm_w(norm, dtype, device):
    return norm.weight.to(dtype=dtype, device=device)


def _fast_block_forward(self, x, x_mask, freqs_cis, adaln_input=None,
                        timestep_zero_index=None, transformer_options={}):
    K = _kernel
    if (
        K is None
        or not self.modulation
        or timestep_zero_index is not None
        or x.device.type != "mps"
        or x.dtype != torch.bfloat16
        or adaln_input is None
        or not x.is_contiguous()
    ):
        return _orig_block_forward(self, x, x_mask, freqs_cis,
                                   adaln_input=adaln_input,
                                   timestep_zero_index=timestep_zero_index,
                                   transformer_options=transformer_options)

    scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
    dt, dev = x.dtype, x.device

    h = K.norm_modulate(x, _norm_w(self.attention_norm1, dt, dev),
                        scale_msa.contiguous(), float(self.attention_norm1.eps))
    attn = self.attention(h, x_mask, freqs_cis, transformer_options=transformer_options)
    x = K.norm_gate_residual(x, attn.contiguous(),
                             _norm_w(self.attention_norm2, dt, dev),
                             gate_msa.tanh().contiguous(), float(self.attention_norm2.eps))

    h = K.norm_modulate(x, _norm_w(self.ffn_norm1, dt, dev),
                        scale_mlp.contiguous(), float(self.ffn_norm1.eps))
    ffn = self.feed_forward(h)
    x = K.norm_gate_residual(x, ffn.contiguous(),
                             _norm_w(self.ffn_norm2, dt, dev),
                             gate_mlp.tanh().contiguous(), float(self.ffn_norm2.eps))
    return x


def _fast_silu_gating(self, x1, x3):
    if (
        _kernel is None
        or x1.device.type != "mps"
        or x1.dtype != torch.bfloat16
        or x1.shape != x3.shape
    ):
        return _orig_silu_gating(self, x1, x3)
    return _kernel.silu_mul(x1.contiguous(), x3.contiguous())


def install():
    global _installed, _orig_forward, _orig_block_forward, _orig_silu_gating, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.lumina.model as L
    except Exception as e:
        print(f"{TAG} lumina model module unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None:
        print(f"{TAG} dit_ext kernel unavailable; Lumina fast path stays off.")
        return

    _orig_forward = L.JointAttention.forward
    L.JointAttention.forward = _fast_forward
    _orig_block_forward = L.JointTransformerBlock.forward
    L.JointTransformerBlock.forward = _fast_block_forward
    _orig_silu_gating = L.FeedForward._forward_silu_gating
    L.FeedForward._forward_silu_gating = _fast_silu_gating
    _installed = True
    print(f"{TAG} Lumina/Z-Image fast path on MPS: fused qkv+QKNorm+RoPE, "
          f"norm+modulate, norm+gate+residual, silu-mul via Metal "
          f"(eager fallback per call).")
