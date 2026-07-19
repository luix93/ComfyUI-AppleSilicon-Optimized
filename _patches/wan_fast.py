"""Patch #28 — Wan block fusion on MPS (opt-in ASFP8_DIT_EXT=1).

Wan 2.x blocks are elementwise/copy-heavy in fp16:

  * repeat_e materializes per-frame modulation [B,F,C] into per-token
    [B,L,C] (220 MB at 21.5k x 5120 fp16) SIX times per block;
  * LayerNorm + addcmul modulate chains at 2 sites, gate addcmuls at 2;
  * self-attention prep: wide (C=5120) weighted q/k RMSNorm, eager flux
    RoPE, and the router's [B,S,H*D] -> [B,H,S,D] contiguous reshapes.

wan_ext kernels (dtype-templated half/bfloat, MTLMathModeSafe):
  ln_mod (LN + (1+scale)+shift with per-frame rows — the repeats never
  materialize), addcmul_rows (gated residual, same trick), rms_rope_wide
  (wide RMSNorm + RoPE written straight into attention layout; v gets a
  pure relayout). Attention then runs via optimized_attention with
  skip_reshape (flash41 fp16).

Numerics mirror eager op-by-op (fp32 math, dtype rounding at each eager
rounding point) — exact-or-1ulp; bounded in tests. Guards fall back per
call: non-MPS, unexpected dtypes/shapes, attn patches, Identity norms.
Kill switch: ASFP8_WAN_FAST=0 (under ASFP8_DIT_EXT=1).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/wan_fast]"

_installed = False
_kernel = None
_orig_self_attn = None
_orig_block = None
_announced = set()


def _announce(which):
    if which not in _announced:
        _announced.add(which)
        sys.stderr.write(f"{TAG} fused {which} engaged.\n")


def _freqs_flat(freqs, S, D):
    """Wan passes flux-format freqs [.., S, D/2, 2, 2] fp32 broadcast over
    batch and heads; kernel wants [S, D/2*4] contiguous."""
    if freqs is None or freqs.dtype != torch.float32:
        return None
    if freqs.numel() != S * (D // 2) * 4:
        return None
    return freqs.reshape(S, -1).contiguous()


def _fast_self_attn(self, x, freqs, transformer_options={}):
    K = _kernel
    patches = transformer_options.get("patches", {})
    b, s, c = x.shape
    n, d = self.num_heads, self.head_dim
    qw = getattr(self.norm_q, "weight", None)
    kw = getattr(self.norm_k, "weight", None)
    ff = _freqs_flat(freqs, s, d) if K is not None else None
    if (
        K is None
        or x.device.type != "mps"
        or x.dtype not in (torch.float16, torch.bfloat16)
        or qw is None or kw is None
        or ff is None
        or d % 2 != 0
        or "attn1_patch" in patches
    ):
        return _orig_self_attn(self, x, freqs, transformer_options=transformer_options)

    from comfy.ldm.modules.attention import optimized_attention
    eps = float(getattr(self.norm_q, "eps", 1e-6) or 1e-6)
    dt, dev = x.dtype, x.device
    q4 = K.rms_rope_wide(self.q(x), qw.to(dt).to(dev), ff, n, 2, eps)
    k4 = K.rms_rope_wide(self.k(x), kw.to(dt).to(dev), ff, n, 2, eps)
    v4 = K.rms_rope_wide(self.v(x), qw, ff, n, 0, 0.0)   # relayout only
    _announce("self-attn")
    out = optimized_attention(q4, k4, v4, heads=n, skip_reshape=True,
                              transformer_options=transformer_options)
    return self.o(out)


def _ppf(L, F):
    return L // F if L % F == 0 else L // F + 1


def _fast_block_forward(self, x, e, freqs, context, context_img_len=257,
                        transformer_options={}):
    import comfy.model_management as mm

    K = _kernel
    patches = transformer_options.get("patches", {})
    ok = (
        K is not None
        and x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and "attn2_patch" not in patches
        and e.dtype == x.dtype
    )
    if not ok:
        return _orig_block(self, x, e, freqs, context,
                           context_img_len=context_img_len,
                           transformer_options=transformer_options)

    if e.ndim < 4:
        e6 = (mm.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
    else:
        e6 = (mm.cast_to(self.modulation, dtype=x.dtype, device=x.device).unsqueeze(0) + e).unbind(2)
    if e6[0].dim() != 3:
        return _orig_block(self, x, e, freqs, context,
                           context_img_len=context_img_len,
                           transformer_options=transformer_options)

    x = x.contiguous()
    L = x.shape[1]
    F = e6[0].shape[1]
    ppf = _ppf(L, F)
    eps1 = float(self.norm1.eps)
    _announce("block")

    y = self.self_attn(K.ln_mod(x, e6[0], e6[1], eps1, ppf), freqs,
                       transformer_options=transformer_options)
    x = K.addcmul_rows(x, y, e6[2], ppf)
    del y

    x = x + self.cross_attn(self.norm3(x), context,
                            context_img_len=context_img_len,
                            transformer_options=transformer_options)

    y = self.ffn(K.ln_mod(x, e6[3], e6[4], float(self.norm2.eps), ppf))
    x = K.addcmul_rows(x, y, e6[5], ppf)
    return x


def install():
    global _installed, _kernel, _orig_self_attn, _orig_block
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_WAN_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.wan.model as WM
    except Exception as e:
        print(f"{TAG} comfy wan model unavailable ({e!r}); not installed.")
        return

    from .wan_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "ln_mod"):
        print(f"{TAG} wan_ext kernels unavailable; wan fast path stays off.")
        return

    _orig_self_attn = WM.WanSelfAttention.forward
    WM.WanSelfAttention.forward = _fast_self_attn
    _orig_block = WM.WanAttentionBlock.forward
    WM.WanAttentionBlock.forward = _fast_block_forward
    _installed = True
    print(f"{TAG} Wan block fast path on MPS: fused LN+modulate with per-frame "
          f"rows (no repeat_e copies), gated residuals, wide RMSNorm+RoPE to "
          f"attention layout (eager fallback per call).")
