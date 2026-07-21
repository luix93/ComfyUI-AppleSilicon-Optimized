"""Patch #25 — GQA-native Krea2 attention (opt-in ASFP8_DIT_EXT=1).

Krea2 is GQA (48 query heads, 12 KV heads, D=128) but ComfyUI's
optimized_attention API takes one head count, so the eager path
repeat_interleaves K and V to 48 heads before every attention call — two
~107 MB materializations per call (224 calls per 8-step run) and 4x the KV
read traffic inside the attention kernel. DrawThings runs GQA natively;
this patch closes that structural gap:

  * dit_ext qkv3_norm_rope fuses the post-linear chain (rearrange -> per-head
    QK RMSNorm -> RoPE -> attention layout) in one pass, keeping k/v at their
    NATIVE 12 heads. Krea2 norm semantics differ from flux: fp32 (1+scale)
    weight with a single final bf16 rounding, eps 1e-5 — the kernel matches
    them exactly.
  * attention goes straight to mfa4_ext flash41, whose kernel has supported
    grouped KV (Hq % Hkv == 0) all along — each query head indexes its KV
    group in-kernel; nothing is repeated.

The sigmoid gate and wo projection stay eager. TextFusion adapter blocks use
the same Attention class but pass masks/no freqs, so they fall back cleanly.

Guarded per call: non-MPS, non-bf16, mask, no/mismatched freqs, D > 128,
Hq % Hkv != 0, or a missing flash41 build all fall back to the original
forward. Kill switch: ASFP8_KREA2_FAST=0 (under ASFP8_DIT_EXT=1;
ASFP8_KREA2_FAST_DEBUG=1 logs each distinct fallback reason once).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/krea2_fast]"

_installed = False
_orig_forward = None
_kernel = None
_fa41 = None
_announced = False
_reasons = set()

# Same rationale as the router default: bf16 score tiles match the stock bmm
# path's rounding; ASFP8_FA41_F32S=1 reverts to fp32 tiles everywhere.
_BF16S = os.environ.get("ASFP8_FA41_F32S") != "1"


def _fallback(reason):
    if os.environ.get("ASFP8_KREA2_FAST_DEBUG") == "1" and reason not in _reasons:
        _reasons.add(reason)
        sys.stderr.write(f"{TAG} fallback: {reason}\n")
    return True


def _announce():
    global _announced
    if not _announced:
        _announced = True
        sys.stderr.write(f"{TAG} GQA-native fused attention engaged.\n")


def _w1(norm, device):
    # Krea2 RMSNorm stores the weight zero-centered; effective weight is 1+scale.
    return (norm.scale.to(dtype=torch.float32, device=device) + 1.0).contiguous()


def _fast_attn_forward(self, x, freqs=None, mask=None, transformer_options={}):
    Hq, Hkv, D = self.heads, self.kvheads, self.headdim
    B, S, _ = x.shape
    if (
        (_kernel is None and _fallback("kernel unavailable"))
        or (_fa41 is None and _fallback("flash41 unavailable"))
        or (x.device.type != "mps" and _fallback("non-MPS"))
        or (x.dtype != torch.bfloat16 and _fallback(f"dtype {x.dtype}"))
        or (mask is not None and _fallback("mask"))
        or ((D > 128 or D % 32 != 0) and _fallback(f"D={D}"))
        or ((Hkv <= 0 or Hq % Hkv != 0) and _fallback(f"heads {Hq}/{Hkv}"))
        or ((freqs is None or freqs.dtype != torch.float32
             or freqs.shape[-3:] != (D // 2, 2, 2)
             or freqs.numel() != B * S * (D // 2) * 4)
            and _fallback("freqs layout"))
    ):
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)

    q_lin, k_lin, v_lin = self.wq(x), self.wk(x), self.wv(x)
    gate = self.gate(x)
    if q_lin.dtype != torch.bfloat16 or k_lin.dtype != torch.bfloat16 \
            or v_lin.dtype != torch.bfloat16:
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)

    dev = x.device
    q, k, v = _kernel.qkv3_norm_rope(
        q_lin.contiguous(), k_lin.contiguous(), v_lin.contiguous(),
        freqs.reshape(B * S, -1).contiguous(),
        _w1(self.qknorm.qnorm, dev), _w1(self.qknorm.knorm, dev),
        Hq, Hkv, D, float(self.qknorm.qnorm.eps),
    )
    del q_lin, k_lin, v_lin
    _announce()

    try:
        # Native GQA: k/v stay at Hkv heads; flash41 groups them in-kernel.
        # Prefer the v2 per-simdgroup kernel where its Lk % 128 gate allows
        # (kill switch shared with the router: ASFP8_ATTN_NA=0).
        if (os.environ.get("ASFP8_ATTN_NA", "1") == "1"
                and hasattr(_fa41, "flash_fwd_na") and k.shape[2] % 128 == 0):
            out = _fa41.flash_fwd_na(q, k, v, D ** -0.5)
        else:
            out = _fa41.flash_fwd_bf16(q, k, v, D ** -0.5, False, 0, 0, _BF16S)
    except Exception:
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)
    del q, k, v
    out = out.transpose(1, 2).reshape(B, S, Hq * D)
    return self.wo(out * torch.sigmoid(gate))


def install():
    global _installed, _orig_forward, _kernel, _fa41
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_KREA2_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.krea2.model as K2
    except Exception as e:
        print(f"{TAG} comfy krea2 model unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "qkv3_norm_rope"):
        print(f"{TAG} dit_ext qkv3 kernel unavailable; krea2 fast path stays off.")
        return

    # flash41 (mfa4_ext) for the native-GQA attention; without it the fused
    # prep would still force a repeat for comfy attention — not worth it.
    try:
        from .mfa4_ext import loader as _fa41_loader
        _fa41 = _fa41_loader.module()
        if _fa41 is not None and not hasattr(_fa41, "flash_fwd_bf16"):
            _fa41 = None
    except Exception:
        _fa41 = None
    if _fa41 is None:
        print(f"{TAG} flash41 unavailable; krea2 GQA fast path stays off.")
        return

    _orig_forward = K2.Attention.forward
    K2.Attention.forward = _fast_attn_forward
    _installed = True
    print(f"{TAG} Krea2 GQA-native attention on MPS: fused q/k/v norm+RoPE at "
          f"native head counts + flash41 grouped-KV (no repeat_interleave; "
          f"eager fallback per call).")
