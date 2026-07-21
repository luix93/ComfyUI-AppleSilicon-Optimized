"""Patch #23 — fast apply_rope on MPS (opt-in via ASFP8_DIT_EXT=1).

comfy.ldm.flux.math.apply_rope is an eager chain (float upcast, reshape into
pairs, two broadcast multiplies, add, reshape back, downcast) that costs
~7.3 ms per call at Flux2-Klein edit shapes ([1, 32, ~8.7k, 128]) — ~0.2 s per
step across 29 attention calls. The dit_ext rope_qk kernel does the identical
math (fp32 pair rotation of the bf16 values, RTE store) in one pass (~0.7 ms).

Applies to every flux-family model that routes through apply_rope (FLUX.2
Klein, Krea2, Chroma, ...). Lumina/Z-Image already fuse rope into their
qkv pre-pass (patch #21) and never reach this seam.

Guarded per call: non-MPS, non-bf16, unexpected shapes/layouts, or a freqs
tensor that doesn't broadcast per-token all fall back to the original.
Same numerics note as patch #21: exact-or-1-ulp (residual ~3e-5 of elements
differ by one bf16 ulp at fp32 rounding boundaries vs MPSGraph's fused chain).

Install rebinds comfy.ldm.flux.math.apply_rope and sweeps sys.modules for
by-value imports (same pattern as the attention router).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/rope_fast]"

_installed = False
_orig = None
_kernel = None


def _fast_apply_rope(xq, xk, freqs_cis):
    K = _kernel
    if (
        K is None
        or xq.device.type != "mps"
        or xq.dtype != torch.bfloat16
        or xk.dtype != torch.bfloat16
        or freqs_cis is None
        or freqs_cis.dtype != torch.float32
        or xq.dim() != 4
        or xk.dim() != 4
        or xq.shape != xk.shape
    ):
        return _orig(xq, xk, freqs_cis)
    B, H, S, D = xq.shape
    if (
        D > 256 or D % 32 != 0
        or freqs_cis.shape[-3:] != (D // 2, 2, 2)
        or freqs_cis.numel() != B * S * (D // 2) * 4  # broadcast over heads only
    ):
        return _orig(xq, xk, freqs_cis)
    qo, ko = K.rope_qk(
        xq.contiguous(), xk.contiguous(),
        freqs_cis.reshape(B * S, -1).contiguous(),
    )
    return qo, ko


def install():
    global _installed, _orig, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.flux.math as FM
    except Exception as e:
        print(f"{TAG} comfy flux math unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "rope_qk"):
        print(f"{TAG} dit_ext rope kernel unavailable; fast rope stays off.")
        return

    _orig = FM.apply_rope
    FM.apply_rope = _fast_apply_rope
    rebound = 0
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "apply_rope", None) is _orig and mod is not FM:
                mod.apply_rope = _fast_apply_rope
                rebound += 1
        except Exception:
            continue
    _installed = True
    print(f"{TAG} apply_rope -> dit_ext Metal kernel on MPS "
          f"(flux-family q/k rotation, eager fallback per call; "
          f"{rebound} module bindings rerouted).")
