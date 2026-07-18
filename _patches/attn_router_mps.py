"""Patch #19 — per-call MPS attention router (opt-in ASFP8_ATTN_ROUTER=1).

ComfyUI picks ONE attention implementation globally at startup. On Apple Silicon
both global choices are bad for someone:

  • default (sub-quadratic): memory-safe but chunked-Python slow — video models
    (Wan at 512x512x81f runs self-attention over ~21k tokens) crawl;
  • --use-pytorch-cross-attention: fast IF mtlflashattn intercepts, but every
    call the flash kernel declines (attn_mask — e.g. EVERY Z-Image/Lumina2
    attention) falls through to stock MPS SDPA, which materializes the full
    score matrix: H*Lq*Lk*2 bytes ≈ 37 GB per call at Wan shapes → swap storm.

This patch replaces the global choice with a per-call route:

  1. unmasked and max seq >= ASFP8_ATTN_FLASH_MIN_SEQ (default 4096, matching
     mtlflashattn's correctness gate so interception is guaranteed) and
     head_dim <= 128 → attention_pytorch → F.scaled_dot_product_attention →
     mtlflashattn's TensorOps flash kernel (fast AND memory-bounded);
  2. masked and max seq >= ASFP8_ATTN_SAFE_MASKED_MIN_SEQ (default 2048) →
     attention_sub_quad, ALWAYS — stock SDPA never sees a long masked sequence,
     regardless of which global attention mode ComfyUI started with;
  3. everything else → the startup default, untouched.

Model modules bind `optimized_attention` by value at import time (before custom
nodes load), so install() also sweeps sys.modules and rebinds any module-level
references to the original dispatch targets.

Requires mtlflashattn (patch #5) for route 1; without it, route 1 stays on the
default path and only the masked-safety route is active. Non-MPS tensors always
take the original path. Kill switch: ASFP8_ATTN_ROUTER=0/unset (opt-in).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/attn_router]"

_installed = False
_default = None        # optimized_attention as chosen by ComfyUI at startup
_default_masked = None
_sub_quad = None
_pytorch = None
_flash_ok = False      # mtlflashattn SDPA reroute confirmed active

_SDPA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _flash_min_seq():
    try:
        return int(os.environ.get("ASFP8_ATTN_FLASH_MIN_SEQ", "4096"))
    except ValueError:
        return 4096


def _flash_max_head_dim():
    """mtlflashattn's fast tier only beats matmul-based attention at small head
    dims (measured M5 Max, S 4-16k, B=1: D=64 -> 1.8x FASTER, D=96 -> 0.8x,
    D=128 -> 0.6-1.0x while sub-quad bmm attention wins outright). Above this,
    route 1 hands off to sub-quad instead."""
    try:
        return int(os.environ.get("ASFP8_ATTN_FLASH_MAX_HEAD_DIM", "64"))
    except ValueError:
        return 64


def _safe_masked_min_seq():
    try:
        return int(os.environ.get("ASFP8_ATTN_SAFE_MASKED_MIN_SEQ", "2048"))
    except ValueError:
        return 2048


def _seq_and_head_dim(q, k, heads, skip_reshape):
    """(max_seq, head_dim) for either input layout ([B,H,L,D] or [B,L,H*D])."""
    if skip_reshape:
        return max(int(q.shape[2]), int(k.shape[2])), int(q.shape[-1])
    return max(int(q.shape[1]), int(k.shape[1])), int(q.shape[-1]) // int(heads)


# Unchunked scores above this size fall back to sub-quad (memory safety: Wan
# at 21k tokens would materialize ~28 GB of scores).
_BMM_MAX_SCORE_BYTES = 6 << 30


def _bmm_attention(q, k, v, heads, skip_reshape=False, skip_output_reshape=False):
    """Lean unmasked attention: one QK^T bmm, in-dtype softmax, one AV bmm.
    Beats sub-quad's chunked path by ~25% at Flux2-edit shapes (47 vs 63 ms)
    — same numerics class (no fp32 upcast, matching comfy's default)."""
    if skip_reshape:
        b, h, s, d = q.shape
        q2, k2, v2 = (t.reshape(b * h, -1, d) for t in (q, k, v))
    else:
        b, s, wide = q.shape
        d = wide // heads
        h = heads
        q2, k2, v2 = (t.reshape(b, -1, h, d).transpose(1, 2).reshape(b * h, -1, d)
                      for t in (q, k, v))
    scale = d ** -0.5
    attn = torch.bmm(q2 * scale, k2.transpose(1, 2)).softmax(dim=-1)
    out = torch.bmm(attn, v2)
    out = out.reshape(b, h, -1, d)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, h * d)
    return out


def _router(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
            skip_output_reshape=False, **kwargs):
    if q.device.type != "mps":
        return _default(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)
    L, D = _seq_and_head_dim(q, k, heads, skip_reshape)
    if mask is None:
        # Route 1: long unmasked attention. Small head dims → SDPA → mtlflashattn
        # flash kernel (its fast tier wins there and is memory-bounded). Large
        # head dims (Flux2/Krea2/Z-Image are all D=128) → sub-quad: its bmm
        # rides the matmul units and beats every fused-SDPA tier on MPS
        # (measured: 63ms vs 101ms flash / 173ms stock at Flux2-edit shapes).
        if (L >= _flash_min_seq() and D <= 128 and D % 8 == 0
                and q.dtype in _SDPA_DTYPES and q.dtype == k.dtype == v.dtype):
            if _flash_ok and D <= _flash_max_head_dim():
                return _pytorch(q, k, v, heads, mask=None, attn_precision=attn_precision,
                                skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape, **kwargs)
            # Default upcast only: honor an explicit fp32 attn_precision request.
            B = q.shape[0] if skip_reshape else q.shape[0]
            H = heads
            score_bytes = 2 * B * H * L * L
            if attn_precision is None and score_bytes <= _BMM_MAX_SCORE_BYTES:
                return _bmm_attention(q, k, v, heads, skip_reshape=skip_reshape,
                                      skip_output_reshape=skip_output_reshape)
            return _sub_quad(q, k, v, heads, mask=None, attn_precision=attn_precision,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
    elif L >= _safe_masked_min_seq():
        # Route 2: long masked attention → sub-quad, never stock SDPA. Protects
        # against the O(L^2) score materialization even when ComfyUI was started
        # with --use-pytorch-cross-attention.
        return _sub_quad(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                         skip_reshape=skip_reshape,
                         skip_output_reshape=skip_output_reshape, **kwargs)
    # Route 3: startup default, untouched.
    fn = _default if mask is None else _default_masked
    return fn(q, k, v, heads, mask=mask, attn_precision=attn_precision,
              skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
              **kwargs)


def install():
    global _installed, _default, _default_masked, _sub_quad, _pytorch, _flash_ok
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_ATTN_ROUTER") != "1":
        return  # opt-in
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.modules.attention as A
    except Exception as e:
        print(f"{TAG} comfy attention module unavailable ({e!r}); not installed.")
        return

    _default = A.optimized_attention
    _default_masked = A.optimized_attention_masked
    _sub_quad = A.attention_sub_quad
    _pytorch = A.attention_pytorch

    # Route 1 requires mtlflashattn's SDPA reroute (patch #5). Confirm it is
    # actually patched in, not merely installed.
    try:
        import metal_flash_attn.sdpa as _mfa_sdpa
        import metal_flash_attn._kernel as _mfa_kernel
        import torch.nn.functional as F
        _flash_ok = (getattr(_mfa_sdpa, "_orig", None) is not None
                     and F.scaled_dot_product_attention is _mfa_sdpa._sdpa
                     # only route when a FAST TensorOps tier exists; without it
                     # mtlflashattn's fallback tier is slower than sub-quad
                     # (measured 3.5-4.7x at Wan/Z-Image head dims on M5 Max)
                     and bool(_mfa_kernel._v2_supported()))
    except Exception:
        _flash_ok = False

    # Rebind the dispatcher names, then sweep already-imported model modules
    # that captured the original objects with `from ... import optimized_attention`.
    A.optimized_attention = _router
    A.optimized_attention_masked = _router
    rebound = 0
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "optimized_attention", None) is _default:
                mod.optimized_attention = _router
                rebound += 1
            if getattr(mod, "optimized_attention_masked", None) is _default_masked:
                mod.optimized_attention_masked = _router
                rebound += 1
        except Exception:
            continue

    _installed = True
    flash = (f"flash(unmasked)>= {_flash_min_seq()} tok, D<={_flash_max_head_dim()} via mtlflashattn"
             if _flash_ok else "flash route OFF (mtlflashattn inactive)")
    print(f"{TAG} per-call attention routing on MPS: {flash}; "
          f"unmasked long-seq D>{_flash_max_head_dim()} -> sub-quad (bmm beats fused SDPA); "
          f"masked >= {_safe_masked_min_seq()} tok -> sub-quad (memory-safe); "
          f"else startup default. ({rebound} module bindings rerouted)")
