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

TAG = "[AppleSilicon-Optimized/attn_router]"

_installed = False
_default = None        # optimized_attention as chosen by ComfyUI at startup
_default_masked = None
_sub_quad = None
_pytorch = None
_flash_ok = False      # mtlflashattn SDPA reroute confirmed active

_SDPA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_fa41 = None           # mfa4_ext flash kernel (bf16, Metal 4.1) if it built


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


def _seqs(q, k, skip_reshape):
    if skip_reshape:
        return int(q.shape[2]), int(k.shape[2])
    return int(q.shape[1]), int(k.shape[1])


_route_seen = set()


def _route_log(route, q, k, heads, mask, attn_precision, skip_reshape):
    """ASFP8_ATTN_DEBUG=1: log each distinct (route, Lq, Lk, D, mask?, prec)
    once — silent misroutes at new model shapes are otherwise invisible."""
    if os.environ.get("ASFP8_ATTN_DEBUG") != "1":
        return
    Lq, Lk = _seqs(q, k, skip_reshape)
    D = int(q.shape[-1]) if skip_reshape else int(q.shape[-1]) // int(heads)
    key = (route, Lq, Lk, D, mask is not None, str(attn_precision), str(q.dtype))
    if key not in _route_seen:
        _route_seen.add(key)
        sys.stderr.write(f"{TAG} route={route} Lq={Lq} Lk={Lk} D={D} H={heads} "
                         f"mask={mask is not None} prec={attn_precision} {q.dtype}\n")


# Unchunked scores above this size fall back to sub-quad (memory safety: Wan
# at 21k tokens would materialize ~28 GB of scores).
_BMM_MAX_SCORE_BYTES = 6 << 30

# Masked attention goes to memory-safe sub-quad only above this score size;
# below it the default path is both safe and much faster (chunked-python
# sub-quad at LTX's 16.6k-query/512-kv text cross-attention was a large
# regression for a 545 MB score matrix stock SDPA handles trivially).
_MASKED_SUB_QUAD_MIN_SCORE_BYTES = 4 << 30


# bf16 score tiles + merged P buffer: 37 TF/s vs 30 for fp32 tiles (measured
# M5 Max, flux/zit/krea shapes). Scores round to bf16 exactly like the stock
# bmm path's materialized score matrix, so the error class matches ComfyUI's
# default attention (~1.7e-3); ASFP8_FA41_F32S=1 forces fp32 score tiles
# (tighter ~4e-4, ~15-20% slower).
_FA41_BF16S = os.environ.get("ASFP8_FA41_F32S") != "1"

# Attention v2 ("na"): per-simdgroup independent flash following DrawThings'
# shipped NAX config (16 SGs x 16 rows, 2x64-col blocks per softmax pass,
# register-chained P — no threadgroup memory in the hot loop, fp32 softmax
# via reduce_rows/map_iterator, relaxed_precision matmuls). Measured M5 Max:
# 41 TF/s at Wan (S=21504 H40 D128) vs 28 for the cooperative kernel (+46%),
# ~34 vs ~29 at flux-edit; parity 2.4e-5 vs fp32 reference (tighter than the
# production kernel's bf16-tile ~1.7e-3). Requires Lk % 128 == 0, non-causal.
# Kill switch: ASFP8_ATTN_NA=0.
_FA41_NA = os.environ.get("ASFP8_ATTN_NA", "1") == "1"


_kern_seen = set()
_na_dead = False   # latched when flash_fwd_na fails (compile/runtime) once


def _kern_log(kern, q4, k4):
    """ASFP8_ATTN_DEBUG=1: log which flash41 kernel serves each shape once."""
    if os.environ.get("ASFP8_ATTN_DEBUG") != "1":
        return
    key = (kern, tuple(q4.shape), tuple(k4.shape))
    if key not in _kern_seen:
        _kern_seen.add(key)
        sys.stderr.write(f"{TAG} flash41 kernel={kern} q={tuple(q4.shape)} "
                         f"k={tuple(k4.shape)} {q4.dtype}\n")


def _fa41_attention(q, k, v, heads, scale=None, skip_reshape=False,
                    skip_output_reshape=False):
    """Metal 4.1 TensorOps flash attention (mfa4_ext): fused online-softmax,
    no materialized score matrix. Beats the bmm path 1.3-1.5x at D=128
    long-seq shapes."""
    if skip_reshape:
        b, h, s, d = q.shape
        q4, k4, v4 = q.contiguous(), k.contiguous(), v.contiguous()
    else:
        b, s, wide = q.shape
        h, d = heads, wide // heads
        q4, k4, v4 = (t.reshape(b, -1, h, d).transpose(1, 2).contiguous()
                      for t in (q, k, v))
    sc = float(scale) if scale else d ** -0.5
    global _na_dead
    if (_FA41_NA and not _na_dead and hasattr(_fa41, "flash_fwd_na")
            and k4.shape[2] % 128 == 0 and d % 32 == 0 and d <= 128):
        try:
            _kern_log("na", q4, k4)
            out = _fa41.flash_fwd_na(q4, k4, v4, sc)
        except Exception as e:
            # Latch: a compile failure (e.g. a macOS beta changing the
            # cooperative-tensor operand API under us) would otherwise
            # re-attempt the expensive Metal compile on EVERY call.
            _na_dead = True
            sys.stderr.write(f"{TAG} attention v2 kernel unavailable, using "
                             f"cooperative kernel ({str(e)[:200]})\n")
            out = _fa41.flash_fwd_bf16(q4, k4, v4, sc, False, 0, 0, _FA41_BF16S)
    else:
        _kern_log("coop", q4, k4)
        out = _fa41.flash_fwd_bf16(q4, k4, v4, sc, False, 0, 0, _FA41_BF16S)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, h * d)
    return out


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
    B = int(q.shape[0])
    H = int(heads)
    if mask is None:
        # Route 1: long unmasked attention. Small head dims → SDPA → mtlflashattn
        # flash kernel (its fast tier wins there and is memory-bounded). Large
        # head dims (Flux2/Krea2/Z-Image are all D=128) → sub-quad: its bmm
        # rides the matmul units and beats every fused-SDPA tier on MPS
        # (measured: 63ms vs 101ms flash / 173ms stock at Flux2-edit shapes).
        if (L >= _flash_min_seq() and D <= 128 and D % 8 == 0
                and q.dtype in _SDPA_DTYPES and q.dtype == k.dtype == v.dtype):
            if _flash_ok and D <= _flash_max_head_dim():
                _route_log("flash", q, k, heads, mask, attn_precision, skip_reshape)
                return _pytorch(q, k, v, heads, mask=None, attn_precision=attn_precision,
                                skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape, **kwargs)
            # Default upcast only: honor an explicit fp32 attn_precision request.
            # flash41 takes bf16 AND fp16 (Wan runs fp16); math is fp32 inside.
            if (_fa41 is not None and attn_precision is None
                    and q.dtype in (torch.bfloat16, torch.float16)):
                try:
                    out = _fa41_attention(q, k, v, heads, scale=kwargs.get("scale"),
                                          skip_reshape=skip_reshape,
                                          skip_output_reshape=skip_output_reshape)
                    _route_log("flash41", q, k, heads, mask, attn_precision, skip_reshape)
                    return out
                except Exception:
                    pass  # fall through to bmm / sub-quad
            # Score size from ACTUAL Lq*Lk — long-query/short-KV cross attention
            # (Wan 21.5k x 512) has a small score matrix despite the long max seq.
            Lq_, Lk_ = _seqs(q, k, skip_reshape)
            score_bytes = 2 * B * H * Lq_ * Lk_
            if attn_precision is None and score_bytes <= _BMM_MAX_SCORE_BYTES:
                _route_log("bmm", q, k, heads, mask, attn_precision, skip_reshape)
                return _bmm_attention(q, k, v, heads, skip_reshape=skip_reshape,
                                      skip_output_reshape=skip_output_reshape)
            _route_log("sub_quad", q, k, heads, mask, attn_precision, skip_reshape)
            return _sub_quad(q, k, v, heads, mask=None, attn_precision=attn_precision,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
    elif L >= _safe_masked_min_seq():
        # Route 2: masked attention whose SCORE MATRIX is dangerously large →
        # sub-quad, never stock SDPA. Gate on Lq*Lk, not max seq: cross-attention
        # with a long query but short KV (e.g. LTX video->text, 16.6k x 512)
        # materializes only a small score matrix and is far faster on the
        # default path than on chunked-python sub-quad.
        Lq, Lk = _seqs(q, k, skip_reshape)
        score_bytes = 2 * B * H * Lq * Lk
        if score_bytes >= _MASKED_SUB_QUAD_MIN_SCORE_BYTES:
            _route_log("sub_quad(masked)", q, k, heads, mask, attn_precision, skip_reshape)
            return _sub_quad(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
    # Route 3: startup default, untouched.
    _route_log("default", q, k, heads, mask, attn_precision, skip_reshape)
    fn = _default if mask is None else _default_masked
    return fn(q, k, v, heads, mask=mask, attn_precision=attn_precision,
              skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
              **kwargs)


def install():
    global _installed, _default, _default_masked, _sub_quad, _pytorch, _flash_ok, _fa41
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

    # Metal 4.1 flash kernel for the unmasked D>64 bf16 route (mfa4_ext);
    # builds lazily here, falls back to bmm/sub-quad when unavailable.
    try:
        from .mfa4_ext import loader as _fa41_loader
        _fa41 = _fa41_loader.module()
        if _fa41 is not None and not hasattr(_fa41, "flash_fwd_bf16"):
            _fa41 = None
    except Exception:
        _fa41 = None

    # Rebind the dispatcher names, then sweep already-imported model modules
    # that captured the original objects with `from ... import optimized_attention`.
    A.optimized_attention = _router
    A.optimized_attention_masked = _router
    rebound = 0
    for mod in list(sys.modules.values()):
        try:
            # __dict__ lookup, NOT getattr — see rope_fast: getattr sweeps
            # fire third-party lazy-module __getattr__ hooks (boot-log spam).
            if vars(mod).get("optimized_attention") is _default:
                mod.optimized_attention = _router
                rebound += 1
            if vars(mod).get("optimized_attention_masked") is _default_masked:
                mod.optimized_attention_masked = _router
                rebound += 1
        except Exception:
            continue

    _installed = True
    flash = (f"flash(unmasked)>= {_flash_min_seq()} tok, D<={_flash_max_head_dim()} via mtlflashattn"
             if _flash_ok else "flash route OFF (mtlflashattn inactive)")
    if _fa41 is not None:
        na = (_FA41_NA and hasattr(_fa41, "flash_fwd_na"))
        wide = ("flash41 attention v2 'na' per-simdgroup kernel where "
                "Lk%128==0 & D%32==0, cooperative kernel otherwise (Metal "
                "4.1 TensorOps)" if na
                else "flash41 cooperative kernel (Metal 4.1 TensorOps; "
                     "attention v2 off)")
    else:
        wide = "bmm (flash41 unavailable)"
    print(f"{TAG} per-call attention routing on MPS: {flash}; "
          f"unmasked long-seq D>{_flash_max_head_dim()} fp16/bf16 -> {wide}; "
          f"masked >= {_safe_masked_min_seq()} tok -> sub-quad (memory-safe); "
          f"else startup default. ({rebound} module bindings rerouted)")
