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
DENSE_BACKEND_KEY = "asfp8_dense_attention_backend"
DENSE_BACKEND_NAME_KEY = "asfp8_dense_attention_backend_name"

_installed = False
_default = None        # optimized_attention as chosen by ComfyUI at startup
_default_masked = None
_sub_quad = None
_pytorch = None
_flash_ok = False      # mtlflashattn SDPA reroute confirmed active

_SDPA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_fa41 = None           # mfa4_ext flash kernel (bf16, Metal 4.1) if it built


def _is_attention_container(value):
    """Recognize ComfyUI's single-owner attention container by protocol.

    The class landed after this package's minimum supported ComfyUI version,
    so importing it at module load would make older installations fail. A
    tensor has ``take`` but not ``peek``; requiring both keeps normal tensors
    out without tying us to one ComfyUI class identity.
    """
    return callable(getattr(value, "peek", None)) and callable(
        getattr(value, "take", None))


def _take_attention_inputs(q, k, v):
    flags = tuple(_is_attention_container(value) for value in (q, k, v))
    if any(flags) and not all(flags):
        raise TypeError("q, k, and v must all be attention tensor containers")
    if all(flags):
        return q.take(), k.take(), v.take(), True
    return q, k, v, False


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


def _short_bmm_min_seq():
    """Below this, sub-quad's chunking overhead is irrelevant and the startup
    default stays untouched (0 disables the short-sequence bmm route)."""
    try:
        return int(os.environ.get("ASFP8_ATTN_SHORT_BMM_MIN_SEQ", "512"))
    except ValueError:
        return 512


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
    # None = SDPA default d**-0.5; an explicit 0.0 is a real (degenerate)
    # scale and must be honored, so no truthiness test here.
    sc = float(scale) if scale is not None else d ** -0.5
    global _na_dead
    # D=256 (Ideogram) was tried on the na kernel (kBlocks head-split) and
    # LOST to the bmm route: 22.9ms best config vs 12.4ms — the 16x256 fp32
    # accumulator spills registers. Wide heads stay on bmm; the kernel keeps
    # D=256 capability for experiments (flash_fwd_na direct call).
    if (_FA41_NA and not _na_dead and hasattr(_fa41, "flash_fwd_na")
            and d % 32 == 0 and d <= 128):
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


def _bmm_attention(q, k, v, heads, scale=None, skip_reshape=False,
                   skip_output_reshape=False):
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
    scale = float(scale) if scale is not None else d ** -0.5
    # NOTE: fp32-upcast softmax was tried for Ideogram (typography concern):
    # output CONTENT was pixel-identical to bf16 softmax at the same seed and
    # it cost +13s/run (1.85GB fp32 score tensor) — bf16 stays.
    attn = torch.bmm(q2 * scale, k2.transpose(1, 2)).softmax(dim=-1)
    out = torch.bmm(attn, v2)
    out = out.reshape(b, h, -1, d)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, h * d)
    return out


def _router_tensors(q, k, v, heads, mask=None, attn_precision=None,
                    skip_reshape=False, skip_output_reshape=False, **kwargs):
    if q.device.type != "mps":
        return _default(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)
    # Third-party attention hooks (comfy's wrap_attn contract): the stock
    # implementations are wrap_attn-wrapped, so delegating with
    # transformer_options intact fires the override exactly once. Our fast
    # paths below would silently bypass it.
    _topts = kwargs.get("transformer_options")
    if _topts is not None and "optimized_attention_override" in _topts:
        _route_log("override", q, k, heads, mask, attn_precision, skip_reshape)
        fn = _default if mask is None else _default_masked
        return fn(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                  skip_reshape=skip_reshape,
                  skip_output_reshape=skip_output_reshape, **kwargs)
    L, D = _seq_and_head_dim(q, k, heads, skip_reshape)
    B = int(q.shape[0])
    H = int(heads)
    # None = SDPA-default d**-0.5; explicit 0.0 is honored (comfy's stock
    # backends read "scale" from kwargs the same way).
    scale = kwargs.get("scale")
    # Score-matrix element size for the memory gates below: fp32 inputs (or a
    # forced fp32 attn_precision upcast) double the materialized score bytes
    # vs the old hardcoded 2-byte assumption.
    esize = q.element_size()
    if attn_precision is torch.float32:
        esize = max(esize, 4)
    if mask is None:
        # GQA guard: K/V with fewer heads than Q (the enable_gqa contract, or
        # raw mismatched shapes) would be silently mis-folded by the reshapes
        # in the bmm/fa41 fast paths. Stock backends handle GQA; keep it off
        # ours. (Krea2's dedicated GQA path never enters the router.)
        if skip_reshape:
            _gqa = (int(q.shape[1]) != int(k.shape[1])
                    or int(q.shape[1]) != int(v.shape[1]))
        else:
            _gqa = (int(q.shape[-1]) != int(k.shape[-1])
                    or int(q.shape[-1]) != int(v.shape[-1]))
        _gqa = _gqa or bool(kwargs.get("enable_gqa", False))
        # Route 1: long unmasked attention. Small head dims → SDPA → mtlflashattn
        # flash kernel (its fast tier wins there and is memory-bounded). Large
        # head dims (Flux2/Krea2/Z-Image are all D=128) → sub-quad: its bmm
        # rides the matmul units and beats every fused-SDPA tier on MPS
        # (measured: 63ms vs 101ms flash / 173ms stock at Flux2-edit shapes).
        # D cap 256 (was 128): Ideogram 4 runs unmasked D=256 — the flash
        # kernels can't take it (their own gates below keep them at D<=128),
        # but the one-shot bmm beats chunked sub-quad there just like at
        # D=128 (measured: sub-quad baddbmm+bmm was 10% of an Ideogram run).
        if (L >= _flash_min_seq() and D <= 256 and D % 8 == 0
                and q.dtype in _SDPA_DTYPES and q.dtype == k.dtype == v.dtype):
            if _flash_ok and D <= _flash_max_head_dim():
                _route_log("flash", q, k, heads, mask, attn_precision, skip_reshape)
                return _pytorch(q, k, v, heads, mask=None, attn_precision=attn_precision,
                                skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape, **kwargs)
            # Default upcast only: honor an explicit fp32 attn_precision request.
            # flash41 takes bf16 AND fp16 (Wan runs fp16); math is fp32 inside.
            if (_fa41 is not None and not _gqa and attn_precision is None
                    and D <= 128
                    and q.dtype in (torch.bfloat16, torch.float16)):
                try:
                    out = _fa41_attention(q, k, v, heads, scale=scale,
                                          skip_reshape=skip_reshape,
                                          skip_output_reshape=skip_output_reshape)
                    _route_log("flash41", q, k, heads, mask, attn_precision, skip_reshape)
                    return out
                except Exception:
                    pass  # fall through to bmm / sub-quad
            # Score size from ACTUAL Lq*Lk — long-query/short-KV cross attention
            # (Wan 21.5k x 512) has a small score matrix despite the long max seq.
            Lq_, Lk_ = _seqs(q, k, skip_reshape)
            score_bytes = esize * B * H * Lq_ * Lk_
            if (not _gqa and attn_precision is None
                    and score_bytes <= _BMM_MAX_SCORE_BYTES):
                _route_log("bmm", q, k, heads, mask, attn_precision, skip_reshape)
                return _bmm_attention(q, k, v, heads, scale=scale,
                                      skip_reshape=skip_reshape,
                                      skip_output_reshape=skip_output_reshape)
            _route_log("sub_quad", q, k, heads, mask, attn_precision, skip_reshape)
            return _sub_quad(q, k, v, heads, mask=None, attn_precision=attn_precision,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)
        # Route 1b: SHORT unmasked attention that would otherwise fall to a
        # chunked-Python sub-quad default. Video VAE decoders live here — the
        # MiniMax H3 decoder alone issues 1080 attention calls at L=1797,
        # H=32, D=64, where sub-quad runs 4.8 TF/s against bmm's 11.9
        # (measured: 5.54 -> 2.23 ms, 3.6 s of a 15 s decode). Only fires when
        # the startup default IS sub-quad; with pytorch attention selected,
        # SDPA is already at bmm speed and we leave the call alone.
        elif (_default is _sub_quad and not _gqa and attn_precision is None
                and _short_bmm_min_seq() > 0 and L >= _short_bmm_min_seq()
                and D <= 256 and D % 8 == 0
                and q.dtype in (torch.bfloat16, torch.float16)
                and q.dtype == k.dtype == v.dtype
                and esize * B * H * L * L <= _BMM_MAX_SCORE_BYTES):
            _route_log("bmm(short)", q, k, heads, mask, attn_precision, skip_reshape)
            return _bmm_attention(q, k, v, heads, scale=scale,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape)
    elif L >= _safe_masked_min_seq():
        # Route 2: masked attention whose SCORE MATRIX is dangerously large →
        # sub-quad, never stock SDPA. Gate on Lq*Lk, not max seq: cross-attention
        # with a long query but short KV (e.g. LTX video->text, 16.6k x 512)
        # materializes only a small score matrix and is far faster on the
        # default path than on chunked-python sub-quad.
        Lq, Lk = _seqs(q, k, skip_reshape)
        score_bytes = esize * B * H * Lq * Lk
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


def _router(q, k, v, heads, mask=None, attn_precision=None,
            skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Container-compatible public entry point for the tensor router."""
    if _default is None and not _prepare_backend():
        raise RuntimeError("ComfyUI attention backends are unavailable")
    q, k, v, consumed = _take_attention_inputs(q, k, v)
    try:
        return _router_tensors(
            q, k, v, heads, mask=mask, attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
    finally:
        if consumed:
            # The Metal command retains its resources until execution finishes;
            # these are only Python references that would otherwise survive
            # into the caller's output projection.
            q = k = v = None


def workflow_attention(q, k, v, heads, **kwargs):
    """Workflow-selected Apple Silicon backend.

    A ModelPatcher override remains present in transformer_options while it
    invokes this function. Strip that key before entering the router so the
    router does not bounce back into the same override.
    """
    topts = kwargs.get("transformer_options")
    if isinstance(topts, dict) and "optimized_attention_override" in topts:
        stripped = dict(topts)
        stripped.pop("optimized_attention_override", None)
        kwargs = dict(kwargs)
        kwargs["transformer_options"] = stripped
    kwargs.pop("_inside_attn_wrapper", None)
    return _router(q, k, v, heads, **kwargs)


def _workflow_attention_containers(q, k, v, heads, **kwargs):
    q, k, v, consumed = _take_attention_inputs(q, k, v)
    if not consumed:
        raise TypeError("container_function requires q, k, and v containers")
    try:
        return workflow_attention(q, k, v, heads, **kwargs)
    finally:
        q = k = v = None


workflow_attention.container_function = _workflow_attention_containers


def comfy_default_attention(q, k, v, heads, mask=None, **kwargs):
    """Call the startup ComfyUI backend, bypassing our global router."""
    if _default is None and not _prepare_backend():
        raise RuntimeError("ComfyUI attention backends are unavailable")
    topts = kwargs.get("transformer_options")
    if isinstance(topts, dict) and "optimized_attention_override" in topts:
        stripped = dict(topts)
        stripped.pop("optimized_attention_override", None)
        kwargs = dict(kwargs)
        kwargs["transformer_options"] = stripped
    kwargs.pop("_inside_attn_wrapper", None)
    fn = _default if mask is None else _default_masked
    return fn(q, k, v, heads, mask=mask, **kwargs)


def _comfy_default_attention_containers(q, k, v, heads, **kwargs):
    q, k, v, consumed = _take_attention_inputs(q, k, v)
    if not consumed:
        raise TypeError("container_function requires q, k, and v containers")
    try:
        return comfy_default_attention(q, k, v, heads, **kwargs)
    finally:
        q = k = v = None


comfy_default_attention.container_function = (
    _comfy_default_attention_containers)


def dense_without_override(comfy_fn, q, k, v, heads, **kwargs):
    """Run one dense attention call with this router's fast paths available.

    An `optimized_attention_override` installed by a node (e.g. H3 sparse
    attention) must be able to decline a call — short sequences, protected
    blocks, unsupported shapes — and get the *fast* dense path back. Calling the
    `func` comfy hands the override would land on the stock implementation and
    silently skip flash41.

    This re-enters the router with only the override key removed from a shallow
    copy of transformer_options, so exactly one dense call happens and every
    other argument (scale, skip_reshape, skip_output_reshape, mask, ...) is
    preserved. Without the router installed, the supplied comfy function is
    used, which is the correct default behaviour there.
    """
    topts = kwargs.get("transformer_options")
    selected_backend = (
        topts.get(DENSE_BACKEND_KEY) if isinstance(topts, dict) else None)
    if isinstance(topts, dict) and "optimized_attention_override" in topts:
        stripped = dict(topts)
        stripped.pop("optimized_attention_override", None)
        kwargs = dict(kwargs)
        kwargs["transformer_options"] = stripped
    # `_inside_attn_wrapper` is wrap_attn's re-entry guard. It is set on the
    # kwargs comfy passes an override; leaving it in would make the stock
    # wrappers skip their own override check, which is what we want, but it
    # would also be passed down to backends that do not accept it.
    kwargs.pop("_inside_attn_wrapper", None)
    if callable(selected_backend):
        return selected_backend(q, k, v, heads, **kwargs)
    if _installed:
        return _router(q, k, v, heads, **kwargs)
    if comfy_fn is not None:
        return comfy_fn(q, k, v, heads, **kwargs)
    import comfy.ldm.modules.attention as A
    return A.optimized_attention(q, k, v, heads, **kwargs)


def _prepare_backend():
    """Capture ComfyUI backends and load optional Metal routes once.

    Unlike ``install()``, this does not replace global dispatch. It lets the
    workflow-local backend use the router even when ASFP8_ATTN_ROUTER is off.
    """
    global _default, _default_masked, _sub_quad, _pytorch, _flash_ok, _fa41
    if all(value is not None for value in (
        _default, _default_masked, _sub_quad, _pytorch,
    )):
        return True
    try:
        import comfy.ldm.modules.attention as A
    except Exception as e:
        print(f"{TAG} comfy attention module unavailable ({e!r}).")
        return False

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
    return True


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_ATTN_ROUTER") != "1":
        return  # opt-in
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return
    if not _prepare_backend():
        return

    import comfy.ldm.modules.attention as A

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
        wide = ("flash41 attention v2 'na' per-simdgroup kernel "
                "(ragged sequences zero-padded and tail-masked; Metal 4.1 "
                "TensorOps)" if na
                else "flash41 cooperative kernel (Metal 4.1 TensorOps; "
                     "attention v2 off)")
    else:
        wide = "bmm (flash41 unavailable)"
    print(f"{TAG} per-call attention routing on MPS: {flash}; "
          f"unmasked long-seq D>{_flash_max_head_dim()} fp16/bf16 -> {wide}; "
          f"masked >= {_safe_masked_min_seq()} tok -> sub-quad (memory-safe); "
          f"else startup default. ({rebound} module bindings rerouted)")
