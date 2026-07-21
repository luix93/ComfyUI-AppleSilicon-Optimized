"""Patch #18 — standard-fp8 Linear via the fp8-native W8A16 Metal kernel (opt-in).

The standard fp8 path (UNETLoader "weight_dtype: fp8_e4m3fn", no scales) keeps raw
fp8 weights and, on every forward, comfy.ops Linear.forward_comfy_cast_weights calls
cast_bias_weight — which (via patch #6) LUT-decodes the whole [N,K] fp8 weight to a
fresh bf16 tensor, then F.linear consumes it. The decode is pure per-call overhead:
it reads N*K fp8 bytes and writes 2*N*K bf16 bytes, up to ~4x the matmul cost at
small M (M5 Max bench, dev/bench notes).

This patch routes eligible Linears through fp8_ext.fp8_matmul2d_nt (Metal 4.1
tensor-ops matmul2d, half activations x fp8-e4m3 weight in its stored [N,K] layout
-> f32): the weight is consumed as raw fp8 bytes — 1 B/elem, no decode, no bf16
materialization — then bias is added in f32 and the result cast to the input dtype.

Numerics: bf16 -> half activation cast is exact within half's range (bf16 has 7
mantissa bits, half 10; only |x| > 65504 would saturate, and activations are clamped
to that bound first), the kernel accumulates in f32, and a one-time self-check on
the first eligible layer verifies parity against the decode path (rel < 5e-2, in
practice ~1e-4 = bf16-matmul rounding). Bench (M5 Max, FLUX shapes): 1.1-4.2x over
the decode path on MLP-shaped weights across M=64..4096.

Gating (bench-driven): the win is weight-size-gated — route only when
max(N, K) >= ASFP8_FP8_EXT_MIN_DIM (default 8192, shared with patch #3) and
(M <= ASFP8_FP8_W8A16_M_MAX (default 4096) or K >= MIN_DIM); square attention-proj
weights and very-wide-M up-projections stay on the decode path where MPS's own bf16
GEMM wins. Ineligible calls (QuantizedTensor weights, e5m2, LoRA weight/bias
functions, vbar/offloaded weights, non-MPS, fp32 inputs) delegate untouched.

Opt-in: ASFP8_FP8_EXT=1 (the fp8 kernel build flag; same switch as patch #3's fast
path). Kill just this patch with ASFP8_FP8_W8A16=0. Any kernel failure permanently
falls back to the decode path for the rest of the process — a render never breaks.
"""

import os
import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-Optimized/w8a16]"

_installed = False
_orig_forward = None    # disable_weight_init.Linear.forward_comfy_cast_weights
_kernel = None          # the simple fp8_ext module, or False once known-bad
_sgemm = None           # the staged fp8s_ext module (preferred), or False
_self_checked = False
_HALF_MAX = 65504.0

_ELIGIBLE_INPUT_DTYPES = (torch.bfloat16, torch.float16)


def _min_dim():
    """Weight-size threshold (shared knob with patch #3's _scaled_mm fast path)."""
    try:
        return int(os.environ.get("ASFP8_FP8_EXT_MIN_DIM", "8192"))
    except ValueError:
        return 8192


def _sg_min_dim():
    """Weight-size threshold for the staged backend. Lower than the simple
    kernel's: the staged geometry+swizzle wins on square 3072^2 attention
    projections too (M5 Max bench), which the simple kernel had to avoid."""
    try:
        return int(os.environ.get("ASFP8_FP8_SGEMM_MIN_DIM", "3072"))
    except ValueError:
        return 3072


def _sg_variant(M):
    """Bench-derived geometry pick: 32x128 small-M, 64x128 mid, 128x128 large."""
    if M <= 64:
        return 3
    if M <= 2048:
        return 1
    return 0


def _m_max():
    """M above which the N-large (up-proj) shape stops winning (M5 Max bench).
    K-large (down-proj) shapes win at every M and bypass this cap."""
    try:
        return int(os.environ.get("ASFP8_FP8_W8A16_M_MAX", "4096"))
    except ValueError:
        return 4096


def _eligible(module, input):
    """Cheap per-call predicate; no device work. True only when the kernel path is
    both applicable and (per bench) a win."""
    w = module.weight
    if w is None or getattr(w, "dtype", None) != torch.float8_e4m3fn:
        return False  # covers QuantizedTensor (dtype hidden) + e5m2 + non-fp8
    if type(w) not in (torch.Tensor, torch.nn.Parameter):
        return False  # tensor subclasses (QuantizedTensor etc.) keep their own path
    if w.dim() != 2 or w.device.type != "mps" or not w.is_contiguous():
        return False
    if input.dtype not in _ELIGIBLE_INPUT_DTYPES or input.device.type != "mps":
        return False
    if input.shape[-1] != w.shape[1]:
        return False
    if len(module.weight_function) > 0 or len(module.bias_function) > 0:
        return False  # LoRA & co. need the decoded weight
    if hasattr(module, "_v") or hasattr(module, "_prefetch"):
        return False  # vbar / async-offload weights: don't fight the machinery
    N, K = int(w.shape[0]), int(w.shape[1])
    M = input.numel() // input.shape[-1]
    if _sgemm:
        # Staged backend: wins on any large-dim weight (square included) up to
        # m_max; past that the native bf16 GEMM catches up (M5 Max bench).
        return max(N, K) >= _sg_min_dim() and M <= _m_max() and K % 16 == 0
    if max(N, K) < _min_dim():
        return False
    if M > _m_max() and K < _min_dim():
        return False
    return True


def _self_check(module, x2d):
    """One-time parity check of the active kernel backend vs the decode path on
    the first eligible layer's real weight. Returns True if it may be used."""
    global _kernel, _sgemm
    try:
        rows = x2d[: min(int(x2d.shape[0]), 32)].contiguous()
        w = module.weight
        wb = decode_fp8(w, torch.float32)
        ref = rows.float() @ wb.t()
        a_h = torch.clamp(rows, -_HALF_MAX, _HALF_MAX).to(torch.float16).contiguous()
        w_u8 = w.view(torch.uint8).contiguous()
        if _sgemm:
            out = _sgemm.w8a16_f32(a_h, w_u8, -1)
        else:
            out = _kernel.fp8_matmul2d_nt(a_h, w_u8, w.shape[0])
        rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        if rel >= 5e-2:
            print(f"{TAG} self-check failed (rel={rel:.4f}); w8a16 disabled.")
            _kernel = False
            _sgemm = False
            return False
        which = "staged" if _sgemm else "simple"
        print(f"{TAG} self-check ok (rel={rel:.2e}); w8a16 fast path live ({which} kernel).")
        return True
    except Exception as e:
        print(f"{TAG} self-check raised; w8a16 disabled: {e!r}")
        _kernel = False
        _sgemm = False
        return False


def _w8a16_forward(module, input):
    """Kernel-backed replacement for Linear.forward_comfy_cast_weights on the
    standard-fp8 path. Bit-comparable to decode+F.linear (f32 accumulate)."""
    orig_shape = input.shape
    x2d = input.reshape(-1, orig_shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()

    global _self_checked
    if not _self_checked:
        _self_checked = True
        if not _self_check(module, x2d):
            return _orig_forward(module, input)

    w = module.weight
    # Exact within half's range; clamp saturates the (pathological) overflow case
    # instead of producing inf. fp16 inputs are already in range.
    if x2d.dtype == torch.float16:
        a_h = x2d
    else:
        a_h = torch.clamp(x2d, -_HALF_MAX, _HALF_MAX).to(torch.float16)

    bias = module.bias
    if bias is not None:
        if bias.dtype in FP8_DTYPES:
            bias = decode_fp8(bias, torch.float32)
        else:
            bias = bias.to(dtype=torch.float32)

    if _sgemm:
        variant = _sg_variant(int(a_h.shape[0]))
        if input.dtype == torch.bfloat16:
            # Fused epilogue: bf16(acc_f32 + bias_f32) in the kernel store — the
            # f32 [M,N] product never round-trips through global memory.
            out = _sgemm.w8a16_bias_bf16(
                a_h, w.view(torch.uint8), bias.contiguous() if bias is not None else None,
                variant)
            return out.reshape(*orig_shape[:-1], w.shape[0])
        out = _sgemm.w8a16_f32(a_h, w.view(torch.uint8), variant)  # [M,N] f32
    else:
        out = _kernel.fp8_matmul2d_nt(a_h, w.view(torch.uint8), w.shape[0])  # [M,N] f32

    if bias is not None:
        # f32 bias add (matches patch #6's unconditional f32 bias behaviour).
        out = out + bias
    return out.to(input.dtype).reshape(*orig_shape[:-1], w.shape[0])


def _patched_forward(self, input):
    global _kernel, _sgemm
    if (_sgemm or _kernel) and _eligible(self, input):
        try:
            return _w8a16_forward(self, input)
        except Exception as e:
            # Failure chain: staged → simple → decode. Never break a render.
            if _sgemm:
                _sgemm = False
                print(f"{TAG} staged kernel failed ({e!r}); "
                      f"falling back to the simple kernel/decode path.")
            else:
                _kernel = False
                print(f"{TAG} kernel path failed ({e!r}); w8a16 disabled, decode path resumes.")
    return _orig_forward(self, input)


def install():
    global _installed, _orig_forward, _kernel, _sgemm
    if _installed:
        return
    if sys.platform != "darwin":
        return
    fp8_ext_on = os.environ.get("ASFP8_FP8_EXT") == "1"
    sgemm_on = os.environ.get("ASFP8_FP8_SGEMM") == "1"
    if not (fp8_ext_on or sgemm_on):
        return  # opt-in: either kernel switch arms this patch
    if os.environ.get("ASFP8_FP8_W8A16", "1") == "0":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    # Build (or load cached) at startup, not mid-render. Staged is preferred;
    # the simple kernel remains as an in-process fallback.
    if sgemm_on:
        from .fp8s_ext.loader import module as _load_sg
        _sgemm = _load_sg() or False
    if fp8_ext_on:
        from .fp8_ext.loader import module as _load
        _kernel = _load() or False
    if not (_sgemm or _kernel):
        print(f"{TAG} no fp8 kernel backend available; standard-fp8 stays on the decode path.")
        return

    try:
        import comfy.ops as ops
    except Exception as e:
        print(f"{TAG} comfy.ops unavailable ({e!r}); not installed.")
        return

    _orig_forward = ops.disable_weight_init.Linear.forward_comfy_cast_weights
    ops.disable_weight_init.Linear.forward_comfy_cast_weights = _patched_forward
    _installed = True
    if _sgemm:
        print(
            f"{TAG} standard-fp8 Linear routed via staged Metal 4.1 half×fp8 W8A16 kernel "
            f"(fused bias epilogue; min_dim={_sg_min_dim()}, m_max={_m_max()})."
        )
    else:
        print(
            f"{TAG} standard-fp8 Linear routed via Metal 4.1 half×fp8 W8A16 kernel "
            f"(no per-forward weight decode; min_dim={_min_dim()}, m_max={_m_max()})."
        )
