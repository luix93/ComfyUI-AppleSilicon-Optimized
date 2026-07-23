"""Fix: INT8-quantized models silently run on CPU (and appear to hang) on MPS.

INT8 nodes (e.g. the `int8-fast` custom node used by Krea2 "int8_mixed"
checkpoints) quantize activations per-row and do the matmul with the integer
GEMM `torch._int_mm` (int8 x int8 -> int32):

    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)            # <- no MPS kernel
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)

PyTorch has no Metal kernel for `aten::_int_mm`. With PYTORCH_ENABLE_MPS_FALLBACK
on (ComfyUI sets it on Mac), the call doesn't crash — it silently copies both
operands MPS->CPU, runs the matmul on the CPU, and copies the int32 result back.
Per Linear, per layer, per sampling step. On a 12 GB Krea2/FLUX model that is
hundreds of CPU round-trips per step, so generation sits at 0/N with the GPU
idle — looks like a hang. (Without the fallback it would instead raise
"not implemented for MPS".)

Triton is the other code path these nodes ship, but Triton has no Metal backend
and usually isn't installed on a Mac, so the pure-PyTorch `_int_mm` path above is
what actually runs.

Fix: on MPS, do the integer matmul on the GPU instead of bouncing to the CPU.

Two tiers:

  * Preferred (ASFP8_INT8_EXT=1, M5-class GPU): the patch #17 int8 W8A8 Metal
    kernel (i8_matmul2d_nt, int8 x int8 -> int32, ~105 TF/s) — EXACT integer
    arithmetic AND much faster than the fp32 emulation. Gated by a one-time
    adversarial parity canary vs an int64 CPU reference (torch.equal); any
    mismatch or build failure latches the tier off for the process.
  * Fallback: float32 GPU emulation. int8 x int8 partial sums are
    integer-valued and |val| <= 127*127*K (~2.5e8 for the widest FLUX layers,
    vs int32's 2.1e9) — but fp32 accumulation is only guaranteed exact below
    2^24, so a running sum beyond that can round its low bits. Real quantized
    activations are zero-mean and land far below that in practice, and the
    result is immediately rescaled to bf16 downstream, so this tier is kept
    as the fallback (never the CPU round-trip, which is the disease this
    patch cures). Non-MPS devices keep the stock kernel untouched.
"""

import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/int_mm]"

_orig = None
_installed = False
_native = None   # int8_ext module once parity-proven; False = latched off


def _native_mod():
    """Lazily build + parity-gate the int8 Metal kernel for _int_mm duty."""
    global _native
    if _native is not None:
        return _native or None
    _native = False
    if os.environ.get("ASFP8_INT8_EXT") != "1":
        return None
    try:
        from .int8_ext import loader
        mod = loader.module()
        if mod is None or not hasattr(mod, "i8_matmul2d_nt"):
            return None
        # Adversarial parity canary: extreme corners (-128/127), ragged K,
        # non-multiple tiles. Reference in int64 on CPU (always exact).
        g = torch.Generator().manual_seed(0)
        for M, K, N in ((3, 129, 5), (16, 4096, 32)):
            a = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8)
            b = torch.randint(-128, 128, (K, N), generator=g, dtype=torch.int8)
            a[:, 0] = -128
            b[0, :] = 127
            ref = (a.long() @ b.long()).to(torch.int32)
            out = mod.i8_matmul2d_nt(a.to("mps"), b.t().contiguous().to("mps"))
            torch.mps.synchronize()
            if not torch.equal(out.cpu(), ref):
                print(f"{TAG} int8 kernel parity canary FAILED at "
                      f"({M},{K},{N}); staying on fp32 emulation.")
                return None
        _native = mod
        print(f"{TAG} torch._int_mm routed to the exact native int8 kernel "
              f"(parity-proven; fp32 emulation remains the fallback).")
    except Exception:
        pass  # stay latched off; fp32 emulation serves every call
    return _native or None


def _int_mm_mps(a, b):
    # Only intervene when an operand is on MPS; everything else keeps the exact
    # native integer kernel (CUDA/CPU).
    if a.device.type == "mps" or b.device.type == "mps":
        mod = _native_mod()
        if (mod is not None
                and a.device.type == "mps" and b.device.type == "mps"
                and a.dtype == torch.int8 and b.dtype == torch.int8
                and a.dim() == 2 and b.dim() == 2):
            try:
                # Kernel wants A[M,K] x B[N,K] contiguous. b arrives [K,N];
                # the common caller pattern is weight.T on a [N,K]-stored
                # weight, so b.t().contiguous() is usually free.
                return mod.i8_matmul2d_nt(a.contiguous(), b.t().contiguous())
            except Exception:
                pass  # fp32 emulation serves this call
        out = torch.mm(a.to(torch.float32), b.to(torch.float32))
        # Sums are integer-valued; round before narrowing so float rounding noise
        # can't truncate e.g. 14.9999997 -> 14. Safe: |out| << 2**31.
        return out.round_().to(torch.int32)
    return _orig(a, b)


def install():
    global _orig, _installed
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if not hasattr(torch, "_int_mm"):
        return

    _orig = torch._int_mm
    torch._int_mm = _int_mm_mps
    _installed = True
    print(f"{TAG} torch._int_mm runs on GPU on MPS instead of falling back to CPU "
          f"(exact int8 kernel where available, float32 emulation otherwise).")
