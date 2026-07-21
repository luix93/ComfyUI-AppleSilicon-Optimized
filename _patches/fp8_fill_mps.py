"""Patch: make fp8 fill/factory ops work on MPS (zeros / ones / full / eye,
their _like variants, and Tensor.zero_ / fill_).

MPS has no fill kernel for float8. Every op that *fills* an fp8 tensor on MPS
fails; only the allocation (`empty` / `empty_like`) and cpu->mps transfer
(`.to()` / `copy_`) work. torch 2.13 made this worse: `torch.zeros` for fp8 on
MPS — which worked through 2.12 — now raises

    RuntimeError: Undefined type Float8_e4m3fn

and `ones` / `full` / `eye` / `fill_` have never worked (pre-2.13 they raised
"Trying to convert Float8_e4m3fn to the MPS backend ..." instead).

Fix the primitive once, same shape as the strided-copy patch (fp8_mps_strided):
try the op natively so any future torch that gains a real fill kernel is used
as-is; on the Float8 error, build the values on CPU (where all fp8 fills work)
and transfer. Bit-exact — a fill has one well-defined bit pattern per value and
the cpu->mps copy is raw bytes. These are cold-path allocations (zero-init
buffers, padding fills), never per-step hot loops.

Disable with ASFP8_DISABLE=fp8_fill_mps if it ever needs to be ruled out.
"""

import sys

import torch

TAG = "[AppleSilicon-Optimized/fp8_fill]"

_installed = False
_FP8 = tuple(
    dt for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dt := getattr(torch, name, None)) is not None
)
# Factories take dtype/device as kwargs; _like variants inherit from the input
# tensor unless overridden; the in-place pair fills an existing mps tensor.
_FACTORIES = ("zeros", "ones", "full", "eye")
_LIKES = ("zeros_like", "ones_like", "full_like")
_INPLACE = ("zero_", "fill_")


def _is_mps(device):
    return device is not None and "mps" in str(device)


def _wrap_factory(orig):
    def fn(*args, **kwargs):
        # Cheap guard: fp8-on-mps only ever arrives via explicit kwargs.
        if kwargs.get("dtype") in _FP8 and _is_mps(kwargs.get("device")):
            try:
                return orig(*args, **kwargs)
            except (RuntimeError, TypeError) as e:
                if "Float8" not in str(e):
                    raise  # a real error, not the MPS fp8 fill limitation
                device = kwargs.pop("device")
                return orig(*args, **kwargs).to(device)
        return orig(*args, **kwargs)

    return fn


def _wrap_like(orig):
    def fn(input, *args, **kwargs):
        dtype = kwargs.get("dtype") or input.dtype
        device = kwargs.get("device") or input.device
        if dtype in _FP8 and _is_mps(device):
            try:
                return orig(input, *args, **kwargs)
            except (RuntimeError, TypeError) as e:
                if "Float8" not in str(e):
                    raise
                kwargs["device"] = "cpu"
                return orig(input, *args, **kwargs).to(device)
        return orig(input, *args, **kwargs)

    return fn


def _wrap_inplace(orig):
    def method(self, *args, **kwargs):
        if self.dtype in _FP8 and self.device.type == "mps":
            try:
                return orig(self, *args, **kwargs)
            except (RuntimeError, TypeError) as e:
                if "Float8" not in str(e):
                    raise
                cpu = torch.empty(self.shape, dtype=self.dtype)
                orig(cpu, *args, **kwargs)
                return self.copy_(cpu)
        return orig(self, *args, **kwargs)

    return method


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    for name in _FACTORIES:
        setattr(torch, name, _wrap_factory(getattr(torch, name)))
    for name in _LIKES:
        setattr(torch, name, _wrap_like(getattr(torch, name)))
    for name in _INPLACE:
        setattr(torch.Tensor, name, _wrap_inplace(getattr(torch.Tensor, name)))
    _installed = True
    print(f"{TAG} fp8 fill/factory ops ({', '.join(_FACTORIES + _LIKES + _INPLACE)}) "
          f"fall back to CPU on MPS (no Metal fp8 fill kernel; torch>=2.13 zeros regression).")
