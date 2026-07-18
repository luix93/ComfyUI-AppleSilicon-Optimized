"""Shared helpers: MPS-safe FP8 decoding via a lookup table.

PyTorch's MPS backend has no FP8 dtype support, so you cannot cast a
float8_e4m3fn / float8_e5m2 tensor to/from float on the GPU. But you *can*:
  - create FP8 tensors on CPU and move them to MPS,
  - bit-view an FP8 tensor as uint8 on MPS,
  - gather/index on MPS.

So we build a 256-entry table mapping every possible FP8 byte to its float
value (decoded on CPU, where the cast works), move that tiny table to MPS once,
then decode any FP8 tensor with a gather: lut[x.view(uint8)]. This is bit-exact
with a real FP8->float cast and runs entirely on the GPU.
"""

import os

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# The opt-in native fast paths implied by the ASFP8_NATIVE=1 master switch.
# Every one auto-falls-back where unsupported (older macOS, pre-M5 GPU, missing
# toolchain), so turning them all on together is safe by construction.
NATIVE_IMPLIES = (
    "ASFP8_FP8_EXT",      # patch #3 fp8-native _scaled_mm + patch #18 build flag
    "ASFP8_FP8_SGEMM",    # patch #18 staged W8A16 backend (preferred; auto-fallback)
    "ASFP8_INT8_EXT",     # patch #17 int8 W8A8 kernel + quant + channelwise epilogue
    "ASFP8_ATTN_ROUTER",  # patch #19 per-call attention routing
    "ASFP8_DIT_EXT",      # patch #21 Lumina/Z-Image fused DiT kernels
)


def apply_native_master_switch(environ=None):
    """ASFP8_NATIVE=1 turns on every opt-in native fast path at once.

    Per-feature vars stay authoritative: any explicitly set value (including
    "0") wins over the master switch, so e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`
    enables everything except the DiT kernels. Returns the list of vars the
    switch actually implied (for the startup log)."""
    if environ is None:
        environ = os.environ
    if environ.get("ASFP8_NATIVE") != "1":
        return []
    implied = []
    for var in NATIVE_IMPLIES:
        if var not in environ:
            environ[var] = "1"
            implied.append(var)
    return implied

_lut_cache = {}


def fp8_to_float_lut(dtype, device, out_dtype=torch.float32):
    """Return a cached 256-entry LUT (`out_dtype`) for FP8 `dtype`, on `device`."""
    key = (dtype, out_dtype, device.type, getattr(device, "index", None))
    lut = _lut_cache.get(key)
    if lut is None:
        # Decode every FP8 byte on CPU (where the cast works), then move to device.
        lut = torch.arange(256, dtype=torch.uint8).view(dtype).to(out_dtype).to(device)
        _lut_cache[key] = lut
    return lut


def decode_fp8(t, out_dtype=torch.float32):
    """Decode an FP8 tensor to `out_dtype` on its own device (MPS-safe).

    bf16 and float32 both represent every FP8 value exactly, so either target is
    bit-exact with a real FP8->float cast.

    A *contiguous* FP8 tensor can be bit-viewed as uint8 and gathered entirely on
    its own device (verified on MPS) — no host round-trip. Only the non-contiguous
    case needs CPU, because MPS cannot run .contiguous() on an FP8 dtype.
    """
    device = t.device
    lut = fp8_to_float_lut(t.dtype, device, out_dtype)
    if t.is_contiguous():
        idx = t.view(torch.uint8).to(torch.long)
    else:
        # MPS can't .contiguous() an FP8 tensor; make it contiguous on CPU, then
        # bring the integer indices back to the target device for the gather.
        idx = t.cpu().contiguous().view(torch.uint8).to(torch.long).to(device)
    return lut[idx]
