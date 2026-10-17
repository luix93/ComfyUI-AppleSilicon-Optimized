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
import re
import subprocess
import sys

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Native fast paths implied by the master switch, enabled automatically on MPS.
# Individual routes keep their build, input and hardware guards and fallbacks.
NATIVE_IMPLIES = (
    "ASFP8_FP8_EXT",      # patch #3 fp8-native _scaled_mm + patch #18 build flag
    "ASFP8_FP8_SGEMM",    # patch #18 staged W8A16 backend (preferred; auto-fallback)
    "ASFP8_INT8_EXT",     # patch #17 int8 W8A8 kernel + quant + channelwise epilogue
    "ASFP8_ATTN_ROUTER",  # patch #19 per-call attention routing
    "ASFP8_DIT_EXT",      # patch #21 Lumina/Z-Image fused DiT kernels
    "ASFP8_CONV3D",       # patch #26 Conv3d -> sliced conv2d on MPS (video VAEs)
)


def apply_native_master_switch(environ=None):
    """Default to native acceleration on macOS/MPS; explicit values win.

    Per-feature vars stay authoritative: any explicitly set value (including
    "0") wins over the master switch, so e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`
    enables everything except the DiT kernels. Returns the list of vars the
    switch actually implied (for the startup log)."""
    if environ is None:
        environ = os.environ
    if "ASFP8_NATIVE" not in environ:
        if sys.platform != "darwin" or not torch.backends.mps.is_available():
            return []
        environ["ASFP8_NATIVE"] = "1"
    if environ.get("ASFP8_NATIVE") != "1":
        return []
    implied = []
    for var in NATIVE_IMPLIES:
        if var not in environ:
            environ[var] = "1"
            implied.append(var)
    return implied


def apply_default_attention_backend(environ=None):
    """Prefer native PyTorch 2.14 on M5 independently of custom-kernel gates.

    The query-cache's Ultra/build/header guards validate our modified shader,
    not PyTorch's own backend. Reusing those guards here sent M5 Max back to
    the older attention router merely because the pack was installed.
    """
    if environ is None:
        environ = os.environ
    if (environ.get("ASFP8_NATIVE") != "1"
            or "ASFP8_ATTN_BACKEND" in environ or sys.platform != "darwin"
            or not torch.backends.mps.is_available()):
        return []
    # Keep other Torch generations and prereleases on their existing policy.
    # Native SDPA does not need the private-header/source hash used by qcache.
    if not re.fullmatch(r"2\.14\.\d+", torch.__version__.split("+")[0]):
        return []
    try:
        brand = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        return []
    if re.search(r"\bM5\b", brand) is None:
        return []
    environ["ASFP8_ATTN_BACKEND"] = "pytorch"
    implied = ["ASFP8_ATTN_BACKEND"]
    # Explicit modern and legacy SDPA choices retain priority.
    if "MTLFLASHATTN_SDPA" not in environ and "APPLESILICON_FP8_SDPA" not in environ:
        environ["MTLFLASHATTN_SDPA"] = "off"
        implied.append("MTLFLASHATTN_SDPA")
    return implied

def ensure_native_prereqs(environ=None):
    """Self-heal the JIT-kernel build prerequisites (opt-out ASFP8_AUTOFIX=0).

    Fresh Desktop-provisioned venvs ship without ninja and with torch's C++
    headers stripped, which silently disables every Metal extension. Two
    fixes, both idempotent and version-safe:

      * ninja missing  -> pip-install it into the running venv (tiny wheel).
      * torch headers  -> per-torch-version cache under
        ~/.cache/asfp8/torch_headers/<version>/: a HEALTHY venv seeds the
        cache once (snapshot of site-packages/torch/include), a stripped venv
        with a matching cache entry restores from it. A stripped venv with no
        cache stays broken (the loud banner in __init__ reports it).

    Returns a list of human-readable action strings for the startup log.
    """
    if environ is None:
        environ = os.environ
    if not any(environ.get(v) == "1"
               for v in ("ASFP8_INT8_EXT", "ASFP8_DIT_EXT",
                         "ASFP8_FP8_EXT", "ASFP8_FP8_SGEMM")):
        return []
    if environ.get("ASFP8_AUTOFIX", "1") != "1":
        return []
    import shutil
    import subprocess
    import sys as _sys
    actions = []

    try:
        import ninja  # noqa: F401
    except Exception:
        try:
            subprocess.run(
                [_sys.executable, "-m", "pip", "install", "--quiet", "ninja"],
                check=True, timeout=180, capture_output=True)
            import importlib
            importlib.invalidate_caches()
            import ninja  # noqa: F401
            actions.append("installed ninja into this venv (was missing)")
        except Exception as e:
            actions.append(f"could NOT auto-install ninja ({e!r}); "
                           f"run: {_sys.executable} -m pip install ninja")

    try:
        inc = os.path.join(os.path.dirname(torch.__file__), "include")
        sentinel = os.path.join(inc, "torch", "extension.h")
        cache = os.path.join(os.path.expanduser("~"), ".cache", "asfp8",
                             "torch_headers", torch.__version__)
        cache_ok = os.path.exists(os.path.join(cache, "torch", "extension.h"))
        if os.path.exists(sentinel) and not cache_ok:
            # Healthy venv: seed the cache (copy to tmp, rename = atomic-ish).
            tmp = cache + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            shutil.copytree(inc, tmp)
            shutil.rmtree(cache, ignore_errors=True)
            os.rename(tmp, cache)
            actions.append(f"cached torch {torch.__version__} C++ headers "
                           f"for future venvs ({cache})")
        elif not os.path.exists(sentinel) and cache_ok:
            # Stripped venv + matching cache: restore.
            shutil.rmtree(inc, ignore_errors=True)
            shutil.copytree(cache, inc)
            actions.append(f"restored stripped torch {torch.__version__} C++ "
                           f"headers from cache (Desktop venvs ship without "
                           f"them; Metal kernels need torch/extension.h)")
    except Exception as e:
        actions.append(f"header cache/restore failed ({e!r})")
    return actions


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
