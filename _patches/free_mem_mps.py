"""Patch #30 — MPS free-memory accounting includes torch's allocator pool.

comfy.model_management.get_free_memory on MPS returns
psutil.virtual_memory().available — which does NOT count torch's MPS
allocator cache (driver_allocated - current_allocated). That pool is
genuinely reusable by new torch allocations, and our prestartup watermark
tuning deliberately keeps it resident, so after a big model loads comfy
sees a nearly-full machine and EVICTS other models it could have kept:
the Wan 2.2 high/low-noise pair (13.9 GB each on a 128 GB machine!)
re-uploads on every sampler switch, ~14 s per run.

Fix: wrap get_free_memory and add the reusable pool back on MPS. This is
honest accounting, not an override — real memory pressure still unloads.
Kill switch: ASFP8_FREE_MEM=0.
"""

import os
import sys

TAG = "[AppleSilicon-FP8/free_mem]"

_installed = False
_orig = None


def _pool_bytes():
    import torch
    try:
        return max(0, torch.mps.driver_allocated_memory()
                   - torch.mps.current_allocated_memory())
    except Exception:
        return 0


def _patched_get_free_memory(dev=None, torch_free_too=False):
    import comfy.model_management as MM
    res = _orig(dev, torch_free_too)
    d = dev if dev is not None else MM.get_torch_device()
    if getattr(d, "type", None) != "mps":
        return res
    pool = _pool_bytes()
    if pool <= 0:
        return res
    if torch_free_too:
        return (res[0] + pool, res[1] + pool)
    return res + pool


def install():
    global _installed, _orig
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_FREE_MEM", "1") != "1":
        return  # kill switch
    import torch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return
    if not hasattr(getattr(torch, "mps", None), "driver_allocated_memory"):
        return

    try:
        import comfy.model_management as MM
    except Exception as e:
        print(f"{TAG} comfy.model_management unavailable ({e!r}); not installed.")
        return

    _orig = MM.get_free_memory
    MM.get_free_memory = _patched_get_free_memory
    _installed = True
    print(f"{TAG} MPS free-memory accounting includes torch's reusable "
          f"allocator pool (stops needless model eviction on unified memory).")
