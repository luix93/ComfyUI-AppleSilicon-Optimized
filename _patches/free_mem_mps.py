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

TAG = "[AppleSilicon-Optimized/free_mem]"

_installed = False
_orig = None


def _pool_bytes():
    import torch
    try:
        return max(0, torch.mps.driver_allocated_memory()
                   - torch.mps.current_allocated_memory())
    except Exception:
        return 0


def _reserve_bytes(total):
    try:
        gb = float(os.environ.get("ASFP8_MPS_RESERVE_GB", "0"))
    except ValueError:
        gb = 0.0
    if gb > 0:
        return int(gb * (1 << 30))
    return max(16 << 30, int(total * 0.15))


def _patched_get_free_memory(dev=None, torch_free_too=False):
    import psutil
    import torch
    import comfy.model_management as MM
    d = dev if dev is not None else MM.get_torch_device()
    if getattr(d, "type", None) != "mps":
        return _orig(dev, torch_free_too)
    # psutil 'available' can't see torch's resident allocator pool and swings
    # with file-cache noise, so comfy would evict / partially offload models
    # a 128GB unified-memory machine can trivially hold. The truthful figure
    # for "what torch can still allocate" is total RAM minus torch's LIVE
    # tensors minus an OS reserve (ASFP8_MPS_RESERVE_GB; default
    # max(16GB, 15%)). The allocator's cached-but-free pool is reusable and
    # therefore NOT subtracted.
    vm = psutil.virtual_memory()
    total = vm.total
    try:
        live = torch.mps.current_allocated_memory()
    except Exception:
        return _orig(dev, torch_free_too)
    reserve = _reserve_bytes(total)
    free = max(0, total - reserve - live)
    # PRESSURE-GATED cap by what the OS says is obtainable (available memory
    # plus torch's reclaimable pool). macOS `available` UNDERSTATES what a
    # busy-but-healthy box can free up (compressed pages, quittable apps), so
    # steering by it in normal operation re-creates the eviction churn this
    # patch exists to prevent — observed 2026-07-23: an ungated cap put the
    # Wan pair into partial-unload/lowvram mode at 1024², and the rerun's
    # offload/reload churn pushed 20 GB into swap. Only clamp when the OS is
    # genuinely near exhaustion: available below the reserve itself.
    if vm.available < reserve:
        free = min(free, max(0, vm.available + _pool_bytes()))
    if torch_free_too:
        # mem_free_torch's ONLY consumer is free_memory's "torch hoards >25%
        # of free memory -> soft_empty_cache()" branch. Stock MPS code sets
        # mem_free_torch = mem_free_total, making that condition ALWAYS true:
        # every load_models_gpu dumped the entire MPS allocator pool and the
        # next sampler re-grew it from scratch. On unified memory the pool IS
        # the working set — report 0 so the cache survives.
        #
        # Size gate: on <64 GB machines the parked high-water pool is wired
        # memory the OS cannot reclaim, and it squeezes everything else on
        # the box — there, report the REAL reusable pool so the hoard-trim
        # branch can still fire when torch genuinely sits on >25% of free.
        if total < (64 << 30):
            return (free, _pool_bytes())
        return (free, 0)
    return free


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

    # Interleaved-drain unload (default on, ASFP8_FREE_INTERLEAVE=0 disables):
    # comfy's unload_all_models detaches every model — which on unified memory
    # materializes a CPU copy of each weight set — and only drains the MPS
    # pool once at the very end, so an explicit "free everything" transiently
    # holds pool + copies of ALL models (observed: memory-pressure spike to
    # orange after LTX). Draining the pool after EACH model caps the transient
    # at roughly one model. unload_all_models is only called on explicit
    # full-unload requests (the /free endpoint / our Free RAM button), never
    # during normal render-time eviction, so render perf is unaffected.
    if os.environ.get("ASFP8_FREE_INTERLEAVE", "1") == "1":
        try:
            _install_interleaved_unload(MM, torch)
        except Exception as e:
            print(f"{TAG} interleaved unload not installed ({e!r}).")

    _installed = True
    print(f"{TAG} MPS free-memory accounting includes torch's reusable "
          f"allocator pool (stops needless model eviction on unified memory).")


def _install_interleaved_unload(MM, torch):
    import threading

    orig_unload_all = MM.unload_all_models
    orig_model_unload = MM.LoadedModel.model_unload
    tls = threading.local()

    def unload_all_wrapped(*a, **kw):
        tls.draining = True
        try:
            return orig_unload_all(*a, **kw)
        finally:
            tls.draining = False

    def model_unload_wrapped(self, *a, **kw):
        r = orig_model_unload(self, *a, **kw)
        if getattr(tls, "draining", False):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return r

    MM.unload_all_models = unload_all_wrapped
    MM.LoadedModel.model_unload = model_unload_wrapped
