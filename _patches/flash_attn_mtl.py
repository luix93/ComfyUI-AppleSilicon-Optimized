"""Wire mtlflashattn into ComfyUI: a guarded `flash_attn` drop-in + an improved
F.scaled_dot_product_attention on MPS.

Replaces the old vendored v0 Metal kernel (memory-safe but ~3x slower than stock
fused SDPA, so it only fired as an OOM rescue). mtlflashattn ships fast
simdgroup_matrix (v1) and TensorOps (v2 / v2r) kernels that beat stock fused SDPA
by 3-4x at length (more on causal), so attention can now route to it for SPEED and
CORRECTNESS, not just to avoid OOM.

Two integrations, each independently guarded and never fatal:
  1. flash_attn shim  -- `import flash_attn` resolves to the Metal kernels on MPS
     (metal_flash_attn._shim), so models that call flash_attn_func get a native path.
  2. improved SDPA    -- F.scaled_dot_product_attention reroutes to mtlflashattn when
     its gate fires: correctness (max seq >= MTLFLASHATTN_SDPA_MIN_SEQ, default 4096,
     since stock MPS fused SDPA is silently wrong past ~4k tokens), a fast TensorOps
     tier (max seq >= MTLFLASHATTN_SDPA_FAST_MIN_SEQ, default 1024), or OOM rescue
     (score bytes >= MTLFLASHATTN_SDPA_MIN_GB, default 12). Everything else, and any
     kernel error, falls back to stock SDPA. Never crashes the caller.

Env (kill switches / tuning):
  MTLFLASHATTN_SDPA=off    disable the SDPA patch   (legacy alias: APPLESILICON_FP8_SDPA=off)
  MTLFLASHATTN_SHIM=off    disable the flash_attn shim
  MTLFLASHATTN_SDPA_MIN_SEQ / _FAST_MIN_SEQ / _MIN_GB   gate thresholds
     (legacy alias: APPLESILICON_FP8_SDPA_MIN_GB -> MTLFLASHATTN_SDPA_MIN_GB)
"""
from __future__ import annotations

import os

TAG = "[AppleSilicon-Optimized/flash]"

# Map the old node's env knobs onto mtlflashattn's so existing setups keep working.
_LEGACY_ENV = {
    "APPLESILICON_FP8_SDPA": "MTLFLASHATTN_SDPA",
    "APPLESILICON_FP8_SDPA_MIN_GB": "MTLFLASHATTN_SDPA_MIN_GB",
}


def _alias_legacy_env():
    """Mirror legacy APPLESILICON_FP8_SDPA* vars onto MTLFLASHATTN_SDPA* (only if
    the new name isn't already set, so an explicit new-name value always wins)."""
    for old, new in _LEGACY_ENV.items():
        val = os.environ.get(old)
        if val is not None and new not in os.environ:
            os.environ[new] = val


def _fix_v2_template_syntax():
    """Fix mtlflashattn's v2 MSL for the macOS 27 Metal compiler.

    The newer GPUCompiler enforces C++ dependent-template-name rules inside the
    v2 kernel templates: `op.get_destination_cooperative_tensor<...>()` fails to
    parse without the `.template` keyword, so `_v2_supported()` probes False and
    every long sequence silently drops to the slow chunked-torch tier (measured
    3.5-4.7x SLOWER than sub-quad at Wan/Z-Image head dims). Rewrite the MSL
    source strings in-module before anything compiles, and reset the compile
    caches. Idempotent; becomes a no-op once upstream ships the fix."""
    try:
        from metal_flash_attn import _kernel as k
    except Exception:
        return
    pats = ("get_destination_cooperative_tensor<", "get_left_input_cooperative_tensor<",
            "get_right_input_cooperative_tensor<")
    fixed = 0
    for name, val in list(vars(k).items()):
        if not isinstance(val, str) or not any("." + p in val for p in pats):
            continue
        new = val
        for p in pats:
            # "." + p never matches an already-fixed ".template " + p site
            # (the char before the call name there is a space, not a dot).
            new = new.replace("." + p, ".template " + p)
        if new != val:
            setattr(k, name, new)
            fixed += 1
    if fixed:
        try:
            k._v2_support = None  # re-probe with the fixed source
            for cache_name in ("_v2_libs", "_v2_dtype_libs", "_v2_reuse"):
                cache = getattr(k, cache_name, None)
                if isinstance(cache, dict):
                    cache.clear()
        except Exception:
            pass
        print(f"{TAG} patched mtlflashattn v2 MSL for the macOS 27 Metal compiler "
              f"({fixed} source blocks: dependent-template `.template` keyword).",
              flush=True)


def install():
    _alias_legacy_env()

    try:
        import torch
    except Exception:
        return  # no torch -> nothing to patch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return  # not Apple Silicon / MPS -> no-op (keeps this a no-op everywhere else)

    try:
        from metal_flash_attn import _shim
        from metal_flash_attn import sdpa as mfa_sdpa
    except Exception:
        print(
            f"{TAG} mtlflashattn not installed -- flash_attn drop-in and fast SDPA "
            f"are off. Install it with:  pip install mtlflashattn",
            flush=True,
        )
        return

    _fix_v2_template_syntax()

    shim_on = False
    try:
        shim_on = _shim.install()  # appends the guarded flash_attn meta-path finder
    except Exception as e:
        print(f"{TAG} flash_attn shim failed to install ({e})", flush=True)

    sdpa_on = False
    try:
        sdpa_on = mfa_sdpa.install()  # gated F.scaled_dot_product_attention reroute
    except Exception as e:
        print(f"{TAG} SDPA patch failed to install ({e})", flush=True)

    parts = []
    if shim_on:
        parts.append("flash_attn drop-in active")
    if sdpa_on:
        parts.append(
            "F.scaled_dot_product_attention -> mtlflashattn on MPS "
            f"(correctness>={os.environ.get('MTLFLASHATTN_SDPA_MIN_SEQ', '4096')} tok, "
            f"fast-tier>={os.environ.get('MTLFLASHATTN_SDPA_FAST_MIN_SEQ', '1024')} tok, "
            f"oom>={os.environ.get('MTLFLASHATTN_SDPA_MIN_GB', '12')} GB)"
        )
    if parts:
        print(f"{TAG} {'; '.join(parts)}.", flush=True)
    else:
        # Already active (e.g. shim auto-loaded via .pth) or disabled by a kill switch.
        print(
            f"{TAG} mtlflashattn present; flash_attn/SDPA already active or "
            f"disabled via env (MTLFLASHATTN_SHIM/MTLFLASHATTN_SDPA).",
            flush=True,
        )
