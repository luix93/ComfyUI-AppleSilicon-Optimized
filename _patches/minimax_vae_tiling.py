"""Optional approximate MiniMax H3 video-VAE tiling controls.

The reference decoder uses 256-pixel tiles with at least 64 pixels of overlap.
That overlap deliberately recomputes border regions before blending them.  A
smaller overlap can reduce the number of tiles for some output shapes, but the
decoded pixels are no longer reference-equivalent and seams can become visible.

Nothing is changed unless at least one variable is explicitly set:

  ASFP8_MINIMAX_VAE_TILE_SIZE=<multiple of 16, 64..320>
  ASFP8_MINIMAX_VAE_OVERLAP=<multiple of 16, 16..tile_size-16>

These are process-start flags because the VAE is constructed when its weights
are loaded.  The conservative experimental setting is overlap=32 with the
reference tile size left at 256.  It should only be used after a matched visual
comparison for the intended resolution.
"""

from __future__ import annotations

import functools
import inspect
import os


TAG = "[AppleSilicon-Optimized/minimax_vae_tiling]"

_installed = False


def _parse_multiple(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum or value % 16:
        raise ValueError(
            f"{name} must be a multiple of 16 in [{minimum}, {maximum}], "
            f"got {value}")
    return value


def _requested_tiling() -> tuple[int, int] | None:
    tile_set = "ASFP8_MINIMAX_VAE_TILE_SIZE" in os.environ
    overlap_set = "ASFP8_MINIMAX_VAE_OVERLAP" in os.environ
    if not tile_set and not overlap_set:
        return None
    # H3's decoder grid/quilt artifacts become repeatable above 320px.  Larger
    # tiles may look faster on a single easy clip while producing an invalid
    # blend geometry, so reject them instead of exposing a known-bad range.
    tile_size = _parse_multiple(
        "ASFP8_MINIMAX_VAE_TILE_SIZE", 256, 64, 320)
    overlap = _parse_multiple(
        "ASFP8_MINIMAX_VAE_OVERLAP", 64, 16, 1008)
    if overlap >= tile_size:
        raise ValueError(
            "ASFP8_MINIMAX_VAE_OVERLAP must be smaller than "
            "ASFP8_MINIMAX_VAE_TILE_SIZE")
    return tile_size, overlap


def install():
    global _installed
    if _installed:
        return
    try:
        requested = _requested_tiling()
    except ValueError as exc:
        print(f"{TAG} invalid approximate tiling request ({exc}); disabled.")
        return
    if requested is None:
        return

    try:
        from comfy.ldm.minimax import vae as minimax_vae
    except Exception as exc:
        print(f"{TAG} MiniMax H3 VAE unavailable ({exc!r}); not installed.")
        return

    cls = minimax_vae.MiniMaxH3VideoVAE
    original = cls.__init__
    if getattr(original, "_asfp8_approx_tiling", False):
        _installed = True
        return

    signature = inspect.signature(original)
    tile_size, overlap = requested

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments["tile_size"] = tile_size
        bound.arguments["tile_overlap_min"] = overlap
        return original(*bound.args, **bound.kwargs)

    wrapped._asfp8_approx_tiling = True
    wrapped._asfp8_original = original
    cls.__init__ = wrapped
    _installed = True
    print(f"{TAG} APPROXIMATE MiniMax H3 VAE tiling enabled: "
          f"tile_size={tile_size}, overlap={overlap} "
          "(reference: 256/64).")
