"""Unified, opt-in MiniMax H3 acceleration node for Apple Silicon.

The individual mechanisms keep their existing implementation boundaries; this
module is deliberately a small coordinator.  Applying them in one place gives
the workflow a stable order and lets the frontend hide settings for mechanisms
that are disabled without duplicating any inference code.
"""

from __future__ import annotations

from typing import Any

from . import minimax_h3_structure
from . import minimax_h3cache
from . import minimax_h3_sparse


DISABLED = "disabled"

# These are the cache modes worth exposing in the production node.  The two
# first-block modes are the useful accelerators; the oracle modes are
# non-skipping diagnostics. Earlier full-model forecasting experiments and block Taylor
# remain loadable through the deprecated research node, but are intentionally
# not promoted here.
CACHE_MODES = (
    DISABLED,
    minimax_h3cache.FIRST_BLOCK_CACHE_MODE,
    minimax_h3cache.FIRST_BLOCK_CACHE_PER_STEP_MODE,
    minimax_h3cache.ORACLE_PREDICTION_MODE,
    minimax_h3cache.BLOCK_ORACLE_PREDICTION_MODE,
)

# corrected is the production sparse route. Dense reference and oracle are
# retained as non-sparse diagnostic choices; the visibly unsafe hard-drop route is
# not offered by the unified node.
SPARSE_MODES = (
    "corrected",
    "dense_reference",
    "oracle",
)


def apply_h3_acceleration(
    model: Any,
    cache_mode: str,
    cache_reuse_threshold: float,
    cache_start_percent: float,
    cache_end_percent: float,
    cache_max_consecutive_skips: int,
    cache_audio_weight: float,
    cache_verbose: bool,
    token_reduction: bool,
    token_reduction_scale: float,
    token_early_evaluations: int,
    token_detail_restore_block: int,
    token_coarse_restore_block: int,
    token_verbose: bool,
    sparse_attention: bool,
    sparse_mode: str,
    sparse_tau: float,
    sparse_start_percent: float,
    sparse_end_percent: float,
    sparse_dense_leading_blocks: int,
    sparse_min_sequence_tokens: int,
    sparse_verbose: bool,
):
    """Attach the selected H3 mechanisms in their supported composition order."""
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"unsupported H3 cache mode: {cache_mode!r}")
    if sparse_mode not in SPARSE_MODES:
        raise ValueError(f"unsupported H3 sparse mode: {sparse_mode!r}")

    patched = model
    if cache_mode != DISABLED:
        patched = minimax_h3cache.apply_h3cache(
            patched,
            reuse_threshold=float(cache_reuse_threshold),
            start_percent=float(cache_start_percent),
            end_percent=float(cache_end_percent),
            max_consecutive_skips=int(cache_max_consecutive_skips),
            audio_weight=float(cache_audio_weight),
            cached_suffix_blocks=0,
            prediction_mode=cache_mode,
            verbose=bool(cache_verbose),
        )

    if token_reduction:
        patched = minimax_h3_structure.apply_h3_token_reduction(
            patched,
            reduction_scale=float(token_reduction_scale),
            early_evaluations=int(token_early_evaluations),
            detail_restore_block=int(token_detail_restore_block),
            coarse_restore_block=int(token_coarse_restore_block),
            verbose=bool(token_verbose),
        )

    if sparse_attention:
        patched = minimax_h3_sparse.apply_h3_sparse(
            patched,
            mode=sparse_mode,
            tau=float(sparse_tau),
            start_percent=float(sparse_start_percent),
            dense_leading_blocks=int(sparse_dense_leading_blocks),
            min_sequence_tokens=int(sparse_min_sequence_tokens),
            verbose=bool(sparse_verbose),
            block_size=minimax_h3_sparse.BLOCK_SIZE,
            end_percent=float(sparse_end_percent),
            temporal_taps=minimax_h3_sparse.TEMPORAL_TAPS,
        )

    return patched


class AppleSiliconH3Acceleration:
    """Single production-facing control surface for MiniMax H3 acceleration."""

    CATEGORY = "Apple Silicon/H3"
    DESCRIPTION = (
        "Configure MiniMax H3 FirstBlockCache, horizontal target-video token "
        "reduction, and corrected Sol-style sparse attention in one node. "
        "Cache and sparse modes are approximate; oracle choices diagnose their "
        "own stage without skipping it. Layer thinning is intentionally unavailable "
        "because it damaged H3 audio. Disabled sections collapse in the "
        "frontend."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),

            # Cache ---------------------------------------------------------
            "cache_mode": (list(CACHE_MODES), {
                "default": minimax_h3cache.FIRST_BLOCK_CACHE_MODE,
            }),
            "cache_reuse_threshold": ("FLOAT", {
                "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.001,
            }),
            "cache_start_percent": ("FLOAT", {
                "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
            }),
            "cache_end_percent": ("FLOAT", {
                "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
            }),
            "cache_max_consecutive_skips": ("INT", {
                "default": 2, "min": 1, "max": 20, "step": 1,
            }),
            "cache_audio_weight": ("FLOAT", {
                "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
            }),
            "cache_verbose": ("BOOLEAN", {"default": True}),

            # Structural acceleration -------------------------------------
            "token_reduction": ("BOOLEAN", {"default": True}),
            "token_reduction_scale": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
            }),
            "token_early_evaluations": ("INT", {
                "default": 10, "min": 1, "max": 20, "step": 1,
            }),
            "token_detail_restore_block": ("INT", {
                "default": 30, "min": 8, "max": 44, "step": 1,
            }),
            "token_coarse_restore_block": ("INT", {
                "default": 40, "min": 12, "max": 48, "step": 1,
            }),
            "token_verbose": ("BOOLEAN", {"default": False}),

            # Sparse attention --------------------------------------------
            "sparse_attention": ("BOOLEAN", {"default": True}),
            "sparse_mode": (list(SPARSE_MODES), {"default": "corrected"}),
            "sparse_tau": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
            }),
            "sparse_start_percent": ("FLOAT", {
                "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01,
            }),
            "sparse_end_percent": ("FLOAT", {
                "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
            }),
            "sparse_dense_leading_blocks": ("INT", {
                "default": 2, "min": 0, "max": 49, "step": 1,
            }),
            "sparse_min_sequence_tokens": ("INT", {
                "default": 0, "min": 0, "max": 131072, "step": 512,
            }),
            "sparse_verbose": ("BOOLEAN", {"default": True}),
        }}

    def apply(self, model, **settings):
        return (apply_h3_acceleration(model, **settings),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3UnifiedAcceleration": AppleSiliconH3Acceleration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3UnifiedAcceleration": "H3 Apple Silicon Acceleration",
}
