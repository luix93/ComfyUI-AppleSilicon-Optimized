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
from . import minimax_h3_veda


DISABLED = "disabled"
SPECTRUM_MODES = ("spectrum_protected", "spectrum_fast")
MOTION_CACHE_MODE = "motion_cache"


def apply_spectrum_preset(model: Any, mode: str, verbose: bool,
                          blend_weight: float = 0.5):
    """Apply our Spectrum runtime directly, with the measured preset."""
    from .h3_runtime.spectrum.api import apply
    return apply(model, protected=mode == "spectrum_protected",
                 blend_weight=float(blend_weight), verbose=bool(verbose))


def apply_motion_cache_preset(model: Any, verbose: bool,
                             reuse_threshold: float = 0.15,
                             motion_strength: float = 1.0,
                             warmup_steps: int = 4,
                             max_consecutive_skips: int = 2,
                             start_percent: float = 0.15,
                             end_percent: float = 0.95):
    from .h3_runtime.motion.runtime import apply

    return apply(
        model, reuse_threshold=float(reuse_threshold), motion_strength=float(motion_strength),
        warmup_steps=int(warmup_steps), max_consecutive_skips=int(max_consecutive_skips),
        start_percent=float(start_percent), end_percent=float(end_percent),
        subsample_factor=8, verbose=bool(verbose),
    )

# First-block reuse and our Spectrum runtime accelerate sampling;
# oracle modes are non-skipping diagnostics. Earlier in-house forecasting and
# block Taylor remain research functions without standalone public nodes.
CACHE_MODES = (
    DISABLED,
    minimax_h3cache.FIRST_BLOCK_CACHE_MODE,
    minimax_h3cache.FIRST_BLOCK_CACHE_PER_STEP_MODE,
    minimax_h3cache.ORACLE_PREDICTION_MODE,
    minimax_h3cache.BLOCK_ORACLE_PREDICTION_MODE,
    *SPECTRUM_MODES,
    MOTION_CACHE_MODE,
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
    sparse_backend: str = "bf16",
    sparse_dense_layers: str = "",
    sparse_window_mode: str = "sigma_percent",
    motion_reuse_threshold: float = 0.15,
    motion_strength: float = 1.0,
    motion_warmup_steps: int = 4,
    motion_max_consecutive_skips: int = 2,
    motion_start_percent: float = 0.15,
    motion_end_percent: float = 0.95,
    spectrum_blend_weight: float = 0.5,
    sparse_memory_budget_gib: float = 0.0,
    veda_predictor: str = minimax_h3_veda.DEFAULT_PREDICTOR,
    veda_sparsity: float = 90.0,
    veda_reference_sparsity: float = 0.0,
    veda_dense_steps: str = "",
    sparse_topk_ratio: float = 0.0,
    token_start_percent: float = 0.0,
    token_end_percent: float = 1.0,
):
    """Attach the selected H3 mechanisms in their supported composition order."""
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"unsupported H3 cache mode: {cache_mode!r}")
    if sparse_mode not in SPARSE_MODES:
        raise ValueError(f"unsupported H3 sparse mode: {sparse_mode!r}")

    # One temporal predictor, followed by optional per-evaluation mechanisms.
    # Register it first so skipped calls never enter the token/sparse wrappers.
    patched = model
    if cache_mode == MOTION_CACHE_MODE:
        patched = apply_motion_cache_preset(
            patched, cache_verbose, motion_reuse_threshold, motion_strength,
            motion_warmup_steps, motion_max_consecutive_skips,
            motion_start_percent, motion_end_percent)
    elif cache_mode in SPECTRUM_MODES:
        patched = apply_spectrum_preset(patched, cache_mode, cache_verbose, spectrum_blend_weight)
    elif cache_mode != DISABLED:
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
            start_percent=float(token_start_percent),
            end_percent=float(token_end_percent),
        )

    if sparse_attention and sparse_backend in ("veda_mlx", "veda_metal"):
        patched = minimax_h3_veda.apply_veda(
            patched, backend=sparse_backend, predictor=veda_predictor,
            sparsity=float(veda_sparsity), reference_sparsity=float(veda_reference_sparsity),
            dense_steps=veda_dense_steps, dense_layers=sparse_dense_layers,
            dense_leading_blocks=int(sparse_dense_leading_blocks),
            start_percent=float(sparse_start_percent), end_percent=float(sparse_end_percent),
            window_mode=sparse_window_mode, min_sequence_tokens=int(sparse_min_sequence_tokens),
            memory_budget_gib=float(sparse_memory_budget_gib), verbose=bool(sparse_verbose))
    elif sparse_attention:
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
            backend=sparse_backend,
            dense_layers=sparse_dense_layers,
            window_mode=sparse_window_mode,
            memory_budget_gib=float(sparse_memory_budget_gib),
            topk_ratio=float(sparse_topk_ratio) if sparse_backend in ("int8", "int8_hybrid") else 0.0,
        )

    return patched


class AppleSiliconH3Acceleration:
    """Advanced control surface for MiniMax H3 acceleration."""

    CATEGORY = "Apple Silicon/H3"
    DESCRIPTION = (
        "Advanced MiniMax H3 acceleration with measured presets and custom controls. "
        "Use H3 Acceleration · Simple for three compact recipe selectors. "
        "New nodes default to MotionCache 0.15 alone: closest content preservation "
        "in our 20-step tests, not bit-exact. MotionCache and Spectrum are bundled. "
        "Combine one cache method with sparse attention and/or token reduction. "
        "Select Exact to disable this node's approximations. "
        "Saved workflows keep their settings. Presets are not validated for 4/6/8-step Turbo LoRAs."
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
                "default": MOTION_CACHE_MODE,
                "tooltip": "Choose one temporal predictor; combine it with sparse attention and/or token reduction. MotionCache and Spectrum are bundled. Spectrum protected keeps first 4/final 2 evaluations actual; fast keeps first/final 1. MotionCache uses the tested 0.15 preset. Tested at 20 steps. Oracle modes do not skip work.",
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
            "cache_verbose": ("BOOLEAN", {"default": False}),

            # Structural acceleration -------------------------------------
            "token_reduction": ("BOOLEAN", {"default": False}),
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
            "sparse_attention": ("BOOLEAN", {"default": False}),
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
            "sparse_verbose": ("BOOLEAN", {"default": False}),
        }, "optional": {
            # Appended for compatibility with existing saved widget positions.
            "sparse_backend": (["bf16", "int8", "dense_int8", "int8_hybrid", "veda_metal", "veda_mlx"], {
                "default": "bf16",
                "tooltip": "BF16/INT8 select SOL. Dense INT8 keeps every token but quantizes QK/PV. SOL hybrid uses dense INT8 in protected layers inside its sampling window; protected steps stay exact. Veda Metal is an optional M5 Ultra tile-list backend; Veda MLX needs the optional mlx package. Both Veda modes need a predictor in models/veda. Each method approximates attention; choose one.",
            }),
            "sparse_dense_layers": ("STRING", {
                "default": "",
                "tooltip": "Additional zero-based layers kept dense, e.g. 0,1,33,34,35. Empty preserves the existing leading-layer policy.",
            }),
            "sparse_window_mode": (["sigma_percent", "step_fraction"], {
                "default": "sigma_percent",
                "tooltip": "sigma_percent preserves older workflows. step_fraction uses the actual sampler schedule: 0.20–0.90 protects the first 4 and last 2 of 20 steps, with 14 sparse steps in between.",
            }),
            "motion_reuse_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1}),
            "motion_warmup_steps": ("INT", {"default": 4, "min": 2, "max": 20, "step": 1}),
            "motion_max_consecutive_skips": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
            "motion_start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "motion_end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "spectrum_blend_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "sparse_memory_budget_gib": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 4096.0, "step": 1.0,
                "tooltip": "SOL/Veda temporary-memory ceiling, not preallocated. 0 = Auto: up to 12.5% of RAM (32 GiB on a 256 GiB Mac), limited by free-memory headroom. Positive values set a manual GiB ceiling. Memory limits do not change attention math.",
            }),
            "veda_predictor": (minimax_h3_veda.predictor_options(), {
                "default": minimax_h3_veda.DEFAULT_PREDICTOR,
                "tooltip": "Veda predictor checkpoint in models/veda. Used only by Veda attention.",
            }),
            "veda_sparsity": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 99.0, "step": 1.0,
                "tooltip": "Percentage of generated video attention tiles skipped. Higher is faster and more approximate."}),
            "veda_reference_sparsity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 99.0, "step": 1.0,
                "tooltip": "0 keeps reference images/video dense. Higher skips reference connections too."}),
            "veda_dense_steps": ("STRING", {"default": "",
                "tooltip": "Additional zero-based steps to keep dense, e.g. 0,1,18,19. Schedule fractions also apply."}),
            "sparse_topk_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01,
                "tooltip": "INT8 SOL only. 0 keeps existing tau routing. Otherwise tau is ignored and this fraction of video blocks is selected, plus global/local protection and pooled tail. Top-k supports up to 262,144 packed tokens (Metal sort shared-memory limit)."}),
            "token_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Fraction of sampler steps to protect at the start. 0 preserves the original token recipes. Protection rounds outward on short schedules."}),
            "token_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Stop token reduction at this fraction of sampler steps; the remaining steps retain all tokens."}),
        }}

    def apply(self, model, **settings):
        return (apply_h3_acceleration(model, **settings),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3UnifiedAcceleration": AppleSiliconH3Acceleration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3UnifiedAcceleration": "H3 Apple Silicon Acceleration",
}
