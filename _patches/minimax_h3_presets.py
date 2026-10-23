"""Small, independently selectable H3 recipes backed by the advanced runtime.

Labels describe relative aggressiveness, not a guarantee of output quality.
Only one temporal cache and one sparse attention implementation are attached.
"""
from __future__ import annotations

import logging

OFF = "Off"
FAST_PROTECTED = "Fast protected · experimental"
FAST_AGGRESSIVE = "Fast aggressive · experimental"
LEGACY_FAST = "Fast · experimental"
CACHE_PRESETS = {
    OFF: {},
    "MotionCache · protected": dict(cache_mode="motion_cache", motion_reuse_threshold=0.10,
        motion_start_percent=0.25, motion_end_percent=0.85, motion_max_consecutive_skips=1),
    "MotionCache · balanced": dict(cache_mode="motion_cache", motion_reuse_threshold=0.15),
    "FirstBlockCache · protected": dict(cache_mode="first_block_cache", cache_reuse_threshold=0.10,
        cache_start_percent=0.25, cache_end_percent=0.85, cache_max_consecutive_skips=1),
    "FirstBlockCache · fast": dict(cache_mode="first_block_cache", cache_reuse_threshold=0.20),
    "Spectrum · protected": dict(cache_mode="spectrum_protected"),
    "Spectrum · fast draft": dict(cache_mode="spectrum_fast"),
}
SPARSE_BASE = dict(sparse_attention=True, sparse_mode="corrected", sparse_backend="int8",
    sparse_dense_leading_blocks=2, sparse_dense_layers="0,1,33,34,35",
    sparse_min_sequence_tokens=8192, sparse_memory_budget_gib=0.0, sparse_topk_ratio=0.0)
SPARSE_PRESETS = {
    OFF: {},
    "SOL Metal · conservative": dict(SPARSE_BASE, sparse_tau=0.25,
        sparse_window_mode="sigma_percent", sparse_start_percent=0.50, sparse_end_percent=0.85),
    "SOL Metal · balanced": dict(SPARSE_BASE, sparse_tau=0.50,
        sparse_window_mode="step_fraction", sparse_start_percent=0.20, sparse_end_percent=0.90),
    "SOL Metal · fast": dict(SPARSE_BASE, sparse_tau=1.30,
        sparse_window_mode="step_fraction", sparse_start_percent=0.20, sparse_end_percent=0.90),
    "SOL Metal · hybrid fast": dict(SPARSE_BASE, sparse_backend="int8_hybrid", sparse_tau=1.30,
        sparse_window_mode="step_fraction", sparse_start_percent=0.20, sparse_end_percent=0.90),
    "SOL Metal · top-k wide": dict(SPARSE_BASE, sparse_tau=1.3, sparse_topk_ratio=0.15,
        sparse_window_mode="step_fraction", sparse_start_percent=0.10, sparse_end_percent=0.95),
    "SOL Metal · top-k all-step draft": dict(SPARSE_BASE, sparse_tau=1.3, sparse_topk_ratio=0.10,
        sparse_window_mode="step_fraction", sparse_start_percent=0.0, sparse_end_percent=1.0,
        sparse_dense_leading_blocks=0, sparse_dense_layers=""),
    "Dense Metal · protected": dict(SPARSE_BASE, sparse_backend="dense_int8_v3",
        sparse_window_mode="step_fraction", sparse_start_percent=0.20, sparse_end_percent=0.90),
    "Dense Metal · balanced": dict(SPARSE_BASE, sparse_backend="dense_int8_v3",
        sparse_window_mode="step_fraction", sparse_start_percent=0.10, sparse_end_percent=0.95),
    "Dense Metal · full experimental": dict(SPARSE_BASE, sparse_backend="dense_int8_v3",
        sparse_window_mode="step_fraction", sparse_start_percent=0.0, sparse_end_percent=1.0,
        sparse_dense_leading_blocks=0, sparse_dense_layers=""),
    "VSA Metal · FastH3 Omni (128)": dict(SPARSE_BASE, sparse_backend="vsa_metal",
        sparse_start_percent=0., sparse_end_percent=1., sparse_dense_leading_blocks=0,
        sparse_dense_layers="", sparse_min_sequence_tokens=12288, vsa_tile_size=128),
}
# Removed or renamed attention recipes. Saved workflows run the nearest maintained recipe.
SPARSE_ALIASES = {
    "VSA Metal · ComfyUI (64)": "VSA Metal · FastH3 Omni (128)",
    "Dense INT8 · protected": "Dense Metal · protected",
    "Dense INT8 · full draft": "Dense Metal · full experimental",
    "Dense INT8 v2 · protected": "Dense Metal · protected",
    "Dense INT8 v2 · full experimental": "Dense Metal · full experimental",
    "Dense QK INT8 / PV BF16 · protected": "Dense Metal · protected",
    "Dense QK INT8 / PV BF16 · full experimental": "Dense Metal · full experimental",
    "SOL INT8 · hybrid balanced": "SOL Metal · balanced",
    "Veda Metal · protected": "SOL Metal · fast",
    "Veda Metal · fast draft": "SOL Metal · top-k wide",
    # Renamed October 23, 2026: same recipes.
    "SOL INT8 · conservative": "SOL Metal · conservative",
    "SOL INT8 · balanced": "SOL Metal · balanced",
    "SOL INT8 · fast": "SOL Metal · fast",
    "SOL INT8 · top-k wide": "SOL Metal · top-k wide",
    "SOL INT8 · top-k all-step draft": "SOL Metal · top-k all-step draft",
    "SOL INT8 · hybrid fast": "SOL Metal · hybrid fast",
    "Dense INT8 v3 · protected": "Dense Metal · protected",
    "Dense INT8 v3 · balanced": "Dense Metal · balanced",
    "Dense INT8 v3 · full experimental": "Dense Metal · full experimental",
}
TOKEN_PRESETS = {
    OFF: {},
    "Protected · experimental": dict(token_reduction=True, token_detail_restore_block=8,
        token_coarse_restore_block=12, token_start_percent=0.20, token_end_percent=0.90),
    "Balanced · experimental": dict(token_reduction=True, token_detail_restore_block=20,
        token_coarse_restore_block=30, token_start_percent=0.10, token_end_percent=0.95),
    "Light · experimental": dict(token_reduction=True, token_detail_restore_block=12,
        token_coarse_restore_block=20),
    # The old 40-block restoration produces repeated edges/bands in Turbo I2V.
    # Restore before the protected late layers; keep the later 30-block phase.
    FAST_PROTECTED: dict(token_reduction=True, token_detail_restore_block=30,
        token_coarse_restore_block=33),
    FAST_AGGRESSIVE: dict(token_reduction=True, token_detail_restore_block=30,
        token_coarse_restore_block=40, token_phase_policy="capped_steps",
        token_early_step_fraction=0.5),
}


def resolve_presets(cache, sparse, tokens, log_activity=False, approximate_references=False):
    """Resolve complete settings without mutating a recipe or the advanced node."""
    from .minimax_h3_unified import AppleSiliconH3Acceleration
    if tokens == LEGACY_FAST:
        tokens = FAST_PROTECTED
    if sparse in SPARSE_ALIASES:
        logging.warning("[AppleSilicon-Optimized/H3] attention preset %r is no longer listed; using %r.",
                        sparse, SPARSE_ALIASES[sparse])
        sparse = SPARSE_ALIASES[sparse]
    settings = {key: spec[1].get("default", spec[0][0] if isinstance(spec[0], list) else None)
                for group in AppleSiliconH3Acceleration.INPUT_TYPES().values()
                for key, spec in group.items() if key != "model"}
    settings.update(cache_mode="disabled", sparse_attention=False, token_reduction=False,
        cache_reuse_threshold=0.20, cache_start_percent=0.15, cache_end_percent=0.95,
        cache_max_consecutive_skips=2, cache_audio_weight=1.0,
        motion_strength=1.0, motion_warmup_steps=4, motion_max_consecutive_skips=2,
        motion_start_percent=0.15, motion_end_percent=0.95,
        spectrum_blend_weight=0.5, token_reduction_scale=1.0, token_early_evaluations=10,
        token_start_percent=0.0, token_end_percent=1.0, sparse_topk_ratio=0.0)
    for label, recipes, mechanism in ((cache, CACHE_PRESETS, "cache"),
                                      (sparse, SPARSE_PRESETS, "sparse"),
                                      (tokens, TOKEN_PRESETS, "tokens")):
        if label not in recipes:
            raise ValueError(f"Unknown H3 {mechanism} preset: {label!r}")
        settings.update(recipes[label])
    settings["sparse_approximate_references"] = bool(approximate_references) and settings.get("sparse_backend") in ("int8", "int8_hybrid", "dense_int8", "dense_floatpv", "dense_int8_v2", "dense_int8_v3")
    settings.update(cache_verbose=bool(log_activity), sparse_verbose=bool(log_activity),
                    token_verbose=bool(log_activity))
    return settings


class AppleSiliconH3QuickAcceleration:
    CATEGORY = "Apple Silicon/H3"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Compact H3 acceleration: combine one cache recipe, one attention recipe and optional token reduction. "
        "Default: MotionCache balanced alone. All enabled recipes approximate inference. "
        "Protected recipes compute more of the original model; fast draft recipes can substantially change appearance/motion. "
        "Use the advanced H3 Acceleration node for custom settings. Cache/attention recipes target 20-step sampling; "
        "short-step Turbo combinations need separate quality checks. Fast token recipes have limited Turbo I2V screening, not a quality guarantee. Dense Metal (INT8 v3 kernel) retains all tokens but changes arithmetic (Kitchen-style numerics); protected keeps the step/layer guards, balanced narrows the step guard, full applies at every step/layer. SOL Metal hybrid fast quantizes protected layers only within the sparse window. "
        "VSA Metal requires FastH3 trained gates, with cache and tokens Off; 128-token tiles use the Omni reference-video policy. "
        "Temporary memory uses Auto, and unrelated nodes/environment options still apply when all three selectors are Off."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "cache": (list(CACHE_PRESETS), {"default": "MotionCache · balanced",
                "tooltip": "MotionCache: protected 0.10 / one skip, balanced 0.15 / two skips. FirstBlockCache: protected 0.10, fast 0.20. Spectrum: protected first 4 / last 2 actual evaluations, fast first / last 1. Select one; combine with sparse and tokens below."}),
            "sparse": (list(SPARSE_PRESETS), {"default": OFF,
                "tooltip": "VSA Metal: FastH3 trained gates required; cache and tokens must be Off. 128-token tiles follow Omni's regional video policy and keep 10% of target-video tiles, with all steps/layers eligible. Dense Metal (INT8 v3): protected keeps first 4/last 2 steps and protected layers exact; balanced keeps first 2/last 1; full quantizes video-query attention in every DiT call, retaining every token. Prefix queries stay exact unless visual reference approximation is enabled, but video and audio outputs can change. SOL Metal hybrid fast adds dense INT8 to protected layers inside the SOL window (tau 1.3). SOL Metal (INT8): conservative tau 0.25 late sigma window, balanced tau 0.5 and fast tau 1.3 over steps 20–90%. Top-k wide: experimental, background morphing reported where INT8 Fast was stable; 15% video budget over steps 10–95%, protected layers; all-step draft: 10%, all steps/layers. Budgets also retain global/local blocks and pooled tail. Auto memory budget. Removed recipes from older workflows map to the nearest remaining one."}),
            "tokens": (list(TOKEN_PRESETS), {"default": OFF,
                "tooltip": "Experimental horizontal pairing from block 4. Protected restores 12/8 over steps 20–90%; Balanced restores 30/20 over 10–95%; Light restores 20/12. Fast protected keeps the corrected 33/30 recipe. Fast aggressive restores at 40 for the first half of sampler steps (at most 10), then at 30; greater quality risk. Old Fast selections map to Fast protected. Full spatial output is restored, but content/detail may change. Incompatible mask pairs run unreduced."}),
        }, "optional": {"log_activity": ("BOOLEAN", {"default": False, "tooltip": "Detailed startup, routing and per-step diagnostics. Final work/speedup summaries remain enabled when off."}),
            "approximate_references": ("BOOLEAN", {"default": False, "tooltip": "SOL Metal and Dense Metal presets. Approximate visual references and full-video guides as well as generated video. Text/audio query rows remain protected, but likeness, motion and sound can change indirectly. Off retains the original protection."})}}

    @classmethod
    def VALIDATE_INPUTS(cls, tokens, sparse=None):
        # Keep workflows saved under former labels executable without adding
        # redundant legacy entries to the visible dropdowns.
        # Linked widgets have no value during ComfyUI's pre-execution validation;
        # resolve_presets validates their actual value when the node executes.
        if tokens is not None and tokens not in TOKEN_PRESETS and tokens != LEGACY_FAST:
            return f"Unknown H3 tokens preset: {tokens!r}"
        if sparse is not None and sparse not in SPARSE_PRESETS and sparse not in SPARSE_ALIASES:
            return f"Unknown H3 attention preset: {sparse!r}"
        return True

    def apply(self, model, cache="MotionCache · balanced", sparse=OFF, tokens=OFF, log_activity=False, approximate_references=False):
        from .minimax_h3_unified import apply_h3_acceleration
        return (apply_h3_acceleration(model, **resolve_presets(cache, sparse, tokens, log_activity, approximate_references)),)


NODE_CLASS_MAPPINGS = {"AppleSiliconH3QuickAcceleration": AppleSiliconH3QuickAcceleration}
NODE_DISPLAY_NAME_MAPPINGS = {"AppleSiliconH3QuickAcceleration": "H3 Acceleration · Simple"}
