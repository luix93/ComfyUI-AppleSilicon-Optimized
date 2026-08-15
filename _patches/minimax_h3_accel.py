"""Preset-driven MiniMax H3 accelerator: one node over the tuned mechanisms.

The low-level nodes (H3Cache, H3 Sparse Attention) stay available for tuning;
this node exists so the common case is a single choice between "don't change my
output", "go faster but stay strictly close", and "go as fast as the EasyCache
quality envelope allows".

Presets do not invent algorithms. They select an already-implemented cache
engine and/or the sparse attention route with calibrated constants, apply them
in a fixed order (cache first, sparse second), and log exactly what was chosen.
The constants in `H3_PRESETS` are placeholders until the Stage 2 calibration
matrix fills them in; until then `balanced` and `aggressive` refuse to run rather
than silently shipping unvalidated numbers, and `exact` — the default — is the
current optimized path with no approximation at all.
"""

from __future__ import annotations

import logging
from typing import Any

from . import minimax_h3_sparse as sparse
from . import minimax_h3cache as h3cache


TAG = "[AppleSilicon-Optimized/H3Accel]"

PRESETS = ("exact", "balanced", "aggressive")

# Filled in by the Stage 2 calibration matrix. `cache` is None or a dict of
# H3Cache kwargs; `easycache` is None or (threshold, start, end); `sparse` is
# None or a dict of sparse-attention kwargs. `calibrated` gates the preset.
H3_PRESETS: dict[str, dict[str, Any]] = {
    "exact": {
        "calibrated": True,
        "cache": None,
        "easycache": None,
        "sparse": None,
        "note": "no approximation; current optimized flash41/INT8 path",
    },
    "balanced": {
        "calibrated": False,
        "cache": {
            "reuse_threshold": 0.04,
            "start_percent": 0.20,
            "end_percent": 0.85,
            "max_consecutive_skips": 1,
            "audio_weight": 1.0,
            "prediction_mode": h3cache.FIRST_BLOCK_CACHE_MODE,
        },
        "easycache": None,
        # diag64 starting point, NOT a calibrated result. The old values here
        # (start_percent 0.20, and tau 1.25 for `aggressive`) were tuned for the
        # rejected empirical router, whose tau is not comparable to this one.
        "sparse": {
            "mode": "corrected",
            "tau": 1.0,
            "start_percent": 0.50,
            "end_percent": 0.85,
            "dense_leading_blocks": 2,
            "min_sequence_tokens": 8192,
        },
        "note": "pending Stage 2 quality gates",
    },
    "aggressive": {
        "calibrated": False,
        "cache": None,
        "easycache": (0.30, 0.20, 0.90),
        "sparse": {
            "mode": "corrected",
            "tau": 1.0,
            "start_percent": 0.40,
            "end_percent": 0.85,
            "dense_leading_blocks": 2,
            "min_sequence_tokens": 8192,
        },
        "note": "pending Stage 2 quality gates",
    },
}


def _apply_easycache(model, threshold, start, end, verbose):
    """Attach ComfyUI's own EasyCache holder and wrappers to a clone.

    This binds to the native implementation rather than the node class: the
    EasyCache node is a V3 `io.ComfyNode` whose call signature is an API detail,
    while the holder and the three wrappers are the actual mechanism and are what
    a preset needs to reproduce exactly. No part of the algorithm is forked.
    """
    try:
        import comfy.patcher_extension
        from comfy_extras.nodes_easycache import (
            EasyCacheHolder, easycache_sample_wrapper,
            easycache_calc_cond_batch_wrapper, easycache_forward_wrapper,
        )
    except Exception as exc:
        raise ValueError(
            "this ComfyUI build does not provide the native EasyCache "
            f"implementation the selected preset needs ({exc})") from None

    patched = model.clone()
    patched.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
        threshold, start, end, subsample_factor=8, offload_cache_diff=False,
        verbose=verbose,
        output_channels=patched.model.latent_format.latent_channels,
    )
    wrappers = comfy.patcher_extension.WrappersMP
    patched.add_wrapper_with_key(
        wrappers.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
    patched.add_wrapper_with_key(
        wrappers.CALC_COND_BATCH, "easycache", easycache_calc_cond_batch_wrapper)
    patched.add_wrapper_with_key(
        wrappers.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
    return patched


def _reject_existing_state(model):
    transformer_options = model.model_options.get("transformer_options", {})
    if h3cache.STATE_KEY in transformer_options:
        raise ValueError(
            "H3Cache is already attached to this model; remove it before using "
            "the combined accelerator node")
    if sparse.STATE_KEY in transformer_options:
        raise ValueError(
            "H3 sparse attention is already attached to this model; remove it "
            "before using the combined accelerator node")
    if "easycache" in transformer_options:
        raise ValueError(
            "EasyCache is already attached to this model; remove it before "
            "using the combined accelerator node")
    existing_attention = transformer_options.get(
        "optimized_attention_override")
    if (existing_attention is not None
            and not getattr(existing_attention, "_asfp8_dense_backend", False)):
        raise ValueError(
            "another attention override is already installed on this model")


def apply_h3_acceleration(model, preset: str, verbose: bool):
    if preset not in H3_PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    config = H3_PRESETS[preset]
    _reject_existing_state(model)
    if not config["calibrated"]:
        raise ValueError(
            f"the '{preset}' preset is not calibrated yet ({config['note']}). "
            f"Use preset 'exact', or compose the low-level H3Cache and H3 Sparse "
            f"Attention nodes directly while calibration is in progress.")

    selected = []
    patched = model
    if config["easycache"] is not None:
        threshold, start, end = config["easycache"]
        patched = _apply_easycache(patched, threshold, start, end, verbose)
        selected.append(f"EasyCache(threshold={threshold}, start={start}, end={end})")
    elif config["cache"] is not None:
        kwargs = config["cache"]
        patched = h3cache.apply_h3cache(
            patched,
            reuse_threshold=kwargs["reuse_threshold"],
            start_percent=kwargs["start_percent"],
            end_percent=kwargs["end_percent"],
            max_consecutive_skips=kwargs["max_consecutive_skips"],
            audio_weight=kwargs["audio_weight"],
            cached_suffix_blocks=0,
            prediction_mode=kwargs["prediction_mode"],
            verbose=verbose,
        )
        selected.append(
            f"H3Cache({kwargs['prediction_mode']}, threshold={kwargs['reuse_threshold']}, "
            f"window={kwargs['start_percent']}..{kwargs['end_percent']})")

    if config["sparse"] is not None:
        kwargs = config["sparse"]
        patched = sparse.apply_h3_sparse(
            patched,
            mode=kwargs["mode"],
            tau=kwargs["tau"],
            start_percent=kwargs["start_percent"],
            dense_leading_blocks=kwargs["dense_leading_blocks"],
            min_sequence_tokens=kwargs["min_sequence_tokens"],
            verbose=verbose,
            end_percent=kwargs["end_percent"],
        )
        selected.append(
            f"SparseAttention({kwargs['mode']}, tau={kwargs['tau']}, "
            f"start={kwargs['start_percent']})")

    if patched is model:
        # 'exact' adds nothing, but must still hand back its own clone so the
        # caller's model object is never shared with the graph downstream.
        patched = model.clone()
    logging.info("%s preset=%s -> %s", TAG, preset,
                 ", ".join(selected) if selected else "no approximation")
    return patched


class AppleSiliconH3Acceleration:
    """Single-choice accelerator for MiniMax H3 on Apple Silicon."""

    CATEGORY = "Apple Silicon/approximate"
    DESCRIPTION = (
        "One node for the MiniMax H3 acceleration stack. 'exact' (default) "
        "changes nothing and keeps the current optimized path; 'balanced' and "
        "'aggressive' select calibrated cache and sparse-attention settings. "
        "Uncalibrated presets refuse to run rather than ship unvalidated "
        "numbers. Compose the low-level H3Cache and H3 Sparse Attention nodes "
        "for custom tuning."
    )
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (list(PRESETS), {"default": "exact"}),
                "verbose": ("BOOLEAN", {"default": True}),
            },
        }

    def patch(self, model, preset, verbose):
        return (apply_h3_acceleration(model, preset, verbose),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3Acceleration": AppleSiliconH3Acceleration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3Acceleration": "H3 Acceleration (Apple Silicon)",
}
