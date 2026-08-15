"""Workflow-local attention backend selection for Apple Silicon.

ComfyUI PR #15479 added named attention backends and single-owner Q/K/V
containers. This node uses the same ModelPatcher override contract while
remaining compatible with older ComfyUI builds. It also records the selected
dense backend separately so H3 sparse attention can decline a call without
losing the user's dense-backend choice.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

import torch

from . import attn_router_mps


TAG = "[AppleSilicon-Optimized/attention_backend]"

BACKEND_CURRENT = "automatic/current"
BACKEND_APPLE = "apple_silicon_mps_auto"
BACKEND_PYTORCH = "pytorch"
BACKEND_COMFY = "comfy_default"
BACKENDS = (
    BACKEND_CURRENT,
    BACKEND_APPLE,
    BACKEND_PYTORCH,
    BACKEND_COMFY,
)

_registered = False


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return sys.platform == "darwin" and mps is not None and mps.is_available()


def _pytorch_attention() -> Callable[..., Any]:
    import comfy.ldm.modules.attention as attention

    function = None
    get_function = getattr(attention, "get_attention_function", None)
    if callable(get_function):
        function = get_function("pytorch", None)
    return function or attention.attention_pytorch


def _backend_function(name: str) -> Callable[..., Any]:
    if name == BACKEND_APPLE:
        if not _mps_available():
            raise ValueError(
                "apple_silicon_mps_auto requires an MPS Apple Silicon device")
        return attn_router_mps.workflow_attention
    if name == BACKEND_PYTORCH:
        return _pytorch_attention()
    if name == BACKEND_COMFY:
        return attn_router_mps.comfy_default_attention
    raise ValueError(f"backend must be one of {BACKENDS}")


def _make_override(backend: Callable[..., Any], name: str):
    def optimized_attention_override(_, *args, **kwargs):
        return backend(*args, **kwargs)

    container_function = getattr(backend, "container_function", None)
    if callable(container_function):
        optimized_attention_override.container_function = container_function
    optimized_attention_override._asfp8_dense_backend = True
    optimized_attention_override._asfp8_dense_backend_name = name
    return optimized_attention_override


def is_our_dense_override(value: Any) -> bool:
    return bool(getattr(value, "_asfp8_dense_backend", False))


def apply_attention_backend(model: Any, backend: str):
    """Clone ``model`` and select one exact dense-attention dispatcher."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend == BACKEND_CURRENT:
        return model.clone()

    function = _backend_function(backend)
    current = model.model_options.get("transformer_options", {})
    existing = current.get("optimized_attention_override")
    sparse_attached = "asfp8_h3_sparse" in current
    sparse_override = bool(
        getattr(existing, "_asfp8_sparse_override", False))
    if (existing is not None and not is_our_dense_override(existing)
            and not (sparse_attached and sparse_override)):
        raise ValueError(
            "another attention override is already installed on this model; "
            "place the Apple Silicon backend before a supported composing node "
            "or remove the conflicting backend")

    patched = model.clone()
    options = patched.model_options.setdefault("transformer_options", {})
    options[attn_router_mps.DENSE_BACKEND_KEY] = function
    options[attn_router_mps.DENSE_BACKEND_NAME_KEY] = backend
    # Sparse attention remains the outer approximation. Its dense fallback
    # consults DENSE_BACKEND_KEY, so node order does not change the result.
    if not sparse_attached:
        options["optimized_attention_override"] = _make_override(
            function, backend)
    logging.info("%s selected %s", TAG, backend)
    return patched


def install():
    """Register with ComfyUI's new backend registry when it is available."""
    global _registered
    if _registered or not _mps_available():
        return
    try:
        import comfy.ldm.modules.attention as attention
    except Exception:
        return
    register = getattr(attention, "register_attention_function", None)
    if not callable(register):
        return
    register("apple_silicon_mps", attn_router_mps.workflow_attention)
    get_function = getattr(attention, "get_attention_function", None)
    if (callable(get_function)
            and get_function("apple_silicon_mps", None)
            is not attn_router_mps.workflow_attention):
        logging.warning(
            "%s name 'apple_silicon_mps' is already owned by another backend; "
            "registry integration disabled", TAG)
        return
    _registered = True
    print(f"{TAG} registered workflow-local backend 'apple_silicon_mps'.")


class AppleSiliconAttentionBackend:
    """Select dense attention for one MODEL without changing global flags."""

    CATEGORY = "Apple Silicon/model"
    DESCRIPTION = (
        "Select dense attention for this model only. automatic/current leaves "
        "the incoming model unchanged; apple_silicon_mps_auto uses the measured "
        "per-call Metal router; pytorch selects ComfyUI's PyTorch backend; "
        "comfy_default bypasses the Apple router and uses the startup default. "
        "Compatible with H3 sparse attention: sparse calls remain outermost and "
        "declined calls use the selected dense backend. These are dense backends; "
        "Comfy Kitchen CUDA INT8 attention is intentionally not exposed on MPS."
    )
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (list(BACKENDS), {"default": BACKEND_CURRENT}),
            },
        }

    def patch(self, model, backend):
        return (apply_attention_backend(model, backend),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconAttentionBackend": AppleSiliconAttentionBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconAttentionBackend": "Attention Backend (Apple Silicon)",
}
