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
BACKEND_INT8 = "dense_metal_v2"
BACKEND_INT8_ACCURATE = "dense_metal"
# Former names; saved workflows keep running the same backend.
LEGACY_BACKENDS = {"int8_dense_accurate": BACKEND_INT8_ACCURATE,
                   "int8_dense_approximate": BACKEND_INT8}
BACKENDS = (
    BACKEND_CURRENT,
    BACKEND_APPLE,
    BACKEND_PYTORCH,
    BACKEND_COMFY,
    BACKEND_INT8_ACCURATE,
    BACKEND_INT8,
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


def _int8_attention_backend(numerics: str = "running") -> Callable[..., Any]:
    """Approximate dense INT8 for eligible calls; others use the Apple router."""
    from . import int8_attention_mps
    if not _mps_available() or not int8_attention_mps.supported():
        raise ValueError("INT8 attention backends require an M5 Apple Silicon GPU")

    def int8_attention(q, k, v, heads, **kwargs):
        return int8_attention_mps.comfy_attention(
            q, k, v, heads, fallback=attn_router_mps.workflow_attention,
            numerics=numerics, **kwargs)

    def containers(q, k, v, heads, **kwargs):
        q, k, v, consumed = attn_router_mps._take_attention_inputs(q, k, v)
        if not consumed:
            raise TypeError("container_function requires q, k, and v containers")
        try:
            return int8_attention(q, k, v, heads, **kwargs)
        finally:
            q = k = v = None

    int8_attention.container_function = containers
    return int8_attention


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
    if name == BACKEND_INT8:
        return _int8_attention_backend()
    if name == BACKEND_INT8_ACCURATE:
        return _int8_attention_backend("tiled")
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
    backend = LEGACY_BACKENDS.get(backend, backend)
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
        "dense_metal (M5, opt-in; formerly int8_dense_accurate) quantizes unmasked "
        "D=128 attention with at least 8192 keys to INT8 while keeping every key; "
        "it is the Metal counterpart to Comfy Kitchen's INT8 attention, with "
        "Kitchen-style numerics (Q/K rotation, per-tile probabilities). It changes "
        "outputs and suits long video sequences. dense_metal_v2 (formerly "
        "int8_dense_approximate) is the earlier kernel: about 8% less attention "
        "time, with about 2.4x the error on long clips. Other calls "
        "use the Apple router. "
        "Compatible with H3 sparse attention: sparse calls remain outermost and "
        "declined calls use the selected dense backend, so with INT8 selected those "
        "protected calls are quantized too. Comfy Kitchen CUDA attention is not "
        "available on MPS."
    )
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (list(BACKENDS), {
                    "default": BACKEND_CURRENT,
                    "tooltip": "dense_metal (recommended) and dense_metal_v2 are the approximate choices: M5 only, never a default.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, backend=None):
        # Workflows saved under a former name stay executable.
        if backend is None or backend in BACKENDS or backend in LEGACY_BACKENDS:
            return True
        return f"backend must be one of {BACKENDS}"

    def patch(self, model, backend):
        return (apply_attention_backend(model, backend),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconAttentionBackend": AppleSiliconAttentionBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconAttentionBackend": "Attention Backend (Apple Silicon)",
}
