"""Supply upstream conditioning omitted by raw SeedVR2 7B FP16 checkpoints.

ComfyUI detects their architecture but refuses the model class without the
two fixed conditioning buffers. Inject those buffers before class matching;
leave all existing tensors, dtype selection and weight loading to ComfyUI.
"""
from functools import wraps
import hashlib
import os
from pathlib import Path
import sys

TAG = "[AppleSilicon-Optimized/seedvr2_loader]"
_installed = False
_ASSET = Path(__file__).with_name("seedvr2_assets") / "conditioning.safetensors"
_SHA256 = "41faea5bb586cf806237b660e80fc66c3d8472268806863fd1e2d5ebec00ba3a"
_SHAPES = {"positive_conditioning": (58, 5120), "negative_conditioning": (64, 5120)}
_CONFIG = {"image_model": "seedvr2", "vid_dim": 3072, "heads": 24,
           "num_layers": 36, "mm_layers": 36, "mlp_type": "normal"}


def _conditioning():
    import torch
    from safetensors.torch import load

    data = _ASSET.read_bytes()
    if hashlib.sha256(data).hexdigest() != _SHA256:
        raise RuntimeError(f"{TAG} bundled upstream conditioning checksum mismatch: {_ASSET}")
    tensors = load(data)
    if set(tensors) != set(_SHAPES) or any(
        tuple(tensors[k].shape) != shape or tensors[k].dtype != torch.bfloat16
        or not torch.isfinite(tensors[k]).all().item()
        for k, shape in _SHAPES.items()
    ):
        raise RuntimeError(f"{TAG} invalid bundled upstream conditioning")
    return tensors


def _wrap_match(original):
    @wraps(original)
    def match(unet_config, state_dict=None, unet_key_prefix=""):
        # Only the tested original 7B architecture with BOTH buffers absent.
        # Embedded conditioning (including customized conditioning), other
        # model families and partially populated files retain stock behavior.
        if (state_dict is not None
                and all(unet_config.get(k) == v for k, v in _CONFIG.items())
                and all(unet_key_prefix + k not in state_dict for k in _SHAPES)):
            tensors = _conditioning()
            state_dict.update({unet_key_prefix + k: v for k, v in tensors.items()})
            print(f"{TAG} supplied official positive/negative conditioning for raw "
                  "SeedVR2 7B; existing checkpoint tensors unchanged.")
        return original(unet_config, state_dict, unet_key_prefix)
    return match


def install():
    global _installed
    if _installed or sys.platform != "darwin" or os.environ.get("ASFP8_SEEDVR2_LOADER", "1") != "1":
        return
    import comfy.model_detection
    comfy.model_detection.model_config_from_unet_config = _wrap_match(
        comfy.model_detection.model_config_from_unet_config)
    _installed = True
    print(f"{TAG} raw SeedVR2 7B conditioning compatibility enabled.")
