"""Patch #22 — default missing `format` in per-layer comfy_quant metadata.

Community int8-convrot checkpoints (e.g. Flux2-Klein-9B-True-V3-INT8-ConvRot)
ship per-layer descriptors like

    {"convrot": true, "convrot_groupsize": 256, "per_row": true}

without the `"format"` key, so comfy's `_load_quantized_module` raises
"Unknown quantization format" even though the tensor layout (int8 weight +
[N,1] fp32 weight_scale + convrot) is byte-identical to files that declare
`"format": "int8_tensorwise"` (Z-Image / Krea2 convrot do, and load fine).

This wraps `comfy.ops._load_quantized_module`: when a layer's descriptor lacks
`format` AND the weight is int8 AND a weight_scale is present, inject
`"format": "int8_tensorwise"` before the original loader runs. Anything else
(w4 packings, fp8, unknown layouts) is left untouched and fails as before —
we only label data whose layout we can positively identify.

Always on (it changes nothing for well-formed files); disable with
ASFP8_DISABLE=quant_meta_fix.
"""

import json
import sys

import torch

TAG = "[AppleSilicon-Optimized/quant_meta]"

_installed = False
_fixed_layers = 0
_announced = False


def _maybe_fix(state_dict, prefix):
    global _fixed_layers, _announced
    key = f"{prefix}comfy_quant"
    blob = state_dict.get(key)
    if blob is None:
        return
    try:
        conf = json.loads(bytes(blob.cpu().numpy().tobytes()))
    except Exception:
        return  # unparseable descriptor: let comfy handle/report it
    if not isinstance(conf, dict) or "format" in conf:
        return
    weight = state_dict.get(f"{prefix}weight")
    scale = state_dict.get(f"{prefix}weight_scale")
    if weight is None or scale is None or weight.dtype != torch.int8:
        return  # only label layouts we can positively identify
    conf["format"] = "int8_tensorwise"
    fixed = torch.frombuffer(bytearray(json.dumps(conf).encode()), dtype=torch.uint8)
    state_dict[key] = fixed
    _fixed_layers += 1
    if not _announced:
        _announced = True
        print(f"{TAG} checkpoint ships comfy_quant metadata without 'format'; "
              f"defaulting int8+weight_scale layers to int8_tensorwise "
              f"(first: {prefix.rstrip('.')}).")


def install():
    global _installed
    if _installed:
        return
    # Metadata repair is platform-independent, but this node targets MPS; stay
    # scoped to macOS like every other patch.
    if sys.platform != "darwin":
        return
    try:
        import comfy.ops as ops
    except Exception as e:
        print(f"{TAG} comfy.ops unavailable ({e!r}); not installed.")
        return
    orig = getattr(ops, "_load_quantized_module", None)
    if orig is None:
        print(f"{TAG} comfy.ops._load_quantized_module not found; not installed.")
        return

    def wrapper(module, load_fn, state_dict, prefix, *args, **kwargs):
        _maybe_fix(state_dict, prefix)
        return orig(module, load_fn, state_dict, prefix, *args, **kwargs)

    ops._load_quantized_module = wrapper
    _installed = True
    print(f"{TAG} armed: missing comfy_quant 'format' defaults to int8_tensorwise "
          f"for int8+scale layers (community convrot checkpoints).")
