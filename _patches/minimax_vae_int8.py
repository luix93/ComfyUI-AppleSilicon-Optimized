"""Native W8A8 for the H3 INT8 ConvRot VAE, with FP16 activation/output rounds.

Selected by loading the quantized VAE under native mode; VAE_INT8=0 restores
ComfyUI's previous decoder route. This is not bit-exact to FP16 weights or to
ComfyUI's MPS weight-only W8A16 fallback. Dense attention is a separate option.
"""
import os

import torch

from . import int8_linear_kernel_mps as native

TAG = "[AppleSilicon-Optimized/minimax_vae_int8]"
_dead = False
_announced = False
_shapes = {(6144, 2048), (2048, 2048), (16384, 2048), (2048, 8192)}


def eligible(layer):
    if (_dead or os.environ.get("ASFP8_MINIMAX_VAE_INT8", "1") != "1"
            or native._kernel is None
            or not hasattr(native._kernel, "i8_matmul2d_nt_fused_cs_half")):
        return False
    import comfy.ops as ops
    if ops.comfy.model_management.in_training:
        return False
    weight = getattr(layer, "weight", None)
    if (not isinstance(weight, ops.QuantizedTensor)
            or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"
            or tuple(weight.shape) not in _shapes
            or not vars(type(layer)).get("_asfp8_int8_patched", False)
            or getattr(layer, "_full_precision_mm", False)
            or getattr(layer, "comfy_force_cast_weights", False)
            or "forward" in vars(layer) or hasattr(layer, "_v")
            or getattr(layer, native._RUNTIME_LORA_ATTR, None) is not None
            or any(getattr(layer, name, ()) for name in (
                "weight_function", "bias_function", "_forward_hooks", "_forward_pre_hooks"))):
        return False
    params = weight._params
    if (getattr(params, "transposed", False)
            or getattr(params, "orig_dtype", None) != torch.float16
            or not getattr(params, "convrot", False)
            or getattr(params, "convrot_groupsize", None) != 256):
        return False
    qdata, scale = weight._qdata, params.scale
    bias = getattr(layer, "bias", None)
    return (qdata.device.type == "mps" and qdata.dtype == torch.int8
            and qdata.is_contiguous() and scale.device == qdata.device
            and scale.dtype == torch.float32 and scale.numel() == weight.shape[0]
            and scale.is_contiguous()
            and (bias is None or (bias.device == qdata.device
                 and bias.dtype == torch.float16 and bias.is_contiguous()
                 and bias.shape == (weight.shape[0],))))


def linear(layer, x, input_act, act_weight=None, act_eps=0.0,
           residual=None, residual_scale=None):
    """Return the native result, or None without consuming/mutating inputs."""
    global _dead, _announced
    if (torch.is_grad_enabled() or not isinstance(x, torch.Tensor)
            or x.device.type != "mps" or x.dtype != torch.float16
            or x.ndim != 3 or not x.is_contiguous() or not eligible(layer)):
        return None
    import comfy.ops as ops
    if input_act not in (None, "rms_norm", "swiglu"):
        return None
    try:
        h = ops._eager_input_act(x, input_act, act_weight, act_eps)
        q, row_scale = native._quant_cached(h, True, 256)
        weight = layer.weight
        out = native._kernel.i8_matmul2d_nt_fused_cs_half(
            q, weight._qdata, row_scale.reshape(-1),
            weight._params.scale.reshape(-1), layer.bias)
        out = out.reshape(*h.shape[:-1], weight.shape[0])
        if residual is not None:
            out = torch.addcmul(residual, out, residual_scale)
    except Exception as exc:
        _dead = True
        print(f"{TAG} native decoder disabled ({str(exc)[:180]}); keeping core forwards.")
        return None
    if not _announced:
        _announced = True
        print(f"{TAG} INT8 ConvRot decoder uses native W8A8 with an FP16 output epilogue; "
              "ASFP8_MINIMAX_VAE_INT8=0 restores the prior MPS route.")
    return out
