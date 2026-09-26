"""Keep already-dequantized embedding weights out of the INT8 decoder.

ComfyUI's cast/offload helper may return a plain floating tensor when moving a
QuantizedTensor between devices. The mixed-precision Embedding assumes that
its INT8 layout survives that cast and tries to dequantize it a second time.
Preserve the cast/offload context, and gather directly when it returns floats.
"""

import functools
import os
import sys

import torch.nn.functional as F

_installed = False


def _patch_embedding(ops, cls):
    original = cls.forward_comfy_cast_weights

    @functools.wraps(original)
    def forward(self, input, out_dtype=None):
        weight = self.weight
        if (not isinstance(weight, ops.QuantizedTensor)
                or getattr(self, "quant_format", None) != "int8_tensorwise"
                or self.weight_function):
            return original(self, input, out_dtype=out_dtype)
        with ops.CastBiasWeightContext(self, device=input.device,
                                      dtype=weight.dtype, offloadable=True) as (cast, _):
            if isinstance(cast, ops.QuantizedTensor):
                out = ops.get_layout_class(self.layout_type).dequantize_embedding(
                    cast._qdata, cast._params, input)
            else:
                # cast_bias_weight already applied scale and inverse ConvRot.
                out = F.embedding(input, cast, self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq, self.sparse)
            return out if out_dtype is None else out.to(dtype=out_dtype)

    cls.forward_comfy_cast_weights = forward


def install():
    global _installed
    if _installed or sys.platform != "darwin" or os.environ.get("ASFP8_INT8_EMBEDDING", "1") != "1":
        return
    import comfy.ops as ops
    original = ops.mixed_precision_ops
    if getattr(original, "_asfp8_int8_embedding", False):
        _installed = True
        return

    @functools.wraps(original)
    def factory(*args, **kwargs):
        result = original(*args, **kwargs)
        _patch_embedding(ops, result.Embedding)
        return result

    factory._asfp8_int8_embedding = True
    ops.mixed_precision_ops = factory
    _installed = True
    print("[AppleSilicon-Optimized/int8_embedding] preserve dequantized embeddings across device casts.")
