"""Exact FP16 output epilogue for Wan 14B's native INT8 projections.

Reuse the existing INT32 GEMM and FP16 epilogue. Scaling rounds to FP16
before bias addition, matching the preceding materialized implementation.
"""
import os

import torch

from . import torch_mpp_qcache

_calls = 0
_dead = False
_shapes = {(5120, 5120), (13824, 5120), (5120, 13824)}


def maybe_prequant(kernel, x, row_scale, weight, col_scale, bias, shape):
    global _calls, _dead
    if (_dead or torch.is_grad_enabled()
            or os.environ.get('ASFP8_DIT_EXT') != '1'
            or os.environ.get('ASFP8_WAN_FAST', '1') != '1'
            or os.environ.get('ASFP8_WAN_EXACT', '1') != '1'
            or os.environ.get('ASFP8_WAN_FP16_GEMM', '1') != '1'
            or not torch_mpp_qcache._torch_build_supported()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'
            or kernel is None or not hasattr(kernel, 'i8_matmul2d_nt_fused_cs_half')
            or tuple(weight.shape) not in _shapes
            or any(type(t) is not torch.Tensor or not t.is_mps for t in
                   (x, row_scale, weight, col_scale))
            or x.dtype != torch.int8 or weight.dtype != torch.int8
            or row_scale.dtype != torch.float32 or col_scale.dtype != torch.float32
            or x.ndim != 2 or x.shape[1] != weight.shape[1]
            or row_scale.numel() != x.shape[0] or col_scale.numel() != weight.shape[0]
            or (bias is not None and (type(bias) not in (torch.Tensor, torch.nn.Parameter)
                or not bias.is_mps or bias.dtype != torch.float16
                or bias.shape != (weight.shape[0],)))
            or not torch_mpp_qcache._hardware()):
        return None
    try:
        out = kernel.i8_matmul2d_nt_fused_cs_half(
            x.contiguous(), weight.contiguous(), row_scale.reshape(-1).contiguous(),
            col_scale.reshape(-1).contiguous(), bias)
    except Exception as exc:
        _dead = True
        print(f'[AppleSilicon-Optimized/wan_fp16] keeping materialized output ({exc!r}).')
        return None
    _calls += 1
    return out.reshape(*shape[:-1], weight.shape[0])
