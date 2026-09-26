"""Exact Klein 9B BF16 SwiGLU and single-block output-layout fusions.

Keep projection and attention arithmetic unchanged. A 128 KiB SiLU table
retains PyTorch's BF16 activation rounding before the gate multiplication.
"""
import os

import torch
import torch.nn.functional as F

from . import torch_mpp_qcache

_installed = False
_library = _table = None
_original = _activation_type = None
_dead = False
_activation_calls = _merge_calls = 0

_SOURCE = '''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
inline bfloat rte(float f) {
  uint u=as_type<uint>(f); u+=0x7fff+((u>>16)&1);
  return as_type<bfloat>(ushort(u>>16));
}
kernel void activation(device const bfloat* x, device const bfloat* lut,
 device bfloat* out, constant uint& W, constant uint& stride,
 uint i [[thread_position_in_grid]]) {
  uint row=i/W, c=i%W;
  out[i]=rte(float(lut[as_type<ushort>(x[row*stride+c])])*float(x[row*stride+W+c]));
}
kernel void merge(device const bfloat* x, device const bfloat* attn,
 device const bfloat* lut, device bfloat* out, constant uint& D,
 constant uint& W, constant uint& stride, uint i [[thread_position_in_grid]]) {
  uint row=i/(D+W), c=i%(D+W);
  if(c<D) out[i]=attn[row*D+c];
  else {
    uint off=row*stride+3*D+c-D;
    out[i]=rte(float(lut[as_type<ushort>(x[off])])*float(x[off+W]));
  }
}
'''


def _enabled(part):
    return (_installed and not _dead and not torch.is_grad_enabled()
            and os.environ.get('ASFP8_FLUX2_GLUE', '1') == '1'
            and os.environ.get('ASFP8_FLUX2_' + part, '1') == '1')


def _tensor(x, width):
    return (type(x) is torch.Tensor and x.is_mps and x.dtype == torch.bfloat16
            and x.ndim == 3 and x.shape[0] == 1 and x.shape[-1] == width
            and x.stride(-1) == 1 and x.stride(-2) >= width
            and 0 < x.shape[1] * x.stride(-2) < 2**31)


def _plain_activation(module):
    return (type(module) is _activation_type
            and type(module.gate_fn) is torch.nn.SiLU
            and not module._forward_hooks and not module._forward_pre_hooks
            and not module.gate_fn._forward_hooks and not module.gate_fn._forward_pre_hooks
            and getattr(module.gate_fn.forward, '__func__', None) is torch.nn.SiLU.forward)


def _ready():
    global _library, _table, _dead
    if _library is None:
        try:
            library = torch.mps.compile_shader(_SOURCE)
            values = torch.arange(65536, device='mps', dtype=torch.int32)
            table = F.silu(values.to(torch.int16).view(torch.bfloat16))
            _library, _table = library, table
        except Exception as exc:
            _dead = True
            print(f'[AppleSilicon-Optimized/flux2_glue] keeping eager activation ({exc!r}).')
    return _library is not None


def _activation(self, x):
    if (not _enabled('SWIGLU') or not _tensor(x, 24576)
            or not _plain_activation(self) or not _ready()):
        return _original(self, x)
    out = torch.empty((*x.shape[:-1], 12288), device=x.device, dtype=x.dtype)
    _library.activation(x, _table, out, 12288, x.stride(-2),
                        threads=out.numel(), group_size=256)
    global _activation_calls
    _activation_calls += 1
    return out


def single_mlp_input(block, projected, attention):
    """Return the fused activation/concatenation, or None for the caller's fallback."""
    if (not _enabled('MLP_LAYOUT') or block.yak_mlp
            or block.hidden_size != 4096 or block.mlp_hidden_dim != 12288
            or block.mlp_hidden_dim_first != 24576
            or not _plain_activation(block.mlp_act)
            or getattr(block.mlp_act.forward, '__func__', None) is not _activation
            or not _tensor(projected, 36864) or not projected.is_contiguous()
            or not _tensor(attention, 4096) or not attention.is_contiguous()
            or projected.shape[:2] != attention.shape[:2] or not _ready()):
        return None
    out = torch.empty((*projected.shape[:-1], 16384), device=projected.device,
                      dtype=projected.dtype)
    _library.merge(projected, attention, _table, out, 4096, 12288,
                   projected.stride(-2), threads=out.numel(), group_size=256)
    global _merge_calls
    _merge_calls += 1
    return out


def install():
    global _installed, _original, _activation_type
    if (_installed or os.environ.get('ASFP8_DIT_EXT') != '1'
            or torch.__version__.split('+')[0] != '2.14.0'
            or torch.version.git_version != torch_mpp_qcache._TORCH_GIT
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'
            or not torch.backends.mps.is_available() or not torch_mpp_qcache._hardware()):
        return
    from comfy.ldm.flux.layers import SiLUActivation
    _activation_type = SiLUActivation
    _original = SiLUActivation.forward
    SiLUActivation.forward = _activation
    _installed = True
