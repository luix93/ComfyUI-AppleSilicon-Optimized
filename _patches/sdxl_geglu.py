"""Exact FP16 GEGLU activation/layout fusion on PyTorch 2.14 MPS.

The 128 KiB table evaluates every FP16 bit pattern through the running
PyTorch GELU once. A single pass reads both projection halves, looks up the
rounded GELU value and multiplies, retaining the intermediate FP16 rounding.
No approximation of GELU, weight packing or replacement GEMM is involved.
"""
import os

import torch
import torch.nn.functional as F

_installed = False
_original = None
_library = None
_table = None
_dead = False

_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void geglu(device const half* X, device const half* T, device half* Y,
                  constant uint& W, uint i [[thread_position_in_grid]]) {
  uint row = i / W, col = i % W;
  half a = X[row * W * 2 + col], g = X[row * W * 2 + W + col];
  Y[i] = half(float(a) * float(T[as_type<ushort>(g)]));
}
"""


def _fused(x):
    global _library, _table
    if _library is None:
        library = torch.mps.compile_shader(_SOURCE)
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        table = F.gelu(values.to(x.device))
        _library, _table = library, table
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    _library.geglu(x, _table, out, out.shape[-1], threads=out.numel(), group_size=256)
    return out


def _forward(self, x):
    global _dead
    if (_dead or os.environ.get("ASFP8_SDXL_GEGLU", "1") != "1"
            or type(x) is not torch.Tensor or not x.is_mps
            or x.dtype != torch.float16 or torch.is_grad_enabled()
            or x.numel() < 1024):
        return _original(self, x)
    projected = self.proj(x)
    if (type(projected) is torch.Tensor and projected.is_mps
            and projected.dtype == torch.float16 and projected.is_contiguous()
            and 0 < projected.numel() < 2**32 and projected.shape[-1] % 2 == 0):
        try:
            return _fused(projected)
        except Exception as exc:
            _dead = True
            print(f"[AppleSilicon-Optimized/sdxl_geglu] disabled ({exc!r}); using eager GEGLU.")
    a, gate = projected.chunk(2, dim=-1)
    return a * F.gelu(gate)


def install():
    global _original, _installed
    if (_installed or os.environ.get("ASFP8_DIT_EXT") != "1"
            or os.environ.get("ASFP8_SDXL_GEGLU", "1") != "1"
            or not torch.__version__.split("+")[0].startswith("2.14.")
            or not torch.backends.mps.is_available()):
        return
    from comfy.ldm.modules.attention import GEGLU
    _original = GEGLU.forward
    GEGLU.forward = _forward
    _installed = True
    print("[AppleSilicon-Optimized/sdxl_geglu] exact FP16 GELU+multiply fusion enabled.")
