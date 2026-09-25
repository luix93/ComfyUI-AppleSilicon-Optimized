"""Explicit PyTorch 2.14 MPS attention, without mtlflashattn interception.

ASFP8_ATTN_BACKEND=pytorch selects this route. The existing router remains
the default, and handles shapes outside the MPP prefill implementation.
The selection changes attention arithmetic; it is not a bit-exact mode.
"""

import os
import sys

import torch
import torch.nn.functional as F

from . import torch_mpp_qcache


def enabled():
    return (os.environ.get("ASFP8_ATTN_BACKEND") == "pytorch"
            and torch.__version__.split("+")[0].startswith("2.14."))


def supports(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    seq = q.shape[-2]
    return (enabled() and q.device.type == "mps"
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.dtype == k.dtype == v.dtype and attn_precision is None
            and dim in (64, 96, 128, 256) and seq > 8
            and (not skip_reshape or k.shape[-1] == v.shape[-1] == dim)
            and (mask is None or mask.dtype in (torch.bool, q.dtype)))


def sdpa(q, k, v, **kwargs):
    out = torch_mpp_qcache.maybe_attention(q, k, v, **kwargs)
    if out is not None:
        return out
    # flash_attn_mtl may already have wrapped F.scaled_dot_product_attention.
    # Its saved original is the PyTorch builtin, including on GQA calls.
    wrapper = sys.modules.get("metal_flash_attn.sdpa")
    raw = getattr(wrapper, "_orig", None) or F.scaled_dot_product_attention
    return raw(q, k, v, **kwargs)


def attention(q, k, v, heads, mask=None, skip_reshape=False,
              skip_output_reshape=False, scale=None, enable_gqa=False):
    b = q.shape[0]
    dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if not skip_reshape:
        kh = k.shape[-1] // dim if enable_gqa else heads
        vh = v.shape[-1] // dim if enable_gqa else heads
        q = q.reshape(b, -1, heads, dim).transpose(1, 2)
        k = k.reshape(b, -1, kh, dim).transpose(1, 2)
        v = v.reshape(b, -1, vh, dim).transpose(1, 2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = sdpa(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False,
               scale=scale, enable_gqa=enable_gqa)
    return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * dim)
