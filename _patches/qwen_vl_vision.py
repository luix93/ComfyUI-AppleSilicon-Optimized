"""Bound Qwen3-VL/Qwen3.5 vision's basic-attention scratch memory on MPS.

The vision tower requests small_input=True even for ~50k patch tokens. That
helper can bypass the global router and allocate an entire quadratic score
matrix. Slice queries only, retaining all keys and native basic-attention
arithmetic/precision. Different matrix shapes can change floating-point bytes;
this is a large-input memory safeguard, not an exact decoding optimization.
"""
import functools
import logging
import math
import os

import psutil
import torch

_installed = False
_logged = False


def _budget():
    try:
        mb = float(os.environ.get('ASFP8_QWEN_VISION_MEMORY_MB', '512'))
        if not math.isfinite(mb):
            raise ValueError
    except ValueError:
        mb = 512
    return int(min(1024, max(64, mb)) * 2**20)


def _rows(q, k, heads, precision, budget):
    # Four score-sized buffers allow for matmul, scale, softmax, and a cast.
    # Keep head/sequence products below MPSGraph's signed indexing limit too.
    size = max(q.element_size(), 4 if precision == torch.float32 else 0)
    row_elements = q.shape[0] * heads * k.shape[-2]
    rows = min(budget // (4 * size * row_elements), (2**30) // row_elements)
    if rows < 1:
        raise RuntimeError('Qwen vision attention cannot fit even one query row in its '
                           'memory budget. Resize the image branch feeding TextGenerate.')
    return max(1, rows // 32 * 32) if rows >= 32 else rows


def _bounded(basic, q, k, v, heads, mask=None, attn_precision=None,
             skip_reshape=False, skip_output_reshape=False, **kwargs):
    global _logged
    from comfy.ldm.modules.attention import get_attn_precision
    from comfy import model_management

    arguments = dict(mask=mask, attn_precision=attn_precision,
                     skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
    if q.device.type != 'mps' or os.environ.get('ASFP8_QWEN_VISION_SAFE', '1') == '0':
        return basic(q, k, v, heads, **arguments)
    # This seam is specific to Qwen35VisionAttention: [B,H,N,D], no mask/GQA.
    if (not skip_reshape or q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or mask is not None or kwargs.get('enable_gqa', False)
            or q.shape[:2] != k.shape[:2] or k.shape != v.shape
            or q.shape[1] != heads or q.shape[-1] != k.shape[-1]
            or q.dtype != k.dtype or k.dtype != v.dtype
            or q.device != k.device or k.device != v.device):
        raise RuntimeError('Unsupported Qwen vision attention layout; cannot apply the '
                           'MPS memory safeguard. Please report the input tensor shapes.')
    precision = get_attn_precision(attn_precision, q.dtype)
    budget = _budget()
    rows = _rows(q, k, heads, precision, budget)
    if q.shape[-2] <= rows:
        return basic(q, k, v, heads, **arguments)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError('Bounded Qwen vision attention supports inference only.')
    # Finish pending work before estimating headroom. Never retry a failed giant
    # allocation or silently drop pixels/keys. Reserve space for output and copies.
    torch.mps.synchronize()
    available = min(torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory(),
                    psutil.virtual_memory().available)
    reserve = q.numel() * q.element_size() * 2 + 256 * 2**20
    budget = min(budget, int((available - reserve) / 2))
    if budget < 16 * 2**20:
        raise RuntimeError('Insufficient MPS memory headroom for Qwen image encoding. '
                           'Unload other models or resize the TextGenerate image input.')
    rows = _rows(q, k, heads, precision, budget)
    if not _logged:
        logging.info('[AppleSilicon-Optimized/QwenVision] bounded dense attention: '
                     '%s query tokens, %s rows/chunk, %.0f MiB scratch budget; '
                     'all image patches retained (floating-point rounding can differ).',
                     q.shape[-2], rows, budget / 2**20)
        _logged = True
    b, h, n, d = q.shape
    out = torch.empty((b, h, n, d) if skip_output_reshape else (b, n, h*d),
                      dtype=v.dtype, device=v.device)
    for start in range(0, n, rows):
        model_management.throw_exception_if_processing_interrupted()
        end = min(n, start + rows)
        part = basic(q[:, :, start:end], k, v, heads, **arguments)
        if skip_output_reshape:
            out[:, :, start:end].copy_(part)
        else:
            out[:, start:end].copy_(part)
        del part
        # Bound outstanding GPU temporaries as well as Python references.
        torch.mps.synchronize()
    return out


def install():
    global _installed
    if _installed or not torch.backends.mps.is_available():
        return
    from comfy.text_encoders import qwen35
    from comfy.ldm.modules import attention
    original = qwen35.optimized_attention_for_device
    basic = attention.attention_basic

    @attention.wrap_attn
    def safe(*args, **kwargs):
        return _bounded(basic, *args, **kwargs)

    @functools.wraps(original)
    def select(device, mask=False, small_input=False):
        selected = original(device, mask=mask, small_input=small_input)
        if (torch.device(device).type == 'mps' and small_input
                and selected is basic and not mask):
            return safe
        return selected

    qwen35.optimized_attention_for_device = select
    _installed = True
