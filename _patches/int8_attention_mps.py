"""Opt-in dense INT8 attention for M5 GPUs; numerically approximate.

Every query attends to every key: this is quantization, not sparsity. Q/K
use INT8 with per-16-query and per-64-key scales after midrange K centering;
V uses per-channel INT8 scales after mean centering. Two numerics:

- ``running`` (v2): P quantized to 255 levels against the running row maximum
  and accumulated in INT32 across key tiles. Fastest; error grows with length.
- ``tiled`` (v3, Kitchen numerics): randomized Hadamard Q/K rotation and P
  quantized against each 64-key tile's own maximum, merged into FP32
  accumulators. About 2.4x lower error at 44.7k tokens for ~8% more time.

Unmasked, non-causal, equal-head, D=128 BF16/FP16 calls only.
Importing this module changes no dispatch.
"""
from __future__ import annotations

import functools
import logging
import math
import os
from pathlib import Path
import subprocess

import torch

TAG = "[AppleSilicon-Optimized/int8_attention]"
HEAD_DIM = 128
MIN_KEYS = 1024
DEFAULT_MIN_TOKENS = 8192  # shorter calls measured no faster than exact paths
MAX_KEYS = 65536  # v2 ("running"): INT32 PV accumulates across all key tiles; 255 * 127 * keys < 2**31
# v3 ("tiled", Dense Metal): INT32 PV spans one 64-key tile before the FP32 merge,
# so no overflow bound applies; the cap matches the largest validated H3 length.
MAX_KEYS_TILED = 262144
_STATS_ROWS = 256
_DTYPES = {torch.bfloat16: "bfloat", torch.float16: "half"}


@functools.lru_cache(maxsize=1)
def supported():
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        return False
    try:
        brand = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return False
    return "M5" in brand


# Kitchen-numerics ("tiled") variants: probability levels per key tile.
TILED_LEVELS = {255: ("255.0f", "128"), 127: ("127.0f", "0")}


@functools.lru_cache(maxsize=8)
def _library(in_t, out_t, levels=255):
    source = (Path(__file__).with_suffix(".metal")).read_text()
    tile_levels, offset = TILED_LEVELS[levels]
    source = (source.replace("IN_T", in_t).replace("OUT_T", out_t)
              .replace("TILE_LEVELS", tile_levels).replace("P_OFFSET", offset))
    return torch.mps.compile_shader(source)


def max_keys(numerics="running"):
    return MAX_KEYS_TILED if numerics == "tiled" else MAX_KEYS


def eligible(q, k, v, numerics="running"):
    """True when ``attention`` accepts these [B,H,N,D] tensors."""
    return (supported()
            and all(type(t) is torch.Tensor and t.ndim == 4 and t.device.type == "mps"
                    and t.dtype == q.dtype and t.stride(-1) == 1 for t in (q, k, v))
            and q.dtype in _DTYPES and q.shape[-1] == HEAD_DIM
            and k.shape == v.shape and q.shape[:2] == k.shape[:2] and k.shape[-1] == HEAD_DIM
            and q.shape[2] > 0 and MIN_KEYS <= k.shape[2] <= max_keys(numerics))


def scratch_bytes(batch_heads, queries, keys):
    lqp, lkp = (queries + 63) // 64 * 64, (keys + 63) // 64 * 64
    chunks = (keys + _STATS_ROWS - 1) // _STATS_ROWS
    return (batch_heads * HEAD_DIM * (lqp + 2 * lkp)
            + 4 * batch_heads * (lqp // 16 + lkp // 64 + 3 * HEAD_DIM + 5 * chunks * HEAD_DIM
                                 + (lkp // 64 + 1) * HEAD_DIM))


def _params(q, k, v, out):
    b, h, lq, _ = q.shape
    values = [b, h, lq, k.shape[2]]
    for t in (q, k, v, out):
        values += [t.stride(0), t.stride(1), t.stride(2)]
    return torch.tensor(values, dtype=torch.int64, device=q.device)


def attention(q, k, v, *, scale=None, out=None, headroom=0.0, numerics="running", rotate=None, levels=255):
    """Return dense INT8 attention as a [B,H,Lq,128] view; strides are honored.

    ``out`` may be any [B,H,Lq,128] tensor with unit channel stride, for example
    a transposed [B,Lq,H,128] buffer, so callers avoid layout copies.
    """
    if not eligible(q, k, v, numerics):
        raise ValueError("INT8 attention requires M5 MPS BF16/FP16 [B,H,N,128] tensors with "
                         f"{MIN_KEYS}..{max_keys(numerics)} keys and equal Q/K/V heads; got {k.shape[2] if k.ndim == 4 else '?'} keys")
    b, h, lq, _ = q.shape
    lk = k.shape[2]
    scale = HEAD_DIM ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    elif (out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
          or out.stride(-1) != 1):
        raise ValueError("out must match q's shape/dtype with unit channel stride")
    bh = b * h
    lqp, lkp = (lq + 63) // 64 * 64, (lk + 63) // 64 * 64
    chunks = (lk + _STATS_ROWS - 1) // _STATS_ROWS
    dev = q.device
    if numerics not in ("running", "tiled"):
        raise ValueError("numerics must be 'running' or 'tiled'")
    tiled = numerics == "tiled"
    rotate = tiled if rotate is None else bool(rotate)
    lib = _library(_DTYPES[q.dtype], _DTYPES[q.dtype], levels if tiled else 255)
    params = _params(q, k, v, out)
    f32 = lambda *shape: torch.empty(shape, dtype=torch.float32, device=dev)
    partial = f32(bh, chunks, HEAD_DIM, 4)
    partial_sum = f32(bh, chunks, HEAD_DIM)
    vpre = torch.empty((bh, lkp // 64 + 1, HEAD_DIM), dtype=torch.int32, device=dev)
    kc, vc, vsc = f32(bh, HEAD_DIM), f32(bh, HEAD_DIM), f32(bh, HEAD_DIM)
    qi = torch.empty((bh, lqp, HEAD_DIM), dtype=torch.int8, device=dev)
    ki = torch.empty((bh, lkp, HEAD_DIM), dtype=torch.int8, device=dev)
    vi = torch.empty_like(ki)
    qs, ks = f32(bh, lqp // 16), f32(bh, lkp // 64)
    lib.i8a_stats(k, v, partial, params, partial_sum, threads=(chunks * 128, bh, 1), group_size=(128, 1, 1))
    lib.i8a_finalize(partial, kc, vc, vsc, params, partial_sum, threads=(bh * 128, 1, 1), group_size=(128, 1, 1))
    quant_q = lib.i8a_quant_q_rot if rotate else lib.i8a_quant_q
    quant_q(q, qi, qs, params, threads=(lqp // 16 * 128, bh, 1), group_size=(128, 1, 1))
    if tiled:
        # Per-block V sums only; the tiled kernel restores each tile's offset.
        # The rotated quantizer writes them as [BH, blocks, 128]; the plain one
        # writes block b at prefix slot b + 1.
        quant_kv = lib.i8a_quant_kv_rot if rotate else lib.i8a_quant_kv
        quant_kv(k, v, ki, vi, ks, kc, vc, vsc, params, vpre,
                 threads=(lkp // 64 * 128, bh, 1), group_size=(128, 1, 1))
        lib.i8a_attention_tiled(qi, ki, vi, qs, ks, vsc, vc, out, params, scale * 1.4426950408889634,
                                vpre[:, 1:].contiguous() if not rotate else vpre,
                                threads=(lqp // 64 * bh * 128, 1, 1), group_size=(128, 1, 1))
        return out
    if rotate:
        raise ValueError("rotation requires tiled numerics")
    lib.i8a_quant_kv(k, v, ki, vi, ks, kc, vc, vsc, params, vpre,
                     threads=(lkp // 64 * 128, bh, 1), group_size=(128, 1, 1))
    lib.i8a_prefix(vpre, params, threads=(bh * 128, 1, 1), group_size=(128, 1, 1))
    lib.i8a_attention(qi, ki, vi, qs, ks, vsc, vc, out, params, scale * 1.4426950408889634, vpre, float(headroom),
                      threads=(lqp // 64 * bh * 128, 1, 1), group_size=(128, 1, 1))
    return out


def min_tokens():
    """Smallest key count routed to INT8 by the workflow backend."""
    try:
        value = int(os.environ.get("ASFP8_INT8_ATTN_MIN_TOKENS", DEFAULT_MIN_TOKENS))
    except ValueError:
        value = DEFAULT_MIN_TOKENS
    return max(MIN_KEYS, value)


_calls = {"int8": 0, "fallback": 0}
_announced = False


def call_counts():
    return dict(_calls)


def _as_bhsd(t, heads, skip_reshape):
    if skip_reshape:
        return t
    if t.ndim != 3 or t.shape[-1] % heads:
        return None
    b, n, width = t.shape
    return t.view(b, n, heads, width // heads).transpose(1, 2)


def comfy_attention(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                    skip_output_reshape=False, fallback=None, numerics="running", **kwargs):
    """ComfyUI optimized-attention signature; ineligible calls use ``fallback``.

    Masked, GQA, short, non-D128 or non-MPS calls are delegated unchanged.
    ``attn_precision`` upcast requests are ignored for eligible calls: selecting
    this backend explicitly opts into approximate INT8 arithmetic.
    """
    global _announced
    q4, k4, v4 = (_as_bhsd(t, heads, skip_reshape) for t in (q, k, v))
    if (mask is None and not kwargs.get("enable_gqa", False)
            and q4 is not None and k4 is not None and v4 is not None
            and k4.shape[2] >= min_tokens() and eligible(q4, k4, v4, numerics)):
        b, h, lq, d = q4.shape
        if skip_output_reshape:
            out = torch.empty((b, h, lq, d), dtype=q.dtype, device=q.device)
            result, view = out, out
        else:
            out = torch.empty((b, lq, h, d), dtype=q.dtype, device=q.device)
            result, view = out.view(b, lq, h * d), out.transpose(1, 2)
        attention(q4, k4, v4, scale=kwargs.get("scale"), out=view, numerics=numerics)
        _calls["int8"] += 1
        if not _announced:
            _announced = True
            logging.info("%s approximate dense INT8 attention active (>= %d keys)", TAG, min_tokens())
        return result
    _calls["fallback"] += 1
    if fallback is None:
        import comfy.ldm.modules.attention as comfy_attention_module
        fallback = comfy_attention_module.attention_pytorch
    return fallback(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)


def h3_attention(q, k, v, *, tau=0.0, video_start=0, scale=None, exact_prefix=True,
                 schedule="linear", memory_budget_gib=0.0, topk_ratio=0.0, protected_ranges=None,
                 numerics="running"):
    """H3 sparse-holder API: (BHSD output, selected-block counts, total slots).

    Protected query rows (text/audio prefix or explicit ranges) are recomputed
    with exact PyTorch attention against every key, as in the earlier dense
    INT8 backend. ``tau`` and ``schedule`` are accepted for ABI compatibility.
    """
    from . import h3_dense_mps
    if topk_ratio != 0:
        raise ValueError("Dense INT8 retains all tokens; topk_ratio must be zero")
    name = "Dense Metal" if numerics == "tiled" else "Dense Metal v2"
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
        raise ValueError(f"{name} requires matching [1,H,S,128] tensors")
    if not eligible(q, k, v, numerics):
        raise ValueError(f"{name} requires M5 MPS BF16/FP16 D128 inputs with "
                         f"{MIN_KEYS}..{max_keys(numerics)} tokens; got {q.shape[2]}")
    heads, tokens = q.shape[1], q.shape[2]
    if protected_ranges is not None:
        from .h3_sol_mps import normalize_protected_ranges
        if not exact_prefix:
            raise ValueError("Explicit protected ranges require exact_prefix=True")
        protected_ranges = normalize_protected_ranges(protected_ranges, tokens)
    required = scratch_bytes(heads, tokens, tokens)
    budget, headroom = h3_dense_mps.workspace_budget(memory_budget_gib)
    if required > budget:
        raise h3_dense_mps.WorkspaceBudgetExceeded(
            f"Dense INT8 v2 needs {required / 2**30:.2f} GiB temporary memory; budget "
            f"{budget / 2**30:.2f} GiB, safe headroom {headroom / 2**30:.2f} GiB.")
    ranges = (protected_ranges if protected_ranges is not None
              else ((0, min(tokens, (video_start + 63) // 64 * 64)),))
    if exact_prefix and len(ranges) == 1 and ranges[0][0] == 0 and ranges[0][1] > 0:
        # A single leading exact range (the usual text/reference/audio prefix):
        # quantize only the rows after it instead of computing and discarding
        # the prefix in INT8. On OmniRef that is 7% of the rows, ~30 ms a call.
        prefix = ranges[0][1]
        out = torch.empty_like(q)
        if prefix < tokens:
            attention(q[:, :, prefix:], k, v, scale=scale, out=out[:, :, prefix:], numerics=numerics)
    else:
        out = attention(q, k, v, scale=scale, numerics=numerics)
    if exact_prefix:
        for start, end in ranges:
            if end > start:
                out[:, :, start:end].copy_(torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, start:end], k, v, scale=scale))
    blocks = (tokens + 63) // 64
    return out, torch.full((heads, blocks), blocks, device=q.device, dtype=torch.int64), heads * blocks * blocks
