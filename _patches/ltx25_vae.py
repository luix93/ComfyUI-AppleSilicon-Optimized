# SPDX-License-Identifier: Apache-2.0
# Portions copyright (c) 2025 Comfy Org. All rights reserved.
"""Exact LTX neighborhood-decoder scheduling, scoped to its attention calls.

Keep tile geometry, masks, key order and PyTorch attention unchanged. Batch
independent tiles with a 128 MiB K/V-copy budget and fuse their copies. Integer
row maps have a 32 MiB cache bound; no image activations or masks are cached.
Elementwise fusions retain each BF16 rounding point.
"""
import functools
import math
import os
import struct
from contextvars import ContextVar
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from . import ltx25_fast

_scope = ContextVar('ltx25_vae_attention', default=False)
_installed = False
_calls = 0
_original_na = _original_rope = None
_geometry = None
_attention_library = None
_attention_dead = False
_silu_table = None
_original_swiglu = _original_modulate = None
_decoder = None
_elementwise_calls = 0
_layout_calls = 0
_index_cache = OrderedDict()
_index_bytes = 0
_INDEX_BUDGET = 32 * 2**20


@functools.lru_cache(maxsize=128)
def _mask_strides(keys):
    return torch.tensor(list(struct.pack('=4q', 0, 0, keys, 1)),
                        dtype=torch.uint8, device='mps')


def _attention(q, k, v, mask):
    global _attention_library, _attention_dead
    if _enabled('ATTN') and not _attention_dead:
        try:
            qc = ltx25_fast.torch_mpp_qcache
            if _attention_library is None:
                source = qc._source()  # validates the installed PyTorch headers
                old = 'bfloat, 64, 32, 128, 4, 1, false, false, true, bfloat'
                assert source.count(old) == 1
                source = source.replace(old, 'bfloat, 64, 32, 64, 4, 1, true, false, true, bfloat')
                _attention_library = torch.mps.compile_shader(source)
            b, heads, seq, dim = q.shape
            params = qc._parameters(tuple(q.shape), tuple(k.shape), q.stride(),
                                    k.stride(), v.stride(), 1., True)
            out = torch.empty((b, seq, heads, dim), device=q.device, dtype=q.dtype).transpose(1, 2)
            _attention_library.custom_attn(q, k, v, out, params, _mask_strides(k.shape[2]), mask,
                                           threads=(((seq + 63) // 64) * 128, heads, b),
                                           group_size=(128, 1, 1))
            return out
        except Exception as exc:
            _attention_dead = True
            _mask_strides.cache_clear()
            print(f'[AppleSilicon-Optimized/ltx25] decoder attention fallback ({exc!r}).')
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.)


def _enabled(part):
    return (os.environ.get('ASFP8_LTX25_FAST', '1') == '1'
            and os.environ.get('ASFP8_LTX25_VAE', '1') == '1'
            and os.environ.get('ASFP8_LTX25_VAE_' + part, '1') == '1')


def _tensor(x, dtype=torch.bfloat16):
    return (type(x) in (torch.Tensor, torch.nn.Parameter) and x.is_mps
            and x.dtype == dtype and x.is_contiguous() and 0 < x.numel() < 2**31)


@functools.lru_cache(maxsize=128)
def _mask_parameters(rel):
    dims = [len(starts) for starts, ends in rel]
    keys = [max(ends) for starts, ends in rel]
    packed = dims + keys + [v for pair in rel for part in pair for v in part]
    return torch.tensor(packed, dtype=torch.int32, device='mps')


def _group_mask(rel, dtype, device):
    if not _enabled('MASK'):
        return _geometry._group_mask(rel, dtype, device)
    nq = math.prod(len(starts) for starts, ends in rel)
    nk = math.prod(max(ends) for starts, ends in rel)
    out = torch.empty((1, 1, nq, nk), device=device, dtype=dtype)
    ltx25_fast._library.mask3d(_mask_parameters(rel), out,
                              threads=nq * nk, group_size=256)
    return out


def _rms_rope(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    if k_scale is None:
        k_scale = q_scale
    if (_scope.get() and _enabled('ROPE') and not torch.is_grad_enabled()
            and all(_tensor(t) for t in (q, k, q_scale, k_scale))
            and q.shape == k.shape and q.ndim == 4 and q.shape[0] == 1
            and q.shape[2] in (4, 8, 16, 32) and q.shape[-1] == 64
            and q.data_ptr() != k.data_ptr()
            and q_scale.shape == k_scale.shape == (64,)
            and _tensor(freqs_cis, torch.float32)
            and freqs_cis.shape == (1, q.shape[1], 1, 32, 2, 2)
            and epsilon == 1e-6 and ltx25_fast._lib() is not None):
        ltx25_fast._library.rms_rope64(q, k, q_scale, k_scale, freqs_cis,
                                     q.shape[2], float(epsilon),
                                     threads=(q.shape[1] * 32, q.shape[2], 2),
                                     group_size=(32, 1, 1))
        return q, k
    return _original_rope(q, k, freqs_cis, q_scale, k_scale, epsilon)


def _na3d(q, k, v, kernel_size, is_causal=None, scale=None):
    if (not _scope.get() or not any(_enabled(part) for part in ('BATCH', 'MASK', 'ATTN', 'LAYOUT'))
            or torch.is_grad_enabled() or not all(_tensor(x) for x in (q, k, v))
            or q.ndim != 6 or q.shape != k.shape or q.shape != v.shape
            or q.shape[0] != 1 or q.shape[-1] != 64
            or q.shape[-2] not in (4, 8, 16, 32)
            or not isinstance(kernel_size, (list, tuple))
            or len(kernel_size) != 3 or any(type(n) is not int or n <= 0 for n in kernel_size)
            or is_causal is not None and (not isinstance(is_causal, (list, tuple))
                                         or len(is_causal) != 3 or any(is_causal))
            or ltx25_fast._lib() is None):
        return _original_na(q, k, v, kernel_size, is_causal, scale)
    global _calls
    _calls += 1
    if _enabled('LAYOUT'):
        return _packed_na(q, k, v, kernel_size, scale)
    return _batched_na(q, k, v, kernel_size, is_causal, scale)


@functools.lru_cache(maxsize=128)
def _tile_plan(dims, kernels):
    bounds = [_geometry._window_bounds(d, k, False) for d, k in zip(dims, kernels)]
    tt, th, tw = _geometry._pick_tiles(dims, kernels)
    groups = {}
    for t0 in range(0, dims[0], tt):
        for h0 in range(0, dims[1], th):
            for w0 in range(0, dims[2], tw):
                starts = (t0, h0, w0)
                ends = tuple(min(s + z, d) for s, z, d in zip(starts, (tt, th, tw), dims))
                origin = tuple(b[0][s] for b, s in zip(bounds, starts))
                rel = tuple((tuple(v - o for v in b[0][s:e]), tuple(v - o for v in b[1][s:e]))
                            for b, s, e, o in zip(bounds, starts, ends, origin))
                groups.setdefault(rel, []).append(starts + origin)
    return tuple((rel, tuple(tiles)) for rel, tiles in groups.items())


def _index_map(h, w, heads, tiles, rel):
    global _index_bytes
    key = (h, w, heads, tiles, rel)
    if key in _index_cache:
        out = _index_cache.pop(key)
        _index_cache[key] = out
        return out
    shapes = [tuple(len(a) for a, z in rel), tuple(max(z) for a, z in rel)]
    parts = [np.array([len(tiles), 0, 0, 0], dtype=np.int32)]
    for j, shape in enumerate(shapes):
        it, iy, ix = np.indices(shape, dtype=np.int32)
        for tile in tiles:
            t0, h0, w0 = tile[j * 3:j * 3 + 3]
            parts.append(((((it + t0) * h + iy + h0) * w + ix + w0) * heads * 64).reshape(-1))
    out = torch.from_numpy(np.concatenate(parts)).to('mps')
    size = out.numel() * out.element_size()
    if size <= _INDEX_BUDGET:
        while _index_bytes + size > _INDEX_BUDGET or len(_index_cache) >= 128:
            _, prior = _index_cache.popitem(last=False)
            _index_bytes -= prior.numel() * prior.element_size()
        _index_cache[key] = out
        _index_bytes += size
    return out


def _packed_na(q, k, v, kernel_size, scale):
    _, t, h, w, nh, hd = q.shape
    scale = hd ** -0.5 if scale is None else scale
    if scale != 1.0:
        q = q * scale
    out = torch.empty_like(v)
    kernels = tuple(min(d, z) for d, z in zip((t, h, w), kernel_size))
    for rel, tiles in _tile_plan((t, h, w), kernels):
        mask = _group_mask(rel, q.dtype, q.device)
        nq, nk = mask.shape[2:]
        g_max = max(1, (2**26) // (nh * nk * hd * 2)) if _enabled('BATCH') else 1
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0:c0 + g_max]
            g = len(chunk)
            mapping = _index_map(h, w, nh, chunk, rel)
            qs = torch.empty((g, nq, nh, hd), dtype=q.dtype, device=q.device)
            ks = torch.empty((g, nk, nh, hd), dtype=q.dtype, device=q.device)
            vs = torch.empty_like(ks)
            ltx25_fast._library.vae_gather(q, k, v, qs, ks, vs, mapping, nq, nk, nh * hd,
                                          threads=(max(qs.numel(), ks.numel()), 3), group_size=(256, 1))
            o = _attention(qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2), mask)
            o = o.transpose(1, 2).contiguous()
            ltx25_fast._library.vae_scatter(o, out, mapping, nh * hd,
                                           threads=o.numel(), group_size=256)
    global _layout_calls
    _layout_calls += 1
    return out


def _swiglu(self, x, pre=None, add_to=None):
    if (not _enabled('SWIGLU') or torch.is_grad_enabled()
            or _decoder.comfy.model_management.in_training
            or not _tensor(x) or x.ndim != 5 or x.shape[0] != 1
            or ltx25_fast._lib() is None):
        return _original_swiglu(self, x, pre, add_to)
    global _silu_table, _elementwise_calls
    if _silu_table is None:
        bits = torch.arange(65536, dtype=torch.int32, device=x.device)
        _silu_table = F.silu(bits.to(torch.int16).view(torch.bfloat16))
    _, t, h, w, _ = x.shape
    chunk = max(1, _decoder.MLP_TOKEN_CHUNK // max(h * w, 1))
    out = add_to if add_to is not None else torch.empty_like(x)
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
        gate, up = self.w_gate(sl), self.w_up(sl)
        if (_tensor(gate) and _tensor(up) and gate.shape == up.shape):
            mid = torch.empty_like(gate)
            ltx25_fast._library.vae_swiglu(gate, up, _silu_table, mid,
                                          threads=gate.numel(), group_size=256)
            _elementwise_calls += 1
        else:
            mid = F.silu(gate) * up
        del gate, up
        y = self.w_down(mid)
        del mid
        if add_to is not None:
            out[:, t0:t1] += y
        else:
            out[:, t0:t1] = y
    return out


def _modulate(x, scale, shift):
    if (_enabled('MODULATE') and not torch.is_grad_enabled()
            and not _decoder.comfy.model_management.in_training
            and all(_tensor(t) for t in (x, scale, shift))
            and x.ndim == 5 and x.shape[0] == 1
            and scale.shape == shift.shape == (1, 1, 1, 1, x.shape[-1])
            and ltx25_fast._lib() is not None):
        out = torch.empty_like(x)
        ltx25_fast._library.vae_modulate(x, scale, shift, out, x.shape[-1],
                                        threads=x.numel(), group_size=256)
        global _elementwise_calls
        _elementwise_calls += 1
        return out
    return _original_modulate(x, scale, shift)


def install():
    global _installed, _original_na, _original_rope, _geometry
    global _original_swiglu, _original_modulate, _decoder
    if _installed:
        return
    try:
        import comfy_kitchen as ck
        import comfy_kitchen.backends.eager.na as geometry
        import comfy.ldm.lightricks.vae.na_diffusion_decoder as decoder
    except ImportError:
        return
    _geometry = geometry
    _decoder = decoder
    original = decoder.NeighborhoodAttention3D.forward
    _original_na, _original_rope = ck.na3d, ck.rms_rope_
    _original_swiglu, _original_modulate = decoder.SwiGLU.forward, decoder.modulate

    @functools.wraps(original)
    def attention(self, x, *args, **kwargs):
        enabled = any(_enabled(part) for part in ('BATCH', 'MASK', 'ROPE', 'ATTN', 'LAYOUT'))
        enabled = (enabled and not torch.is_grad_enabled()
                   and not decoder.comfy.model_management.in_training
                   and _tensor(x) and x.ndim == 5 and x.shape[0] == 1)
        token = _scope.set(enabled)
        try:
            return original(self, x, *args, **kwargs)
        finally:
            _scope.reset(token)

    decoder.NeighborhoodAttention3D.forward = attention
    ck.na3d, ck.rms_rope_ = _na3d, _rms_rope
    decoder.SwiGLU.forward, decoder.modulate = _swiglu, _modulate
    _installed = True


# Adapted from Comfy Kitchen eager/na.py (Apache-2.0).
# Copyright (c) 2025 Comfy Org. See THIRD_PARTY_NOTICES.md.
def _batched_na(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` tensors."""
    batch, t, h, w, nh, hd = q.shape
    dims = (t, h, w)
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kernels = [k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, dims, strict=True)]
    if scale is None:
        scale = hd ** -0.5
    device = q.device
    if scale != 1.0:
        q = q * scale

    bounds = [_geometry._window_bounds(d, k_, c) for d, k_, c in zip(dims, kernels, causal, strict=True)]
    tile_t, tile_h, tile_w = _geometry._pick_tiles(dims, [min(k_, d) for k_, d in zip(kernels, dims, strict=True)])

    # Group tiles by relative window geometry so each group shares one mask
    # and stacks into batched SDPA calls.
    groups = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (tuple(s - rt0 for s in bounds[0][0][t0:t1]), tuple(e - rt0 for e in bounds[0][1][t0:t1]))
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (tuple(s - rh0 for s in bounds[1][0][h0:h1]), tuple(e - rh0 for e in bounds[1][1][h0:h1]))
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (tuple(s - rw0 for s in bounds[2][0][w0:w1]), tuple(e - rw0 for e in bounds[2][1][w0:w1]))
                groups.setdefault((rel_t, rel_h, rel_w), []).append((
                    (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                    (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                ))

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        if _enabled("BATCH"):
            g_max = max(1, (2 ** 26) // max(1, batch * nh * nk * hd * 2))
        else:
            g_max = 1  # CPU math backend materializes [G*B, NH, Nq, Nk]
        qs0, _ = tiles[0]
        tq, th, tw = (qs0[0].stop - qs0[0].start, qs0[1].stop - qs0[1].start, qs0[2].stop - qs0[2].start)
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0:c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            # [G, B, t, h, w, NH, HD] -> [G*B, NH, N, HD]
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = _attention(q_s, k_s, v_s, mask)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out
