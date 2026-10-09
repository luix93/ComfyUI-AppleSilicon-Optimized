"""Veda tile-list attention using PyTorch's M5 MPP arithmetic on Metal.

The selected 128-token tiles are read in place. No expanded K/V batches or
PyTorch/MLX bridge are needed. Attention remains approximate to dense H3.
PyTorch/MLX source licenses are in krea2_ext; Veda's license is in h3_runtime/veda.
"""

import functools
import struct

import torch

from . import torch_mpp_qcache
from .h3_runtime.veda.backends import base
from .h3_runtime.veda.core import selection


_GATHER_POOL = r'''
#include <metal_stdlib>
using namespace metal;
kernel void veda_gather_pool(
    device const DTYPE* q [[buffer(0)]], device const DTYPE* k [[buffer(1)]],
    device const DTYPE* v [[buffer(2)]], device const long* perm [[buffer(3)]],
    device const int* valid [[buffer(4)]], device const long* heads [[buffer(5)]],
    constant long* p [[buffer(6)]], device DTYPE* qt [[buffer(7)]],
    device DTYPE* kt [[buffer(8)]], device DTYPE* vt [[buffer(9)]],
    device float* fq [[buffer(10)]], device float* fk [[buffer(11)]],
    uint d [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const uint tile = group / p[7], h = group % p[7];
    const long source_head = heads[h];
    float sq = 0, sk = 0, maxq = -INFINITY, maxk = -INFINITY;
    float minq = INFINITY, mink = INFINITY;
    for (uint row = 0; row < 128; ++row) {
        const long packed = perm[tile * 128 + row];
        const long out = ((long(tile) * 128 + row) * p[7] + h) * 128 + d;
        DTYPE qval = 0, kval = 0, vval = 0;
        if (packed >= 0) {
            qval = q[packed * p[0] + source_head * p[1] + d];
            kval = k[packed * p[2] + source_head * p[3] + d];
            vval = v[packed * p[4] + source_head * p[5] + d];
            sq += float(qval); sk += float(kval);
            maxq = max(maxq, float(qval)); maxk = max(maxk, float(kval));
            minq = min(minq, float(qval)); mink = min(mink, float(kval));
        }
        qt[out] = qval; kt[out] = kval; vt[out] = vval;
    }
    if (tile < p[8]) {
        const long offset = (long(h) * p[8] + tile) * 384 + d;
        const int count = valid[tile];
        fq[offset] = count ? sq / count : 0;
        fk[offset] = count ? sk / count : 0;
        fq[offset + 128] = count ? maxq : 0;
        fk[offset + 128] = count ? maxk : 0;
        fq[offset + 256] = count ? minq : 0;
        fk[offset + 256] = count ? mink : 0;
    }
}
'''


@functools.lru_cache(maxsize=2)
def _gather_library(dtype):
    return torch.mps.compile_shader(_GATHER_POOL.replace('DTYPE', dtype))


def gather_pool(q, k, v, layout, heads):
    if q.shape[-1] != 128 or any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError('Veda fused pooling requires contiguous 128-channel heads')
    shape = (layout.num_slots, len(heads), 128)
    qt, kt, vt = [torch.empty(shape, dtype=q.dtype, device=q.device) for _ in range(3)]
    fq, fk = [torch.empty((len(heads), layout.n_video_tiles, 384), device=q.device, dtype=torch.float32) for _ in range(2)]
    params = torch.tensor([*q.stride()[:2], *k.stride()[:2], *v.stride()[:2],
                           layout.num_slots, len(heads), layout.n_video_tiles],
                          dtype=torch.int64, device=q.device)
    impl = _gather_library('bfloat' if q.dtype == torch.bfloat16 else 'half')
    impl.veda_gather_pool(q, k, v, layout.perm, layout.valid_count, heads, params,
                         qt, kt, vt, fq, fk,
                         threads=(layout.n_tiles * len(heads) * 128, 1, 1), group_size=(128, 1, 1))
    return qt, kt, vt, fq, fk


def _remove_branches(source, marker):
    while marker in source:
        start = source.index(marker)
        opening = source.index('{', start)
        depth, end = 1, opening + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        source = source[:start] + source[end:]
    return source


def _source(dtype='bfloat', key_block=32):
    source = torch_mpp_qcache._source(dtype)
    start = source.index('prefill_attention_mpp(')
    prefix, body = source[:start], source[start:]
    body = _remove_branches(body, 'if constexpr (HAS_MASK)')
    body = _remove_branches(body, 'if constexpr (DO_CAUSAL)')
    body = body.replace(
        'const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],\n'
        '    const device MaskType* mask [[buffer(6)]],',
        'const constant int* sparse_params [[buffer(5)]],\n'
        '    const device int* indices [[buffer(6)]],\n'
        '    const device int* counts [[buffer(7)]],\n'
        '    const device int* valid_counts [[buffer(8)]],')
    body = body.replace('  const bool align_K = (kL_rem == 0);',
                        '  const bool align_K_initial = (kL_rem == 0);')
    body = body.replace('const int NK = align_K ?', 'const int NK = align_K_initial ?')
    before = '  int kb_lim = NK;'
    after = '''  const int query_tile = block_idx * BQ / 128;
  const int video_tiles = sparse_params[1];
  const int global_tiles = sparse_params[0] - video_tiles;
  const int width = sparse_params[2];
  const bool video_query = query_tile < video_tiles;
  const int count = video_query ? counts[head_idx * video_tiles + query_tile] : 0;
  const device T* Kbase = K;
  const device T* Vbase = V;
  int kb_lim = (video_query ? count + global_tiles : sparse_params[0]) * (128 / BK);'''
    assert body.count(before) == 1
    body = body.replace(before, after)
    body = body.replace('  const short lim_rows_k = kL_rem;', '')
    before = '    const bool is_last_k = (kb == NK_aligned);'
    after = '''    const int selected = kb / (128 / BK);
    const int tile = video_query
        ? (selected < count ? indices[(head_idx * video_tiles + query_tile) * width + selected]
                            : video_tiles + selected - count)
        : selected;
    const int within_tile = (kb % (128 / BK)) * BK;
    const short lim_rows_k = short(valid_counts[tile] - within_tile);
    if (lim_rows_k <= 0) continue;
    const bool align_K = lim_rows_k >= BK;
    const bool is_last_k = !align_K;
    K = Kbase + (tile * 128 + within_tile) * k_seq_stride;
    V = Vbase + (tile * 128 + within_tile) * v_seq_stride;'''
    assert body.count(before) == 1
    body = body.replace(before, after).replace('>= kL_rem', '>= lim_rows_k')
    body = body.replace('    K += BK * k_seq_stride;\n    V += BK * v_seq_stride;', '')
    body = body.replace(f'{dtype}, 64, 32, 128,', f'{dtype}, 64, {key_block}, 128,')
    return prefix + body


@functools.lru_cache(maxsize=4)
def _library(dtype, key_block):
    return torch.mps.compile_shader(_source(dtype, key_block))


def attend_selected(q, k, v, indices, keep, layout, key_block=32):
    if not torch_mpp_qcache._hardware():
        raise base.BackendUnavailable('Veda Metal MPP currently requires M5 Ultra')
    if q.dtype not in (torch.bfloat16, torch.float16) or q.shape[-1] != 128:
        raise base.BackendUnavailable('Veda Metal needs FP16/BF16 QKV with head dimension 128')
    n, heads, dim = q.shape
    # Walk selected tiles in sequence order to improve K/V locality.
    selected = indices.masked_fill(~keep, layout.n_tiles).sort(-1).values.to(torch.int32).contiguous()
    counts = keep.sum(-1, dtype=torch.int32).contiguous()
    params = torch.tensor([layout.n_tiles, layout.n_video_tiles, selected.shape[-1]],
                          dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    bhq = q.transpose(0, 1).unsqueeze(0)
    bhk = k.transpose(0, 1).unsqueeze(0)
    bhv = v.transpose(0, 1).unsqueeze(0)
    raw = struct.pack('=6i2f12q', 1, heads, dim, n, n, 1, dim ** -0.5, 1.,
                      *bhq.stride()[:3], *bhk.stride()[:3], *bhv.stride()[:3],
                      n * heads * dim, dim, heads * dim)
    attn_params = torch.tensor(list(raw), dtype=torch.uint8, device=q.device)
    library = _library('bfloat' if q.dtype == torch.bfloat16 else 'half', key_block)
    library.custom_attn(q, k, v, out, attn_params, params, selected, counts,
                        layout.valid_count,
                        threads=((n // 64) * 128, heads, 1), group_size=(128, 1, 1))
    return out


class MetalBackend(base.Backend):
    name = 'veda-metal'
    display = 'Veda Metal MPP'
    dtypes = (torch.bfloat16, torch.float16)

    def __init__(self, key_block=32):
        self.key_block = key_block

    def attend(self, q, k, v, block_mask, layout):
        index, counts = selection.tile_index_list(block_mask[:, :layout.n_video_tiles, :layout.n_video_tiles])
        keep = torch.arange(index.shape[-1], device=q.device)[None, None, :] < counts[..., None]
        return self.attend_selected(q, k, v, index, keep, layout)

    def attend_selected(self, q, k, v, index, keep, layout):
        return attend_selected(q, k, v, index, keep, layout, self.key_block)
