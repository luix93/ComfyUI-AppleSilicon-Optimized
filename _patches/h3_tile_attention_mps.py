"""Tile-list attention using PyTorch's M5 MPP arithmetic on Metal (used by VSA).

Each query tile attends to its own list of selected 128-token key tiles, read
in place; no expanded K/V batches are built. The kernel is the hash-checked
PyTorch 2.14 MPP prefill kernel with masking/causal branches removed and the
key loop walking a tile list. PyTorch/MLX source licenses are in krea2_ext.
Attention over a subset of tiles is approximate to dense attention.
"""

import functools
import struct

import torch

from . import torch_mpp_qcache


class TileAttentionUnavailable(RuntimeError):
    """The device, build or tensor layout cannot run the tile-list kernel."""


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
        raise TileAttentionUnavailable('Tile-list Metal attention currently requires M5 Ultra')
    if q.dtype not in (torch.bfloat16, torch.float16) or q.shape[-1] != 128:
        raise TileAttentionUnavailable('Tile-list Metal attention needs FP16/BF16 QKV with head dimension 128')
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
