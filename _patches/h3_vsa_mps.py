"""Experimental FastH3 Omni VSA operator, independent of SOL selection.

Implements BF16/FP16 exempt-prefix VSA with 128-token Omni tiles or 64-token
ComfyUI-style tiles. Reference-video selection is independent per region.
Algorithm reference: FastVideo cdda1b7535fe32699335e39db82c1d4db21ceb88,
fastvideo/attention/backends/video_sparse_attn_h3.py (Apache-2.0).
See third_party/FastVideo-Apache-2.0.txt and THIRD_PARTY_NOTICES.md.

The existing PyTorch/MLX-derived Metal tile-list kernel supplies fine attention;
the learned coarse branch and routing run through PyTorch on the same MPS
device. No learned predictor, SOL correction, forced diagonal, quantized attention,
model monkey patch, CPU attention fallback, or runtime download is involved.
Workflow integration is owned by minimax_h3_vsa.py; this module owns no model
weights, sampling schedule, or persistent activation cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math
import struct
from typing import Sequence

import torch

TILE = 128
CUBE = (4, 4, 8)
Segment = int | tuple[int, int, int]

_STAGING = r'''
#include <metal_stdlib>
using namespace metal;
kernel void vsa_gather_pool(
    device const DTYPE* q [[buffer(0)]], device const DTYPE* k [[buffer(1)]],
    device const DTYPE* v [[buffer(2)]], device const DTYPE* g [[buffer(3)]],
    device const long* perm [[buffer(4)]], device const int* valid [[buffer(5)]],
    constant long* p [[buffer(6)]], device DTYPE* qt [[buffer(7)]],
    device DTYPE* kt [[buffer(8)]], device DTYPE* vt [[buffer(9)]],
    device DTYPE* gt [[buffer(10)]], device float* pq [[buffer(11)]],
    device float* pk [[buffer(12)]], device float* pv [[buffer(13)]],
    uint d [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    const long tile = group / p[12], h = group % p[12];
    float sq = 0, sk = 0, sv = 0;
    for (uint row = 0; row < VSA_TILE; ++row) {
        const long source = perm[tile*VSA_TILE+row];
        const long dest = ((tile*VSA_TILE+row)*p[12]+h)*128+d;
        DTYPE qv = 0, kv = 0, vv = 0, gv = 0;
        if (source >= 0) {
            qv = q[source*p[0]+h*p[1]+d*p[2]];
            kv = k[source*p[3]+h*p[4]+d*p[5]];
            vv = v[source*p[6]+h*p[7]+d*p[8]];
            gv = g[source*p[9]+h*p[10]+d*p[11]];
            sq += float(qv); sk += float(kv); sv += float(vv);
        }
        qt[dest] = qv; kt[dest] = kv; vt[dest] = vv; gt[dest] = gv;
    }
    const long dest = (h*p[13]+tile)*128+d;
    pq[dest] = sq / valid[tile];
    pk[dest] = sk / valid[tile];
    pv[dest] = sv / valid[tile];
}
kernel void vsa_merge_scatter(
    device const DTYPE* fine [[buffer(0)]], device const float* coarse [[buffer(1)]],
    device const DTYPE* gate [[buffer(2)]], device const long* inverse [[buffer(3)]],
    constant long* p [[buffer(4)]], device DTYPE* out [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p[0]*p[1]*128) return;
    const long d = i % 128, h = (i/128) % p[1], row = i/(128*p[1]);
    const long slot = inverse[row], source = (slot*p[1]+h)*128+d;
    // Upstream rounds the coarse branch to activation dtype BEFORE addcmul.
    const DTYPE c = DTYPE(coarse[(h*p[2]+slot/VSA_TILE)*128+d]);
    out[row*p[3]+h*p[4]+d*p[5]] = DTYPE(fma(float(c), float(gate[source]), float(fine[source])));
}
'''


@functools.lru_cache(maxsize=4)
def _staging_library(dtype, tile_size):
    return torch.mps.compile_shader(_STAGING.replace("DTYPE", dtype).replace("VSA_TILE", str(tile_size)))


def _metal_gather_pool(q, k, v, gate, plan):
    heads = q.shape[1]
    shape = (plan.num_slots, heads, 128)
    qt, kt, vt, gt = [torch.empty(shape, device=q.device, dtype=q.dtype) for _ in range(4)]
    pq, pk, pv = [torch.empty((heads, plan.n_tiles, 128), device=q.device, dtype=torch.float32) for _ in range(3)]
    params = torch.tensor([*q.stride(), *k.stride(), *v.stride(), *gate.stride(), heads, plan.n_tiles],
                          dtype=torch.int64, device=q.device)
    lib = _staging_library("bfloat" if q.dtype == torch.bfloat16 else "half", plan.tile_size)
    lib.vsa_gather_pool(q, k, v, gate, plan.perm, plan.valid_count, params,
                        qt, kt, vt, gt, pq, pk, pv,
                        threads=plan.n_tiles*heads*128, group_size=128)
    return qt, kt, vt, gt, pq, pk, pv


def _metal_merge_scatter(fine, coarse, gate, plan, out):
    heads = fine.shape[1]
    params = torch.tensor([plan.seq_len, heads, plan.n_tiles, *out.stride()], dtype=torch.int64, device=out.device)
    lib = _staging_library("bfloat" if out.dtype == torch.bfloat16 else "half", plan.tile_size)
    lib.vsa_merge_scatter(fine, coarse, gate, plan.inverse, params, out,
                          threads=plan.seq_len*heads*128, group_size=256)


@dataclass(frozen=True)
class Plan:
    """Video tiles first, then segment-pure dense tiles (a kernel permutation).

    Within each video region the upstream t/h/w tile and row order is retained.
    Dense segments never share a partially occupied tile.
    ``inverse`` maps packed real rows to their unique padded tile slots.
    """

    segments: tuple[Segment, ...]
    perm: torch.Tensor
    inverse: torch.Tensor
    valid_count: torch.Tensor
    regions: tuple[tuple[int, int], ...]
    seq_len: int
    n_video_tiles: int
    n_tiles: int
    tile_size: int = TILE

    @property
    def num_slots(self):
        return self.n_tiles * self.tile_size


def build_plan(segments: Sequence[Segment], device="cpu", tile_size=128) -> Plan:
    """Each int is a dense segment; each (T,H,W) is a sparse video region.

    Supply token-grid dimensions, not pixel/latent-channel dimensions. List
    every reference video separately, followed eventually by the target video.
    ComfyUI's ambiguous ref_img labels are deliberately not guessed here.
    """
    if tile_size not in (64, 128):
        raise ValueError("VSA tile size must be 64 (ComfyUI) or 128 (FastH3 Omni)")
    cube = (4, 4, tile_size // 16)
    segments = tuple(segments)
    video, dense, regions = [], [], []
    cursor = tile_cursor = 0
    for seg in segments:
        if type(seg) is int:
            if seg < 0:
                raise ValueError("dense segment lengths must be non-negative")
            for start in range(cursor, cursor + seg, tile_size):
                rows = torch.full((tile_size,), -1, dtype=torch.int64)
                count = min(tile_size, cursor + seg - start)
                rows[:count] = torch.arange(start, start + count)
                dense.append(rows)
            cursor += seg
        else:
            if (not isinstance(seg, tuple) or len(seg) != 3
                    or any(type(n) is not int or n <= 0 for n in seg)):
                raise ValueError("video regions must be positive integer (T,H,W) tuples")
            t, h, w = seg
            grid = torch.arange(cursor, cursor + t*h*w).view(t, h, w)
            begin = tile_cursor
            for ti in range(0, t, cube[0]):
                for hi in range(0, h, cube[1]):
                    for wi in range(0, w, cube[2]):
                        live = grid[ti:ti+4, hi:hi+4, wi:wi+cube[2]].reshape(-1)
                        rows = torch.full((tile_size,), -1, dtype=torch.int64)
                        rows[:live.numel()] = live
                        video.append(rows)
                        tile_cursor += 1
            regions.append((begin, tile_cursor))
            cursor += t*h*w
    if not regions or not isinstance(segments[-1], tuple):
        raise ValueError("segments must end with the target video region")
    rows = torch.stack(video + dense)
    perm = rows.reshape(-1)
    real = perm >= 0
    inverse = torch.empty(cursor, dtype=torch.int64)
    inverse[perm[real]] = torch.arange(perm.numel())[real]
    return Plan(segments, perm.to(device), inverse.to(device),
                (rows >= 0).sum(-1).to(device=device, dtype=torch.int32),
                tuple(regions), cursor, tile_cursor, rows.shape[0], tile_size)


def gather(x: torch.Tensor, plan: Plan) -> torch.Tensor:
    """Packed [S,H,D] -> zero-padded tiled [N,H,D]."""
    return x[plan.perm.clamp_min(0)].masked_fill((plan.perm < 0)[:, None, None], 0)


def pool(x: torch.Tensor, plan: Plan) -> torch.Tensor:
    """True valid-row mean, FP32 accumulation; result [H,tiles,D]."""
    return (x.reshape(plan.n_tiles, plan.tile_size, *x.shape[1:]).sum(1, dtype=torch.float32)
            / plan.valid_count[:, None, None]).permute(1, 0, 2)


def _sparsities(plan, sparsity, reference_keep):
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("sparsity must be finite and in [0,1)")
    if not math.isfinite(reference_keep) or not 0 < reference_keep <= 1:
        raise ValueError("reference_keep must be finite and in (0,1]")
    # Upstream's zero-sparsity override makes every video region dense.
    return ((1-reference_keep,) * (len(plan.regions)-1) + (sparsity,)
            if sparsity > 0 else (0.,) * len(plan.regions))


def select(scores, plan, sparsity=.9, reference_keep=.1):
    """Per-query/per-head top-k independently in EACH video-key region.

    No diagonal/local rescue or fractional-budget rule is applied.
    Dense query tiles and dense keys are implicit in the native fine kernel.
    Ties use torch.topk, as upstream does; cross-device tie parity is not promised.
    """
    parts = []
    for (start, stop), sparse in zip(plan.regions, _sparsities(plan, sparsity, reference_keep)):
        count = max(1, min(math.ceil((1-sparse)*(stop-start)), stop-start))
        part = scores[:, :plan.n_video_tiles, start:stop].topk(count, dim=-1).indices
        parts.append(part + start)
    index = torch.cat(parts, -1)
    return index, torch.ones_like(index, dtype=torch.bool)


def reference_fine(q, k, v, index, keep, plan):
    """Small-shape oracle only: materializes token-level masks and FP32 scores."""
    if plan.num_slots > 4096:
        raise ValueError("reference attention is limited to 4096 padded tokens")
    mask = torch.ones((q.shape[1], plan.n_tiles, plan.n_tiles), dtype=torch.bool, device=q.device)
    mask[:, :plan.n_video_tiles, :plan.n_video_tiles] = False
    video_mask = mask[:, :plan.n_video_tiles, :plan.n_video_tiles]
    video_mask.scatter_(-1, index, keep)
    allowed = mask.repeat_interleave(plan.tile_size, 1).repeat_interleave(plan.tile_size, 2)
    valid = (plan.perm >= 0)
    allowed = allowed & valid[None, None, :]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k.float()) * q.shape[-1]**-.5
    probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return torch.einsum('hqk,khd->qhd', probs, v.float()).to(q.dtype)


@functools.lru_cache(maxsize=2)
def _fine64_library(dtype):
    from .h3_tile_attention_mps import _source
    source = _source(dtype)
    for old, new in (("block_idx * BQ / 128", "block_idx * BQ / 64"),
                     ("(128 / BK)", "(64 / BK)"),
                     ("tile * 128 + within_tile", "tile * 64 + within_tile")):
        if old not in source:
            raise RuntimeError("VSA64 fine kernel source no longer matches the validated tile-list kernel")
        source = source.replace(old, new)
    return torch.mps.compile_shader(source)


def _fine_metal(q, k, v, index, keep, plan):
    from .h3_tile_attention_mps import attend_selected
    if plan.tile_size == 128:
        return attend_selected(q, k, v, index, keep, plan)
    selected = index.masked_fill(~keep, plan.n_tiles).sort(-1).values.to(torch.int32).contiguous()
    counts = keep.sum(-1, dtype=torch.int32).contiguous()
    params = torch.tensor([plan.n_tiles, plan.n_video_tiles, selected.shape[-1]], dtype=torch.int32, device=q.device)
    n, heads, dim = q.shape
    qs, ks, vs = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    raw = struct.pack('=6i2f12q', 1, heads, dim, n, n, 1, dim**-.5, 1.,
                      *qs.stride()[:3], *ks.stride()[:3], *vs.stride()[:3], n*heads*dim, dim, heads*dim)
    attn_params = torch.tensor(list(raw), dtype=torch.uint8, device=q.device)
    out = torch.empty_like(q)
    library = _fine64_library('bfloat' if q.dtype == torch.bfloat16 else 'half')
    library.custom_attn(q, k, v, out, attn_params, params, selected, counts, plan.valid_count,
                        threads=((n//64)*128, heads, 1), group_size=(128, 1, 1))
    return out


def _validate_inputs(q, k, v, gate, plan):
    if (q.ndim != 3 or q.shape[0] != plan.seq_len or q.shape[1] < 1 or q.shape[-1] != 128
            or any(t.shape != q.shape or t.dtype != q.dtype or t.device != q.device for t in (k, v, gate))):
        raise ValueError("VSA requires matching Q/K/V/gate [S,H,128] tensors")
    if q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("VSA requires FP32 reference or FP16/BF16 activations")
    if any(t.device != q.device for t in (plan.perm, plan.valid_count, plan.inverse)):
        raise ValueError("VSA plan and activations must use the same device")
    if torch.is_grad_enabled():
        raise ValueError("VSA prototype is inference-only; use torch.inference_mode()")


def attention(q, k, v, gate, plan, *, backend="metal", sparsity=.9,
              reference_keep=.1, head_chunk=4, workspace_mib=256):
    """Full VSA fine + learned coarse branch, packed [S,H,128] -> same.

    ``gate`` is the output of the checkpoint's raw to_gate_compress projection
    on normalized/modulated input, before RoPE. It is NOT a sigmoid and MUST
    NOT be replaced by external predictor scores. Q/K are already normalized/RoPE'd.
    The operator does not accept a model or alter its weights/sampling schedule.
    Unsupported Metal hardware/builds fail explicitly, never silently go dense.
    """
    _validate_inputs(q, k, v, gate, plan)
    _sparsities(plan, sparsity, reference_keep)
    if type(head_chunk) is not int or head_chunk <= 0:
        raise ValueError("head_chunk must be a positive integer")
    if not math.isfinite(workspace_mib) or workspace_mib <= 0:
        raise ValueError("workspace_mib must be finite and positive")
    if backend == "metal":
        from . import torch_mpp_qcache
        if (q.device.type != "mps" or q.dtype not in (torch.bfloat16, torch.float16)
                or not torch_mpp_qcache._hardware() or not torch_mpp_qcache._torch_build_supported()):
            raise RuntimeError("Metal VSA currently requires M5 Ultra and reviewed PyTorch 2.14.0/2.14.1")
        fine = _fine_metal
    elif backend == "reference":
        fine = reference_fine
    else:
        raise ValueError(f"unknown VSA backend: {backend}")
    # Conservative scratch estimate: tiled QKVG + fine output; FP32 coarse
    # scores/probabilities/topk workspace; pooled vectors and output/scatter.
    # Caller-owned packed QKVG and final output are outside this scratch budget.
    per_head = (plan.num_slots * 128 * q.element_size() * 7
                + plan.n_tiles**2 * 32 + plan.n_tiles * 128 * 4 * 4)
    fit = int(workspace_mib * 2**20) // per_head
    if fit < 1:
        raise ValueError("VSA scratch ceiling cannot hold one head; increase workspace_mib")
    chunk = min(head_chunk, fit, q.shape[1])
    out = torch.empty_like(q)
    for start in range(0, q.shape[1], chunk):
        stop = min(start + chunk, q.shape[1])
        if backend == "metal":
            qt, kt, vt, gt, pq, pk, pv = _metal_gather_pool(
                *(t[:, start:stop] for t in (q, k, v, gate)), plan)
        else:
            qt, kt, vt, gt = (gather(t[:, start:stop], plan) for t in (q, k, v, gate))
            pq, pk, pv = (pool(t, plan) for t in (qt, kt, vt))
        scores = torch.matmul(pq, pk.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        index, keep = select(scores, plan, sparsity, reference_keep)
        refined = fine(qt, kt, vt, index, keep, plan)
        # The upstream branch casts pooled-attention output BEFORE addcmul.
        coarse = torch.matmul(scores.softmax(-1), pv)
        if backend == "metal":
            _metal_merge_scatter(refined, coarse, gt, plan, out[:, start:stop])
        else:
            coarse = coarse.permute(1, 0, 2).to(refined.dtype)
            merged = torch.addcmul(refined.view(plan.n_tiles, plan.tile_size, stop-start, 128),
                                  coarse[:, None], gt.view(plan.n_tiles, plan.tile_size, stop-start, 128))
            out[:, start:stop] = merged.view(plan.num_slots, stop-start, 128)[plan.inverse]
    return out
