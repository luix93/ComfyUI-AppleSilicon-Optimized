"""Pack-owned bounded Veda execution using trained predictor arithmetic.

Derived from Veda-on-ComfyUI (MIT); see LICENSE and ../ORIGINS.json.
Native layout/pooling and memory guards are specific to our Apple runtime.
"""
import os
import torch
from ... import h3_sol_mps
from .backends import mlx_gather
from .core import engine, selection, tiling, predictor as scorer

class BoundedMlxBackend(mlx_gather.MlxGatherBackend):
    def __init__(self, mx):
        super().__init__(mx, chunk_bytes=128 * 2**20)
        self.previous_cache_limit = mx.set_cache_limit(256 * 2**20)
        mx.clear_cache()

    def close(self):
        self.mx.clear_cache()
        self.mx.set_cache_limit(self.previous_cache_limit)

class AppleVedaEngine(engine.GeometryState):
    def __init__(self, *args, route, memory_budget_gib, **kwargs):
        super().__init__(*args, **kwargs)
        self.projections = {}
        self.projection_bytes = 0
        self.route = route
        self.memory_budget_gib = memory_budget_gib

    def _weights(self, layer):
        if layer not in self.projections:
            self.projections[layer] = tuple(t.float().to(self.device)
                for t in (self.bundle.proj_q[layer], self.bundle.proj_k[layer]))
            self.projection_bytes += sum(t.numel() * t.element_size() for t in self.projections[layer])
        return self.projections[layer]

    def reset(self):
        super().reset()
        self.projections.clear()
        self.projection_bytes = 0

    def attention(self, q, k, v, layer, spec, plan):
        seq_len, heads, dim = q.shape
        budget, _ = h3_sol_mps.workspace_budget(self.memory_budget_gib)
        ratio = float(os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '1.7'))
        if ratio > 0:
            allocator_free = max(0, int(torch.mps.recommended_max_memory() * ratio)
                                 - torch.mps.driver_allocated_memory())
            budget = min(budget, allocator_free // 2)
        output_bytes = (seq_len + 1) * heads * dim * q.element_size()
        if output_bytes >= budget:
            raise h3_sol_mps.WorkspaceBudgetExceeded(
                f'Veda output alone needs {output_bytes / 2**30:.2f} GiB; '
                f'available temporary budget {budget / 2**30:.2f} GiB. Free memory or reduce the sequence.')
        out = q.new_empty(seq_len + 1, heads, dim)
        pq, pk = self._weights(layer)
        chunks = 0
        for group in plan.head_groups(layer, self.device):
            layout = self._tile_layout(spec, group.shape)
            blocks = selection.column_blocks(layout, self.generated, self.reference)
            per_head = (layout.num_slots * dim * q.element_size() * 6
                        + layout.n_video_tiles ** 2 * 16)
            # Include resident predictor projections and the MLX gather workspace.
            resident = self.projection_bytes
            available = budget - output_bytes - resident - (512 * 2**20 if self.route == 'veda_mlx' else 0)
            if available < per_head:
                raise h3_sol_mps.WorkspaceBudgetExceeded(
                    f'Veda needs at least {(per_head + resident) / 2**30:.2f} GiB temporary memory; '
                    f'available budget {budget / 2**30:.2f} GiB. Increase Temporary memory or reduce the sequence.')
            step = max(1, min(len(group.heads), available // per_head))
            self.chunking['heads_per_chunk'] = step
            for hs in group.heads.split(step):
                chunks += 1
                if self.route == 'veda_metal':
                    from ...h3_veda_mps import gather_pool
                    qt, kt, vt, fq, fk = gather_pool(q, k, v, layout, hs)
                else:
                    qt, kt, vt = (tiling.gather_tiles(t, layout, hs) for t in (q, k, v))
                    fq, fk = scorer.pool_video_tiles(qt, layout), scorer.pool_video_tiles(kt, layout)
                scores = scorer.tile_logits(fq, fk, pq.index_select(0, hs), pk.index_select(0, hs))
                del fq, fk
                index, keep = selection.select(scores, layout, blocks)
                del scores
                self.stats.add(keep, layout)
                if self.route == 'veda_metal':
                    ot = self.backend.attend_selected(qt, kt, vt, index, keep, layout)
                else:
                    mask = selection.block_mask(index, keep, layout)
                    ot = self.backend.attend(qt, kt, vt, mask, layout)
                    del mask
                tiling.scatter_tiles_(out, ot, layout, hs)
                del qt, kt, vt, ot, index, keep
        self.chunking['chunks_per_layer'] = chunks
        return out[:seq_len]

