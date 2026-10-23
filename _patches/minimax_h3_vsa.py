"""ModelPatcher integration for FastH3's trained Metal VSA attention."""
from collections import Counter
import logging
import math

import torch

from . import h3_vsa_mps

STATE_KEY = "asfp8_h3_vsa"


def reference_grids(payload):
    """Keep image refs dense and describe each reference VIDEO independently."""
    grids = []
    for ref in (payload or {}).get("refs", ()):
        kind = ref["kind"]
        if kind == "image":
            grids.append(None)
        elif kind in ("video", "video_audio"):
            grids.append((int(ref["latent_t"]), int(ref["latent_h"])//2, int(ref["latent_w"])//2))
        elif kind != "audio":
            raise ValueError(f"Metal VSA does not recognize reference kind {kind!r}")
    return tuple(grids)


def layout_segments(layout, grids):
    """Resolve ComfyUI's ref_img spans using metadata from the actual payload."""
    target = tuple(int(n) for n in layout.signature[1:4])
    target = (target[0], target[1]//2, target[2]//2)
    segments, ref_index, cursor = [], 0, 0
    for start, stop, kind in layout.segments:
        if start != cursor or stop < start:
            raise ValueError("Metal VSA needs contiguous packed H3 segments")
        count = stop-start
        if kind == "video":
            segment = target
        elif kind == "ref_img":
            if ref_index >= len(grids):
                raise ValueError("Metal VSA is missing reference metadata; reconnect its model output")
            segment = grids[ref_index] or count
            ref_index += 1
        elif kind in ("text", "cond", "cond_audio", "ref_audio", "audio"):
            segment = count
        else:
            raise ValueError(f"Metal VSA does not recognize packed segment {kind!r}")
        if isinstance(segment, tuple) and math.prod(segment) != count:
            raise ValueError("Metal VSA reference/target grid does not match its packed rows")
        segments.append(segment)
        cursor = stop
    if ref_index != len(grids) or cursor != layout.seq_len:
        raise ValueError("Metal VSA metadata does not match the current H3 layout")
    return tuple(segments)


class VSAPatch:
    def __init__(self, *, tile_size, keep_percent, reference_keep_percent, start_sigma,
                 end_sigma, min_tokens, dense_layers, head_chunk, workspace_mib, verbose):
        self.tile_size = tile_size
        self.sparsity = 1-keep_percent/100
        self.reference_keep = reference_keep_percent/100
        self.start_sigma, self.end_sigma = start_sigma, end_sigma
        self.min_tokens, self.dense_layers = min_tokens, dense_layers
        self.head_chunk, self.workspace_mib = head_chunk, workspace_mib
        self.verbose = verbose
        self.plans = {}
        self.stats = Counter()
        self.last_stats = {}

    def model_wrapper(self, executor, x, timestep, context, transformer_options=None,
                      minimax_payload=None, **kwargs):
        options = dict(transformer_options or {})
        grids = reference_grids(minimax_payload)
        options[STATE_KEY+"_refs"] = tuple(None for _ in grids) if self.tile_size == 64 else grids
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    def plan(self, layout, grids, device):
        segments = layout_segments(layout, grids)
        key = (segments, str(device))
        if key not in self.plans:
            if len(self.plans) >= 4:
                self.plans.clear()
            self.plans[key] = h3_vsa_mps.build_plan(segments, device, self.tile_size)
            self.stats["packed_tokens"] = layout.seq_len
            self.stats["video_regions"] = len(self.plans[key].regions)
            self.stats["tile_size"] = self.tile_size
            if self.verbose:
                logging.info("[AppleSilicon-Optimized/H3VSA] %d tokens, %d video region(s), %d-token tiles",
                             layout.seq_len, len(self.plans[key].regions), self.tile_size)
        return self.plans[key]

    def attention(self, attn, index, x, rope_freqs, options):
        from comfy import model_management, quant_ops
        from . import int8_linear_kernel_mps as i8k
        sigma = options.get("sigmas")
        outside = sigma is not None and not self.end_sigma <= float(sigma.flatten()[0]) <= self.start_sigma
        if outside or x.shape[0] < self.min_tokens or index in self.dense_layers:
            self.stats["dense_calls"] += 1
            return attn(x, rope_freqs=rope_freqs, transformer_options=options)
        layout = options.get("minimax_h3_layout")
        if layout is None or layout.seq_len != x.shape[0]:
            raise ValueError("Metal VSA needs the current native H3 packed layout")
        plan = self.plan(layout, options.get(STATE_KEY+"_refs", ()), x.device)
        s, heads, dim = x.shape[0], attn.heads, attn.head_dim
        direct = None
        if rope_freqs is not None:
            qw = model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = model_management.cast_to(attn.k_norm.weight, device=x.device)
            direct = i8k.int8_qkv_rms_rope_prequant(attn.qkv_proj, x, rope_freqs,
                qw.float().contiguous(), kw.float().contiguous(), heads, dim, attn.q_norm.eps)
        if direct is not None:
            q, k, v = (t[0].transpose(0, 1) for t in direct)
            self.stats["qkv_epilogue_calls"] += 1
        else:
            q, k, v = attn.qkv_proj(x).split(heads*dim, dim=-1)
            q, k = q.view(1, s, heads, dim), k.view(1, s, heads, dim)
            if rope_freqs is not None:
                quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3]*2)
            else:
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k, v = q[0], k[0], v.view(s, heads, dim)
        gate = attn.to_gate_compress(x).view(s, heads, dim)
        out = h3_vsa_mps.attention(q, k, v, gate, plan,
            sparsity=self.sparsity, reference_keep=self.reference_keep,
            head_chunk=self.head_chunk, workspace_mib=self.workspace_mib)
        self.stats["vsa_calls"] += 1
        projected = i8k.int8_o_proj_prequant(attn.out_proj, out.transpose(0,1).unsqueeze(0))
        if projected is not None:
            self.stats["o_projection_calls"] += 1
            return projected.squeeze(0)
        return attn.out_proj(out.reshape(s, heads*dim))

    def block_patch(self, block, index):
        def attention(x, rope_freqs=None, transformer_options=None):
            return self.attention(block.attn, index, x, rope_freqs, transformer_options or {})
        def patch(args, extra):
            return extra["original_block"]({**args, "attention": attention})
        return patch

    def cleanup(self):
        self.last_stats = dict(self.stats)
        if self.stats:
            logging.info("[AppleSilicon-Optimized/H3VSA] %d Metal VSA calls, %d dense calls; tile size %d",
                         self.stats["vsa_calls"], self.stats["dense_calls"], self.tile_size)
        self.stats.clear()
        self.plans.clear()


def apply_vsa(model, *, tile_size=128, keep_percent=10., reference_keep_percent=10.,
              start_percent=0., end_percent=1., min_tokens=12288, dense_layers="",
              head_chunk=4, memory_budget_gib=0., verbose=False):
    import comfy.patcher_extension as pe
    from .minimax_h3_sparse import parse_dense_layers
    diffusion = model.get_model_object("diffusion_model")
    if type(diffusion).__name__ != "MiniMaxH3Model":
        raise ValueError("Metal VSA requires a FastH3 model")
    if any(block.attn.to_gate_compress is None for block in diffusion.blocks):
        raise ValueError("Metal VSA requires a checkpoint with trained to_gate_compress layers")
    options = model.model_options.get("transformer_options", {})
    if (STATE_KEY in options or "asfp8_h3_sparse" in options
            or options.get("patches_replace", {}).get("dit")):
        raise ValueError("Metal VSA needs one attention node; remove other sparse/cache/token patches")
    if not (tile_size in (64, 128) and 0 < keep_percent <= 100 and 0 < reference_keep_percent <= 100):
        raise ValueError("Metal VSA needs 64/128-token tiles and keep percentages in (0,100]")
    if not (0 <= start_percent <= end_percent <= 1 and min_tokens >= 0 and head_chunk > 0
            and math.isfinite(memory_budget_gib) and memory_budget_gib >= 0):
        raise ValueError("Invalid Metal VSA window, token threshold or temporary-memory settings")
    sampling = model.get_model_object("model_sampling")
    patch = VSAPatch(tile_size=tile_size, keep_percent=keep_percent,
        reference_keep_percent=reference_keep_percent,
        start_sigma=float(sampling.percent_to_sigma(start_percent)),
        end_sigma=float(sampling.percent_to_sigma(end_percent)), min_tokens=min_tokens,
        dense_layers=parse_dense_layers(dense_layers), head_chunk=head_chunk,
        workspace_mib=memory_budget_gib*1024 if memory_budget_gib else 256., verbose=verbose)
    patched = model.clone()
    patched.model_options.setdefault("transformer_options", {})[STATE_KEY] = patch
    patched.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, STATE_KEY, patch.model_wrapper)
    for i, block in enumerate(diffusion.blocks):
        patched.set_model_patch_replace(patch.block_patch(block, i), "dit", "double_block", i)
    patched.add_callback_with_key(pe.CallbacksMP.ON_CLEANUP, STATE_KEY, lambda mp: patch.cleanup())
    return patched
