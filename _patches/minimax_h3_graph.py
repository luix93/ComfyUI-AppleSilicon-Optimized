"""Guarded MiniMax H3 block-graph rewrites for residual eager sequences.

This is deliberately a small architecture rewriter, not a general compiler.
It recognizes stable H3 module graphs, verifies runtime tensor geometry, and
replaces only patterns backed by a purpose-built Metal kernel.  Unknown ComfyUI
revisions fall through to their original Python forwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import sys
from typing import Any, Callable

import torch


TAG = "[AppleSilicon-Optimized/h3_graph]"


@dataclass(frozen=True)
class RewriteResult:
    pattern: str
    applied: bool
    reason: str


def _forward_names(cls) -> set[str]:
    forward = getattr(cls, "forward", None)
    code = getattr(forward, "__code__", None)
    return set(getattr(code, "co_names", ()))


def _valid_segment(segment, length: int):
    if not isinstance(segment, (tuple, list)) or len(segment) != 3:
        return None
    try:
        start, stop, row = map(int, segment)
    except (TypeError, ValueError):
        return None
    if not (0 <= start < stop <= length and row >= 0):
        return None
    return start, stop, row


def rewrite_final_layer(
    module: Any,
    kernel: Any,
    segment_rows: Callable[..., torch.Tensor | None],
) -> RewriteResult:
    """Fuse the two target-tail norm/modulate/cast chains into one pass."""
    cls = getattr(module, "FinalLayer", None)
    if cls is None:
        return RewriteResult("final-rms-mod-cast", False, "FinalLayer absent")
    if getattr(cls.forward, "_asfp8_h3_graph", False):
        return RewriteResult("final-rms-mod-cast", True, "already applied")
    if tuple(inspect.signature(cls.forward).parameters) != (
            "self", "x", "t_emb", "video_seg", "audio_seg"):
        return RewriteResult("final-rms-mod-cast", False,
                             "unrecognized FinalLayer signature; retaining native PDD/schedule handling")
    required = {"adaln_proj", "norm", "video_out", "audio_out"}
    missing = required - _forward_names(cls)
    if missing:
        return RewriteResult(
            "final-rms-mod-cast", False,
            f"unrecognized FinalLayer graph; missing {sorted(missing)}")
    if not hasattr(kernel, "rms_mod_seg_f32"):
        return RewriteResult(
            "final-rms-mod-cast", False, "Metal kernel unavailable")

    original = cls.forward
    state = {"dead": False, "announced": False}

    def rewritten(self, x, t_emb, video_seg, audio_seg):
        if (state["dead"] or not isinstance(x, torch.Tensor)
                or x.device.type != "mps" or x.dtype != torch.bfloat16
                or x.dim() != 2 or not x.is_contiguous()):
            return original(self, x, t_emb, video_seg, audio_seg)
        video = _valid_segment(video_seg, x.shape[0])
        audio = _valid_segment(audio_seg, x.shape[0])
        if video is None or audio is None:
            return original(self, x, t_emb, video_seg, audio_seg)
        va, vb, vrow = video
        aa, ab, arow = audio
        ordered = sorted((video, audio), key=lambda value: value[0])
        if ordered[0][1] != ordered[1][0]:
            return original(self, x, t_emb, video_seg, audio_seg)
        start, stop = ordered[0][0], ordered[1][1]
        target = x[start:stop]
        if not target.is_contiguous():
            return original(self, x, t_emb, video_seg, audio_seg)

        try:
            shift, scale = self.adaln_proj(t_emb)
            if (not isinstance(shift, torch.Tensor)
                    or not isinstance(scale, torch.Tensor)
                    or shift.shape != scale.shape or shift.dim() != 2
                    or max(vrow, arow) >= shift.shape[0]
                    or shift.shape[1] != x.shape[1]):
                return original(self, x, t_emb, video_seg, audio_seg)
            local_segments = [
                (seg_start - start, seg_stop - start, row)
                for seg_start, seg_stop, row in ordered
            ]
            rows = segment_rows(local_segments, stop - start, x.device)
            weight = getattr(self.norm, "weight", None)
            if rows is None or not isinstance(weight, torch.Tensor):
                return original(self, x, t_emb, video_seg, audio_seg)
            import comfy.model_management as mm
            weight_bf16 = mm.cast_to(
                weight, dtype=torch.bfloat16, device=x.device).contiguous()
            fused = kernel.rms_mod_seg_f32(
                target, weight_bf16,
                scale.to(device=x.device, dtype=torch.bfloat16).contiguous(),
                shift.to(device=x.device, dtype=torch.bfloat16).contiguous(),
                rows, float(self.norm.eps))
        except Exception as exc:
            state["dead"] = True
            sys.stderr.write(
                f"{TAG} final-layer rewrite disabled ({str(exc)[:200]})\n")
            return original(self, x, t_emb, video_seg, audio_seg)

        if not state["announced"]:
            state["announced"] = True
            sys.stderr.write(
                f"{TAG} final target RMSNorm + modulation + fp32 casts fused.\n")
        hv = fused[va - start:vb - start]
        ha = fused[aa - start:ab - start]
        return self.video_out(hv), self.audio_out(ha)

    rewritten._asfp8_h3_graph = True
    rewritten._asfp8_original = original
    cls.forward = rewritten
    return RewriteResult("final-rms-mod-cast", True, "matched and replaced")


def rewrite_h3_graph(module, kernel, segment_rows):
    """Apply every known exact H3 graph pattern and return an audit report."""
    return [rewrite_final_layer(module, kernel, segment_rows)]
