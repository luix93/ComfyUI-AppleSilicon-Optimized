"""Opt-in H3 horizontal target-video token reduction.

The coordinator owns DiT block boundaries, composes an existing block
replacement (notably FirstBlockCache or Spectrum) instead of overwriting it,
and keeps the sparse-attention holder's reduced-sequence metadata synchronized.

Token reduction follows h3.c's released policy: after block 3, adjacent target
video tokens are averaged horizontally while the packed text/condition/audio
prefix stays exact. The first ten evaluations restore before block 40; later
evaluations restore before block 30. Expansion retains the original within-pair
detail and adds each reduced token's update relative to its pooled baseline.

The former gate-ranked layer-thinning experiment was removed after real H3
renders showed broken audio. The legacy node signature is retained only so old
workflow JSON can load; its active-layer setting can no longer omit blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Hashable

import torch


TAG = "[AppleSilicon-Optimized/H3Structure]"
STATE_KEY = "asfp8_h3_structure"
WRAPPER_KEY = "asfp8_h3_structure"
BLOCK_COUNT = 50


def _stream_key(transformer_options: dict[str, Any]) -> Hashable:
    uuids = transformer_options.get("uuids")
    if isinstance(uuids, (list, tuple)) and len(uuids) == 1:
        try:
            hash(uuids[0])
            return ("uuid", uuids[0])
        except TypeError:
            return ("uuid-repr", repr(uuids[0]))
    return ("direct",)


def _pool_video(video: torch.Tensor, frames: int, height: int, width: int):
    """Pair horizontal rows with one bf16 rounding, preserving odd tails."""
    hidden = video.shape[-1]
    grid = video.reshape(frames, height, width, hidden)
    paired_width = width // 2
    if paired_width:
        left = grid[:, :, :paired_width * 2:2]
        right = grid[:, :, 1:paired_width * 2:2]
        paired = (left + right) * 0.5
    else:
        paired = grid[:, :, :0]
    if width % 2:
        paired = torch.cat((paired, grid[:, :, -1:]), dim=2)
    return paired.reshape(-1, hidden).contiguous()


def _pool_rope(rope: torch.Tensor, video_start: int, frames: int,
               height: int, width: int):
    """Build midpoint rotations for paired rows from their complex matrices."""
    if (not isinstance(rope, torch.Tensor) or rope.dim() != 6
            or rope.shape[0] != 1 or rope.shape[2] != 1
            or rope.shape[1] < video_start + frames * height * width
            or rope.shape[-2:] != (2, 2)):
        raise ValueError("unexpected H3 RoPE table layout")
    prefix = rope[:, :video_start]
    rows = rope[:, video_start:video_start + frames * height * width]
    rows = rows.reshape(1, frames, height, width, 1, rope.shape[-3], 2, 2)
    paired_width = width // 2
    if paired_width:
        left = rows[:, :, :, :paired_width * 2:2]
        right = rows[:, :, :, 1:paired_width * 2:2]
        # A rotation matrix stores [[cos,-sin],[sin,cos]]. Normalizing the
        # complex sum yields the matrix at the midpoint angle without atan2.
        c = left[..., 0, 0].float() + right[..., 0, 0].float()
        s = left[..., 1, 0].float() + right[..., 1, 0].float()
        norm = (c.square() + s.square()).sqrt().clamp_min(1.0e-12)
        c, s = c / norm, s / norm
        midpoint = torch.stack((c, -s, s, c), dim=-1).reshape(
            1, frames, height, paired_width, 1, rope.shape[-3], 2, 2)
        midpoint = midpoint.to(rope.dtype)
    else:
        midpoint = rows[:, :, :, :0]
    if width % 2:
        midpoint = torch.cat((midpoint, rows[:, :, :, -1:]), dim=3)
    reduced = midpoint.reshape(
        1, frames * height * ((width + 1) // 2), 1,
        rope.shape[-3], 2, 2)
    return torch.cat((prefix, reduced), dim=1).contiguous()


def _reduced_segments(segments, video_start: int, full_rows: int,
                      reduced_rows: int):
    if not isinstance(segments, (list, tuple)) or not segments:
        raise ValueError("missing H3 modulation segments")
    result = []
    cursor = 0
    found_video = False
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) != 3:
            raise ValueError("invalid H3 modulation segment")
        a, b, row = map(int, segment)
        if a != cursor or b < a:
            raise ValueError("H3 modulation segments do not tile the sequence")
        if a == video_start and b == full_rows:
            result.append((a, video_start + reduced_rows, row))
            cursor = full_rows
            found_video = True
        else:
            if b > video_start:
                raise ValueError("target video must be the final modulation segment")
            result.append((a, b, row))
            cursor = b
    if cursor != full_rows or not found_video:
        raise ValueError("could not identify H3's final target-video segment")
    return result


@dataclass
class _ReductionState:
    original: torch.Tensor
    baseline_video: torch.Tensor
    video_start: int
    frames: int
    height: int
    width: int
    full_rope: torch.Tensor
    full_segments: Any
    reduced_rope: torch.Tensor
    reduced_segments: Any
    restore_block: int
    sparse_metadata: tuple[Any, ...] | None = None


@dataclass
class H3StructureHolder:
    reduction_scale: float
    early_evaluations: int
    begin_block: int
    detail_restore_block: int
    coarse_restore_block: int
    verbose: bool = False
    evaluations: int = 0
    current_geometry: tuple[int, int, int, int] | None = None
    current_stream: Hashable | None = None
    reduction: _ReductionState | None = None

    def clone(self):
        return H3StructureHolder(
            self.reduction_scale, self.early_evaluations,
            self.begin_block, self.detail_restore_block,
            self.coarse_restore_block, self.verbose)

    def begin_evaluation(self, video, audio, transformer_options):
        self.evaluations += 1
        self.current_stream = _stream_key(transformer_options)
        self.reduction = None
        self.current_geometry = None
        if (isinstance(video, torch.Tensor) and video.dim() == 5
                and isinstance(audio, torch.Tensor) and audio.dim() >= 3):
            frames = int(video.shape[-3])
            height = (int(video.shape[-2]) + 1) // 2
            width = (int(video.shape[-1]) + 1) // 2
            audio_rows = int(audio.shape[-2] * audio.shape[-1])
            if min(frames, height, width, audio_rows) > 0:
                self.current_geometry = (frames, height, width, audio_rows)

    def end_evaluation(self):
        self.reduction = None
        self.current_geometry = None
        self.current_stream = None

    def _fbc_skip_active(self, transformer_options) -> bool:
        try:
            from . import minimax_h3cache
            cache = transformer_options.get(minimax_h3cache.STATE_KEY)
            if not isinstance(cache, minimax_h3cache.H3CacheHolder):
                return False
            key = minimax_h3cache._stream_key(transformer_options)
            state = cache.first_block_states.get(key)
            return bool(state is not None and state.skip_active)
        except Exception:
            return False

    def enter_reduction(self, args):
        if self.reduction is not None or self.current_geometry is None:
            return args
        transformer_options = args.get("transformer_options", {})
        if self._fbc_skip_active(transformer_options):
            return args
        hidden = args.get("img")
        rope = args.get("rope_freqs")
        segments = args.get("mod_segments")
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 2:
            return args
        frames, height, width, audio_rows = self.current_geometry
        video_rows = frames * height * width
        video_start = hidden.shape[0] - video_rows
        if video_start <= audio_rows or width < 2:
            return args
        video = hidden[video_start:]
        pooled_video = _pool_video(video, frames, height, width)
        reduced_rows = pooled_video.shape[0]
        reduced = torch.empty(
            (video_start + reduced_rows, hidden.shape[1]),
            dtype=hidden.dtype, device=hidden.device)
        reduced[:video_start].copy_(hidden[:video_start])
        reduced[video_start:].copy_(pooled_video)
        reduced_rope = _pool_rope(
            rope, video_start, frames, height, width)
        reduced_segments = _reduced_segments(
            segments, video_start, hidden.shape[0], reduced_rows)
        restore = (self.coarse_restore_block
                   if self.evaluations <= self.early_evaluations
                   else self.detail_restore_block)
        self.reduction = _ReductionState(
            original=hidden.detach().clone(),
            baseline_video=pooled_video.detach().clone(),
            video_start=video_start, frames=frames, height=height, width=width,
            full_rope=rope, full_segments=segments,
            reduced_rope=reduced_rope, reduced_segments=reduced_segments,
            restore_block=restore)

        # Sparse attention derives its sink from target-row counts. Replace the
        # full-grid counts while reduced, then restore them at expansion.
        try:
            from . import minimax_h3_sparse
            sparse = transformer_options.get(minimax_h3_sparse.STATE_KEY)
            if isinstance(sparse, minimax_h3_sparse.H3SparseHolder):
                state = self.reduction
                state.sparse_metadata = (
                    sparse.target_rows, sparse.audio_rows, sparse.rows_per_frame)
                sparse.target_rows = audio_rows + reduced_rows
                sparse.audio_rows = audio_rows
                sparse.rows_per_frame = height * ((width + 1) // 2)
        except Exception:
            pass
        local = dict(args)
        local["img"] = reduced
        local["rope_freqs"] = reduced_rope
        local["mod_segments"] = reduced_segments
        return local

    def continue_reduction(self, args):
        """Reapply reduced static metadata to every block in the reduced span.

        MiniMax's outer block loop reconstructs each replacement's argument
        dictionary from its original full-length RoPE and modulation tables.
        Only the hidden stream is carried from one block to the next, so these
        two entries must be substituted again after the reduction boundary.
        """
        state = self.reduction
        if state is None:
            return args
        hidden = args.get("img")
        if (not isinstance(hidden, torch.Tensor)
                or hidden.shape[0] != state.reduced_rope.shape[1]):
            raise RuntimeError("H3 token reduction lost its reduced sequence layout")
        local = dict(args)
        local["rope_freqs"] = state.reduced_rope
        local["mod_segments"] = state.reduced_segments
        return local

    def leave_reduction(self, args, index):
        state = self.reduction
        if state is None or index != state.restore_block:
            return args
        hidden = args.get("img")
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("H3 token reduction lost its hidden tensor")
        reduced_video = hidden[state.video_start:]
        update = reduced_video - state.baseline_video
        expanded_update = update.reshape(
            state.frames, state.height, (state.width + 1) // 2, -1
        ).repeat_interleave(2, dim=2)[:, :, :state.width].reshape(
            -1, hidden.shape[-1])
        full = torch.empty_like(state.original)
        full[:state.video_start].copy_(hidden[:state.video_start])
        full[state.video_start:].copy_(
            state.original[state.video_start:] + expanded_update * self.reduction_scale)
        local = dict(args)
        local["img"] = full
        local["rope_freqs"] = state.full_rope
        local["mod_segments"] = state.full_segments
        transformer_options = args.get("transformer_options", {})
        if state.sparse_metadata is not None:
            try:
                from . import minimax_h3_sparse
                sparse = transformer_options.get(minimax_h3_sparse.STATE_KEY)
                if isinstance(sparse, minimax_h3_sparse.H3SparseHolder):
                    (sparse.target_rows, sparse.audio_rows,
                     sparse.rows_per_frame) = state.sparse_metadata
            except Exception:
                pass
        self.reduction = None
        return local

    def advance_sparse_for_skipped_block(self, transformer_options):
        try:
            from . import minimax_h3_sparse
            sparse = transformer_options.get(minimax_h3_sparse.STATE_KEY)
            if isinstance(sparse, minimax_h3_sparse.H3SparseHolder):
                sparse.block_index += 1
        except Exception:
            pass


def _find_transformer_options(args, kwargs):
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    for value in reversed(args):
        if isinstance(value, dict) and STATE_KEY in value:
            return value
    return None


def structure_diffusion_wrapper(executor, *args, **kwargs):
    options = _find_transformer_options(args, kwargs)
    holder = options.get(STATE_KEY) if isinstance(options, dict) else None
    if not isinstance(holder, H3StructureHolder):
        return executor(*args, **kwargs)
    inputs = args[0] if args else kwargs.get("x")
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        return executor(*args, **kwargs)
    holder.begin_evaluation(inputs[0], inputs[1], options)
    try:
        return executor(*args, **kwargs)
    finally:
        holder.end_evaluation()


def structure_sample_wrapper(executor, *args, **kwargs):
    import comfy.model_patcher

    guider = executor.class_obj
    original_options = guider.model_options
    active = None
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(
            original_options)
        options = guider.model_options.setdefault("transformer_options", {})
        template = options.get(STATE_KEY)
        if not isinstance(template, H3StructureHolder):
            return executor(*args, **kwargs)
        active = template.clone()
        options[STATE_KEY] = active
        logging.info(
            "%s enabled scale=%.3g early_evaluations=%d restore_blocks=%d/%d",
            TAG, active.reduction_scale, active.early_evaluations,
            active.coarse_restore_block, active.detail_restore_block)
        return executor(*args, **kwargs)
    finally:
        if active is not None:
            logging.info("%s evaluations=%d", TAG, active.evaluations)
            active.end_evaluation()
        guider.model_options = original_options


def _make_block_wrapper(index, block, previous):
    def wrapper(args, extra_options):
        options = args.get("transformer_options")
        holder = options.get(STATE_KEY) if isinstance(options, dict) else None

        def invoke(local_args):
            if previous is not None:
                return previous(local_args, extra_options)
            return extra_options["original_block"](local_args)

        if not isinstance(holder, H3StructureHolder):
            return invoke(args)
        local = holder.leave_reduction(args, index)
        if index == holder.begin_block:
            local = holder.enter_reduction(local)
        elif holder.reduction is not None:
            local = holder.continue_reduction(local)

        fbc_tail_skip = index > 0 and holder._fbc_skip_active(options)
        if fbc_tail_skip:
            # FirstBlockCache already represented this entire tail with its
            # cached residual. Let its wrapper clear bookkeeping at block 49,
            # while advancing sparse attention's logical block counter for the
            # omitted attention call.
            result = invoke(local)
            holder.advance_sparse_for_skipped_block(options)
            return result

        return invoke(local)

    # FirstBlockCache uses this marker to distinguish our deliberately
    # composable coordinator from an unrelated third-party block replacement.
    wrapper._asfp8_h3_structure = True
    return wrapper


def apply_h3_token_reduction(model, reduction_scale, early_evaluations,
                             detail_restore_block, coarse_restore_block,
                             verbose):
    """Attach only the supported H3 target-video token reduction."""
    import comfy.patcher_extension

    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(dm, "blocks", None)
    if blocks is None or len(blocks) != BLOCK_COUNT:
        raise ValueError("H3 structural acceleration requires a 50-block MiniMax H3 model")
    if not 0.0 <= float(reduction_scale) <= 2.0:
        raise ValueError("reduction_scale must be in [0, 2]")
    if int(early_evaluations) < 1:
        raise ValueError("early_evaluations must be at least 1")
    begin = 4
    detail = int(detail_restore_block)
    coarse = int(coarse_restore_block)
    if not begin < detail < coarse < BLOCK_COUNT:
        raise ValueError("restore blocks must satisfy 4 < detail < coarse < 50")
    patched = model.clone()
    options = patched.model_options.setdefault("transformer_options", {})
    if STATE_KEY in options:
        raise ValueError("H3 structural acceleration is already attached")
    options[STATE_KEY] = H3StructureHolder(
        float(reduction_scale), int(early_evaluations), begin, detail,
        coarse, bool(verbose))
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY, structure_sample_wrapper)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY, structure_diffusion_wrapper)

    existing = dict(options.get("patches_replace", {}).get("dit", {}))
    for index, block in enumerate(blocks):
        previous = existing.get(("double_block", index))
        patched.set_model_patch_replace(
            _make_block_wrapper(index, block, previous),
            "dit", "double_block", index)
    return patched


def apply_h3_structure(model, token_reduction, active_layers,
                       reduction_scale, calibration_evaluations,
                       early_evaluations, detail_restore_block,
                       coarse_restore_block, verbose):
    """Compatibility boundary for workflows saved with the former node."""
    requested_active_layers = int(active_layers)
    if not 40 <= requested_active_layers <= BLOCK_COUNT:
        raise ValueError("active_layers must be in [40, 50]")
    if int(calibration_evaluations) < 1:
        raise ValueError("calibration_evaluations must be at least 1")
    if not token_reduction:
        if requested_active_layers < BLOCK_COUNT:
            raise ValueError(
                "H3 layer thinning was removed because it damages audio; "
                "enable token reduction instead")
        raise ValueError("enable token reduction")
    if requested_active_layers < BLOCK_COUNT:
        logging.warning(
            "%s active_layers=%d ignored: H3 layer thinning was removed "
            "because it damages audio", TAG, requested_active_layers)
    return apply_h3_token_reduction(
        model, reduction_scale, early_evaluations, detail_restore_block,
        coarse_restore_block, verbose)


class AppleSiliconH3StructuralAcceleration:
    # Kept registered so old graphs load. The unified H3 node exposes only the
    # validated token-reduction mechanism.
    DEPRECATED = True
    CATEGORY = "Apple Silicon/approximate"
    DESCRIPTION = (
        "Legacy loader for h3.c-style horizontal target-video token reduction. "
        "Layer thinning has been removed because real H3 renders damaged "
        "audio; saved active_layers values below 50 are ignored when token "
        "reduction is enabled. Use H3 Apple Silicon Acceleration for new "
        "workflows."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "token_reduction": ("BOOLEAN", {"default": True}),
            "active_layers": ("INT", {
                "default": 50, "min": 40, "max": 50, "step": 1}),
            "reduction_scale": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "calibration_evaluations": ("INT", {
                "default": 1, "min": 1, "max": 4, "step": 1}),
            "early_evaluations": ("INT", {
                "default": 10, "min": 1, "max": 20, "step": 1}),
            "detail_restore_block": ("INT", {
                "default": 30, "min": 8, "max": 44, "step": 1}),
            "coarse_restore_block": ("INT", {
                "default": 40, "min": 12, "max": 48, "step": 1}),
            "verbose": ("BOOLEAN", {"default": False}),
        }}

    def apply(self, model, token_reduction, active_layers, reduction_scale,
              calibration_evaluations, early_evaluations,
              detail_restore_block, coarse_restore_block, verbose):
        return (apply_h3_structure(
            model, token_reduction, active_layers, reduction_scale,
            calibration_evaluations, early_evaluations,
            detail_restore_block, coarse_restore_block, verbose),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3StructuralAcceleration":
        AppleSiliconH3StructuralAcceleration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3StructuralAcceleration":
        "H3 Structural Acceleration (Apple Silicon)",
}
