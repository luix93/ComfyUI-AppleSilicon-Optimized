"""Audio/video-aware full-model cache for MiniMax H3.

This is an explicitly approximate, opt-in ComfyUI model patch.  Exact model
evaluations learn how changes in H3's video and audio inputs map to changes in
their respective outputs. A full-model cache hit either predicts the next raw
video/audio velocity or the sampler-facing denoised estimate and avoids all
transformer blocks. The default predictor holds H3's previous network velocity
while analytically correcting the audio stream for its shifted sigma schedule;
simpler predictors are retained for controlled comparison.

Unlike generic EasyCache, the decision is made from separately normalized
video and audio estimates, either modality can veto reuse, and consecutive
hits are bounded.  Threshold zero is an exact-compute calibration mode.  State
is cloned for one OUTER_SAMPLE invocation and cleared in ``finally``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Hashable

import torch


TAG = "[AppleSilicon-Optimized/H3Cache]"
STATE_KEY = "asfp8_h3cache"
WRAPPER_KEY = "asfp8_h3cache"
DENOISED_PREDICTION_MODES = {"denoised_hold", "denoised_linear"}
RAW_PREDICTION_MODES = {
    "secant_velocity", "velocity_hold", "output_hold", "residual_carry",
}
ORACLE_CANDIDATE_MODES = RAW_PREDICTION_MODES | {
    "taylor_step_1", "taylor_step_2", "taylor_sigma_1",
    "taylor_sigma_2",
}
ORACLE_DIAGNOSTIC_MODES = ORACLE_CANDIDATE_MODES | {
    "projection_ceiling_1", "projection_ceiling_2",
}
ORACLE_PREDICTION_MODE = "oracle"
BLOCK_ORACLE_PREDICTION_MODE = "block_oracle"
BLOCK_TAYLOR_PREDICTION_MODE = "block_taylor"
FIRST_BLOCK_CACHE_MODE = "first_block_cache"
FIRST_BLOCK_CACHE_PER_STEP_MODE = "first_block_cache_per_step"
FIRST_BLOCK_CACHE_MODES = {
    FIRST_BLOCK_CACHE_MODE,
    FIRST_BLOCK_CACHE_PER_STEP_MODE,
}
# Modes that own transformer blocks directly and leave run_forward exact.
BLOCK_LEVEL_MODES = {
    BLOCK_ORACLE_PREDICTION_MODE,
    BLOCK_TAYLOR_PREDICTION_MODE,
    *FIRST_BLOCK_CACHE_MODES,
}


def _as_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().flatten()[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stream_key(transformer_options: dict[str, Any]) -> Hashable | None:
    """Return the conditioning-stream key, or None when batching is unsafe."""
    uuids = transformer_options.get("uuids")
    if uuids is None:
        return ("direct",)
    if not isinstance(uuids, (list, tuple)) or len(uuids) != 1:
        return None
    try:
        hash(uuids[0])
        return ("uuid", uuids[0])
    except TypeError:
        return ("uuid-repr", repr(uuids[0]))


def _subsample_video(x: torch.Tensor, factor: int) -> torch.Tensor:
    if x.dim() != 5:
        raise ValueError("H3 video tensor must be BCTHW")
    return x[..., ::factor, ::factor].detach().contiguous()


def _subsample_audio(x: torch.Tensor, factor: int) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError("H3 audio tensor must include a time dimension")
    return x[..., ::factor].detach().contiguous()


def _relative_change(current: torch.Tensor, previous: torch.Tensor) -> float:
    if current.shape != previous.shape:
        raise ValueError("sample shape changed")
    current = current.float()
    previous = previous.float()
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean().clamp_min(1.0e-6)
    return float((numerator / denominator).detach().item())


def _least_squares_scale(delta_input: torch.Tensor, delta_output: torch.Tensor):
    """Best scalar mapping delta_input to delta_output in least squares."""
    if delta_input.shape != delta_output.shape:
        raise ValueError("secant tensors must have matching shapes")
    x = delta_input.float()
    y = delta_output.float()
    denominator = x.square().sum()
    if float(denominator.detach().item()) <= 1.0e-12:
        return None
    numerator = (x * y).sum()
    return float((numerator / denominator).detach().item())


def _prediction_metrics(
    prediction: torch.Tensor, actual: torch.Tensor
) -> tuple[float, float]:
    """Return normalized RMSE and cosine similarity without flattening copies."""
    if prediction.shape != actual.shape:
        raise ValueError("prediction and actual output shapes must match")
    predicted = prediction.float()
    reference = actual.float()
    delta = predicted - reference
    rmse = delta.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-8)
    denominator = (
        predicted.square().sum().sqrt()
        * reference.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    cosine = (predicted * reference).sum() / denominator
    return (
        float((rmse / reference_rms).detach().item()),
        float(cosine.detach().item()),
    )


def _projection_ceiling(
    base: torch.Tensor,
    actual: torch.Tensor,
    directions: list[torch.Tensor],
) -> tuple[torch.Tensor, tuple[float, ...]] | None:
    """Hindsight least-squares projection used only as a predictor lower bound."""
    if not directions or len(directions) > 2:
        raise ValueError("projection ceiling supports one or two directions")
    if any(direction.shape != base.shape for direction in directions):
        raise ValueError("projection directions must match the base shape")
    target = actual.float() - base.float()
    vectors = [direction.float() for direction in directions]
    if len(vectors) == 1:
        denominator = float(vectors[0].square().sum().detach().item())
        if denominator <= 1.0e-12:
            return None
        coefficient = float(
            (vectors[0] * target).sum().detach().item()) / denominator
        return base + directions[0] * coefficient, (coefficient,)

    gram00 = float(vectors[0].square().sum().detach().item())
    gram11 = float(vectors[1].square().sum().detach().item())
    gram01 = float((vectors[0] * vectors[1]).sum().detach().item())
    rhs0 = float((vectors[0] * target).sum().detach().item())
    rhs1 = float((vectors[1] * target).sum().detach().item())
    ridge = max(1.0e-12, 1.0e-7 * (gram00 + gram11))
    gram00 += ridge
    gram11 += ridge
    determinant = gram00 * gram11 - gram01 * gram01
    if abs(determinant) <= 1.0e-20:
        return None
    coefficient0 = (rhs0 * gram11 - rhs1 * gram01) / determinant
    coefficient1 = (rhs1 * gram00 - rhs0 * gram01) / determinant
    prediction = (
        base + directions[0] * coefficient0
        + directions[1] * coefficient1
    )
    return prediction, (coefficient0, coefficient1)


def _time_shift_slope(
    sigma: float, from_shift: float = 12.0, to_shift: float = 3.0
) -> float:
    """Scalar form of MiniMax H3's d(sigma_audio)/d(sigma_video)."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base) ** 2
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    return numerator / denominator


def _extract_h3_pair(value: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not isinstance(value[0], torch.Tensor)
            or not isinstance(value[1], torch.Tensor)):
        return None
    return value[0], value[1]


def _sample_block_targets(
    hidden: torch.Tensor,
    audio_rows: int,
    video_rows: int,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy bounded audio/video samples from H3's packed target-row tail."""
    if hidden.dim() < 2 or factor < 1:
        raise ValueError("invalid H3 block sample")
    video_start = hidden.shape[0] - video_rows
    audio_start = video_start - audio_rows
    if audio_start < 0 or audio_rows <= 0 or video_rows <= 0:
        raise ValueError("invalid H3 block target layout")
    audio = hidden[audio_start:video_start:factor, ..., ::factor]
    video = hidden[video_start::factor, ..., ::factor]
    return (
        video.detach().clone().contiguous(),
        audio.detach().clone().contiguous(),
    )


def _extract_packed_h3_pair(
    value: Any, latent_shapes: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """View ComfyUI's packed sampler tensor as H3 video and audio tensors."""
    if (not isinstance(value, torch.Tensor)
            or not isinstance(latent_shapes, (list, tuple))
            or len(latent_shapes) != 2):
        return None
    try:
        video_shape = tuple(int(x) for x in latent_shapes[0])
        audio_shape = tuple(int(x) for x in latent_shapes[1])
    except (TypeError, ValueError):
        return None
    if (len(video_shape) != 5 or len(audio_shape) < 3
            or video_shape[0] != value.shape[0]
            or audio_shape[0] != value.shape[0]):
        return None
    video_elements = math.prod(video_shape[1:])
    audio_elements = math.prod(audio_shape[1:])
    flattened = value.reshape(value.shape[0], -1)
    if flattened.shape[1] != video_elements + audio_elements:
        return None
    video = flattened[:, :video_elements].reshape(video_shape)
    audio = flattened[:, video_elements:].reshape(audio_shape)
    return video, audio


def _find_transformer_options(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    candidate = kwargs.get("transformer_options")
    if isinstance(candidate, dict):
        return candidate
    for candidate in reversed(args):
        if isinstance(candidate, dict) and STATE_KEY in candidate:
            return candidate
    return None


@dataclass
class _OracleStats:
    count: int = 0
    video_nrmse_sum: float = 0.0
    audio_nrmse_sum: float = 0.0
    video_nrmse_min: float = float("inf")
    audio_nrmse_min: float = float("inf")
    video_nrmse_max: float = 0.0
    audio_nrmse_max: float = 0.0
    video_cosine_min: float = 1.0
    audio_cosine_min: float = 1.0

    def update(
        self,
        video_nrmse: float,
        video_cosine: float,
        audio_nrmse: float,
        audio_cosine: float,
    ) -> None:
        self.count += 1
        self.video_nrmse_sum += video_nrmse
        self.audio_nrmse_sum += audio_nrmse
        self.video_nrmse_min = min(self.video_nrmse_min, video_nrmse)
        self.audio_nrmse_min = min(self.audio_nrmse_min, audio_nrmse)
        self.video_nrmse_max = max(self.video_nrmse_max, video_nrmse)
        self.audio_nrmse_max = max(self.audio_nrmse_max, audio_nrmse)
        self.video_cosine_min = min(self.video_cosine_min, video_cosine)
        self.audio_cosine_min = min(self.audio_cosine_min, audio_cosine)


@dataclass
class _BlockOracleState:
    metadata: tuple[Any, ...]
    evaluations: int = 0
    video_residual_history: list[torch.Tensor] = field(default_factory=list)
    audio_residual_history: list[torch.Tensor] = field(default_factory=list)
    stats: dict[str, _OracleStats] = field(default_factory=dict)


@dataclass
class _BlockTaylorState:
    metadata: tuple[Any, ...]
    evaluations: int = 0
    refreshes: int = 0
    skips: int = 0
    consecutive_skips: int = 0
    residual_history: list[torch.Tensor] = field(default_factory=list)
    evaluation_history: list[int] = field(default_factory=list)


@dataclass
class _FirstBlockCacheState:
    """State for one stream's FirstBlockCache.

    Block 0 runs exactly on every evaluation and its residual is the change
    detector; the aggregate residual of blocks 1..N-1 is what gets reused. The
    head clone only lives while an exact tail is executing.
    """

    metadata: tuple[Any, ...]
    previous_video: torch.Tensor | None = None
    previous_audio: torch.Tensor | None = None
    cached_tail: torch.Tensor | None = None
    saved_head: torch.Tensor | None = None
    cumulative_change: float = 0.0
    consecutive_skips: int = 0
    evaluations: int = 0
    refreshes: int = 0
    skips: int = 0
    skip_active: bool = False
    peak_bytes: int = 0

    def clear_tensors(self) -> None:
        self.previous_video = None
        self.previous_audio = None
        self.cached_tail = None
        self.saved_head = None
        self.cumulative_change = 0.0
        self.consecutive_skips = 0
        self.skip_active = False


@dataclass
class _StreamState:
    metadata: tuple[Any, ...] | None = None
    last_input_video: torch.Tensor | None = None
    last_input_audio: torch.Tensor | None = None
    refresh_input_video: torch.Tensor | None = None
    refresh_input_audio: torch.Tensor | None = None
    refresh_output_video: torch.Tensor | None = None
    refresh_output_audio: torch.Tensor | None = None
    cached_video_output: torch.Tensor | None = None
    cached_audio_output: torch.Tensor | None = None
    cached_sigma: float | None = None
    cached_video_residual: torch.Tensor | None = None
    cached_audio_residual: torch.Tensor | None = None
    cached_denoised: torch.Tensor | None = None
    denoised_slope: torch.Tensor | None = None
    video_transform_rate: float | None = None
    audio_transform_rate: float | None = None
    video_secant_scale: float | None = None
    audio_secant_scale: float | None = None
    cumulative_change: float = 0.0
    consecutive_skips: int = 0
    evaluations: int = 0
    refreshes: int = 0
    skips: int = 0
    last_input_video_change: float | None = None
    last_input_audio_change: float | None = None
    last_estimated_video_change: float | None = None
    last_estimated_audio_change: float | None = None
    last_actual_video_change: float | None = None
    last_actual_audio_change: float | None = None
    suffix_last_probe_video: torch.Tensor | None = None
    suffix_last_probe_audio: torch.Tensor | None = None
    suffix_current_probe: torch.Tensor | None = None
    suffix_cached_delta: torch.Tensor | None = None
    suffix_skip_active: bool = False
    suffix_last_video_change: float | None = None
    suffix_last_audio_change: float | None = None
    oracle_stats: dict[str, _OracleStats] = field(default_factory=dict)
    oracle_video_history: list[torch.Tensor] = field(default_factory=list)
    oracle_audio_history: list[torch.Tensor] = field(default_factory=list)
    oracle_sigma_history: list[float] = field(default_factory=list)

    def clear_tensors(self) -> None:
        self.last_input_video = None
        self.last_input_audio = None
        self.refresh_input_video = None
        self.refresh_input_audio = None
        self.refresh_output_video = None
        self.refresh_output_audio = None
        self.cached_video_output = None
        self.cached_audio_output = None
        self.cached_sigma = None
        self.cached_video_residual = None
        self.cached_audio_residual = None
        self.cached_denoised = None
        self.denoised_slope = None
        self.video_transform_rate = None
        self.audio_transform_rate = None
        self.video_secant_scale = None
        self.audio_secant_scale = None
        self.suffix_last_probe_video = None
        self.suffix_last_probe_audio = None
        self.suffix_current_probe = None
        self.suffix_cached_delta = None
        self.suffix_skip_active = False
        self.suffix_last_video_change = None
        self.suffix_last_audio_change = None
        self.oracle_stats.clear()
        self.oracle_video_history.clear()
        self.oracle_audio_history.clear()
        self.oracle_sigma_history.clear()
        self.cumulative_change = 0.0
        self.consecutive_skips = 0


class H3CacheHolder:
    """Per-sampling-run configuration and H3 cache state."""

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int = 1,
        video_subsample: int = 8,
        audio_subsample: int = 8,
        audio_weight: float = 1.0,
        cached_suffix_blocks: int = 0,
        prediction_mode: str = "velocity_hold",
        verbose: bool = False,
        block_oracle_indices: tuple[int, ...] = (),
        block_taylor_indices: tuple[int, ...] = (),
    ):
        if reuse_threshold < 0:
            raise ValueError("reuse_threshold must be non-negative")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError("start/end percent must satisfy 0 <= start <= end <= 1")
        if max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be at least 1")
        if video_subsample < 1 or audio_subsample < 1:
            raise ValueError("subsample factors must be positive")
        if audio_weight <= 0:
            raise ValueError("audio_weight must be positive")
        if cached_suffix_blocks < 0:
            raise ValueError("cached_suffix_blocks must be non-negative")
        if prediction_mode not in {
            *RAW_PREDICTION_MODES, *DENOISED_PREDICTION_MODES,
            ORACLE_PREDICTION_MODE, *BLOCK_LEVEL_MODES,
        }:
            raise ValueError(
                "prediction_mode must be secant_velocity, velocity_hold, "
                "output_hold, residual_carry, denoised_hold, denoised_linear, "
                "first_block_cache, or first_block_cache_per_step; oracle "
                "and block_oracle are exact-compute diagnostics; block_taylor "
                "is an approximate early-block predictor")
        self.reuse_threshold = float(reuse_threshold)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.video_subsample = int(video_subsample)
        self.audio_subsample = int(audio_subsample)
        self.audio_weight = float(audio_weight)
        self.cached_suffix_blocks = int(cached_suffix_blocks)
        self.prediction_mode = prediction_mode
        self.verbose = bool(verbose)
        self.block_oracle_indices = tuple(int(i) for i in block_oracle_indices)
        self.block_taylor_indices = tuple(int(i) for i in block_taylor_indices)
        self.start_sigma: float | None = None
        self.end_sigma: float | None = None
        self.streams: dict[Hashable, _StreamState] = {}
        self.suffix_layouts: dict[Hashable, tuple[int, int]] = {}
        self.block_oracle_states: dict[
            tuple[Hashable, int], _BlockOracleState
        ] = {}
        self.block_taylor_states: dict[
            tuple[Hashable, int], _BlockTaylorState
        ] = {}
        self.first_block_states: dict[Hashable, _FirstBlockCacheState] = {}

    def clone(self) -> "H3CacheHolder":
        return H3CacheHolder(
            self.reuse_threshold,
            self.start_percent,
            self.end_percent,
            self.max_consecutive_skips,
            self.video_subsample,
            self.audio_subsample,
            self.audio_weight,
            self.cached_suffix_blocks,
            self.prediction_mode,
            self.verbose,
            self.block_oracle_indices,
            self.block_taylor_indices,
        )

    def prepare_timesteps(self, model_sampling: Any) -> "H3CacheHolder":
        self.start_sigma = float(model_sampling.percent_to_sigma(self.start_percent))
        self.end_sigma = float(model_sampling.percent_to_sigma(self.end_percent))
        return self

    def set_sigma_window(self, start_sigma: float, end_sigma: float) -> None:
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)

    def in_cache_window(self, transformer_options: dict[str, Any]) -> bool:
        if self.start_sigma is None or self.end_sigma is None:
            return False
        sigma = _as_float(transformer_options.get("sigmas"))
        if sigma is None:
            return False
        upper = max(self.start_sigma, self.end_sigma)
        lower = min(self.start_sigma, self.end_sigma)
        return lower < sigma <= upper

    def reset(self) -> None:
        for state in self.streams.values():
            state.clear_tensors()
        self.streams.clear()
        self.suffix_layouts.clear()
        self.block_oracle_states.clear()
        self.block_taylor_states.clear()
        for first_state in self.first_block_states.values():
            first_state.clear_tensors()
        self.first_block_states.clear()

    def first_block_totals(self) -> tuple[int, int, int, int]:
        return (
            sum(s.evaluations for s in self.first_block_states.values()),
            sum(s.refreshes for s in self.first_block_states.values()),
            sum(s.skips for s in self.first_block_states.values()),
            max((s.peak_bytes for s in self.first_block_states.values()), default=0),
        )

    def totals(self) -> tuple[int, int, int]:
        evaluations = sum(s.evaluations for s in self.streams.values())
        refreshes = sum(s.refreshes for s in self.streams.values())
        skips = sum(s.skips for s in self.streams.values())
        return evaluations, refreshes, skips

    def oracle_summaries(self):
        for stream_key, state in self.streams.items():
            for predictor, stats in sorted(state.oracle_stats.items()):
                yield stream_key, predictor, stats

    def block_oracle_summaries(self):
        for (stream_key, block_index), state in sorted(
            self.block_oracle_states.items(), key=lambda item: item[0][1]
        ):
            for predictor, stats in sorted(state.stats.items()):
                yield stream_key, block_index, predictor, stats

    def block_taylor_totals(self) -> tuple[int, int, int]:
        return (
            sum(state.evaluations for state in self.block_taylor_states.values()),
            sum(state.refreshes for state in self.block_taylor_states.values()),
            sum(state.skips for state in self.block_taylor_states.values()),
        )

    def _samples(
        self, video: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _subsample_video(video, self.video_subsample),
            _subsample_audio(audio, self.audio_subsample),
        )

    def _metadata(
        self, video: torch.Tensor, audio: torch.Tensor
    ) -> tuple[Any, ...]:
        return (
            tuple(video.shape), video.dtype, video.device.type, video.device.index,
            tuple(audio.shape), audio.dtype, audio.device.type, audio.device.index,
        )

    @staticmethod
    def _residual_valid(
        residual: torch.Tensor | None, current: torch.Tensor
    ) -> bool:
        return (
            residual is not None
            and residual.shape == current.shape
            and residual.dtype == current.dtype
            and residual.device == current.device
        )

    def _prediction_valid(
        self, state: _StreamState, video: torch.Tensor, audio: torch.Tensor
    ) -> bool:
        if self.prediction_mode in {
            "secant_velocity", "velocity_hold", "output_hold"
        }:
            return (
                self._residual_valid(state.cached_video_output, video)
                and self._residual_valid(state.cached_audio_output, audio)
                and (self.prediction_mode == "output_hold"
                     or state.cached_sigma is not None)
                and (self.prediction_mode != "secant_velocity" or (
                    state.video_secant_scale is not None
                    and state.audio_secant_scale is not None
                    and self._residual_valid(
                        state.cached_video_residual, video)
                    and self._residual_valid(
                        state.cached_audio_residual, audio)
                ))
            )
        return (
            self._residual_valid(state.cached_video_residual, video)
            and self._residual_valid(state.cached_audio_residual, audio)
        )

    def _raw_predictions(
        self,
        state: _StreamState,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        transformer_options: dict[str, Any],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Construct every currently valid raw-output prediction."""
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        outputs_valid = (
            self._residual_valid(state.cached_video_output, video_input)
            and self._residual_valid(state.cached_audio_output, audio_input)
        )
        residuals_valid = (
            self._residual_valid(state.cached_video_residual, video_input)
            and self._residual_valid(state.cached_audio_residual, audio_input)
        )
        if outputs_valid:
            predictions["output_hold"] = (
                state.cached_video_output, state.cached_audio_output)
        if residuals_valid:
            predictions["residual_carry"] = (
                video_input + state.cached_video_residual,
                audio_input + state.cached_audio_residual,
            )

        sigma = _as_float(transformer_options.get("sigmas"))
        if outputs_valid and sigma is not None and state.cached_sigma is not None:
            shift_v = float(transformer_options.get(
                "minimax_h3_sigma_shift_video", 12.0))
            shift_a = float(transformer_options.get(
                "minimax_h3_sigma_shift_audio", 3.0))
            slope_now = _time_shift_slope(sigma, shift_v, shift_a)
            slope_cached = _time_shift_slope(
                state.cached_sigma, shift_v, shift_a)
            predictions["velocity_hold"] = (
                state.cached_video_output,
                state.cached_audio_output * (slope_now / slope_cached),
            )
            if (residuals_valid
                    and state.video_secant_scale is not None
                    and state.audio_secant_scale is not None):
                video_delta = (
                    video_input + state.cached_video_residual
                    - state.cached_video_output
                )
                audio_delta = (
                    audio_input + state.cached_audio_residual
                    - state.cached_audio_output
                )
                predictions["secant_velocity"] = (
                    state.cached_video_output
                    + state.video_secant_scale * video_delta,
                    (
                        state.cached_audio_output / slope_cached
                        + state.audio_secant_scale * audio_delta
                    ) * slope_now,
                )

        video_history = state.oracle_video_history
        audio_history = state.oracle_audio_history
        sigma_history = state.oracle_sigma_history
        if len(video_history) >= 2 and len(audio_history) >= 2:
            video_delta = video_history[-1] - video_history[-2]
            audio_delta = audio_history[-1] - audio_history[-2]
            predictions["taylor_step_1"] = (
                video_history[-1] + video_delta,
                audio_history[-1] + audio_delta,
            )
            if (sigma is not None and len(sigma_history) >= 2
                    and abs(sigma_history[-1] - sigma_history[-2]) > 1.0e-12):
                ratio = (
                    (sigma - sigma_history[-1])
                    / (sigma_history[-1] - sigma_history[-2])
                )
                predictions["taylor_sigma_1"] = (
                    video_history[-1] + ratio * video_delta,
                    audio_history[-1] + ratio * audio_delta,
                )

        if len(video_history) >= 3 and len(audio_history) >= 3:
            video_d1 = video_history[-1] - video_history[-2]
            audio_d1 = audio_history[-1] - audio_history[-2]
            video_d2 = video_d1 - (
                video_history[-2] - video_history[-3])
            audio_d2 = audio_d1 - (
                audio_history[-2] - audio_history[-3])
            predictions["taylor_step_2"] = (
                video_history[-1] + video_d1 + 0.5 * video_d2,
                audio_history[-1] + audio_d1 + 0.5 * audio_d2,
            )
            if sigma is not None and len(sigma_history) >= 3:
                x0, x1, x2 = sigma_history[-3:]
                denominators = (
                    (x0 - x1) * (x0 - x2),
                    (x1 - x0) * (x1 - x2),
                    (x2 - x0) * (x2 - x1),
                )
                if all(abs(value) > 1.0e-12 for value in denominators):
                    weights = (
                        ((sigma - x1) * (sigma - x2)) / denominators[0],
                        ((sigma - x0) * (sigma - x2)) / denominators[1],
                        ((sigma - x0) * (sigma - x1)) / denominators[2],
                    )
                    predictions["taylor_sigma_2"] = (
                        sum(weight * value for weight, value in zip(
                            weights, video_history[-3:])),
                        sum(weight * value for weight, value in zip(
                            weights, audio_history[-3:])),
                    )
        return predictions

    @staticmethod
    def _archive_oracle_output(
        state: _StreamState,
        sigma: float | None,
        video_output: torch.Tensor,
        audio_output: torch.Tensor,
    ) -> None:
        if sigma is None:
            return
        state.oracle_video_history.append(video_output.detach())
        state.oracle_audio_history.append(audio_output.detach())
        state.oracle_sigma_history.append(float(sigma))
        del state.oracle_video_history[:-3]
        del state.oracle_audio_history[:-3]
        del state.oracle_sigma_history[:-3]

    def _record_oracle_predictions(
        self,
        state: _StreamState,
        transformer_options: dict[str, Any],
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
        video_output: torch.Tensor,
        audio_output: torch.Tensor,
    ) -> None:
        sigma = _as_float(transformer_options.get("sigmas"))
        for predictor, (video_prediction, audio_prediction) in predictions.items():
            try:
                video_nrmse, video_cosine = _prediction_metrics(
                    video_prediction, video_output)
                audio_nrmse, audio_cosine = _prediction_metrics(
                    audio_prediction, audio_output)
            except (RuntimeError, ValueError):
                continue
            state.oracle_stats.setdefault(predictor, _OracleStats()).update(
                video_nrmse, video_cosine, audio_nrmse, audio_cosine)
            logging.info(
                "%s oracle eval=%d sigma=%s predictor=%s "
                "video_nrmse=%.7g video_cosine=%.7g "
                "audio_nrmse=%.7g audio_cosine=%.7g",
                TAG, state.evaluations,
                "?" if sigma is None else f"{sigma:.7g}", predictor,
                video_nrmse, video_cosine, audio_nrmse, audio_cosine,
            )

        video_history = state.oracle_video_history
        audio_history = state.oracle_audio_history
        if len(video_history) < 2 or len(audio_history) < 2:
            return
        video_directions = [video_history[-1] - video_history[-2]]
        audio_directions = [audio_history[-1] - audio_history[-2]]
        projection_orders = [("projection_ceiling_1", 1)]
        if len(video_history) >= 3 and len(audio_history) >= 3:
            video_directions.append(video_history[-2] - video_history[-3])
            audio_directions.append(audio_history[-2] - audio_history[-3])
            projection_orders.append(("projection_ceiling_2", 2))

        for predictor, order in projection_orders:
            try:
                video_projection = _projection_ceiling(
                    video_history[-1], video_output,
                    video_directions[:order])
                audio_projection = _projection_ceiling(
                    audio_history[-1], audio_output,
                    audio_directions[:order])
                if video_projection is None or audio_projection is None:
                    continue
                video_prediction, video_coefficients = video_projection
                audio_prediction, audio_coefficients = audio_projection
                video_nrmse, video_cosine = _prediction_metrics(
                    video_prediction, video_output)
                audio_nrmse, audio_cosine = _prediction_metrics(
                    audio_prediction, audio_output)
            except (RuntimeError, ValueError):
                continue
            state.oracle_stats.setdefault(predictor, _OracleStats()).update(
                video_nrmse, video_cosine, audio_nrmse, audio_cosine)
            logging.info(
                "%s oracle eval=%d sigma=%s predictor=%s "
                "video_nrmse=%.7g video_cosine=%.7g video_coefficients=%s "
                "audio_nrmse=%.7g audio_cosine=%.7g audio_coefficients=%s",
                TAG, state.evaluations,
                "?" if sigma is None else f"{sigma:.7g}", predictor,
                video_nrmse, video_cosine,
                ",".join(f"{value:.7g}" for value in video_coefficients),
                audio_nrmse, audio_cosine,
                ",".join(f"{value:.7g}" for value in audio_coefficients),
            )

    def _log_decision(
        self,
        state: _StreamState,
        transformer_options: dict[str, Any],
        action: str,
        reason: str,
    ) -> None:
        if not self.verbose:
            return
        sigma = _as_float(transformer_options.get("sigmas"))
        logging.info(
            "%s eval=%d sigma=%s input_video=%s input_audio=%s "
            "estimate_video=%s estimate_audio=%s cumulative=%.6g "
            "threshold=%.6g action=%s reason=%s",
            TAG,
            state.evaluations,
            "?" if sigma is None else f"{sigma:.7g}",
            "?" if state.last_input_video_change is None else f"{state.last_input_video_change:.6g}",
            "?" if state.last_input_audio_change is None else f"{state.last_input_audio_change:.6g}",
            "?" if state.last_estimated_video_change is None else f"{state.last_estimated_video_change:.6g}",
            "?" if state.last_estimated_audio_change is None else f"{state.last_estimated_audio_change:.6g}",
            state.cumulative_change,
            self.reuse_threshold,
            action,
            reason,
        )

    def run_forward(
        self,
        executor: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        transformer_options: dict[str, Any],
    ):
        if self.prediction_mode in DENOISED_PREDICTION_MODES:
            return executor(*args, **kwargs)
        if self.cached_suffix_blocks or self.prediction_mode in BLOCK_LEVEL_MODES:
            inputs = _extract_h3_pair(args[0] if args else None)
            key = _stream_key(transformer_options)
            if inputs is not None and key is not None:
                video_input, audio_input = inputs
                if video_input.dim() == 5 and audio_input.dim() >= 4:
                    patch_h = (video_input.shape[-2] + 1) // 2
                    patch_w = (video_input.shape[-1] + 1) // 2
                    video_rows = (
                        video_input.shape[0] * video_input.shape[-3]
                        * patch_h * patch_w
                    )
                    audio_rows = audio_input.shape[-2] * audio_input.shape[-1]
                    self.suffix_layouts[key] = (audio_rows, video_rows)
            return executor(*args, **kwargs)
        inputs = _extract_h3_pair(args[0] if args else None)
        key = _stream_key(transformer_options)
        if inputs is None or key is None:
            return executor(*args, **kwargs)
        video_input, audio_input = inputs
        try:
            current_video, current_audio = self._samples(video_input, audio_input)
        except ValueError:
            return executor(*args, **kwargs)

        metadata = self._metadata(video_input, audio_input)
        state = self.streams.setdefault(key, _StreamState(metadata=metadata))
        if state.metadata != metadata:
            state.clear_tensors()
            state = _StreamState(metadata=metadata)
            self.streams[key] = state

        state.evaluations += 1
        input_video_change = input_audio_change = None
        if state.last_input_video is not None and state.last_input_audio is not None:
            try:
                input_video_change = _relative_change(
                    current_video, state.last_input_video)
                input_audio_change = _relative_change(
                    current_audio, state.last_input_audio)
                state.last_input_video_change = input_video_change
                state.last_input_audio_change = input_audio_change
            except (RuntimeError, ValueError):
                state.clear_tensors()

        estimated_video = estimated_audio = None
        if (input_video_change is not None and input_audio_change is not None
                and state.video_transform_rate is not None
                and state.audio_transform_rate is not None):
            estimated_video = state.video_transform_rate * input_video_change
            estimated_audio = state.audio_transform_rate * input_audio_change
            state.last_estimated_video_change = estimated_video
            state.last_estimated_audio_change = estimated_audio

        combined_estimate = (
            max(estimated_video, self.audio_weight * estimated_audio)
            if estimated_video is not None and estimated_audio is not None
            else None
        )
        candidate_cumulative = (
            state.cumulative_change + combined_estimate
            if combined_estimate is not None else float("inf")
        )
        can_reuse = (
            combined_estimate is not None
            and self.prediction_mode != ORACLE_PREDICTION_MODE
            and self.reuse_threshold > 0.0
            and self.in_cache_window(transformer_options)
            and self._prediction_valid(state, video_input, audio_input)
            and state.consecutive_skips < self.max_consecutive_skips
            and candidate_cumulative < self.reuse_threshold
        )

        if can_reuse:
            state.cumulative_change = candidate_cumulative
            state.consecutive_skips += 1
            state.skips += 1
            state.last_input_video = current_video
            state.last_input_audio = current_audio
            self._log_decision(
                state, transformer_options, "skip", "below_threshold")
            if self.prediction_mode in {
                "secant_velocity", "velocity_hold", "output_hold"
            }:
                audio_output = state.cached_audio_output.clone()
                if self.prediction_mode != "output_hold":
                    sigma = _as_float(transformer_options.get("sigmas"))
                    shift_v = float(transformer_options.get(
                        "minimax_h3_sigma_shift_video", 12.0))
                    shift_a = float(transformer_options.get(
                        "minimax_h3_sigma_shift_audio", 3.0))
                    slope_now = _time_shift_slope(sigma, shift_v, shift_a)
                    slope_cached = _time_shift_slope(
                        state.cached_sigma, shift_v, shift_a)
                    if self.prediction_mode == "velocity_hold":
                        audio_output.mul_(slope_now / slope_cached)
                        return [state.cached_video_output.clone(), audio_output]
                    video_delta = (
                        video_input + state.cached_video_residual
                        - state.cached_video_output
                    )
                    audio_delta = (
                        audio_input + state.cached_audio_residual
                        - state.cached_audio_output
                    )
                    video_output = state.cached_video_output + (
                        state.video_secant_scale * video_delta)
                    audio_output.div_(slope_cached).add_(
                        audio_delta, alpha=state.audio_secant_scale
                    ).mul_(slope_now)
                    return [video_output, audio_output]
                return [state.cached_video_output.clone(), audio_output]
            return [video_input + state.cached_video_residual,
                    audio_input + state.cached_audio_residual]

        reason = (
            "oracle_exact"
            if self.prediction_mode == ORACLE_PREDICTION_MODE else "cold"
        )
        if (self.prediction_mode != ORACLE_PREDICTION_MODE
                and combined_estimate is not None):
            if not self.in_cache_window(transformer_options):
                reason = "protected_window"
            elif not self._prediction_valid(state, video_input, audio_input):
                reason = "no_prediction"
            elif state.consecutive_skips >= self.max_consecutive_skips:
                reason = "max_consecutive"
            elif candidate_cumulative >= self.reuse_threshold:
                reason = "threshold"
        state.cumulative_change = 0.0
        state.consecutive_skips = 0
        state.refreshes += 1
        self._log_decision(state, transformer_options, "refresh", reason)

        oracle_predictions = (
            self._raw_predictions(
                state, video_input, audio_input, transformer_options)
            if self.prediction_mode == ORACLE_PREDICTION_MODE else {}
        )
        full_output = executor(*args, **kwargs)
        outputs = _extract_h3_pair(full_output)
        if outputs is None:
            state.clear_tensors()
            return full_output
        video_output, audio_output = outputs
        if (video_output.shape != video_input.shape
                or audio_output.shape != audio_input.shape):
            state.clear_tensors()
            return full_output

        if oracle_predictions:
            self._record_oracle_predictions(
                state, transformer_options, oracle_predictions,
                video_output, audio_output)
        if self.prediction_mode == ORACLE_PREDICTION_MODE:
            self._archive_oracle_output(
                state, _as_float(transformer_options.get("sigmas")),
                video_output, audio_output)

        output_video, output_audio = self._samples(video_output, audio_output)
        if (state.refresh_input_video is not None
                and state.refresh_input_audio is not None
                and state.refresh_output_video is not None
                and state.refresh_output_audio is not None):
            try:
                refresh_input_video_change = _relative_change(
                    current_video, state.refresh_input_video)
                refresh_input_audio_change = _relative_change(
                    current_audio, state.refresh_input_audio)
                actual_video_change = _relative_change(
                    output_video, state.refresh_output_video)
                actual_audio_change = _relative_change(
                    output_audio, state.refresh_output_audio)
                state.last_actual_video_change = actual_video_change
                state.last_actual_audio_change = actual_audio_change
                if refresh_input_video_change > 1.0e-12:
                    state.video_transform_rate = (
                        actual_video_change / refresh_input_video_change)
                if refresh_input_audio_change > 1.0e-12:
                    state.audio_transform_rate = (
                        actual_audio_change / refresh_input_audio_change)
                sigma = _as_float(transformer_options.get("sigmas"))
                if sigma is not None and state.cached_sigma is not None:
                    shift_v = float(transformer_options.get(
                        "minimax_h3_sigma_shift_video", 12.0))
                    shift_a = float(transformer_options.get(
                        "minimax_h3_sigma_shift_audio", 3.0))
                    state.video_secant_scale = _least_squares_scale(
                        current_video - state.refresh_input_video,
                        output_video - state.refresh_output_video,
                    )
                    state.audio_secant_scale = _least_squares_scale(
                        current_audio - state.refresh_input_audio,
                        output_audio / _time_shift_slope(
                            sigma, shift_v, shift_a)
                        - state.refresh_output_audio / _time_shift_slope(
                            state.cached_sigma, shift_v, shift_a),
                    )
                if self.verbose:
                    logging.info(
                        "%s eval=%d actual_video=%.6g actual_audio=%.6g "
                        "video_rate=%s audio_rate=%s video_secant=%s "
                        "audio_secant=%s",
                        TAG, state.evaluations, actual_video_change,
                        actual_audio_change,
                        "?" if state.video_transform_rate is None else f"{state.video_transform_rate:.6g}",
                        "?" if state.audio_transform_rate is None else f"{state.audio_transform_rate:.6g}",
                        "?" if state.video_secant_scale is None else f"{state.video_secant_scale:.6g}",
                        "?" if state.audio_secant_scale is None else f"{state.audio_secant_scale:.6g}",
                    )
            except (RuntimeError, ValueError):
                state.video_transform_rate = None
                state.audio_transform_rate = None

        state.cached_video_output = video_output.detach()
        state.cached_audio_output = audio_output.detach()
        state.cached_sigma = _as_float(transformer_options.get("sigmas"))
        state.cached_video_residual = (video_output - video_input).detach()
        state.cached_audio_residual = (audio_output - audio_input).detach()
        state.last_input_video = current_video
        state.last_input_audio = current_audio
        state.refresh_input_video = current_video
        state.refresh_input_audio = current_audio
        state.refresh_output_video = output_video
        state.refresh_output_audio = output_audio
        return full_output

    def run_apply_model(
        self,
        executor: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        transformer_options: dict[str, Any],
    ):
        """Cache the sampler-facing denoised estimate rather than raw velocity."""
        if self.prediction_mode not in DENOISED_PREDICTION_MODES:
            return executor(*args, **kwargs)
        packed_input = args[0] if args else kwargs.get("x")
        inputs = _extract_packed_h3_pair(
            packed_input, kwargs.get("latent_shapes"))
        key = _stream_key(transformer_options)
        if inputs is None or key is None:
            return executor(*args, **kwargs)
        video_input, audio_input = inputs
        try:
            current_video, current_audio = self._samples(
                video_input, audio_input)
        except ValueError:
            return executor(*args, **kwargs)

        metadata = ("denoised",) + self._metadata(video_input, audio_input)
        state = self.streams.setdefault(key, _StreamState(metadata=metadata))
        if state.metadata != metadata:
            state.clear_tensors()
            state = _StreamState(metadata=metadata)
            self.streams[key] = state

        state.evaluations += 1
        input_video_change = input_audio_change = None
        if state.last_input_video is not None and state.last_input_audio is not None:
            try:
                input_video_change = _relative_change(
                    current_video, state.last_input_video)
                input_audio_change = _relative_change(
                    current_audio, state.last_input_audio)
                state.last_input_video_change = input_video_change
                state.last_input_audio_change = input_audio_change
            except (RuntimeError, ValueError):
                state.clear_tensors()

        estimated_video = estimated_audio = None
        if (input_video_change is not None and input_audio_change is not None
                and state.video_transform_rate is not None
                and state.audio_transform_rate is not None):
            estimated_video = state.video_transform_rate * input_video_change
            estimated_audio = state.audio_transform_rate * input_audio_change
            state.last_estimated_video_change = estimated_video
            state.last_estimated_audio_change = estimated_audio

        combined_estimate = (
            max(estimated_video, self.audio_weight * estimated_audio)
            if estimated_video is not None and estimated_audio is not None
            else None
        )
        candidate_cumulative = (
            state.cumulative_change + combined_estimate
            if combined_estimate is not None else float("inf")
        )
        prediction_valid = self._residual_valid(
            state.cached_denoised, packed_input)
        if self.prediction_mode == "denoised_linear":
            prediction_valid = (
                prediction_valid
                and self._residual_valid(state.denoised_slope, packed_input)
                and state.cached_sigma is not None
            )
        can_reuse = (
            combined_estimate is not None
            and self.reuse_threshold > 0.0
            and self.in_cache_window(transformer_options)
            and prediction_valid
            and state.consecutive_skips < self.max_consecutive_skips
            and candidate_cumulative < self.reuse_threshold
        )
        if can_reuse:
            state.cumulative_change = candidate_cumulative
            state.consecutive_skips += 1
            state.skips += 1
            state.last_input_video = current_video
            state.last_input_audio = current_audio
            self._log_decision(
                state, transformer_options, "skip", "below_threshold")
            if self.prediction_mode == "denoised_linear":
                sigma = _as_float(transformer_options.get("sigmas"))
                return state.cached_denoised + state.denoised_slope * (
                    sigma - state.cached_sigma)
            return state.cached_denoised.clone()

        reason = "cold"
        if combined_estimate is not None:
            if not self.in_cache_window(transformer_options):
                reason = "protected_window"
            elif not prediction_valid:
                reason = "no_prediction"
            elif state.consecutive_skips >= self.max_consecutive_skips:
                reason = "max_consecutive"
            elif candidate_cumulative >= self.reuse_threshold:
                reason = "threshold"
        state.cumulative_change = 0.0
        state.consecutive_skips = 0
        state.refreshes += 1
        self._log_decision(state, transformer_options, "refresh", reason)

        full_output = executor(*args, **kwargs)
        outputs = _extract_packed_h3_pair(
            full_output, kwargs.get("latent_shapes"))
        if outputs is None or full_output.shape != packed_input.shape:
            state.clear_tensors()
            return full_output
        video_output, audio_output = outputs
        output_video, output_audio = self._samples(video_output, audio_output)
        if (state.refresh_input_video is not None
                and state.refresh_input_audio is not None
                and state.refresh_output_video is not None
                and state.refresh_output_audio is not None):
            try:
                refresh_input_video_change = _relative_change(
                    current_video, state.refresh_input_video)
                refresh_input_audio_change = _relative_change(
                    current_audio, state.refresh_input_audio)
                actual_video_change = _relative_change(
                    output_video, state.refresh_output_video)
                actual_audio_change = _relative_change(
                    output_audio, state.refresh_output_audio)
                state.last_actual_video_change = actual_video_change
                state.last_actual_audio_change = actual_audio_change
                if refresh_input_video_change > 1.0e-12:
                    state.video_transform_rate = (
                        actual_video_change / refresh_input_video_change)
                if refresh_input_audio_change > 1.0e-12:
                    state.audio_transform_rate = (
                        actual_audio_change / refresh_input_audio_change)
                if self.verbose:
                    logging.info(
                        "%s eval=%d denoised_actual_video=%.6g "
                        "denoised_actual_audio=%.6g video_rate=%s "
                        "audio_rate=%s",
                        TAG, state.evaluations, actual_video_change,
                        actual_audio_change,
                        "?" if state.video_transform_rate is None else f"{state.video_transform_rate:.6g}",
                        "?" if state.audio_transform_rate is None else f"{state.audio_transform_rate:.6g}",
                    )
            except (RuntimeError, ValueError):
                state.video_transform_rate = None
                state.audio_transform_rate = None

        sigma = _as_float(transformer_options.get("sigmas"))
        if (self.prediction_mode == "denoised_linear"
                and state.cached_denoised is not None
                and state.cached_sigma is not None
                and sigma is not None
                and abs(sigma - state.cached_sigma) > 1.0e-12):
            state.denoised_slope = (
                (full_output.detach() - state.cached_denoised)
                / (sigma - state.cached_sigma)
            )
        state.cached_denoised = full_output.detach()
        state.cached_sigma = sigma
        state.last_input_video = current_video
        state.last_input_audio = current_audio
        state.refresh_input_video = current_video
        state.refresh_input_audio = current_audio
        state.refresh_output_video = output_video
        state.refresh_output_audio = output_audio
        return full_output

    def _record_block_oracle(
        self,
        state: _BlockOracleState,
        block_index: int,
        sigma: float | None,
        video_residual: torch.Tensor,
        audio_residual: torch.Tensor,
    ) -> None:
        video_history = state.video_residual_history
        audio_history = state.audio_residual_history
        predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if video_history and audio_history:
            predictions["residual_hold"] = (
                video_history[-1], audio_history[-1])
        if len(video_history) >= 2 and len(audio_history) >= 2:
            predictions["residual_taylor_1"] = (
                video_history[-1]
                + (video_history[-1] - video_history[-2]),
                audio_history[-1]
                + (audio_history[-1] - audio_history[-2]),
            )

        for predictor, (video_prediction, audio_prediction) in predictions.items():
            video_nrmse, video_cosine = _prediction_metrics(
                video_prediction, video_residual)
            audio_nrmse, audio_cosine = _prediction_metrics(
                audio_prediction, audio_residual)
            state.stats.setdefault(predictor, _OracleStats()).update(
                video_nrmse, video_cosine, audio_nrmse, audio_cosine)
            logging.info(
                "%s block_oracle block=%d eval=%d sigma=%s predictor=%s "
                "video_nrmse=%.7g video_cosine=%.7g "
                "audio_nrmse=%.7g audio_cosine=%.7g",
                TAG, block_index, state.evaluations,
                "?" if sigma is None else f"{sigma:.7g}", predictor,
                video_nrmse, video_cosine, audio_nrmse, audio_cosine,
            )

        if len(video_history) >= 2 and len(audio_history) >= 2:
            video_directions = [video_history[-1] - video_history[-2]]
            audio_directions = [audio_history[-1] - audio_history[-2]]
            orders = [("residual_projection_ceiling_1", 1)]
            if len(video_history) >= 3 and len(audio_history) >= 3:
                video_directions.append(video_history[-2] - video_history[-3])
                audio_directions.append(audio_history[-2] - audio_history[-3])
                orders.append(("residual_projection_ceiling_2", 2))
            for predictor, order in orders:
                video_projection = _projection_ceiling(
                    video_history[-1], video_residual,
                    video_directions[:order])
                audio_projection = _projection_ceiling(
                    audio_history[-1], audio_residual,
                    audio_directions[:order])
                if video_projection is None or audio_projection is None:
                    continue
                video_prediction, video_coefficients = video_projection
                audio_prediction, audio_coefficients = audio_projection
                video_nrmse, video_cosine = _prediction_metrics(
                    video_prediction, video_residual)
                audio_nrmse, audio_cosine = _prediction_metrics(
                    audio_prediction, audio_residual)
                state.stats.setdefault(predictor, _OracleStats()).update(
                    video_nrmse, video_cosine, audio_nrmse, audio_cosine)
                logging.info(
                    "%s block_oracle block=%d eval=%d sigma=%s predictor=%s "
                    "video_nrmse=%.7g video_cosine=%.7g video_coefficients=%s "
                    "audio_nrmse=%.7g audio_cosine=%.7g audio_coefficients=%s",
                    TAG, block_index, state.evaluations,
                    "?" if sigma is None else f"{sigma:.7g}", predictor,
                    video_nrmse, video_cosine,
                    ",".join(f"{value:.7g}" for value in video_coefficients),
                    audio_nrmse, audio_cosine,
                    ",".join(f"{value:.7g}" for value in audio_coefficients),
                )

        video_history.append(video_residual.detach())
        audio_history.append(audio_residual.detach())
        del video_history[:-3]
        del audio_history[:-3]

    def run_block_oracle(
        self,
        index: int,
        args: dict[str, Any],
        original_block: Any,
    ) -> dict[str, torch.Tensor]:
        """Measure representative block-residual predictability, returning exact."""
        hidden = args.get("img")
        transformer_options = args.get("transformer_options")
        if not isinstance(hidden, torch.Tensor) or not isinstance(
            transformer_options, dict
        ):
            return original_block(args)
        stream_key = _stream_key(transformer_options)
        target_rows = (
            self.suffix_layouts.get(stream_key)
            if stream_key is not None else None
        )
        if stream_key is None or target_rows is None:
            return original_block(args)
        audio_rows, video_rows = target_rows
        try:
            video_input, audio_input = _sample_block_targets(
                hidden, audio_rows, video_rows, self.video_subsample)
        except ValueError:
            return original_block(args)

        metadata = (
            tuple(hidden.shape), hidden.dtype, hidden.device.type,
            hidden.device.index, audio_rows, video_rows,
            self.video_subsample,
        )
        state_key = (stream_key, index)
        state = self.block_oracle_states.get(state_key)
        if state is None or state.metadata != metadata:
            state = _BlockOracleState(metadata=metadata)
            self.block_oracle_states[state_key] = state
        state.evaluations += 1

        result = original_block(args)
        output = result.get("img") if isinstance(result, dict) else None
        if not isinstance(output, torch.Tensor) or output.shape != hidden.shape:
            return result
        try:
            video_output, audio_output = _sample_block_targets(
                output, audio_rows, video_rows, self.video_subsample)
            self._record_block_oracle(
                state, index,
                _as_float(transformer_options.get("sigmas")),
                video_output - video_input,
                audio_output - audio_input,
            )
        except (RuntimeError, ValueError):
            pass
        return result

    def run_block_taylor(
        self,
        index: int,
        args: dict[str, Any],
        original_block: Any,
    ) -> dict[str, torch.Tensor]:
        """Predict one early block residual from two consecutive exact calls."""
        hidden = args.get("img")
        transformer_options = args.get("transformer_options")
        if not isinstance(hidden, torch.Tensor) or not isinstance(
            transformer_options, dict
        ):
            return original_block(args)
        stream_key = _stream_key(transformer_options)
        if stream_key is None:
            return original_block(args)

        metadata = (
            tuple(hidden.shape), hidden.dtype, hidden.device.type,
            hidden.device.index, "block_taylor", index,
        )
        state_key = (stream_key, index)
        state = self.block_taylor_states.get(state_key)
        if state is None or state.metadata != metadata:
            state = _BlockTaylorState(metadata=metadata)
            self.block_taylor_states[state_key] = state
        state.evaluations += 1

        history_is_consecutive = (
            len(state.residual_history) >= 2
            and state.evaluation_history[-2:] == [
                state.evaluations - 2, state.evaluations - 1]
        )
        history_is_valid = (
            history_is_consecutive
            and all(
                self._residual_valid(residual, hidden)
                for residual in state.residual_history[-2:]
            )
        )
        can_reuse = (
            self.reuse_threshold > 0.0
            and self.in_cache_window(transformer_options)
            and history_is_valid
            and state.consecutive_skips < self.max_consecutive_skips
        )
        if can_reuse:
            prediction = (
                state.residual_history[-1]
                + state.residual_history[-1] - state.residual_history[-2]
            )
            state.consecutive_skips += 1
            state.skips += 1
            if self.verbose:
                logging.info(
                    "%s block_taylor block=%d eval=%d sigma=%s "
                    "action=skip predictor=residual_taylor_1",
                    TAG, index, state.evaluations,
                    _as_float(transformer_options.get("sigmas")),
                )
            return {"img": hidden + prediction}

        reason = "cold"
        if len(state.residual_history) >= 2:
            if not self.in_cache_window(transformer_options):
                reason = "protected_window"
            elif not history_is_consecutive:
                reason = "nonconsecutive_history"
            elif not history_is_valid:
                reason = "invalid_history"
            elif state.consecutive_skips >= self.max_consecutive_skips:
                reason = "max_consecutive"
        state.consecutive_skips = 0
        state.refreshes += 1
        block_input = hidden.detach().clone()
        result = original_block(args)
        output = result.get("img") if isinstance(result, dict) else None
        if not isinstance(output, torch.Tensor) or output.shape != hidden.shape:
            state.residual_history.clear()
            state.evaluation_history.clear()
            return result
        residual = (output - block_input).detach()
        state.residual_history.append(residual)
        state.evaluation_history.append(state.evaluations)
        del state.residual_history[:-2]
        del state.evaluation_history[:-2]
        if self.verbose:
            logging.info(
                "%s block_taylor block=%d eval=%d sigma=%s "
                "action=refresh reason=%s",
                TAG, index, state.evaluations,
                _as_float(transformer_options.get("sigmas")), reason,
            )
        return result

    def run_first_block(
        self,
        index: int,
        last_index: int,
        args: dict[str, Any],
        original_block: Any,
        notify_external_skip: Any = None,
    ) -> dict[str, torch.Tensor]:
        """FirstBlockCache: block 0 is always exact and decides the rest.

        Block 0's own residual is a strong, cheap proxy for how much the whole
        transformer would change this evaluation. When it is stable, blocks
        1..N-1 are replaced by their cached aggregate residual and skipped
        entirely; when it is not, they run exactly and refresh that cache. The
        original mode accumulates change between refreshes; the per-step mode
        implements the Discord policy and tests only the latest relative L1.
        """
        hidden = args.get("img")
        transformer_options = args.get("transformer_options")
        if not isinstance(hidden, torch.Tensor) or not isinstance(
            transformer_options, dict
        ):
            return original_block(args)
        key = _stream_key(transformer_options)
        target_rows = self.suffix_layouts.get(key) if key is not None else None
        if key is None or target_rows is None:
            return original_block(args)
        audio_rows, video_rows = target_rows
        per_step_policy = (
            self.prediction_mode == FIRST_BLOCK_CACHE_PER_STEP_MODE)
        # Match the Discord detector exactly in the per-step policy. The
        # original cumulative policy keeps its lower-overhead sampled detector.
        detector_subsample = 1 if per_step_policy else self.video_subsample

        state = self.first_block_states.get(key)

        if index != 0:
            # Structural token reduction deliberately changes the sequence
            # length inside the transformer, then restores it before the tail.
            # Cache identity is established at block 0's full-sequence boundary;
            # do not replace that state merely because an interior block sees
            # the reduced stream.
            if state is None:
                return original_block(args)
            if state.skip_active:
                # Blocks 1..N-1 were already accounted for by the cached
                # residual block 0 added; do nothing at all.
                if notify_external_skip is not None:
                    notify_external_skip()
                if index == last_index:
                    state.skip_active = False
                return {"img": hidden}
            result = original_block(args)
            output = result.get("img") if isinstance(result, dict) else None
            if index == last_index:
                if (isinstance(output, torch.Tensor)
                        and state.saved_head is not None
                        and output.shape == state.saved_head.shape):
                    state.cached_tail = (output - state.saved_head).detach()
                    state.peak_bytes = max(
                        state.peak_bytes,
                        state.cached_tail.numel() * state.cached_tail.element_size(),
                    )
                else:
                    state.cached_tail = None
                # Release the head clone the moment the tail residual exists.
                state.saved_head = None
            return result

        metadata = (
            tuple(hidden.shape), hidden.dtype, hidden.device.type,
            hidden.device.index, audio_rows, video_rows, last_index,
            self.prediction_mode, detector_subsample,
        )
        if state is None or state.metadata != metadata:
            if state is not None:
                state.clear_tensors()
            state = _FirstBlockCacheState(metadata=metadata)
            self.first_block_states[key] = state

        state.evaluations += 1
        try:
            input_video, input_audio = _sample_block_targets(
                hidden, audio_rows, video_rows, detector_subsample)
        except ValueError:
            state.clear_tensors()
            return original_block(args)

        # H3's optimized block mutates its stream in place, so every sample the
        # detector keeps must be an owned copy taken before the block runs.
        result = original_block(args)
        output = result.get("img") if isinstance(result, dict) else None
        if not isinstance(output, torch.Tensor) or output.shape != hidden.shape:
            state.clear_tensors()
            return result
        try:
            output_video, output_audio = _sample_block_targets(
                output, audio_rows, video_rows, detector_subsample)
        except ValueError:
            state.clear_tensors()
            return result
        current_video = output_video - input_video
        current_audio = output_audio - input_audio

        combined_change = None
        video_change = audio_change = None
        if state.previous_video is not None and state.previous_audio is not None:
            try:
                video_change = _relative_change(current_video, state.previous_video)
                audio_change = _relative_change(current_audio, state.previous_audio)
                combined_change = max(video_change, self.audio_weight * audio_change)
            except (RuntimeError, ValueError):
                state.clear_tensors()
                combined_change = None

        candidate_cumulative = (
            state.cumulative_change + combined_change
            if combined_change is not None else float("inf")
        )
        decision_change = (
            combined_change if per_step_policy else candidate_cumulative)
        cache_valid = self._residual_valid(state.cached_tail, output)
        can_reuse = (
            combined_change is not None
            and self.reuse_threshold > 0.0
            and self.in_cache_window(transformer_options)
            and cache_valid
            and state.consecutive_skips < self.max_consecutive_skips
            and decision_change < self.reuse_threshold
        )

        state.previous_video = current_video
        state.previous_audio = current_audio
        if can_reuse:
            state.cumulative_change = (
                0.0 if per_step_policy else candidate_cumulative)
            state.consecutive_skips += 1
            state.skips += 1
            state.skip_active = last_index > 0
            if self.verbose:
                logging.info(
                    "%s first_block mode=%s eval=%d sigma=%s video=%.6g "
                    "audio=%.6g decision=%.6g cumulative=%.6g "
                    "threshold=%.6g action=skip",
                    TAG, self.prediction_mode, state.evaluations,
                    _as_float(transformer_options.get("sigmas")),
                    video_change, audio_change, decision_change,
                    state.cumulative_change, self.reuse_threshold,
                )
            return {"img": output + state.cached_tail}

        reason = "cold"
        if combined_change is not None:
            if not self.in_cache_window(transformer_options):
                reason = "protected_window"
            elif not cache_valid:
                reason = "no_cached_tail"
            elif state.consecutive_skips >= self.max_consecutive_skips:
                reason = "max_consecutive"
            elif self.reuse_threshold <= 0.0:
                reason = "calibration"
            elif decision_change >= self.reuse_threshold:
                reason = "threshold"
        state.cumulative_change = 0.0
        state.consecutive_skips = 0
        state.refreshes += 1
        state.skip_active = False
        state.saved_head = output.detach().clone()
        state.peak_bytes = max(
            state.peak_bytes,
            state.saved_head.numel() * state.saved_head.element_size(),
        )
        if self.verbose:
            logging.info(
                "%s first_block mode=%s eval=%d sigma=%s video=%s audio=%s "
                "decision=%s cumulative=0 threshold=%.6g action=refresh "
                "reason=%s",
                TAG, self.prediction_mode, state.evaluations,
                _as_float(transformer_options.get("sigmas")),
                "?" if video_change is None else f"{video_change:.6g}",
                "?" if audio_change is None else f"{audio_change:.6g}",
                "?" if combined_change is None else f"{decision_change:.6g}",
                self.reuse_threshold, reason,
            )
        return result

    def run_suffix_block(
        self,
        index: int,
        probe_index: int,
        last_index: int,
        args: dict[str, Any],
        original_block: Any,
    ) -> dict[str, torch.Tensor]:
        """Reuse a cached residual for a short transformer suffix."""
        h = args.get("img")
        transformer_options = args.get("transformer_options")
        if not isinstance(h, torch.Tensor) or not isinstance(
            transformer_options, dict
        ):
            return original_block(args)
        key = _stream_key(transformer_options)
        if key is None:
            return original_block(args)
        target_rows = self.suffix_layouts.get(key)
        if target_rows is None:
            return original_block(args)
        audio_rows, video_rows = target_rows
        if (audio_rows <= 0 or video_rows <= 0
                or audio_rows + video_rows > h.shape[0]):
            return original_block(args)
        metadata = (
            tuple(h.shape), h.dtype, h.device.type, h.device.index,
            "suffix", probe_index, last_index, audio_rows, video_rows,
        )
        state = self.streams.setdefault(key, _StreamState(metadata=metadata))
        if state.metadata != metadata:
            state.clear_tensors()
            state = _StreamState(metadata=metadata)
            self.streams[key] = state

        if index != probe_index:
            if state.suffix_skip_active:
                if index == last_index:
                    state.suffix_skip_active = False
                return {"img": h}
            result = original_block(args)
            output = result["img"]
            if index == last_index and state.suffix_current_probe is not None:
                state.suffix_cached_delta = (
                    output - state.suffix_current_probe).detach()
                state.suffix_current_probe = None
            return result

        state.evaluations += 1
        # Target audio and video are H3's last two contiguous packed segments.
        # Conditioning/text rows are timestep-modulated too, so including them
        # makes a poor cache detector. Keep owned samples because later blocks
        # update the complete hidden state in place.
        video_start = h.shape[0] - video_rows
        audio_start = video_start - audio_rows
        video_sample = h[
            video_start::self.video_subsample,
            ::self.video_subsample,
        ].detach().contiguous()
        audio_sample = h[
            audio_start:video_start:self.audio_subsample,
            ::self.audio_subsample,
        ].detach().contiguous()
        video_change = audio_change = None
        if (state.suffix_last_probe_video is not None
                and state.suffix_last_probe_audio is not None):
            try:
                video_change = _relative_change(
                    video_sample, state.suffix_last_probe_video)
                audio_change = _relative_change(
                    audio_sample, state.suffix_last_probe_audio)
                state.suffix_last_video_change = video_change
                state.suffix_last_audio_change = audio_change
            except (RuntimeError, ValueError):
                state.clear_tensors()

        combined_change = (
            max(video_change, self.audio_weight * audio_change)
            if video_change is not None and audio_change is not None else None
        )

        candidate_cumulative = (
            state.cumulative_change + combined_change
            if combined_change is not None else float("inf")
        )
        cache_valid = self._residual_valid(state.suffix_cached_delta, h)
        can_reuse = (
            combined_change is not None
            and self.reuse_threshold > 0.0
            and self.in_cache_window(transformer_options)
            and cache_valid
            and state.consecutive_skips < self.max_consecutive_skips
            and candidate_cumulative < self.reuse_threshold
        )
        if can_reuse:
            state.cumulative_change = candidate_cumulative
            state.consecutive_skips += 1
            state.skips += 1
            state.suffix_last_probe_video = video_sample
            state.suffix_last_probe_audio = audio_sample
            state.suffix_skip_active = probe_index != last_index
            if self.verbose:
                logging.info(
                    "%s eval=%d sigma=%s suffix_video=%.6g suffix_audio=%.6g "
                    "cumulative=%.6g "
                    "threshold=%.6g cached_blocks=%d action=skip",
                    TAG, state.evaluations,
                    _as_float(transformer_options.get("sigmas")),
                    video_change, audio_change, state.cumulative_change,
                    self.reuse_threshold,
                    self.cached_suffix_blocks,
                )
            return {"img": h + state.suffix_cached_delta}

        reason = "cold"
        if combined_change is not None:
            if not self.in_cache_window(transformer_options):
                reason = "protected_window"
            elif not cache_valid:
                reason = "no_suffix"
            elif state.consecutive_skips >= self.max_consecutive_skips:
                reason = "max_consecutive"
            elif candidate_cumulative >= self.reuse_threshold:
                reason = "threshold"
        state.cumulative_change = 0.0
        state.consecutive_skips = 0
        state.refreshes += 1
        state.suffix_skip_active = False
        state.suffix_last_probe_video = video_sample
        state.suffix_last_probe_audio = audio_sample
        state.suffix_current_probe = h.detach().clone()
        if self.verbose:
            logging.info(
                "%s eval=%d sigma=%s suffix_video=%s suffix_audio=%s "
                "cumulative=0 "
                "threshold=%.6g cached_blocks=%d action=refresh reason=%s",
                TAG, state.evaluations,
                _as_float(transformer_options.get("sigmas")),
                "?" if video_change is None else f"{video_change:.6g}",
                "?" if audio_change is None else f"{audio_change:.6g}",
                self.reuse_threshold, self.cached_suffix_blocks, reason,
            )
        result = original_block(args)
        if probe_index == last_index:
            state.suffix_cached_delta = (
                result["img"] - state.suffix_current_probe).detach()
            state.suffix_current_probe = None
        return result


def h3cache_forward_wrapper(executor, *args, **kwargs):
    transformer_options = _find_transformer_options(args, kwargs)
    if transformer_options is None:
        return executor(*args, **kwargs)
    holder = transformer_options.get(STATE_KEY)
    if not isinstance(holder, H3CacheHolder):
        return executor(*args, **kwargs)
    return holder.run_forward(executor, args, kwargs, transformer_options)


def h3cache_apply_model_wrapper(executor, *args, **kwargs):
    transformer_options = _find_transformer_options(args, kwargs)
    if transformer_options is None:
        return executor(*args, **kwargs)
    holder = transformer_options.get(STATE_KEY)
    if not isinstance(holder, H3CacheHolder):
        return executor(*args, **kwargs)
    return holder.run_apply_model(executor, args, kwargs, transformer_options)


def _make_suffix_block_wrapper(index: int, probe_index: int, last_index: int):
    def suffix_block_wrapper(args, extra_options):
        transformer_options = args.get("transformer_options")
        holder = (
            transformer_options.get(STATE_KEY)
            if isinstance(transformer_options, dict) else None
        )
        original_block = extra_options["original_block"]
        if not isinstance(holder, H3CacheHolder):
            return original_block(args)
        return holder.run_suffix_block(
            index, probe_index, last_index, args, original_block)

    return suffix_block_wrapper


def _make_block_oracle_wrapper(index: int):
    def block_oracle_wrapper(args, extra_options):
        transformer_options = args.get("transformer_options")
        holder = (
            transformer_options.get(STATE_KEY)
            if isinstance(transformer_options, dict) else None
        )
        original_block = extra_options["original_block"]
        if not isinstance(holder, H3CacheHolder):
            return original_block(args)
        return holder.run_block_oracle(index, args, original_block)

    return block_oracle_wrapper


def _make_block_taylor_wrapper(index: int):
    def block_taylor_wrapper(args, extra_options):
        transformer_options = args.get("transformer_options")
        holder = (
            transformer_options.get(STATE_KEY)
            if isinstance(transformer_options, dict) else None
        )
        original_block = extra_options["original_block"]
        if not isinstance(holder, H3CacheHolder):
            return original_block(args)
        return holder.run_block_taylor(index, args, original_block)

    return block_taylor_wrapper


def _make_first_block_wrapper(index: int, last_index: int, previous=None):
    def first_block_wrapper(args, extra_options):
        transformer_options = args.get("transformer_options")
        holder = (
            transformer_options.get(STATE_KEY)
            if isinstance(transformer_options, dict) else None
        )
        if previous is None:
            original_block = extra_options["original_block"]
            notify_external_skip = None
        else:
            original_block = lambda local_args: previous(local_args, extra_options)
            structure = transformer_options.get("asfp8_h3_structure")
            advance = getattr(structure, "advance_sparse_for_skipped_block", None)
            notify_external_skip = (
                (lambda: advance(transformer_options))
                if advance is not None else None)
        if not isinstance(holder, H3CacheHolder):
            return original_block(args)
        return holder.run_first_block(
            index, last_index, args, original_block, notify_external_skip)

    first_block_wrapper._asfp8_h3cache_first_block = True
    return first_block_wrapper


def h3cache_sample_wrapper(executor, *args, **kwargs):
    """Clone H3Cache state for one sample and release it on every exit path."""
    import comfy.model_patcher

    guider = executor.class_obj
    original_options = guider.model_options
    active: H3CacheHolder | None = None
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(
            original_options)
        transformer_options = guider.model_options.setdefault(
            "transformer_options", {})
        template = transformer_options.get(STATE_KEY)
        if not isinstance(template, H3CacheHolder):
            return executor(*args, **kwargs)
        if "easycache" in transformer_options:
            raise RuntimeError("H3Cache and EasyCache cannot be enabled together")
        active = template.clone().prepare_timesteps(
            guider.model_patcher.model.model_sampling)
        transformer_options[STATE_KEY] = active
        if active.verbose:
            logging.info(
                "%s enabled threshold=%.6g window=%.2f..%.2f max_skips=%d "
                "audio_weight=%.3g predictor=%s cached_suffix_blocks=%d",
                TAG, active.reuse_threshold, active.start_percent,
                active.end_percent, active.max_consecutive_skips,
                active.audio_weight, active.prediction_mode,
                active.cached_suffix_blocks,
            )
        return executor(*args, **kwargs)
    finally:
        if active is not None:
            from .h3_logging import call_summary, block_summary
            first_evals, _, first_skips, _ = active.first_block_totals()
            block_evals, _, block_skips = active.block_taylor_totals()
            evaluations, _, skips = active.totals()
            if first_evals:
                blocks = len(guider.model_patcher.model.diffusion_model.blocks)
                logging.info("%s", block_summary("FirstBlockCache", first_skips * (blocks - 1), first_evals * blocks))
            elif block_evals:
                logging.info("%s", block_summary("Block cache", block_skips, block_evals))
            elif evaluations:
                if active.cached_suffix_blocks:
                    blocks = len(guider.model_patcher.model.diffusion_model.blocks)
                    logging.info("%s", block_summary("Suffix cache", skips * active.cached_suffix_blocks, evaluations * blocks))
                else:
                    logging.info("%s", call_summary("Cache", skips, evaluations))
            if active.verbose:
                evaluations, refreshes, skips = active.totals()
                logging.info(
                    "%s evaluations=%d refreshed=%d reused=%d",
                    TAG, evaluations, refreshes, skips,
                )
                for stream_key, predictor, stats in active.oracle_summaries():
                    logging.info(
                        "%s oracle_summary stream=%r predictor=%s count=%d "
                        "video_nrmse_mean=%.7g video_nrmse_min=%.7g "
                        "video_nrmse_max=%.7g video_cosine_min=%.7g "
                        "audio_nrmse_mean=%.7g audio_nrmse_min=%.7g "
                        "audio_nrmse_max=%.7g audio_cosine_min=%.7g",
                        TAG, stream_key, predictor, stats.count,
                        stats.video_nrmse_sum / stats.count,
                        stats.video_nrmse_min, stats.video_nrmse_max,
                        stats.video_cosine_min,
                        stats.audio_nrmse_sum / stats.count,
                        stats.audio_nrmse_min, stats.audio_nrmse_max,
                        stats.audio_cosine_min,
                    )
                for (
                    stream_key, block_index, predictor, stats
                ) in active.block_oracle_summaries():
                    logging.info(
                        "%s block_oracle_summary stream=%r block=%d "
                        "predictor=%s count=%d "
                        "video_nrmse_mean=%.7g video_nrmse_min=%.7g "
                        "video_nrmse_max=%.7g video_cosine_min=%.7g "
                        "audio_nrmse_mean=%.7g audio_nrmse_min=%.7g "
                        "audio_nrmse_max=%.7g audio_cosine_min=%.7g",
                        TAG, stream_key, block_index, predictor, stats.count,
                        stats.video_nrmse_sum / stats.count,
                        stats.video_nrmse_min, stats.video_nrmse_max,
                        stats.video_cosine_min,
                        stats.audio_nrmse_sum / stats.count,
                        stats.audio_nrmse_min, stats.audio_nrmse_max,
                        stats.audio_cosine_min,
                    )
                block_evaluations, block_refreshes, block_skips = (
                    active.block_taylor_totals())
                if block_evaluations:
                    logging.info(
                        "%s block_taylor evaluations=%d refreshed=%d reused=%d",
                        TAG, block_evaluations, block_refreshes, block_skips,
                    )
                first_evaluations, first_refreshes, first_skips, first_bytes = (
                    active.first_block_totals())
                if first_evaluations:
                    logging.info(
                        "%s first_block_cache evaluations=%d refreshed=%d reused=%d "
                        "peak_cached_bytes=%d",
                        TAG, first_evaluations, first_refreshes, first_skips,
                        first_bytes,
                    )
            active.reset()
        guider.model_options = original_options


def apply_h3cache(
    model: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
    max_consecutive_skips: int,
    audio_weight: float,
    cached_suffix_blocks: int,
    prediction_mode: str,
    verbose: bool,
):
    """Attach H3Cache to a cloned MiniMax H3 ModelPatcher."""
    import comfy.patcher_extension

    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(diffusion_model, "blocks", None)
    module_name = getattr(type(diffusion_model), "__module__", "")
    if blocks is None or "comfy.ldm.minimax" not in module_name:
        raise ValueError("H3Cache only supports ComfyUI's MiniMax H3 model")
    if len(blocks) < 2:
        raise ValueError("MiniMax H3 transformer block list is incomplete")
    if cached_suffix_blocks < 0 or cached_suffix_blocks >= len(blocks):
        raise ValueError(
            "cached_suffix_blocks must be 0 or smaller than H3's block count")
    if cached_suffix_blocks and prediction_mode in DENOISED_PREDICTION_MODES:
        raise ValueError(
            "denoised predictors cannot be combined with suffix caching")
    if cached_suffix_blocks and prediction_mode in BLOCK_LEVEL_MODES:
        raise ValueError(
            "block modes select their own transformer blocks; "
            "cached_suffix_blocks must be 0")

    block_oracle_indices: tuple[int, ...] = ()
    block_taylor_indices: tuple[int, ...] = ()
    first_block_indices: tuple[int, ...] = ()
    if prediction_mode == BLOCK_ORACLE_PREDICTION_MODE:
        last_index = len(blocks) - 1
        early_indices = set(range(0, min(13, len(blocks)), 2))
        block_oracle_indices = tuple(sorted(early_indices | {
            last_index // 2, (3 * last_index) // 4, last_index,
        }))
    elif prediction_mode == BLOCK_TAYLOR_PREDICTION_MODE:
        # The exact-compute oracle found only the first block sufficiently
        # smooth to justify an approximation trial.
        block_taylor_indices = (0,)
    elif prediction_mode in FIRST_BLOCK_CACHE_MODES:
        # Every block participates: block 0 decides, the rest are skippable.
        first_block_indices = tuple(range(len(blocks)))

    patched = model.clone()
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {})
    if "easycache" in transformer_options:
        raise ValueError("Remove EasyCache before adding H3Cache")
    if STATE_KEY in transformer_options:
        raise ValueError("H3Cache is already attached to this model")

    transformer_options[STATE_KEY] = H3CacheHolder(
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        max_consecutive_skips=max_consecutive_skips,
        audio_weight=audio_weight,
        cached_suffix_blocks=cached_suffix_blocks,
        prediction_mode=prediction_mode,
        verbose=verbose,
        block_oracle_indices=block_oracle_indices,
        block_taylor_indices=block_taylor_indices,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY,
        h3cache_sample_wrapper,
    )
    if prediction_mode in DENOISED_PREDICTION_MODES:
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            WRAPPER_KEY,
            h3cache_apply_model_wrapper,
        )
    else:
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            h3cache_forward_wrapper,
        )
    if cached_suffix_blocks:
        probe_index = len(blocks) - cached_suffix_blocks
        last_index = len(blocks) - 1
        existing = transformer_options.get(
            "patches_replace", {}).get("dit", {})
        conflicts = [
            i for i in range(probe_index, last_index + 1)
            if ("double_block", i) in existing
        ]
        if conflicts:
            raise ValueError(
                "H3Cache suffix conflicts with existing DiT block patches: "
                + ", ".join(map(str, conflicts)))
        for index in range(probe_index, last_index + 1):
            patched.set_model_patch_replace(
                _make_suffix_block_wrapper(index, probe_index, last_index),
                "dit", "double_block", index,
            )
    elif block_oracle_indices:
        existing = transformer_options.get(
            "patches_replace", {}).get("dit", {})
        conflicts = [
            index for index in block_oracle_indices
            if ("double_block", index) in existing
        ]
        if conflicts:
            raise ValueError(
                "H3Cache block oracle conflicts with existing DiT block "
                "patches: " + ", ".join(map(str, conflicts)))
        for index in block_oracle_indices:
            patched.set_model_patch_replace(
                _make_block_oracle_wrapper(index),
                "dit", "double_block", index,
            )
    elif block_taylor_indices:
        existing = transformer_options.get(
            "patches_replace", {}).get("dit", {})
        conflicts = [
            index for index in block_taylor_indices
            if ("double_block", index) in existing
        ]
        if conflicts:
            raise ValueError(
                "H3Cache block Taylor conflicts with existing DiT block "
                "patches: " + ", ".join(map(str, conflicts)))
        for index in block_taylor_indices:
            patched.set_model_patch_replace(
                _make_block_taylor_wrapper(index),
                "dit", "double_block", index,
            )
    elif first_block_indices:
        existing = transformer_options.get(
            "patches_replace", {}).get("dit", {})
        conflicts = [
            index for index in first_block_indices
            if (("double_block", index) in existing
                and not getattr(existing[("double_block", index)],
                                "_asfp8_h3_structure", False))
        ]
        if conflicts:
            raise ValueError(
                "H3Cache FirstBlockCache conflicts with existing DiT block "
                "patches: " + ", ".join(map(str, conflicts)))
        last_index = first_block_indices[-1]
        for index in first_block_indices:
            previous = existing.get(("double_block", index))
            patched.set_model_patch_replace(
                _make_first_block_wrapper(index, last_index, previous),
                "dit", "double_block", index,
            )
    return patched
