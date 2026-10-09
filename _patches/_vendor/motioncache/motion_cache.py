from __future__ import annotations

import logging

import torch

import comfy.ldm.minimax.model
import comfy.model_patcher
import comfy.patcher_extension
from comfy_api.latest import io


PATCH_KEY = "minimax_h3_motioncache"


class MiniMaxH3MotionCacheState:
    def __init__(
        self,
        reuse_threshold,
        motion_strength,
        warmup_steps,
        max_consecutive_skips,
        start_percent,
        end_percent,
        subsample_factor,
        verbose=False,
    ):
        self.reuse_threshold = reuse_threshold
        self.motion_strength = motion_strength
        self.warmup_steps = warmup_steps
        self.max_consecutive_skips = max_consecutive_skips
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.verbose = verbose
        self.start_sigma = 0.0
        self.end_sigma = 0.0
        self.reset()

    def clone(self):
        return MiniMaxH3MotionCacheState(
            self.reuse_threshold,
            self.motion_strength,
            self.warmup_steps,
            self.max_consecutive_skips,
            self.start_percent,
            self.end_percent,
            self.subsample_factor,
            self.verbose,
        )

    def prepare(self, model_sampling):
        self.start_sigma = float(model_sampling.percent_to_sigma(self.start_percent))
        self.end_sigma = float(model_sampling.percent_to_sigma(self.end_percent))
        return self

    def reset(self):
        self.step = 0
        self.total_calls = 0
        self.total_skipped = 0
        self.consecutive_skips = 0
        self.accumulated_change = 0.0
        self.metadata = None
        self.previous_video_input = None
        self.previous_audio_input = None
        self.previous_video_output = None
        self.previous_audio_output = None
        self.video_residual = None
        self.audio_residual = None
        self.motion_weights = None
        self.video_change_rate = None
        self.audio_change_rate = None
        self.video_output_norm = None
        self.audio_output_norm = None

    def check_metadata(self, video, audio, uuids):
        metadata = (
            video.shape,
            video.dtype,
            video.device,
            audio.shape,
            audio.dtype,
            audio.device,
            tuple(uuids),
        )
        if self.metadata is not None and metadata != self.metadata:
            if self.verbose:
                logging.info(
                    "MiniMax H3 MotionCache - input or conditioning changed, resetting cache"
                )
            self.reset()
        self.metadata = metadata

    def active(self, sigmas):
        sigma = float(sigmas.flatten()[0])
        return self.end_sigma < sigma <= self.start_sigma

    def subsample_video(self, video):
        factor = self.subsample_factor
        return video[..., ::factor, ::factor].detach().clone()

    def subsample_audio(self, audio):
        return audio[..., :: self.subsample_factor].detach().clone()

    def build_motion_weights(self, video):
        if video.shape[2] < 2 or self.motion_strength == 0.0:
            return torch.ones(
                (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
                dtype=torch.float32,
                device=video.device,
            )

        frame_delta = (
            (video[:, :, 1:].float() - video[:, :, :-1].float())
            .abs()
            .mean(dim=1, keepdim=True)
        )
        frame_delta = torch.cat((frame_delta[:, :, :1], frame_delta), dim=2)
        mean = frame_delta.mean(dim=(2, 3, 4), keepdim=True)
        normalized = frame_delta / mean.clamp_min(1e-6)
        weights = 1.0 + self.motion_strength * normalized
        return weights / weights.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def mean_change(current, previous, weights=None):
        change = (current.float() - previous.float()).abs()
        if weights is not None:
            change = change * weights
        return float(change.mean())

    @staticmethod
    def mean_norm(value, weights=None):
        norm = value.float().abs()
        if weights is not None:
            norm = norm * weights
        return max(float(norm.mean()), 1e-6)

    def reuse_score(self, video, audio):
        if self.video_change_rate is None or self.audio_change_rate is None:
            return None

        video = self.subsample_video(video)
        audio = self.subsample_audio(audio)
        video_change = self.mean_change(
            video, self.previous_video_input, self.motion_weights
        )
        audio_change = self.mean_change(audio, self.previous_audio_input)
        video_score = self.video_change_rate * video_change / self.video_output_norm
        audio_score = self.audio_change_rate * audio_change / self.audio_output_norm
        return max(video_score, audio_score)

    def should_reuse(self, video, audio, sigmas):
        if self.step <= self.warmup_steps or not self.active(sigmas):
            self.accumulated_change = 0.0
            self.consecutive_skips = 0
            return False, None
        if self.video_residual is None or self.audio_residual is None:
            return False, None

        score = self.reuse_score(video, audio)
        if score is None:
            return False, None
        self.accumulated_change += score
        reuse = (
            self.accumulated_change < self.reuse_threshold
            and self.consecutive_skips < self.max_consecutive_skips
        )
        if reuse:
            self.consecutive_skips += 1
            self.total_skipped += 1
        else:
            self.accumulated_change = 0.0
            self.consecutive_skips = 0
        return reuse, score

    def apply_residual(self, video, audio):
        return [video + self.video_residual, audio + self.audio_residual]

    def update(self, video, audio, output_video, output_audio, sigmas):
        video_input = self.subsample_video(video)
        audio_input = self.subsample_audio(audio)
        video_output = self.subsample_video(output_video)
        audio_output = self.subsample_audio(output_audio)
        sigma = float(sigmas.flatten()[0])
        denoised_video = video_input.float() - video_output.float() * sigma
        motion_weights = self.build_motion_weights(denoised_video)

        if self.previous_video_input is not None:
            video_input_change = self.mean_change(
                video_input, self.previous_video_input, motion_weights
            )
            audio_input_change = self.mean_change(
                audio_input, self.previous_audio_input
            )
            video_output_change = self.mean_change(
                video_output, self.previous_video_output, motion_weights
            )
            audio_output_change = self.mean_change(
                audio_output, self.previous_audio_output
            )
            if video_input_change > 1e-6:
                self.video_change_rate = video_output_change / video_input_change
            if audio_input_change > 1e-6:
                self.audio_change_rate = audio_output_change / audio_input_change

        self.video_output_norm = self.mean_norm(video_output, motion_weights)
        self.audio_output_norm = self.mean_norm(audio_output)
        self.previous_video_input = video_input
        self.previous_audio_input = audio_input
        self.previous_video_output = video_output
        self.previous_audio_output = audio_output
        self.motion_weights = motion_weights
        self.video_residual = (output_video - video).detach().clone()
        self.audio_residual = (output_audio - audio).detach().clone()


def motioncache_forward_wrapper(executor, *args, **kwargs):
    transformer_options = args[-1]
    if not isinstance(transformer_options, dict):
        transformer_options = kwargs.get("transformer_options")
    state: MiniMaxH3MotionCacheState = transformer_options[PATCH_KEY]

    inputs = args[0]
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
        raise ValueError(
            "MiniMax H3 MotionCache requires the H3 [video, audio] latent input"
        )
    video, audio = inputs
    uuids = transformer_options["uuids"]
    sigmas = transformer_options["sigmas"]

    state.check_metadata(video, audio, uuids)
    state.step += 1
    state.total_calls += 1
    reuse, score = state.should_reuse(video, audio, sigmas)
    if reuse:
        if state.verbose:
            logging.info(
                "MiniMax H3 MotionCache - step %d reused, score %.5f, accumulated %.5f",
                state.step,
                score,
                state.accumulated_change,
            )
        return state.apply_residual(video, audio)

    output = executor(*args, **kwargs)
    if not isinstance(output, (list, tuple)) or len(output) != 2:
        raise ValueError("MiniMax H3 MotionCache expected H3 to return [video, audio]")
    state.update(video, audio, output[0], output[1], sigmas)
    if state.verbose:
        logging.info(
            "MiniMax H3 MotionCache - step %d computed%s",
            state.step,
            "" if score is None else f", score {score:.5f}",
        )
    return output


def motioncache_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original_model_options = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(
        original_model_options
    )
    state = (
        guider.model_options["transformer_options"][PATCH_KEY]
        .clone()
        .prepare(guider.model_patcher.model.model_sampling)
    )
    guider.model_options["transformer_options"][PATCH_KEY] = state
    logging.info(
        "MiniMax H3 MotionCache enabled - threshold %.3f, warmup %d, range %.0f%%-%.0f%%",
        state.reuse_threshold,
        state.warmup_steps,
        state.start_percent * 100,
        state.end_percent * 100,
    )
    try:
        return executor(*args, **kwargs)
    finally:
        computed = state.total_calls - state.total_skipped
        speedup = state.total_calls / computed if computed else 1.0
        logging.info(
            "MiniMax H3 MotionCache - skipped %d/%d model calls (estimated %.2fx DiT speedup)",
            state.total_skipped,
            state.total_calls,
            speedup,
        )
        state.reset()
        guider.model_options = original_model_options


class MiniMaxH3MotionCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MotionCache",
            display_name="MiniMax H3 MotionCache",
            category="model/patch/minimax",
            description="Motion-aware residual caching adapted for MiniMax H3 joint video/audio denoising.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "reuse_threshold",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Higher values reuse more model calls. Start at 0.15.",
                ),
                io.Float.Input(
                    "motion_strength",
                    default=1.0,
                    min=0.0,
                    max=4.0,
                    step=0.1,
                    tooltip="How strongly frame differences prioritize moving regions.",
                ),
                io.Int.Input(
                    "warmup_steps", default=4, min=2, max=20, step=1, advanced=True
                ),
                io.Int.Input(
                    "max_consecutive_skips",
                    default=2,
                    min=1,
                    max=10,
                    step=1,
                    advanced=True,
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "subsample_factor", default=8, min=1, max=32, step=1, advanced=True
                ),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        reuse_threshold,
        motion_strength,
        warmup_steps,
        max_consecutive_skips,
        start_percent,
        end_percent,
        subsample_factor,
        verbose,
    ):
        if start_percent >= end_percent:
            raise ValueError(
                "MiniMax H3 MotionCache start_percent must be lower than end_percent"
            )

        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, comfy.ldm.minimax.model.MiniMaxH3Model):
            raise ValueError(
                "MiniMax H3 MotionCache only supports MiniMax H3 diffusion models"
            )

        transformer_options = patched.model_options.setdefault(
            "transformer_options", {}
        )
        if "easycache" in transformer_options:
            raise ValueError("Remove EasyCache before applying MiniMax H3 MotionCache")
        if patched.get_wrappers(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PATCH_KEY
        ):
            return io.NodeOutput(patched)

        transformer_options[PATCH_KEY] = MiniMaxH3MotionCacheState(
            reuse_threshold=float(reuse_threshold),
            motion_strength=float(motion_strength),
            warmup_steps=int(warmup_steps),
            max_consecutive_skips=int(max_consecutive_skips),
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            subsample_factor=int(subsample_factor),
            verbose=bool(verbose),
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            PATCH_KEY,
            motioncache_sample_wrapper,
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PATCH_KEY,
            motioncache_forward_wrapper,
        )
        return io.NodeOutput(patched)
