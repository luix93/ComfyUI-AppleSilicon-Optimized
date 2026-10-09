"""Focused CPU tests for the MiniMax H3 audio/video cache controller."""

import logging

import pytest
import torch

from _patches import minimax_h3cache as cache


def _options(sigma=0.5):
    return {
        "sigmas": torch.tensor([sigma]),
        "uuids": ["condition-a"],
    }


def _inputs(video_value=1.0, audio_value=1.0, video_shape=(1, 2, 3, 8, 8)):
    return [
        torch.full(video_shape, video_value),
        torch.full((1, 2, 24), audio_value),
    ]


class _Executor:
    def __init__(self, video_residual=2.0, audio_residual=3.0):
        self.video_residual = video_residual
        self.audio_residual = audio_residual
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        video, audio = args[0]
        return [
            video + self.video_residual,
            audio + self.audio_residual,
        ]


def _holder(
    threshold=1.0,
    max_skips=1,
    audio_weight=1.0,
    cached_suffix_blocks=0,
    prediction_mode="velocity_hold",
    verbose=False,
):
    holder = cache.H3CacheHolder(
        threshold,
        start_percent=0.2,
        end_percent=0.85,
        max_consecutive_skips=max_skips,
        video_subsample=2,
        audio_subsample=2,
        audio_weight=audio_weight,
        cached_suffix_blocks=cached_suffix_blocks,
        prediction_mode=prediction_mode,
        verbose=verbose,
    )
    holder.set_sigma_window(0.8, 0.2)
    return holder


def _run(holder, executor, inputs, options):
    return holder.run_forward(
        executor,
        (inputs, torch.zeros(1), torch.zeros(1), options),
        {},
        options,
    )


def _pack(inputs):
    video, audio = inputs
    return torch.cat([
        video.reshape(video.shape[0], 1, -1),
        audio.reshape(audio.shape[0], 1, -1),
    ], dim=-1)


def _run_denoised(holder, executor, inputs, options):
    packed = _pack(inputs)
    latent_shapes = [inputs[0].shape, inputs[1].shape]
    return holder.run_apply_model(
        executor,
        (packed, options["sigmas"]),
        {"transformer_options": options, "latent_shapes": latent_shapes},
        options,
    )


def test_two_refreshes_calibrate_then_previous_velocity_is_reused():
    holder = _holder(threshold=1.0, prediction_mode="output_hold")
    executor = _Executor()
    options = _options(0.7)

    first = _run(holder, executor, _inputs(1.0, 1.0), options)
    assert torch.equal(first[0], _inputs(3.0, 4.0)[0])

    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)

    options["sigmas"] = torch.tensor([0.5])
    third_inputs = _inputs(1.2, 1.2)
    third = _run(holder, executor, third_inputs, options)

    assert executor.calls == 2
    assert torch.allclose(third[0], _inputs(3.1, 4.1)[0])
    assert torch.allclose(third[1], _inputs(3.1, 4.1)[1])
    assert holder.totals() == (3, 2, 1)


def test_velocity_hold_corrects_audio_schedule_slope_only():
    holder = _holder(threshold=1.0, prediction_mode="velocity_hold")
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    second = _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    third = _run(holder, executor, _inputs(1.2, 1.2), options)

    ratio = cache._time_shift_slope(0.5) / cache._time_shift_slope(
        float(torch.tensor(0.6).item()))
    assert torch.equal(third[0], second[0])
    assert torch.allclose(third[1], second[1] * ratio)


def test_residual_carry_predictor_remains_available_for_comparison():
    holder = _holder(threshold=1.0, prediction_mode="residual_carry")
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    third_inputs = _inputs(1.2, 1.2)
    third = _run(holder, executor, third_inputs, options)

    assert torch.allclose(third[0], third_inputs[0] + 2.0)
    assert torch.allclose(third[1], third_inputs[1] + 3.0)


def test_secant_velocity_learns_local_input_to_velocity_direction():
    holder = _holder(threshold=1.0, prediction_mode="secant_velocity")
    executor = _Executor()
    options = _options(0.7)
    options["minimax_h3_sigma_shift_video"] = 1.0
    options["minimax_h3_sigma_shift_audio"] = 1.0

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    third = _run(holder, executor, _inputs(1.2, 1.2), options)

    assert executor.calls == 2
    assert torch.allclose(third[0], _inputs(3.2, 4.2)[0])
    assert torch.allclose(third[1], _inputs(3.2, 4.2)[1])


def test_denoised_hold_reuses_sampler_facing_output():
    holder = _holder(threshold=1.0, prediction_mode="denoised_hold")
    calls = []

    def executor(packed, *args, **kwargs):
        calls.append(packed.clone())
        return packed * 2.0

    options = _options(0.7)
    _run_denoised(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    second = _run_denoised(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    third = _run_denoised(holder, executor, _inputs(1.2, 1.2), options)

    assert len(calls) == 2
    assert torch.equal(third, second)
    assert holder.totals() == (3, 2, 1)


def test_denoised_linear_extrapolates_across_sigma():
    holder = _holder(threshold=1.0, prediction_mode="denoised_linear")
    calls = []

    def executor(packed, *args, **kwargs):
        calls.append(packed.clone())
        sigma = float(kwargs["transformer_options"]["sigmas"][0])
        return torch.full_like(packed, sigma * 10.0)

    options = _options(0.7)
    _run_denoised(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run_denoised(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    third = _run_denoised(holder, executor, _inputs(1.2, 1.2), options)

    assert len(calls) == 2
    assert torch.allclose(third, torch.full_like(third, 5.0))
    assert holder.totals() == (3, 2, 1)


def test_suffix_cache_reuses_only_the_selected_block_tail():
    holder = _holder(threshold=1.0, cached_suffix_blocks=2)
    options = _options(0.7)
    holder.suffix_layouts[("uuid", "condition-a")] = (2, 4)
    calls = []

    def block(args):
        calls.append(args["img"].clone())
        args["img"].add_(1.0)
        return {"img": args["img"]}

    def evaluate(value):
        args = {"img": torch.full((8, 8), value),
                "transformer_options": options}
        first = holder.run_suffix_block(0, 0, 1, args, block)
        second_args = {"img": first["img"],
                       "transformer_options": options}
        return holder.run_suffix_block(1, 0, 1, second_args, block)["img"]

    assert torch.equal(evaluate(1.0), torch.full((8, 8), 3.0))
    options["sigmas"] = torch.tensor([0.6])
    assert torch.equal(evaluate(1.1), torch.full((8, 8), 3.1))

    assert len(calls) == 2
    assert holder.totals() == (2, 1, 1)


def test_block_oracle_scores_residual_history_and_returns_exact_output():
    holder = _holder(
        threshold=0.0, prediction_mode=cache.BLOCK_ORACLE_PREDICTION_MODE)
    options = _options(0.7)
    holder.suffix_layouts[("uuid", "condition-a")] = (2, 4)
    calls = 0

    def block(args):
        nonlocal calls
        calls += 1
        return {"img": args["img"] + float(calls)}

    for expected in range(1, 5):
        hidden = torch.zeros(8, 8)
        result = holder.run_block_oracle(
            12, {"img": hidden, "transformer_options": options}, block)
        assert torch.equal(result["img"], torch.full((8, 8), float(expected)))

    state = holder.block_oracle_states[(("uuid", "condition-a"), 12)]
    assert state.evaluations == 4
    assert state.stats["residual_hold"].count == 3
    assert state.stats["residual_taylor_1"].count == 2
    assert state.stats["residual_taylor_1"].video_nrmse_max < 1.0e-6
    assert state.stats["residual_projection_ceiling_1"].count == 2
    assert state.stats["residual_projection_ceiling_2"].count == 1


def test_block_taylor_predicts_after_two_consecutive_exact_residuals():
    holder = _holder(
        threshold=1.0, prediction_mode=cache.BLOCK_TAYLOR_PREDICTION_MODE)
    options = _options(0.7)
    calls = 0

    def block(args):
        nonlocal calls
        calls += 1
        # Mutate in place, matching H3's block contract and exercising the
        # owned pre-block copy used to measure the exact residual.
        args["img"].add_(float(calls))
        return {"img": args["img"]}

    outputs = []
    for sigma in (0.7, 0.6, 0.5):
        options["sigmas"] = torch.tensor([sigma])
        outputs.append(holder.run_block_taylor(
            0,
            {"img": torch.zeros(8, 8), "transformer_options": options},
            block,
        )["img"])

    assert calls == 2
    assert torch.equal(outputs[0], torch.full((8, 8), 1.0))
    assert torch.equal(outputs[1], torch.full((8, 8), 2.0))
    assert torch.equal(outputs[2], torch.full((8, 8), 3.0))
    state = holder.block_taylor_states[(('uuid', 'condition-a'), 0)]
    assert (state.evaluations, state.refreshes, state.skips) == (3, 2, 1)
    assert holder.block_taylor_totals() == (3, 2, 1)


def test_block_taylor_forces_exact_recovery_after_a_skip():
    holder = _holder(
        threshold=1.0, max_skips=2,
        prediction_mode=cache.BLOCK_TAYLOR_PREDICTION_MODE)
    options = _options(0.7)
    calls = 0

    def block(args):
        nonlocal calls
        calls += 1
        return {"img": args["img"] + float(calls)}

    for sigma in (0.7, 0.6, 0.5, 0.4):
        options["sigmas"] = torch.tensor([sigma])
        holder.run_block_taylor(
            0,
            {"img": torch.zeros(8, 8), "transformer_options": options},
            block,
        )

    # The third evaluation is predicted; the nonconsecutive exact history
    # forces the fourth evaluation to recover even when max_skips permits two.
    assert calls == 3
    assert holder.block_taylor_totals() == (4, 3, 1)


def test_audio_change_vetoes_reuse_when_video_is_stable():
    holder = _holder(threshold=0.05)
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    result = _run(holder, executor, _inputs(1.1, 2.0), options)

    assert executor.calls == 3
    assert torch.allclose(result[1], _inputs(1.1, 5.0)[1])
    assert holder.totals() == (3, 3, 0)
    state = next(iter(holder.streams.values()))
    assert state.last_estimated_audio_change > holder.reuse_threshold


def test_max_consecutive_skip_forces_refresh():
    holder = _holder(threshold=1.0, max_skips=1)
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    _run(holder, executor, _inputs(1.2, 1.2), options)
    options["sigmas"] = torch.tensor([0.4])
    _run(holder, executor, _inputs(1.3, 1.3), options)

    assert executor.calls == 3
    assert holder.totals() == (4, 3, 1)


def test_threshold_zero_is_exact_calibration_only():
    holder = _holder(threshold=0.0)
    executor = _Executor()
    options = _options(0.7)

    for sigma, value in ((0.7, 1.0), (0.6, 1.1), (0.5, 1.2)):
        options["sigmas"] = torch.tensor([sigma])
        _run(holder, executor, _inputs(value, value), options)

    assert executor.calls == 3
    assert holder.totals() == (3, 3, 0)
    state = next(iter(holder.streams.values()))
    assert state.video_transform_rate is not None
    assert state.audio_transform_rate is not None


def test_prediction_metrics_report_relative_error_and_cosine():
    actual = torch.tensor([1.0, 2.0, 3.0])
    nrmse, cosine = cache._prediction_metrics(actual.clone(), actual)
    assert nrmse == 0.0
    assert torch.isclose(torch.tensor(cosine), torch.tensor(1.0))


def test_projection_ceiling_finds_exact_scalar_direction():
    base = torch.tensor([1.0, 2.0, 3.0])
    direction = torch.tensor([2.0, -1.0, 0.5])
    result = cache._projection_ceiling(
        base, base + 1.5 * direction, [direction])
    assert result is not None
    prediction, coefficients = result
    assert torch.allclose(prediction, base + 1.5 * direction)
    assert abs(coefficients[0] - 1.5) < 1.0e-6


def test_oracle_scores_all_raw_predictors_without_skipping(caplog):
    holder = _holder(
        threshold=1.0, prediction_mode=cache.ORACLE_PREDICTION_MODE)
    executor = _Executor()
    options = _options(0.7)
    options["minimax_h3_sigma_shift_video"] = 1.0
    options["minimax_h3_sigma_shift_audio"] = 1.0

    with caplog.at_level(logging.INFO):
        for sigma, value in (
            (0.7, 1.0), (0.6, 1.1), (0.5, 1.2), (0.4, 1.3)
        ):
            options["sigmas"] = torch.tensor([sigma])
            result = _run(holder, executor, _inputs(value, value), options)
            assert torch.equal(result[0], _inputs(value + 2.0, value + 3.0)[0])
            assert torch.equal(result[1], _inputs(value + 2.0, value + 3.0)[1])

    assert executor.calls == 4
    assert holder.totals() == (4, 4, 0)
    stats = next(iter(holder.streams.values())).oracle_stats
    assert set(stats) == cache.ORACLE_DIAGNOSTIC_MODES
    assert stats["residual_carry"].count == 3
    assert stats["residual_carry"].video_nrmse_max < 1.0e-6
    assert stats["residual_carry"].audio_nrmse_max < 1.0e-6
    assert stats["secant_velocity"].count == 2
    assert stats["taylor_step_1"].count == 2
    assert stats["taylor_step_1"].video_nrmse_max < 1.0e-6
    assert stats["taylor_sigma_1"].audio_nrmse_max < 1.0e-6
    assert stats["taylor_step_2"].count == 1
    assert stats["taylor_sigma_2"].count == 1
    assert stats["projection_ceiling_1"].count == 2
    assert stats["projection_ceiling_1"].video_nrmse_max < 1.0e-6
    assert stats["projection_ceiling_2"].count == 1
    assert any(
        "oracle eval=4" in record.getMessage()
        and "predictor=secant_velocity" in record.getMessage()
        for record in caplog.records
    )


def test_verbose_decision_log_formats_all_metrics(caplog):
    holder = _holder(threshold=0.0, verbose=True)

    with caplog.at_level(logging.INFO):
        _run(holder, _Executor(), _inputs(), _options(0.7))

    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(cache.TAG)
    )
    assert "estimate_video=" in message
    assert "estimate_audio=" in message
    assert "action=refresh" in message


def test_protected_end_window_never_skips():
    holder = _holder(threshold=1.0)
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.1])
    _run(holder, executor, _inputs(1.2, 1.2), options)

    assert executor.calls == 3
    assert holder.totals() == (3, 3, 0)


def test_shape_change_resets_state_and_refreshes():
    holder = _holder(threshold=1.0)
    executor = _Executor()
    options = _options(0.7)

    _run(holder, executor, _inputs(1.0, 1.0), options)
    options["sigmas"] = torch.tensor([0.6])
    _run(holder, executor, _inputs(1.1, 1.1), options)
    options["sigmas"] = torch.tensor([0.5])
    _run(holder, executor, _inputs(
        1.2, 1.2, video_shape=(1, 2, 3, 10, 8)), options)

    assert executor.calls == 3
    assert holder.totals() == (1, 1, 0)


def test_unsupported_input_falls_through_without_state():
    holder = _holder()
    options = _options()
    tensor = torch.ones(1)
    calls = []

    def passthrough(*args, **kwargs):
        calls.append((args, kwargs))
        return "fallback"

    result = holder.run_forward(passthrough, (tensor, options), {}, options)

    assert len(calls) == 1
    assert holder.streams == {}
    assert result == "fallback"


def test_streams_do_not_share_cached_residuals():
    holder = _holder(threshold=1.0)
    executor = _Executor()
    options_a = _options(0.7)
    options_b = {"sigmas": torch.tensor([0.7]), "uuids": ["condition-b"]}

    _run(holder, executor, _inputs(), options_a)
    _run(holder, executor, _inputs(), options_b)

    assert len(holder.streams) == 2
    assert holder.totals() == (2, 2, 0)


def test_reset_releases_all_stream_tensors():
    holder = _holder()
    _run(holder, _Executor(), _inputs(), _options())
    assert holder.streams
    holder.reset()
    assert holder.streams == {}


def test_wrapper_locates_options_and_uses_holder():
    holder = _holder(threshold=0.0)
    options = _options()
    options[cache.STATE_KEY] = holder
    executor = _Executor()

    result = cache.h3cache_forward_wrapper(
        executor, _inputs(), torch.zeros(1), torch.zeros(1), options)

    assert executor.calls == 1
    assert torch.all(result[0] == 3.0)
    assert holder.totals() == (1, 1, 0)


# --- FirstBlockCache --------------------------------------------------------
#
# Block 0 runs exactly on every evaluation and its residual decides whether the
# cached aggregate residual of blocks 1..N-1 can stand in for running them.

_AUDIO_ROWS = 4
_VIDEO_ROWS = 6
_SEQ = 12          # 2 prefix rows + 4 audio + 6 video
_WIDTH = 3
_STREAM = ("uuid", "condition-a")


def _first_holder(
    threshold=1.0,
    max_skips=1,
    audio_weight=1.0,
    verbose=False,
    prediction_mode=cache.FIRST_BLOCK_CACHE_MODE,
    video_subsample=1,
):
    holder = cache.H3CacheHolder(
        threshold,
        start_percent=0.2,
        end_percent=0.85,
        max_consecutive_skips=max_skips,
        video_subsample=video_subsample,
        audio_subsample=1,
        audio_weight=audio_weight,
        cached_suffix_blocks=0,
        prediction_mode=prediction_mode,
        verbose=verbose,
    )
    holder.set_sigma_window(0.8, 0.2)
    holder.suffix_layouts[_STREAM] = (_AUDIO_ROWS, _VIDEO_ROWS)
    return holder


class _FakeBlocks:
    """N fake DiT blocks that mutate their stream in place, as H3's do."""

    def __init__(self, count, deltas=None):
        self.count = count
        self.deltas = list(deltas) if deltas is not None else [
            float(i + 1) for i in range(count)]
        self.calls = []

    def block(self, index):
        def run(args):
            self.calls.append(index)
            args["img"].add_(self.deltas[index])
            return {"img": args["img"]}
        return run


class _ChangingHeadBlocks(_FakeBlocks):
    """Fake blocks whose block-0 residual changes on each evaluation."""

    def __init__(self, head_deltas):
        super().__init__(3, deltas=[0.0, 2.0, 3.0])
        self.head_deltas = list(head_deltas)
        self.head_index = 0

    def block(self, index):
        if index != 0:
            return super().block(index)

        def run(args):
            self.calls.append(index)
            delta = self.head_deltas[self.head_index]
            self.head_index += 1
            args["img"].add_(delta)
            return {"img": args["img"]}

        return run


def _drive_blocks(holder, blocks, hidden, options):
    last = blocks.count - 1
    current = hidden
    for index in range(blocks.count):
        current = holder.run_first_block(
            index, last,
            {"img": current, "transformer_options": options},
            blocks.block(index),
        )["img"]
    return current


def _hidden(value=0.0):
    return torch.full((_SEQ, _WIDTH), float(value))


def test_first_block_threshold_zero_runs_every_block_exactly():
    holder = _first_holder(threshold=0.0)
    options = _options()
    blocks = _FakeBlocks(4)
    for _ in range(3):
        result = _drive_blocks(holder, blocks, _hidden(), options)
        assert torch.all(result == sum(blocks.deltas))
    assert blocks.calls == [0, 1, 2, 3] * 3
    evaluations, refreshes, skips, _ = holder.first_block_totals()
    assert (evaluations, refreshes, skips) == (3, 3, 0)


def test_first_block_hit_runs_block_zero_and_skips_the_tail():
    holder = _first_holder(threshold=1.0)
    options = _options()
    blocks = _FakeBlocks(4)

    _drive_blocks(holder, blocks, _hidden(), options)     # cold refresh
    blocks.calls.clear()
    result = _drive_blocks(holder, blocks, _hidden(), options)

    assert blocks.calls == [0]                            # tail never ran
    assert torch.all(result == sum(blocks.deltas))        # but the value is right
    _, refreshes, skips, peak = holder.first_block_totals()
    assert (refreshes, skips) == (1, 1)
    assert peak > 0


def test_first_block_wrapper_composes_structure_when_applied_after_it():
    holder = _first_holder(threshold=1.0)
    options = _options()
    options[cache.STATE_KEY] = holder

    class Structure:
        def __init__(self):
            self.skipped = 0

        def advance_sparse_for_skipped_block(self, _options):
            self.skipped += 1

    structure = Structure()
    options["asfp8_h3_structure"] = structure
    calls = []
    wrappers = []
    for index in range(3):
        def previous(args, extra_options, index=index):
            calls.append(("structure", index))
            return extra_options["original_block"](args)

        previous._asfp8_h3_structure = True
        wrappers.append(cache._make_first_block_wrapper(index, 2, previous))

    def evaluate():
        hidden = _hidden()
        for index, wrapper in enumerate(wrappers):
            def block(args, index=index):
                calls.append(("block", index))
                args["img"].add_(index + 1)
                return {"img": args["img"]}

            hidden = wrapper(
                {"img": hidden, "transformer_options": options},
                {"original_block": block})["img"]
        return hidden

    assert torch.all(evaluate() == 6)
    calls.clear()
    assert torch.all(evaluate() == 6)
    assert calls == [("structure", 0), ("block", 0)]
    assert structure.skipped == 2


def test_first_block_state_survives_structural_interior_sequence_change():
    holder = _first_holder(threshold=1.0)
    options = _options()
    full = _hidden()

    # Block 0 establishes the detector and saves its full-sequence output.
    head = holder.run_first_block(
        0, 2, {"img": full, "transformer_options": options},
        _FakeBlocks(1, deltas=[1.0]).block(0))["img"]
    state = holder.first_block_states[_STREAM]

    # Structural acceleration reduces only the interior of the stack.
    reduced = head[:8].clone()
    holder.run_first_block(
        1, 2, {"img": reduced, "transformer_options": options},
        _FakeBlocks(2).block(1))
    assert holder.first_block_states[_STREAM] is state
    assert state.saved_head is not None

    # Its restore boundary returns to the full layout before the last block, so
    # the aggregate full-stack tail can still be learned and reused.
    restored = head.clone().add_(2.0)
    result = holder.run_first_block(
        2, 2, {"img": restored, "transformer_options": options},
        _FakeBlocks(3).block(2))["img"]
    assert holder.first_block_states[_STREAM] is state
    assert state.saved_head is None
    torch.testing.assert_close(state.cached_tail, result - head)


def test_first_block_hit_reuses_the_cached_tail_not_a_fresh_one():
    """The whole point: a hit must NOT reflect changes in blocks 1..N-1."""
    holder = _first_holder(threshold=1.0)
    options = _options()
    blocks = _FakeBlocks(3, deltas=[1.0, 2.0, 3.0])

    _drive_blocks(holder, blocks, _hidden(), options)     # caches tail = 5.0
    blocks.deltas = [1.0, 100.0, 100.0]                   # tail would now be 200
    result = _drive_blocks(holder, blocks, _hidden(), options)

    assert blocks.calls == [0, 1, 2, 0]
    assert torch.all(result == 1.0 + 5.0)                 # stale tail, as designed


def test_first_block_max_consecutive_skips_forces_a_refresh():
    holder = _first_holder(threshold=10.0, max_skips=1)
    options = _options()
    blocks = _FakeBlocks(3)
    for _ in range(4):
        _drive_blocks(holder, blocks, _hidden(), options)
    _, refreshes, skips, _ = holder.first_block_totals()
    assert skips == 2 and refreshes == 2                  # strict alternation


def test_first_block_per_step_allows_five_then_forces_a_refresh():
    holder = _first_holder(
        threshold=1.0,
        max_skips=5,
        prediction_mode=cache.FIRST_BLOCK_CACHE_PER_STEP_MODE,
    )
    options = _options()
    blocks = _FakeBlocks(3)
    for _ in range(7):
        _drive_blocks(holder, blocks, _hidden(), options)
    _, refreshes, skips, _ = holder.first_block_totals()
    assert (refreshes, skips) == (2, 5)


def test_first_block_per_step_does_not_accumulate_small_changes():
    head_deltas = [1.0, 1.06, 1.12]
    cumulative = _first_holder(threshold=0.10, max_skips=5)
    per_step = _first_holder(
        threshold=0.10,
        max_skips=5,
        prediction_mode=cache.FIRST_BLOCK_CACHE_PER_STEP_MODE,
        video_subsample=8,
    )
    options = _options()

    for holder in (cumulative, per_step):
        blocks = _ChangingHeadBlocks(head_deltas)
        for _ in range(3):
            _drive_blocks(holder, blocks, _hidden(), options)

    assert cumulative.first_block_totals()[:3] == (3, 2, 1)
    assert per_step.first_block_totals()[:3] == (3, 1, 2)
    # Discord mode overrides the configured sample factor and measures every
    # target feature for both video and audio.
    assert per_step.first_block_states[_STREAM].metadata[-1] == 1


def test_first_block_audio_change_alone_vetoes_reuse():
    holder = _first_holder(threshold=0.05, audio_weight=1.0)
    options = _options()
    video_start = _SEQ - _VIDEO_ROWS
    audio_start = video_start - _AUDIO_ROWS

    delta = torch.zeros(_SEQ, _WIDTH)
    delta[video_start:] = 1.0
    delta[audio_start:video_start] = 1.0

    class _AudioBlocks(_FakeBlocks):
        def block(self, index):
            def run(args):
                self.calls.append(index)
                args["img"].add_(delta if index == 0 else self.deltas[index])
                return {"img": args["img"]}
            return run

    blocks = _AudioBlocks(3)
    _drive_blocks(holder, blocks, _hidden(), options)
    _drive_blocks(holder, blocks, _hidden(), options)
    _, _, skips_before, _ = holder.first_block_totals()
    assert skips_before == 1                               # stable: reuse allowed

    delta[audio_start:video_start] = 50.0                  # audio only moves
    blocks.calls.clear()
    _drive_blocks(holder, blocks, _hidden(), options)
    assert 1 in blocks.calls                               # tail ran: reuse vetoed


def test_first_block_protected_window_never_skips():
    holder = _first_holder(threshold=10.0)
    options = _options(sigma=0.1)                          # below the window
    blocks = _FakeBlocks(3)
    for _ in range(3):
        _drive_blocks(holder, blocks, _hidden(), options)
    _, refreshes, skips, _ = holder.first_block_totals()
    assert (refreshes, skips) == (3, 0)


def test_first_block_in_place_mutation_cannot_corrupt_the_detector():
    """Block 0's own in-place write must not alias the stored samples."""
    holder = _first_holder(threshold=1.0)
    options = _options()
    blocks = _FakeBlocks(3)
    _drive_blocks(holder, blocks, _hidden(), options)
    state = holder.first_block_states[_STREAM]
    before = state.previous_video.clone()
    _drive_blocks(holder, blocks, _hidden(), options)
    # samples are owned copies: the later evaluation's writes changed nothing
    assert torch.equal(before, state.previous_video)


def test_first_block_metadata_change_resets_state():
    holder = _first_holder(threshold=1.0)
    options = _options()
    blocks = _FakeBlocks(3)
    _drive_blocks(holder, blocks, _hidden(), options)
    assert holder.first_block_states[_STREAM].cached_tail is not None

    wider = torch.zeros(_SEQ, _WIDTH + 1)
    blocks.calls.clear()
    _drive_blocks(holder, blocks, wider, options)
    assert blocks.calls == [0, 1, 2]                       # exact after the reset
    assert holder.first_block_states[_STREAM].metadata[0] == (_SEQ, _WIDTH + 1)


def test_first_block_reset_releases_cached_tensors():
    holder = _first_holder(threshold=1.0)
    _drive_blocks(holder, _FakeBlocks(3), _hidden(), _options())
    assert holder.first_block_states
    holder.reset()
    assert holder.first_block_states == {}


def test_first_block_without_a_layout_stays_exact():
    holder = _first_holder(threshold=1.0)
    holder.suffix_layouts.clear()
    blocks = _FakeBlocks(3)
    result = _drive_blocks(holder, blocks, _hidden(), _options())
    assert blocks.calls == [0, 1, 2]
    assert torch.all(result == sum(blocks.deltas))


def test_first_block_mode_leaves_run_forward_exact():
    holder = _first_holder(threshold=1.0)
    executor = _Executor()
    options = _options()
    for _ in range(3):
        _run(holder, executor, _inputs(), options)
    assert executor.calls == 3                             # never skipped there
    assert holder.suffix_layouts                           # layout was recorded


def test_first_block_rejects_suffix_blocks():
    with pytest.raises(ValueError):
        cache.H3CacheHolder(
            1.0, 0.2, 0.85, prediction_mode=cache.FIRST_BLOCK_CACHE_MODE,
            cached_suffix_blocks=-1)


def test_node_exposes_both_first_block_policies_and_five_skips():
    from _patches import minimax_h3_unified
    required = minimax_h3_unified.AppleSiliconH3Acceleration.INPUT_TYPES()["required"]
    modes = required["cache_mode"][0]
    assert cache.FIRST_BLOCK_CACHE_MODE in modes
    assert cache.FIRST_BLOCK_CACHE_PER_STEP_MODE in modes
    assert required["cache_max_consecutive_skips"][1]["max"] >= 5
