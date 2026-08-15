"""CPU tests for H3 structural reduction/ranking policy."""

import math

import pytest
import torch

from _patches import minimax_h3_structure as structure


def _rope(angles):
    angles = torch.tensor(angles, dtype=torch.float32).view(1, -1, 1, 1)
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, -s, s, c), dim=-1).reshape(
        1, len(angles.flatten()), 1, 1, 2, 2)


def test_horizontal_pool_preserves_odd_tail():
    video = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    pooled = structure._pool_video(video, frames=1, height=1, width=5)
    expected = torch.stack((
        (video[0] + video[1]) * 0.5,
        (video[2] + video[3]) * 0.5,
        video[4],
    ))
    torch.testing.assert_close(pooled, expected)


def test_rope_pool_uses_pair_midpoint_angle():
    rope = _rope([0.0, 0.2, 0.4, 0.6])
    reduced = structure._pool_rope(
        rope, video_start=0, frames=1, height=1, width=4)
    assert reduced.shape[1] == 2
    got_cos = reduced[0, :, 0, 0, 0, 0]
    got_sin = reduced[0, :, 0, 0, 1, 0]
    torch.testing.assert_close(got_cos, torch.tensor([math.cos(0.1), math.cos(0.5)]))
    torch.testing.assert_close(got_sin, torch.tensor([math.sin(0.1), math.sin(0.5)]))


def test_reduction_expansion_retains_original_pair_detail():
    holder = structure.H3StructureHolder(
        reduction_scale=1.0, early_evaluations=10,
        begin_block=4, detail_restore_block=30, coarse_restore_block=40)
    holder.evaluations = 1
    holder.current_geometry = (1, 1, 4, 1)
    hidden = torch.tensor([
        [10.0, 20.0], [30.0, 40.0],
        [1.0, 2.0], [3.0, 6.0], [8.0, 9.0], [12.0, 15.0],
    ])
    args = {
        "img": hidden,
        "rope_freqs": _rope([0, 0, 0, 0, 0, 0]),
        "mod_segments": [(0, 2, 0), (2, 6, 1)],
        "transformer_options": {},
    }
    reduced = holder.enter_reduction(args)
    assert reduced["img"].shape == (4, 2)
    torch.testing.assert_close(
        reduced["img"][2:], torch.tensor([[2.0, 4.0], [10.0, 12.0]]))
    reduced["img"][2:].add_(1.0)

    restored = holder.leave_reduction(reduced, index=40)
    torch.testing.assert_close(restored["img"][:2], hidden[:2])
    torch.testing.assert_close(
        restored["img"][2:], hidden[2:] + 1.0)
    assert restored["mod_segments"] == args["mod_segments"]
    assert holder.reduction is None


def test_reduced_rope_and_segments_persist_across_outer_block_loop():
    """The model carries hidden between blocks but recreates all static args."""
    holder = structure.H3StructureHolder(
        reduction_scale=1.0, early_evaluations=10,
        begin_block=4, detail_restore_block=30, coarse_restore_block=40)
    holder.evaluations = 1
    holder.current_geometry = (1, 1, 4, 1)
    options = {structure.STATE_KEY: holder}
    full_rope = _rope([0, 0, 0, 0, 0, 0])
    full_segments = [(0, 2, 0), (2, 6, 1)]
    seen = []
    wrappers = [
        structure._make_block_wrapper(index, object(), None)
        for index in range(structure.BLOCK_COUNT)
    ]
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    for index, wrapper in enumerate(wrappers):
        def block(args, index=index):
            seen.append((
                index, args["img"].shape[0], args["rope_freqs"].shape[1],
                args["mod_segments"][-1][1]))
            return {"img": args["img"]}

        # This mirrors MiniMaxH3Model._forward: only hidden is carried; the
        # original static tables are put into every new block argument dict.
        hidden = wrapper({
            "img": hidden,
            "rope_freqs": full_rope,
            "mod_segments": full_segments,
            "transformer_options": options,
        }, {"original_block": block})["img"]

    assert seen[3] == (3, 6, 6, 6)
    assert seen[4] == (4, 4, 4, 4)
    assert seen[5] == (5, 4, 4, 4)
    assert seen[39] == (39, 4, 4, 4)
    assert seen[40] == (40, 6, 6, 6)


def test_layer_thinning_state_is_absent():
    holder = structure.H3StructureHolder(
        reduction_scale=1.0, early_evaluations=10,
        begin_block=4, detail_restore_block=30, coarse_restore_block=40)
    assert not hasattr(holder, "active_layers")
    assert not hasattr(holder, "gate_samples")
    assert not hasattr(holder, "skipped_blocks")


def test_legacy_combined_workflow_ignores_layer_count(monkeypatch, caplog):
    captured = {}

    def apply_token(model, *args):
        captured["call"] = (model, args)
        return "token-only"

    monkeypatch.setattr(structure, "apply_h3_token_reduction", apply_token)
    result = structure.apply_h3_structure(
        "model", True, 45, 1.0, 1, 10, 30, 40, False)
    assert result == "token-only"
    assert captured["call"] == ("model", (1.0, 10, 30, 40, False))
    assert "layer thinning was removed" in caplog.text


def test_legacy_layer_thinning_only_workflow_fails_explicitly():
    with pytest.raises(ValueError, match="layer thinning was removed"):
        structure.apply_h3_structure(
            object(), False, 45, 1.0, 1, 10, 30, 40, False)
