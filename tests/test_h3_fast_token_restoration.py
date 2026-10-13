"""Fast must restore spatial detail before H3's late structure-sensitive blocks."""
import pytest
import torch

from _patches import minimax_h3_presets as presets
from _patches import minimax_h3_structure as structure


@pytest.mark.parametrize("evaluation", [1, 10, 11, 20])
def test_fast_restores_full_tokens_and_metadata_before_late_blocks(evaluation):
    cfg = presets.resolve_presets("Off", "Off", "Fast · experimental")
    holder = structure.H3StructureHolder(
        cfg["token_reduction_scale"], cfg["token_early_evaluations"], 4,
        cfg["token_detail_restore_block"], cfg["token_coarse_restore_block"])
    holder.evaluations = evaluation
    holder.current_geometry = (2, 3, 5, 1)  # odd width exercises the unpaired tail
    options = {structure.STATE_KEY: holder}
    hidden = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4)
    original = hidden.clone()
    rope = torch.eye(2).reshape(1, 1, 1, 1, 2, 2).repeat(1, 32, 1, 1, 1, 1)
    segments = [(0, 2, 0), (2, 32, 1)]
    reduced = False
    for index in range(structure.BLOCK_COUNT):
        def block(args):
            nonlocal reduced
            if args["img"].shape[0] < 32:
                reduced = True
            if index >= 33:
                assert args["img"].shape[0] == 32
                assert args["rope_freqs"] is rope
                assert args["mod_segments"] is segments
            return {"img": args["img"]}

        wrapper = structure._make_block_wrapper(index, object(), None)
        hidden = wrapper({"img": hidden, "rope_freqs": rope,
                          "mod_segments": segments, "transformer_options": options},
                         {"original_block": block})["img"]
    assert reduced
    assert holder.reduction is None
    assert torch.equal(hidden, original)
