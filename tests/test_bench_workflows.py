"""Unit tests for benchmark-only workflow transformations."""

import pytest

from dev import bench_workflows as bench


def _prompt():
    return {
        "loader": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3"}},
        "scheduler": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["loader", 0], "steps": 20},
        },
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["loader", 0]},
        },
        "resolution": {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": "16:9", "megapixels": 1.0},
        },
    }


def test_easycache_injection_rewires_model_consumers_without_mutating_input():
    original = _prompt()
    changed = bench.set_easycache(original, (0.3, 0.2, 0.9))

    assert original["scheduler"]["inputs"]["model"] == ["loader", 0]
    assert changed["scheduler"]["inputs"]["model"] == ["asfp8:easycache", 0]
    assert changed["guider"]["inputs"]["model"] == ["asfp8:easycache", 0]
    assert changed["asfp8:easycache"]["inputs"]["model"] == ["loader", 0]
    assert changed["asfp8:easycache"]["inputs"]["reuse_threshold"] == 0.3


def test_megapixel_override_is_copy_on_write():
    original = _prompt()
    changed = bench.set_megapixels(original, 0.4)
    assert original["resolution"]["inputs"]["megapixels"] == 1.0
    assert changed["resolution"]["inputs"]["megapixels"] == 0.4


def test_easycache_requires_exactly_one_loader():
    with pytest.raises(RuntimeError, match="expected one UNETLoader"):
        bench.set_easycache({}, (0.3, 0.2, 0.9))


def test_h3cache_injection_rewires_model_consumers_without_mutating_input():
    original = _prompt()
    changed = bench.set_h3cache(
        original, (0.04, 0.2, 0.85, 1.0), audio_weight=1.5,
        prediction_mode="first_block_cache")

    assert original["scheduler"]["inputs"]["model"] == ["loader", 0]
    assert changed["scheduler"]["inputs"]["model"] == ["asfp8:h3cache", 0]
    assert changed["guider"]["inputs"]["model"] == ["asfp8:h3cache", 0]
    node = changed["asfp8:h3cache"]
    assert node["class_type"] == "AppleSiliconH3UnifiedAcceleration"
    inputs = node["inputs"]
    assert inputs["model"] == ["loader", 0]
    assert inputs["cache_reuse_threshold"] == 0.04
    assert inputs["cache_audio_weight"] == 1.5
    assert inputs["cache_mode"] == "first_block_cache"
    assert inputs["token_reduction"] is False
    assert inputs["sparse_attention"] is False


def test_retired_cache_experiments_fail_before_submission():
    with pytest.raises(ValueError, match="retired cache"):
        bench.set_h3cache(_prompt(), (0.04, 0.2, 0.85, 1), prediction_mode="velocity_hold")



def test_h3cache_requires_exactly_one_loader():
    with pytest.raises(RuntimeError, match="expected one UNETLoader"):
        bench.set_h3cache({}, (0.04, 0.2, 0.85, 1.0))


def test_spectrum_single_injection_isolates_one_forecast_at_20_steps():
    original = _prompt()
    changed = bench.set_spectrum_h3(original, "single")

    assert original["guider"]["inputs"]["model"] == ["loader", 0]
    assert changed["guider"]["inputs"]["model"] == [
        "asfp8:spectrum_h3", 0]
    inputs = changed["asfp8:spectrum_h3"]["inputs"]
    assert inputs["model"] == ["loader", 0]
    assert inputs["warmup_steps"] == 5
    assert inputs["tail_actual_steps"] == 14
    assert inputs["history_storage"] == "vram"


def test_spectrum_exact_forces_all_supported_workflow_steps_actual():
    changed = bench.set_spectrum_h3(_prompt(), "exact")
    inputs = changed["asfp8:spectrum_h3"]["inputs"]
    assert inputs["warmup_steps"] == 64
    assert inputs["flex_window"] == 0.0


def test_spectrum_single_late_injection_isolates_step_15_at_20_steps():
    changed = bench.set_spectrum_h3(_prompt(), "single-late")
    inputs = changed["asfp8:spectrum_h3"]["inputs"]
    assert inputs["warmup_steps"] == 15
    assert inputs["tail_actual_steps"] == 4
    assert inputs["flex_window"] == 0.0


def test_spectrum_requires_one_scheduler_step_count():
    prompt = _prompt()
    del prompt["scheduler"]
    with pytest.raises(RuntimeError, match="BasicScheduler step count"):
        bench.set_spectrum_h3(prompt, "single")


def test_output_artifacts_extracts_nested_history_file_descriptors():
    entry = {
        "outputs": {
            "save-video": {
                "video": [{
                    "filename": "MiniMax_H3_00001.mp4",
                    "subfolder": "video",
                    "type": "output",
                    "format": "video/mp4",
                }],
            },
            "preview": {
                "images": [{
                    "filename": "preview.png",
                    "subfolder": "",
                    "type": "temp",
                }],
            },
        },
    }

    assert bench.output_artifacts(entry) == [
        {
            "node_id": "save-video",
            "filename": "MiniMax_H3_00001.mp4",
            "subfolder": "video",
            "type": "output",
            "format": "video/mp4",
        },
        {
            "node_id": "preview",
            "filename": "preview.png",
            "subfolder": "",
            "type": "temp",
        },
    ]


# --- H3 sparse attention / preset injection ---------------------------------

def test_h3_sparse_injection_rewires_model_consumers_without_mutating_input():
    original = _prompt()
    changed = bench.set_h3_sparse(original, ("corrected", 1.0, 0.2, 2, 8192))

    assert original["scheduler"]["inputs"]["model"] == ["loader", 0]
    assert changed["scheduler"]["inputs"]["model"] == ["asfp8:h3sparse", 0]
    assert changed["guider"]["inputs"]["model"] == ["asfp8:h3sparse", 0]
    node = changed["asfp8:h3sparse"]
    assert node["class_type"] == "AppleSiliconH3UnifiedAcceleration"
    assert node["inputs"]["model"] == ["loader", 0]
    assert node["inputs"]["sparse_mode"] == "corrected"
    assert node["inputs"]["sparse_tau"] == 1.0
    assert node["inputs"]["sparse_dense_leading_blocks"] == 2


def test_h3_sparse_chains_after_an_injected_cache():
    """loader -> cache -> sparse -> consumers, matching the preset node order."""
    prompt = bench.set_h3cache(_prompt(), (0.04, 0.2, 0.85, 1))
    chained = bench.set_h3_sparse(prompt, ("corrected", 1.0, 0.2, 2, 8192))

    assert chained["asfp8:h3cache"]["inputs"]["model"] == ["loader", 0]
    assert chained["asfp8:h3sparse"]["inputs"]["model"] == ["asfp8:h3cache", 0]
    assert chained["scheduler"]["inputs"]["model"] == ["asfp8:h3sparse", 0]
    assert chained["guider"]["inputs"]["model"] == ["asfp8:h3sparse", 0]


def test_h3_sparse_chains_after_an_injected_easycache():
    prompt = bench.set_easycache(_prompt(), (0.3, 0.2, 0.9))
    chained = bench.set_h3_sparse(prompt, ("corrected", 1.0, 0.2, 2, 8192))

    assert chained["asfp8:h3sparse"]["inputs"]["model"] == ["asfp8:easycache", 0]
    assert chained["guider"]["inputs"]["model"] == ["asfp8:h3sparse", 0]


def test_h3_sparse_is_a_noop_when_not_requested():
    assert bench.set_h3_sparse(_prompt(), None) == _prompt()


def test_h3_sparse_requires_exactly_one_loader():
    prompt = _prompt()
    prompt["loader2"] = {"class_type": "UNETLoader", "inputs": {}}
    with pytest.raises(RuntimeError, match="one UNETLoader"):
        bench.set_h3_sparse(prompt, ("corrected", 1.0, 0.2, 2, 8192))






def test_reserved_injection_ids_cannot_collide():
    prompt = _prompt()
    prompt["asfp8:h3sparse"] = {"class_type": "Whatever", "inputs": {}}
    with pytest.raises(RuntimeError, match="reserved node"):
        bench.set_h3_sparse(prompt, ("corrected", 1.0, 0.2, 2, 8192))
