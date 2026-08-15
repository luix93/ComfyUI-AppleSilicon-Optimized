"""Regression tests for the unified production-facing H3 node."""

from _patches import minimax_h3_structure as structure
from _patches import minimax_h3_unified as unified
from _patches import minimax_h3cache as cache
from _patches import minimax_h3_sparse as sparse


def _defaults():
    inputs = unified.AppleSiliconH3Acceleration.INPUT_TYPES()["required"]
    return {
        name: declaration[1]["default"]
        for name, declaration in inputs.items()
        if name != "model"
    }


def test_unified_defaults_match_validated_workflow_settings():
    defaults = _defaults()
    assert defaults == {
        "cache_mode": "first_block_cache",
        "cache_reuse_threshold": 0.20,
        "cache_start_percent": 0.15,
        "cache_end_percent": 0.95,
        "cache_max_consecutive_skips": 2,
        "cache_audio_weight": 1.0,
        "cache_verbose": True,
        "token_reduction": True,
        "token_reduction_scale": 1.0,
        "token_early_evaluations": 10,
        "token_detail_restore_block": 30,
        "token_coarse_restore_block": 40,
        "token_verbose": False,
        "sparse_attention": True,
        "sparse_mode": "corrected",
        "sparse_tau": 1.0,
        "sparse_start_percent": 0.20,
        "sparse_end_percent": 0.85,
        "sparse_dense_leading_blocks": 2,
        "sparse_min_sequence_tokens": 0,
        "sparse_verbose": True,
    }


def test_unified_ui_omits_rejected_or_fixed_controls():
    inputs = unified.AppleSiliconH3Acceleration.INPUT_TYPES()["required"]
    assert "active_layers" not in inputs
    assert "calibration_evaluations" not in inputs
    assert "cached_suffix_blocks" not in inputs
    assert "sparse_block_size" not in inputs
    assert "sparse_temporal_taps" not in inputs
    assert "hard_drop" not in unified.SPARSE_MODES
    assert "block_taylor" not in unified.CACHE_MODES


def test_disabled_mechanisms_return_input_without_patching(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("disabled mechanism was applied")

    monkeypatch.setattr(cache, "apply_h3cache", unexpected)
    monkeypatch.setattr(structure, "apply_h3_token_reduction", unexpected)
    monkeypatch.setattr(sparse, "apply_h3_sparse", unexpected)
    model = object()
    settings = _defaults()
    settings.update({
        "cache_mode": "disabled",
        "token_reduction": False,
        "sparse_attention": False,
    })
    assert unified.apply_h3_acceleration(model, **settings) is model


def test_mechanisms_apply_in_cache_token_sparse_order(monkeypatch):
    calls = []

    def apply_cache(model, **kwargs):
        calls.append(("cache", model, kwargs))
        return "cached-model"

    def apply_structure(model, **kwargs):
        calls.append(("token", model, kwargs))
        return "reduced-model"

    def apply_sparse(model, **kwargs):
        calls.append(("sparse", model, kwargs))
        return "sparse-model"

    monkeypatch.setattr(cache, "apply_h3cache", apply_cache)
    monkeypatch.setattr(structure, "apply_h3_token_reduction", apply_structure)
    monkeypatch.setattr(sparse, "apply_h3_sparse", apply_sparse)
    settings = _defaults()
    settings["sparse_attention"] = True

    result = unified.apply_h3_acceleration("source-model", **settings)

    assert result == "sparse-model"
    assert [call[:2] for call in calls] == [
        ("cache", "source-model"),
        ("token", "cached-model"),
        ("sparse", "reduced-model"),
    ]
    assert calls[0][2]["cached_suffix_blocks"] == 0
    assert "active_layers" not in calls[1][2]
    assert calls[2][2]["block_size"] == sparse.BLOCK_SIZE
    assert calls[2][2]["temporal_taps"] == sparse.TEMPORAL_TAPS


def test_legacy_nodes_remain_registered_but_deprecated():
    assert cache.AppleSiliconH3Cache.DEPRECATED is True
    assert structure.AppleSiliconH3StructuralAcceleration.DEPRECATED is True
    assert sparse.AppleSiliconH3SparseAttention.DEPRECATED is True
    assert unified.NODE_CLASS_MAPPINGS["AppleSiliconH3UnifiedAcceleration"] \
        is unified.AppleSiliconH3Acceleration
