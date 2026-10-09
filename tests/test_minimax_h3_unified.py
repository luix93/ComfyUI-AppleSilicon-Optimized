"""Regression tests for the unified production-facing H3 node."""

import sys
import types

import pytest

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
        "cache_mode": "motion_cache",
        "cache_reuse_threshold": 0.20,
        "cache_start_percent": 0.15,
        "cache_end_percent": 0.95,
        "cache_max_consecutive_skips": 2,
        "cache_audio_weight": 1.0,
        "cache_verbose": False,
        "token_reduction": False,
        "token_reduction_scale": 1.0,
        "token_early_evaluations": 10,
        "token_detail_restore_block": 30,
        "token_coarse_restore_block": 40,
        "token_verbose": False,
        "sparse_attention": False,
        "sparse_mode": "corrected",
        "sparse_tau": 1.0,
        "sparse_start_percent": 0.20,
        "sparse_end_percent": 0.85,
        "sparse_dense_leading_blocks": 2,
        "sparse_min_sequence_tokens": 0,
        "sparse_verbose": False,
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
    settings.update(cache_mode="first_block_cache", token_reduction=True, sparse_attention=True,
                    sparse_memory_budget_gib=16.0)

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
    assert calls[2][2]["memory_budget_gib"] == 16.0


def test_only_canonical_h3_nodes_are_registered():
    from _patches import attention_backend, minimax_h3_turbo, minimax_h3_presets
    mappings = {**attention_backend.NODE_CLASS_MAPPINGS,
                **minimax_h3_turbo.NODE_CLASS_MAPPINGS,
                **unified.NODE_CLASS_MAPPINGS, **minimax_h3_presets.NODE_CLASS_MAPPINGS}
    assert set(mappings) == {
        "AppleSiliconAttentionBackend", "AppleSiliconH3LoRA",
        "AppleSiliconH3UnifiedAcceleration", "AppleSiliconH3QuickAcceleration",
    }
    for retired in (cache, structure, sparse):
        assert not getattr(retired, "NODE_CLASS_MAPPINGS", {})


@pytest.mark.parametrize("mode,protected", [
    ("spectrum_protected", True), ("spectrum_fast", False),
])
def test_spectrum_calls_owned_runtime(monkeypatch, mode, protected):
    calls = []
    def apply(model, **kwargs):
        calls.append((model, kwargs)); return "forecast-model"
    monkeypatch.setitem(sys.modules, "_patches.h3_runtime.spectrum.api",
                        types.SimpleNamespace(apply=apply))
    settings = _defaults()
    settings.update(cache_mode=mode, token_reduction=False, sparse_attention=False)
    assert unified.apply_h3_acceleration("source", **settings) == "forecast-model"
    assert calls == [("source", dict(protected=protected, blend_weight=0.5, verbose=False))]
    settings["spectrum_blend_weight"] = 0.35
    unified.apply_h3_acceleration("source", **settings)
    assert calls[-1][1]["blend_weight"] == 0.35


@pytest.mark.parametrize("mode", ["motion_cache", "spectrum_protected", "spectrum_fast"])
def test_bundled_predictors_compose_with_token_and_sparse(monkeypatch, mode):
    calls = []
    def predictor(model, *args):
        calls.append(("predictor", model)); return "predicted"
    def tokens(model, **kwargs):
        calls.append(("tokens", model)); return "tokens"
    def attention(model, **kwargs):
        calls.append(("attention", model)); return "combined"
    monkeypatch.setattr(unified, "apply_spectrum_preset", predictor)
    monkeypatch.setattr(unified, "apply_motion_cache_preset", predictor)
    monkeypatch.setattr(structure, "apply_h3_token_reduction", tokens)
    monkeypatch.setattr(sparse, "apply_h3_sparse", attention)
    settings = _defaults()
    settings.update(cache_mode=mode, token_reduction=True, sparse_attention=True)
    assert unified.apply_h3_acceleration("source", **settings) == "combined"
    assert calls == [("predictor", "source"), ("tokens", "predicted"), ("attention", "tokens")]


def test_motion_cache_calls_owned_runtime(monkeypatch):
    calls = []
    def apply(model, **kwargs):
        calls.append((model, kwargs)); return "motion-model"
    monkeypatch.setitem(sys.modules, "_patches.h3_runtime.motion.runtime",
                        types.SimpleNamespace(apply=apply))
    settings = _defaults()
    settings.update(cache_mode="motion_cache", token_reduction=False, sparse_attention=False)
    assert unified.apply_h3_acceleration("source", **settings) == "motion-model"
    assert calls == [("source", dict(reuse_threshold=0.15, motion_strength=1.0,
        warmup_steps=4, max_consecutive_skips=2, start_percent=0.15, end_percent=0.95,
        subsample_factor=8, verbose=False))]
    settings.update(motion_reuse_threshold=0.17, motion_strength=0.8,
        motion_warmup_steps=3, motion_max_consecutive_skips=1,
        motion_start_percent=0.2, motion_end_percent=0.8)
    unified.apply_h3_acceleration("source", **settings)
    assert calls[-1] == ("source", dict(reuse_threshold=0.17, motion_strength=0.8,
        warmup_steps=3, max_consecutive_skips=1, start_percent=0.2, end_percent=0.8,
        subsample_factor=8, verbose=False))


def test_bundled_sources_are_pinned_and_unmodified():
    import hashlib
    import json
    from pathlib import Path
    vendor = Path(unified.__file__).parent / "_vendor"
    manifest = json.loads((vendor / "manifest.json").read_text())
    for package, entry in manifest.items():
        for filename, expected in entry["files"].items():
            assert hashlib.sha256((vendor / package / filename).read_bytes()).hexdigest() == expected


def test_new_optional_tuning_controls_keep_the_existing_widget_order():
    schema = unified.AppleSiliconH3Acceleration.INPUT_TYPES()
    assert list(schema["optional"])[:3] == ["sparse_backend", "sparse_dense_layers", "sparse_window_mode"]
    assert schema["optional"]["motion_reuse_threshold"][1]["default"] == 0.15
    assert schema["optional"]["spectrum_blend_weight"][1]["default"] == 0.5
    assert list(schema["optional"])[10:15] == ["sparse_memory_budget_gib", "veda_predictor", "veda_sparsity", "veda_reference_sparsity", "veda_dense_steps"]
    assert schema["optional"]["sparse_memory_budget_gib"][1]["default"] == 0.0
    assert list(schema["optional"])[15:] == ["sparse_topk_ratio", "token_start_percent", "token_end_percent"]


@pytest.mark.parametrize("backend", ["veda_metal", "veda_mlx"])
def test_veda_is_an_independent_sparse_mode_with_existing_cache_composition(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(unified, "apply_motion_cache_preset", lambda m, *a: "cached")
    def veda(model, **options):
        calls.append((model, options))
        return "veda-model"
    monkeypatch.setattr(unified.minimax_h3_veda, "apply_veda", veda)
    monkeypatch.setattr(sparse, "apply_h3_sparse", lambda *a, **k: pytest.fail("SOL must not run with Veda"))
    settings = _defaults()
    settings.update(sparse_attention=True, sparse_backend=backend, veda_sparsity=80,
                    veda_reference_sparsity=0, veda_dense_steps="0,19", sparse_memory_budget_gib=8)
    assert unified.apply_h3_acceleration("source", **settings) == "veda-model"
    assert calls[0][0] == "cached"
    assert calls[0][1]["backend"] == backend
    assert calls[0][1]["sparsity"] == 80
    assert calls[0][1]["reference_sparsity"] == 0
    assert calls[0][1]["dense_steps"] == "0,19"
    assert calls[0][1]["memory_budget_gib"] == 8


def test_disabled_veda_does_not_load_predictor_or_backend(monkeypatch):
    monkeypatch.setattr(unified.minimax_h3_veda, "apply_veda", lambda *a, **k: pytest.fail("Veda was loaded"))
    settings = _defaults()
    settings.update(cache_mode="disabled", sparse_backend="veda_metal", sparse_attention=False)
    model = object()
    assert unified.apply_h3_acceleration(model, **settings) is model


@pytest.mark.parametrize("backend,expected", [("int8", .15), ("bf16", 0.0)])
def test_hidden_topk_setting_is_only_forwarded_to_int8(monkeypatch, backend, expected):
    calls = []
    def apply_sparse(model, **kwargs):
        calls.append(kwargs)
        return model
    monkeypatch.setattr(sparse, "apply_h3_sparse", apply_sparse)
    settings = _defaults()
    settings.update(cache_mode="disabled", sparse_attention=True, sparse_backend=backend,
                    sparse_topk_ratio=.15)
    unified.apply_h3_acceleration("source", **settings)
    assert calls[0]["topk_ratio"] == expected
