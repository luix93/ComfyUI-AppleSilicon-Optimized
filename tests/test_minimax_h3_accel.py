"""Tests for the combined MiniMax H3 accelerator preset node.

These cover preset bookkeeping and the compatibility rules, which are the parts
that must not silently misbehave. The mechanisms themselves are tested in
test_minimax_h3cache.py and test_minimax_h3_sparse.py.
"""

import pytest

from _patches import minimax_h3_accel as accel
from _patches import minimax_h3_sparse as sparse
from _patches import minimax_h3cache as h3cache


class _FakeModel:
    """Minimal ModelPatcher stand-in: options plus a real clone."""

    def __init__(self, transformer_options=None):
        self.model_options = {
            "transformer_options": dict(transformer_options or {})
        }
        self.clones = 0

    def clone(self):
        clone = _FakeModel(self.model_options["transformer_options"])
        clone.clones = self.clones + 1
        return clone


def test_every_preset_is_declared_and_exact_is_the_calibrated_one():
    assert set(accel.H3_PRESETS) == set(accel.PRESETS)
    assert accel.H3_PRESETS["exact"]["calibrated"] is True
    assert accel.H3_PRESETS["exact"]["cache"] is None
    assert accel.H3_PRESETS["exact"]["sparse"] is None
    for name in ("balanced", "aggressive"):
        config = accel.H3_PRESETS[name]
        assert config["note"]
        assert config["cache"] is not None or config["easycache"] is not None


def test_exact_preset_returns_a_clone_and_adds_nothing():
    model = _FakeModel()
    result = accel.apply_h3_acceleration(model, "exact", verbose=False)
    assert result is not model
    assert result.clones == 1
    assert result.model_options["transformer_options"] == {}


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="preset must be"):
        accel.apply_h3_acceleration(_FakeModel(), "turbo", verbose=False)


@pytest.mark.parametrize("preset", ["balanced", "aggressive"])
def test_uncalibrated_presets_refuse_rather_than_guess(preset):
    """Until Stage 2 fills in measured constants these must not run."""
    assert accel.H3_PRESETS[preset]["calibrated"] is False
    with pytest.raises(ValueError, match="not calibrated"):
        accel.apply_h3_acceleration(_FakeModel(), preset, verbose=False)


@pytest.mark.parametrize("key,message", [
    (h3cache.STATE_KEY, "H3Cache is already attached"),
    (sparse.STATE_KEY, "sparse attention is already attached"),
    ("easycache", "EasyCache is already attached"),
])
def test_existing_cache_or_sparse_state_is_rejected(key, message):
    model = _FakeModel({key: object()})
    with pytest.raises(ValueError, match=message):
        accel.apply_h3_acceleration(model, "exact", verbose=False)


def test_existing_attention_override_is_rejected():
    model = _FakeModel({"optimized_attention_override": lambda *a, **k: None})
    with pytest.raises(ValueError, match="another attention override"):
        accel.apply_h3_acceleration(model, "exact", verbose=False)


def test_exact_preset_preserves_our_composable_dense_backend():
    def override(*args, **kwargs):
        return None

    override._asfp8_dense_backend = True
    model = _FakeModel({"optimized_attention_override": override})
    result = accel.apply_h3_acceleration(model, "exact", verbose=False)
    assert result.model_options["transformer_options"][
        "optimized_attention_override"] is override


def test_rejection_happens_before_any_patching():
    """A rejected combination must leave the incoming model untouched."""
    model = _FakeModel({h3cache.STATE_KEY: object()})
    before = dict(model.model_options["transformer_options"])
    with pytest.raises(ValueError):
        accel.apply_h3_acceleration(model, "exact", verbose=False)
    assert model.model_options["transformer_options"] == before
    assert model.clones == 0


class _FakeDiT:
    def __init__(self):
        self.blocks = list(range(50))


# apply_h3_sparse identifies H3 by the model class's defining module
_FakeDiT.__module__ = "comfy.ldm.minimax.model"


class _FakeH3Model(_FakeModel):
    def __init__(self, transformer_options=None):
        super().__init__(transformer_options)
        self.model = type("_Inner", (), {"diffusion_model": _FakeDiT()})()

    def clone(self):
        clone = _FakeH3Model(self.model_options["transformer_options"])
        clone.clones = self.clones + 1
        return clone


def test_sparse_node_refuses_a_model_with_an_existing_override(monkeypatch):
    """The low-level node enforces the same rule as the combined node."""
    monkeypatch.setattr(sparse, "_kernel", lambda: object())
    monkeypatch.setattr(sparse.torch.backends.mps, "is_available", lambda: True)
    model = _FakeH3Model({"optimized_attention_override": lambda *a, **k: None})
    with pytest.raises(ValueError, match="another attention override"):
        sparse.apply_h3_sparse(
            model, mode="corrected", tau=1.0, start_percent=0.2,
            dense_leading_blocks=2, min_sequence_tokens=8192, verbose=False)
    assert model.clones == 0


def test_sparse_node_reports_a_missing_kernel_clearly(monkeypatch):
    monkeypatch.setattr(sparse, "_kernel", lambda: None)
    monkeypatch.setattr(sparse.torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="sol_ext Metal kernel"):
        sparse.apply_h3_sparse(
            _FakeH3Model(), mode="corrected", tau=1.0, start_percent=0.2,
            dense_leading_blocks=2, min_sequence_tokens=8192, verbose=False)


def test_sparse_node_rejects_double_attachment(monkeypatch):
    monkeypatch.setattr(sparse, "_kernel", lambda: object())
    monkeypatch.setattr(sparse.torch.backends.mps, "is_available", lambda: True)
    model = _FakeH3Model({sparse.STATE_KEY: object()})
    with pytest.raises(ValueError, match="already attached"):
        sparse.apply_h3_sparse(
            model, mode="corrected", tau=1.0, start_percent=0.2,
            dense_leading_blocks=2, min_sequence_tokens=8192, verbose=False)


def test_sparse_node_rejects_non_h3_models():
    with pytest.raises(ValueError, match="MiniMax H3"):
        sparse.apply_h3_sparse(
            _FakeModel(), mode="corrected", tau=1.0, start_percent=0.2,
            dense_leading_blocks=2, min_sequence_tokens=8192, verbose=False)


def test_node_classes_are_registered_under_the_approximate_category():
    for mapping, key in (
        (accel.NODE_CLASS_MAPPINGS, "AppleSiliconH3Acceleration"),
        (sparse.NODE_CLASS_MAPPINGS, "AppleSiliconH3SparseAttention"),
    ):
        node = mapping[key]
        assert node.CATEGORY == "Apple Silicon/approximate"
        assert node.RETURN_TYPES == ("MODEL",)
        assert "model" in node.INPUT_TYPES()["required"]


def test_sparse_node_defaults_are_the_documented_ones():
    required = sparse.AppleSiliconH3SparseAttention.INPUT_TYPES()["required"]
    assert required["mode"][0][0] == "corrected"
    assert required["tau"][1]["default"] == 1.0
    # 0.50/0.85 is the diag64 20-step policy: the first ten of twenty
    # evaluations protected, and the final three.
    assert required["start_percent"][1]["default"] == 0.50
    assert required["end_percent"][1]["default"] == 0.85
    assert required["dense_leading_blocks"][1]["default"] == 2
    assert required["min_sequence_tokens"][1]["default"] == 8192
    assert required["block_size"][1]["default"] == 64
    assert required["temporal_taps"][1]["default"] == 0


def test_accel_node_defaults_to_exact():
    required = accel.AppleSiliconH3Acceleration.INPUT_TYPES()["required"]
    assert required["preset"][1]["default"] == "exact"


def test_sparse_node_widget_order_is_backward_compatible():
    """ComfyUI maps saved workflow values POSITIONALLY.

    Inserting a widget mid-list silently shifts every later value into the wrong
    slot — which is how `min_sequence_tokens=4096` once ended up in
    `dense_leading_blocks` and tripped its max of 49. New widgets must be
    appended, so this pins the prefix that existing workflows rely on.
    """
    order = list(sparse.AppleSiliconH3SparseAttention.INPUT_TYPES()["required"])
    assert order[:7] == [
        "model", "mode", "tau", "start_percent", "dense_leading_blocks",
        "min_sequence_tokens", "verbose",
    ], order
    assert set(order[7:]) == {"end_percent", "temporal_taps", "block_size"}
