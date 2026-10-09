"""Sparse attention attachment guards after node consolidation."""

import pytest
from _patches import minimax_h3_sparse as sparse


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


class _FakeDiT:
    def __init__(self):
        self.blocks = list(range(50))


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
