"""Memory-policy regressions; no GPU allocations required."""
import logging
from types import SimpleNamespace

import pytest
import torch

from _patches import h3_sol_mps as sol
from _patches import minimax_h3_sparse as sparse

GIB = 2**30


def memory(monkeypatch, total=256, available=160, metal_max=192, allocated=32):
    monkeypatch.setattr(sol.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=total * GIB, available=available * GIB))
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: metal_max * GIB)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: allocated * GIB)


@pytest.mark.parametrize("total,expected", [(256, 32), (64, 8), (16, 2)])
def test_auto_scales_with_physical_ram(monkeypatch, total, expected):
    memory(monkeypatch, total=total, available=total, metal_max=total, allocated=0)
    assert sol.workspace_budget()[0] == expected * GIB


def test_both_modes_respect_os_and_metal_headroom(monkeypatch):
    memory(monkeypatch, available=24)
    auto, headroom = sol.workspace_budget()
    assert headroom == 24 * GIB - int(256 * GIB * .05)
    assert auto == headroom // 2
    assert sol.workspace_budget(64)[0] == headroom
    memory(monkeypatch, available=160, allocated=180)
    assert sol.workspace_budget(64) == (0, 0)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_ceiling_rejected_at_both_entry_points(value):
    with pytest.raises(ValueError, match="finite"):
        sol.workspace_budget(value)
    with pytest.raises(ValueError):
        sparse.H3SparseHolder(memory_budget_gib=value)


def test_large_shape_has_actionable_budget_error_and_accepts_override(monkeypatch):
    memory(monkeypatch)
    assert 4 * GIB < sol.scratch_bytes(65536, 48) < 8 * GIB
    with pytest.raises(sol.WorkspaceBudgetExceeded) as caught:
        sol.check_workspace(65536, 48, 4)
    assert "65,536 tokens, 48 heads" in str(caught.value)
    assert "manual 4 GiB" in str(caught.value)
    assert "Temporary memory (GiB)" in str(caught.value)
    sol.check_workspace(65536, 48, 8)
    sol.check_workspace(65536, 48, 0)


def test_budget_fallback_does_not_latch_and_reports_each_shape_once(monkeypatch, caplog):
    holder = sparse.H3SparseHolder(min_sequence_tokens=0, dense_leading_blocks=0)
    options = {sparse.STATE_KEY: holder}
    x = torch.zeros(1, 2, 64, 128, dtype=torch.bfloat16)
    attempts = []

    def run(holder, q, k, v, scale):
        attempts.append(1)
        if len(attempts) <= 2:
            raise sparse._SparseBudgetExceeded(f"changing headroom {len(attempts)}")
        return q[0], torch.tensor([[1]]), 4

    monkeypatch.setattr(sparse, "_eligible", lambda *a: None)
    monkeypatch.setattr(sparse, "_dense", lambda fn, q, k, v, h, kw: q)
    monkeypatch.setattr(sparse, "_run_sparse", run)
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            sparse.sparse_attention_override(None, x, x, x, 2, skip_reshape=True,
                skip_output_reshape=True, transformer_options=options)
    assert not holder.disabled
    assert holder.stats.dense_calls == 2
    assert holder.stats.sparse_calls == 1
    assert caplog.text.count("dense fallback: memory_budget") == 1
