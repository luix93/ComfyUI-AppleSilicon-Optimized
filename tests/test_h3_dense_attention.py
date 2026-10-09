"""Optional quantized dense attention must preserve exact fallback boundaries."""
from types import SimpleNamespace

import pytest
import torch

from _patches import h3_dense_mps as dense
from _patches import minimax_h3_sparse as sparse


def test_switching_topk_to_dense_ignores_hidden_fraction():
    holder = sparse.H3SparseHolder(backend="dense_int8", topk_ratio=0.15)
    assert holder.topk_ratio == 0.0
    assert holder.clone().topk_ratio == 0.0


def invoke(monkeypatch, holder, *, reason=None, flattened=False):
    x = torch.arange(2 * 67 * 128).reshape(1, 2, 67, 128).to(torch.bfloat16)
    monkeypatch.setattr(sparse, "_eligible", lambda *args: reason)
    monkeypatch.setattr(sparse, "_dense", lambda *args: x + 2)
    result = sparse.sparse_attention_override(None, x, x, x, 2,
        skip_reshape=True, skip_output_reshape=not flattened,
        transformer_options={sparse.STATE_KEY: holder})
    return x, result


@pytest.mark.parametrize("flattened", [False, True])
def test_hybrid_protected_layer_uses_dense_quantization_with_correct_layout(monkeypatch, flattened):
    holder = sparse.H3SparseHolder(backend="int8_hybrid", dense_leading_blocks=2)
    monkeypatch.setattr(sparse, "_run_dense_int8", lambda h, q, k, v, scale:
                        (q[0] + 1, None, 0))
    x, got = invoke(monkeypatch, holder, flattened=flattened)
    expected = x + 1
    if flattened:
        expected = expected.transpose(1, 2).reshape(1, 67, 256)
    assert torch.equal(got, expected)
    assert holder.stats.quantized_dense_calls == 1


@pytest.mark.parametrize("reason", ["protected_steps", "short_sequence", "masked_attention", "not_mps"])
def test_hybrid_never_quantizes_other_fallbacks(monkeypatch, reason):
    holder = sparse.H3SparseHolder(backend="int8_hybrid")
    monkeypatch.setattr(sparse, "_run_dense_int8", lambda *args:
                        pytest.fail("quantized a protected or ineligible call"))
    x, got = invoke(monkeypatch, holder, reason=reason)
    assert torch.equal(got, x + 2)
    assert holder.stats.quantized_dense_calls == 0


@pytest.mark.parametrize("failure,latches", [
    (sparse._SparseBudgetExceeded, False), (sparse._SparseUnavailable, True)])
def test_hybrid_dense_failure_preserves_sol_and_fresh_run(monkeypatch, failure, latches):
    holder = sparse.H3SparseHolder(backend="int8_hybrid")
    def fail(*args):
        raise failure("test")
    monkeypatch.setattr(sparse, "_run_dense_int8", fail)
    x, got = invoke(monkeypatch, holder)
    assert torch.equal(got, x + 2)
    assert holder.hybrid_dense_disabled == latches
    assert not holder.disabled
    # A later SOL layer can still execute in the same run.
    holder.block_index = 2
    monkeypatch.setattr(sparse, "_run_sparse", lambda h, q, k, v, scale:
                        (q[0] + 3, torch.ones(2, 2), 8))
    x, got = invoke(monkeypatch, holder)
    assert torch.equal(got, x + 3)
    assert holder.stats.sparse_calls == 1
    fresh = holder.clone()
    assert not fresh.disabled and not fresh.hybrid_dense_disabled
    assert fresh.stats.calls == 0 and fresh.backend == "int8_hybrid"


def test_dense_workspace_budget_scales_and_checks_before_allocation(monkeypatch):
    gib = 2**30
    monkeypatch.setattr(dense.psutil, "virtual_memory", lambda:
                        SimpleNamespace(total=256*gib, available=80*gib))
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 192*gib)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 100*gib)
    assert dense.workspace_budget()[0] == 32*gib
    assert dense.scratch_bytes(37860, 56, 0) < dense.scratch_bytes(37860, 56, 3)
    dense.check_workspace(37860, 56, 0, 0)
    with pytest.raises(dense.WorkspaceBudgetExceeded, match="Temporary memory"):
        dense.check_workspace(37860, 56, .01, 0)


@pytest.mark.parametrize("backend", ["dense_int8", "int8_hybrid"])
def test_dense_reference_never_returns_quantized_output(monkeypatch, backend):
    holder = sparse.H3SparseHolder(backend=backend, mode="dense_reference", dense_leading_blocks=0)
    monkeypatch.setattr(sparse, "_run_sparse", lambda h, q, k, v, scale:
                        (q[0]+10, torch.ones(2, 2), 8))
    x, got = invoke(monkeypatch, holder)
    assert torch.equal(got, x + 2)
    assert holder.stats.routed_calls == 1 and holder.stats.sparse_calls == 0
