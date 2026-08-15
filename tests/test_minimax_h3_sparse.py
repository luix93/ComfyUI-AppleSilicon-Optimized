"""Tests for corrected Sol-style sparse attention on MiniMax H3.

The CPU tests pin the routing and correction algebra; the MPS tests pin the
Metal kernel against the fp32 specification in the module itself.
"""

import math

import pytest
import torch

from _patches import minimax_h3_sparse as sparse

from conftest import requires_mps


def _selection(heads, n_q, n_k, value=False):
    return torch.full((heads, n_q, n_k), value, dtype=torch.bool)


def _qkv(heads, seq, dim, device="cpu", dtype=torch.float32, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda: torch.randn(heads, seq, dim, generator=generator).to(  # noqa: E731
        device=device, dtype=dtype)
    return make(), make(), make()


# --- block statistics -------------------------------------------------------

def test_block_means_use_the_real_length_of_a_ragged_final_block():
    x = torch.arange(10, dtype=torch.float32).view(1, 10, 1)
    means, lengths = sparse.block_means(x, block=4, valid=10)
    assert lengths.tolist() == [4.0, 4.0, 2.0]
    # final block holds tokens 8 and 9 only: mean 8.5, not (8+9)/4
    assert means[0, 2, 0].item() == pytest.approx(8.5)
    assert means[0, 0, 0].item() == pytest.approx(1.5)


def test_block_means_ignore_values_past_the_valid_length():
    x = torch.zeros(1, 8, 1)
    x[0, :5] = 1.0
    x[0, 5:] = 1e6          # padding that must not be averaged in
    means, lengths = sparse.block_means(x, block=4, valid=5)
    assert lengths.tolist() == [4.0, 1.0]
    assert means[0, 1, 0].item() == pytest.approx(1.0)


# --- diag64 routing (NVIDIA's released policy) -------------------------------

def _diag_threshold(q_cent, k_cent, scale, tau):
    """The threshold spelled out term by term, independent of the module."""
    s = scale * math.log2(math.e)
    k_mean = k_cent.mean(dim=1)
    k_var = k_cent.var(dim=1, unbiased=False)
    mu = s * (q_cent * k_mean.unsqueeze(1)).sum(-1)
    sigma2 = (q_cent.square() * k_var.unsqueeze(1)).sum(-1) * (s * s)
    return mu + tau * (sigma2 + 1e-6).sqrt()


def test_diag_threshold_matches_the_analytic_definition():
    """log2 scaling, POPULATION variance, and the 1e-6 guard, term by term."""
    torch.manual_seed(0)
    heads, n_q, n_k, dim = 3, 5, 9, 16
    q_cent = torch.randn(heads, n_q, dim)
    k_cent = torch.randn(heads, n_k, dim)
    scale, tau = 0.25, 1.0

    # radius -1 disables the band entirely, isolating the merit decision;
    # radius 0 would still force the diagonal, which is a rule not a threshold
    got = sparse.route_blocks_diag_reference(q_cent, k_cent, scale, tau,
                                             sink_blocks=0, local_radius=-1)
    s = scale * math.log2(math.e)
    proxy = s * (q_cent @ k_cent.transpose(-1, -2))
    want = proxy > _diag_threshold(q_cent, k_cent, scale, tau).unsqueeze(-1)
    assert torch.equal(got, want)


def test_diag_selection_is_strictly_greater_than_the_threshold():
    """A block scoring EXACTLY at the threshold is rejected, not kept."""
    dim = 4
    # zero query centroid: every proxy is 0, mu is 0, sigma2 is 0, so the
    # threshold at tau=0 is sqrt(1e-6) > 0 and nothing can be selected on merit
    q_cent = torch.zeros(1, 2, dim)
    k_cent = torch.randn(1, 6, dim)
    selected = sparse.route_blocks_diag_reference(
        q_cent, k_cent, scale=1.0, tau=0.0, sink_blocks=0, local_radius=-1)
    assert not selected.any()


def test_a_query_block_always_attends_to_its_own_keys():
    """Radius 0 is still a band: |i - j| <= 0 forces the diagonal."""
    q_cent = torch.zeros(1, 4, 8)
    k_cent = torch.randn(1, 4, 8)
    selected = sparse.route_blocks_diag_reference(
        q_cent, k_cent, scale=1.0, tau=100.0, sink_blocks=0, local_radius=0)
    assert torch.equal(selected[0], torch.eye(4, dtype=torch.bool))


def test_diag_log2_scaling_changes_what_tau_means():
    """Dropping log2(e) would not reorder blocks but WOULD move the threshold.

    One estimated standard deviation is 1.4426x wider in log2 units, so the
    same tau keeps a different set. This pins that the scaling is really there.
    """
    torch.manual_seed(1)
    q_cent = torch.randn(2, 6, 32)
    k_cent = torch.randn(2, 24, 32)
    with_log2 = sparse.route_blocks_diag_reference(
        q_cent, k_cent, 0.2, 1.0, sink_blocks=0, local_radius=0)
    # the same policy computed in natural-log units
    s = 0.2
    proxy = s * (q_cent @ k_cent.transpose(-1, -2))
    k_mean, k_var = k_cent.mean(1), k_cent.var(1, unbiased=False)
    mu = s * (q_cent * k_mean.unsqueeze(1)).sum(-1)
    sigma2 = (q_cent.square() * k_var.unsqueeze(1)).sum(-1) * (s * s)
    without = proxy > (mu + (sigma2 + 1e-6).sqrt()).unsqueeze(-1)
    assert not torch.equal(with_log2, without)


def test_diag_local_band_and_sink_are_always_selected():
    heads, n_blocks = 2, 6
    q_cent = torch.zeros(heads, n_blocks, 4)
    k_cent = torch.zeros(heads, n_blocks, 4)
    # a zero proxy against a zero threshold selects nothing on merit (strict >)
    selected = sparse.route_blocks_diag_reference(
        q_cent, k_cent, scale=1.0, tau=100.0, sink_blocks=2)
    for i in range(n_blocks):
        for j in range(n_blocks):
            local = abs(i - j) <= sparse.LOCAL_RADIUS
            sink = j < 2 or i < 2
            assert bool(selected[0, i, j]) == (local or sink)


def test_diag_sink_query_blocks_are_fully_dense():
    torch.manual_seed(2)
    q_cent = torch.randn(2, 8, 16)
    k_cent = torch.randn(2, 8, 16)
    selected = sparse.route_blocks_diag_reference(
        q_cent, k_cent, scale=0.25, tau=3.0, sink_blocks=3)
    assert selected[:, :3, :].all(), "a query block inside the sink must be exact"
    assert selected[:, :, :3].all(), "the sink is a key sink for every query"


def test_sink_rounds_outward_to_a_block_boundary():
    """Truncating would leave real prefix rows unprotected in the last block."""
    assert sparse.sink_block_count(0, 64) == 0
    assert sparse.sink_block_count(1, 64) == 1
    assert sparse.sink_block_count(64, 64) == 1
    assert sparse.sink_block_count(65, 64) == 2
    assert sparse.sink_block_count(130, 64) == 3


def test_diag_tau_monotonically_reduces_density():
    torch.manual_seed(0)
    q_cent = torch.randn(4, 24, 16)
    k_cent = torch.randn(4, 24, 16)
    densities = [
        sparse.route_blocks_diag_reference(q_cent, k_cent, scale=0.25, tau=tau,
                                           sink_blocks=0).float().mean().item()
        for tau in (0.0, 0.5, 1.0, 2.0)
    ]
    assert densities == sorted(densities, reverse=True)
    assert densities[0] > densities[-1]


def test_diag_threshold_ignores_the_proxy_rows_own_spread():
    """The whole point of the analytic threshold, stated as a test.

    The rejected empirical router took the mean and variance OF THE PROXY ROW,
    so scaling one query block's centroid rescaled its own threshold with it and
    the selection was unchanged. The analytic threshold is built from the KEY
    centroids, so it does not track the row — a query centroid twice as large
    doubles the proxies AND the mean term, but the standard-deviation term also
    doubles, and the selection genuinely depends on the key distribution.
    """
    torch.manual_seed(3)
    k_cent = torch.randn(1, 32, 16)
    q_cent = torch.randn(1, 4, 16)
    base = sparse.route_blocks_diag_reference(q_cent, k_cent, 0.25, 1.0,
                                              sink_blocks=0, local_radius=0)
    # a shifted query centroid moves the proxies and the mean term together,
    # so an empirical row-mean threshold would be blind to it
    shifted = sparse.route_blocks_diag_reference(
        q_cent + 0.5 * k_cent.mean(1, keepdim=True), k_cent, 0.25, 1.0,
        sink_blocks=0, local_radius=0)
    assert not torch.equal(base, shifted)


def test_mask_unpacking_round_trips_a_selection():
    torch.manual_seed(5)
    heads, n_q, n_k = 3, 4, 70          # 70 blocks -> 3 words, ragged
    want = torch.rand(heads, n_q, n_k) > 0.5
    words = torch.zeros(heads, n_q, 32, dtype=torch.int64)
    for j in range(n_k):
        words[..., j // 32] |= want[..., j].to(torch.int64) << (j % 32)
    packed = (words - (words >= 2 ** 31).to(torch.int64) * 2 ** 32).to(torch.int32)
    assert torch.equal(sparse.mask_to_selection(packed, n_k), want)


def test_block_zero_cannot_alias_an_unselected_entry():
    """`selected * index` would make every rejected block look like block 0."""
    selected = _selection(1, 1, 4)
    selected[0, 0, 0] = True        # ONLY block 0 is selected
    indices, counts = sparse.selection_to_lists(selected)
    assert counts.tolist() == [[1]]
    assert indices.shape[-1] == 1
    assert indices[0, 0, 0].item() == 0

    selected = _selection(1, 1, 4)
    selected[0, 0, 3] = True        # ONLY the last block is selected
    indices, counts = sparse.selection_to_lists(selected)
    assert counts.tolist() == [[1]]
    assert indices[0, 0, 0].item() == 3


def test_selection_lists_are_sorted_and_padded_within_the_count():
    selected = _selection(1, 2, 5)
    selected[0, 0, [4, 1]] = True
    selected[0, 1, [0, 2, 3]] = True
    indices, counts = sparse.selection_to_lists(selected)
    assert counts.tolist() == [[2, 3]]
    assert indices[0, 0, :2].tolist() == [1, 4]
    assert indices[0, 1, :3].tolist() == [0, 2, 3]
    # entries past the count are never read by the kernel but must stay in range
    assert int(indices.max().item()) <= 4


# --- correction algebra -----------------------------------------------------

def test_full_selection_reproduces_dense_attention():
    heads, seq, dim, block = 2, 32, 8, 8
    q, k, v = _qkv(heads, seq, dim, seed=3)
    selected = _selection(heads, seq // block, seq // block, value=True)
    scale = 1.0 / math.sqrt(dim)
    got = sparse.sparse_attention_reference(q, k, v, selected, scale, block)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scale)[0]
    assert torch.allclose(got, want, atol=1e-5)


def test_centroid_correction_beats_hard_drop_against_dense():
    heads, seq, dim, block = 4, 128, 16, 16
    q, k, v = _qkv(heads, seq, dim, seed=7)
    scale = 1.0 / math.sqrt(dim)
    q_mean, _ = sparse.block_means(q, block, seq)
    k_mean, _ = sparse.block_means(k, block, seq)
    selected = sparse.route_blocks_diag_reference(q_mean, k_mean, scale, tau=1.0,
                                                  sink_blocks=0)
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scale)[0]
    corrected = sparse.sparse_attention_reference(q, k, v, selected, scale, block,
                                                  correct=True)
    dropped = sparse.sparse_attention_reference(q, k, v, selected, scale, block,
                                                correct=False)
    assert (corrected - dense).norm() < (dropped - dense).norm()


def test_correction_uses_the_ragged_block_length_in_its_multiplicity():
    """A short final block must carry log(real length), not log(block)."""
    heads, dim, block = 1, 4, 8
    seq = 20                      # final block holds 4 tokens
    q, k, v = _qkv(heads, seq, dim, seed=11)
    scale = 1.0 / math.sqrt(dim)
    selected = _selection(heads, 3, 3)
    selected[:, :, 0] = True      # keep block 0 exact, drop blocks 1 and 2
    got = sparse.sparse_attention_reference(q, k, v, selected, scale, block)

    # explicit reconstruction with the correct multiplicities
    k_mean, lengths = sparse.block_means(k, block, seq)
    v_mean, _ = sparse.block_means(v, block, seq)
    assert lengths.tolist() == [8.0, 8.0, 4.0]
    scores = scale * (q @ k.transpose(-1, -2))
    keep = torch.zeros(seq, dtype=torch.bool)
    keep[:block] = True
    scores = scores.masked_fill(~keep.view(1, 1, -1), float("-inf"))
    centroid = scale * (q @ k_mean.transpose(-1, -2)) + lengths.log().view(1, 1, -1)
    centroid[:, :, 0] = float("-inf")
    joint = torch.cat([scores, centroid], dim=-1)
    values = torch.cat([v, v_mean], dim=-2)
    weights = torch.softmax(joint, dim=-1)
    assert torch.allclose(got, weights @ values, atol=1e-5)


def test_merge_matches_the_reference_given_an_exact_lse():
    heads, seq, dim, block = 2, 48, 8, 8
    q, k, v = _qkv(heads, seq, dim, seed=13)
    scale = 1.0 / math.sqrt(dim)
    selected = _selection(heads, seq // block, seq // block)
    selected[:, :, :2] = True
    rows = torch.arange(seq) // block
    scores = scale * (q @ k.transpose(-1, -2))
    token_mask = selected[:, rows, :].repeat_interleave(block, dim=-1)[:, :, :seq]
    scores = scores.masked_fill(~token_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    exact_out = weights @ v
    lse = torch.logsumexp(scores, dim=-1)

    k_mean, lengths = sparse.block_means(k, block, seq)
    v_mean, _ = sparse.block_means(v, block, seq)
    merged = sparse.merge_centroid_correction(
        q, exact_out, lse, k_mean, v_mean, lengths, selected, block, scale)
    want = sparse.sparse_attention_reference(q, k, v, selected, scale, block)
    assert torch.allclose(merged, want, atol=1e-5)


def test_correction_chunking_does_not_change_the_result(monkeypatch):
    heads, seq, dim, block = 2, 64, 8, 8
    q, k, v = _qkv(heads, seq, dim, seed=17)
    scale = 1.0 / math.sqrt(dim)
    selected = _selection(heads, seq // block, seq // block)
    selected[:, :, 0] = True
    whole = sparse.sparse_attention_reference(q, k, v, selected, scale, block)

    monkeypatch.setattr(sparse, "_MIN_CHUNK_ROWS", 7)
    monkeypatch.setattr(sparse, "_CORRECTION_BUDGET_BYTES", 1)
    rows = torch.arange(seq) // block
    scores = scale * (q @ k.transpose(-1, -2))
    token_mask = selected[:, rows, :].repeat_interleave(block, dim=-1)[:, :, :seq]
    scores = scores.masked_fill(~token_mask, float("-inf"))
    exact_out = torch.softmax(scores, dim=-1) @ v
    lse = torch.logsumexp(scores, dim=-1)
    k_mean, lengths = sparse.block_means(k, block, seq)
    v_mean, _ = sparse.block_means(v, block, seq)
    chunked = sparse.merge_centroid_correction(
        q, exact_out, lse, k_mean, v_mean, lengths, selected, block, scale)
    assert torch.allclose(chunked, whole, atol=1e-5)


# --- holder / routing policy ------------------------------------------------

def _holder(**kwargs):
    holder = sparse.H3SparseHolder(**kwargs)
    holder.start_sigma = 0.8
    holder.end_sigma = 0.2
    holder.sigma = 0.5
    holder.target_rows = 100
    return holder


def test_block_size_granularities_are_validated_before_the_router_gate(monkeypatch):
    """The kernel supports three; the diag64 POLICY permits only 64."""
    monkeypatch.setenv("ASFP8_SOL_ROUTER", sparse.ROUTER_LEGACY)
    for size in (32, 64, 128):
        assert sparse.H3SparseHolder(block_size=size).block_size == size
    with pytest.raises(ValueError, match="block_size must be"):
        sparse.H3SparseHolder(block_size=48)


def test_holder_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        sparse.H3SparseHolder(mode="nonsense")
    with pytest.raises(ValueError):
        sparse.H3SparseHolder(tau=-1.0)
    with pytest.raises(ValueError):
        sparse.H3SparseHolder(start_percent=1.5)
    with pytest.raises(ValueError):
        sparse.H3SparseHolder(block_size=96)


def test_sigma_window_protects_the_early_steps():
    holder = _holder()
    holder.sigma = 0.9          # earlier than start_sigma=0.8 in sigma terms
    assert not holder.in_sparse_window()
    holder.sigma = 0.8
    assert holder.in_sparse_window()


def test_begin_evaluation_derives_target_rows_from_the_latents():
    holder = sparse.H3SparseHolder()
    video = torch.zeros(1, 24, 3, 8, 8)     # 3 frames, 4x4 patch rows each
    audio = torch.zeros(1, 32, 2, 10)       # 2 channels x 10 frames
    holder.begin_evaluation(video, audio, sigma=0.5)
    assert holder.target_rows == 3 * 4 * 4 + 2 * 10
    assert holder.block_index == 0
    assert holder.stats.evaluations == 1


def test_begin_evaluation_resets_the_block_counter_every_evaluation():
    holder = sparse.H3SparseHolder()
    video = torch.zeros(1, 24, 1, 4, 4)
    audio = torch.zeros(1, 32, 2, 4)
    holder.begin_evaluation(video, audio, sigma=0.5)
    holder.block_index = 37
    holder.begin_evaluation(video, audio, sigma=0.4)
    assert holder.block_index == 0


def test_unsupported_calls_report_a_reason_and_never_run_sparse():
    holder = _holder()
    seq = 16384
    q = torch.zeros(1, 2, seq, 128)
    cases = {
        "masked_attention": ({"mask": torch.zeros(1), "skip_reshape": True}, q),
        "unexpected_layout": ({"skip_reshape": False}, q),
        "unsupported_dtype": ({"skip_reshape": True}, q.to(torch.float32)),
        "not_mps": ({"skip_reshape": True}, q.to(torch.bfloat16)),
    }
    for expected, (kwargs, tensor) in cases.items():
        reason = sparse._eligible(holder, tensor, tensor, tensor, kwargs)
        assert reason == expected, f"{expected} != {reason}"


def test_short_and_refiner_sequences_are_declined():
    holder = _holder(min_sequence_tokens=8192)
    holder.target_rows = 5000
    kwargs = {"skip_reshape": True}
    short = torch.zeros(1, 2, 4096, 128, dtype=torch.bfloat16)
    assert sparse._eligible(holder, short, short, short, kwargs) == "not_main_sequence"
    holder.target_rows = 1000
    assert sparse._eligible(holder, short, short, short, kwargs) == "short_sequence"


def test_cross_attention_shapes_are_declined():
    holder = _holder()
    q = torch.zeros(1, 2, 16384, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 2, 512, 128, dtype=torch.bfloat16)
    assert sparse._eligible(holder, q, k, k, {"skip_reshape": True}) == "cross_attention"


def test_refiner_calls_do_not_advance_the_block_counter(monkeypatch):
    holder = _holder(min_sequence_tokens=0)
    holder.target_rows = 5000
    calls = []
    monkeypatch.setattr(sparse, "_dense",
                        lambda fn, q, k, v, h, kw: calls.append("dense") or q)
    options = {sparse.STATE_KEY: holder}
    refiner = torch.zeros(1, 2, 256, 128, dtype=torch.bfloat16)
    for _ in range(2):
        sparse.sparse_attention_override(
            None, refiner, refiner, refiner, 2,
            skip_reshape=True, transformer_options=options)
    assert holder.block_index == 0
    assert calls == ["dense", "dense"]


def test_leading_blocks_stay_dense_and_still_advance_the_counter(monkeypatch):
    holder = _holder(min_sequence_tokens=0, dense_leading_blocks=2)
    holder.target_rows = 100
    # eligibility itself is covered above; this pins the counter/route policy
    monkeypatch.setattr(sparse, "_eligible", lambda *a: None)
    monkeypatch.setattr(sparse, "_dense", lambda fn, q, k, v, h, kw: q)
    monkeypatch.setattr(
        sparse, "_run_sparse",
        lambda holder, q, k, v, scale: (q[0], torch.tensor([[4]]), 16))
    options = {sparse.STATE_KEY: holder}
    x = torch.zeros(1, 2, 1024, 128, dtype=torch.bfloat16)
    for _ in range(4):
        sparse.sparse_attention_override(
            None, x, x, x, 2, skip_reshape=True, skip_output_reshape=True,
            transformer_options=options)
    assert holder.block_index == 4
    assert holder.stats.dense_calls == 2      # blocks 0 and 1
    assert holder.stats.sparse_calls == 2     # blocks 2 and 3


def test_kernel_failure_latches_the_route_off_for_the_run(monkeypatch):
    holder = _holder(min_sequence_tokens=0)
    holder.target_rows = 100
    attempts = []

    def boom(holder_, q, k, v, scale):
        attempts.append(1)
        raise sparse._SparseUnavailable("no kernel")

    def eligible(holder_, q, k, v, kwargs):
        return "latched_off" if holder_.disabled else None

    monkeypatch.setattr(sparse, "_eligible", eligible)
    monkeypatch.setattr(sparse, "_dense", lambda fn, q, k, v, h, kw: q)
    monkeypatch.setattr(sparse, "_run_sparse", boom)
    options = {sparse.STATE_KEY: holder}
    x = torch.zeros(1, 2, 1024, 128, dtype=torch.bfloat16)
    for _ in range(5):
        sparse.sparse_attention_override(
            None, x, x, x, 2, skip_reshape=True, transformer_options=options)
    assert len(attempts) == 1
    assert holder.disabled


def test_missing_holder_falls_through_to_dense(monkeypatch):
    seen = []
    monkeypatch.setattr(sparse, "_dense",
                        lambda fn, q, k, v, h, kw: seen.append("dense") or q)
    x = torch.zeros(1, 2, 64, 8)
    sparse.sparse_attention_override(None, x, x, x, 2, transformer_options={})
    assert seen == ["dense"]


# --- MPS: the Metal kernel against the module's own specification -----------

def _kernel_or_skip():
    kernel = sparse._kernel()
    if kernel is None:
        pytest.skip("sol_ext Metal kernel unavailable")
    return kernel


def _run_kernel(kernel, query, key, value, selected, scale, block, correct):
    heads, seq, dim = query.shape
    k_mean, lengths = sparse.block_means(key, block, seq)
    v_mean, _ = sparse.block_means(value, block, seq)
    indices, counts = sparse.selection_to_lists(selected)
    padded = selected.shape[-1] * block
    pad = padded - seq
    parts = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value)]
    out, lse = kernel.sol_sparse_fwd(
        parts[0].unsqueeze(0), parts[1].unsqueeze(0), parts[2].unsqueeze(0),
        indices, counts, scale, seq, seq, block, 0, False)
    out = out[0, :, :seq]
    if not correct:
        return out
    return sparse.merge_centroid_correction(
        query, out, lse[:, :seq], k_mean, v_mean, lengths, selected, block, scale)


@requires_mps
@pytest.mark.parametrize("seq,block", [(1024, 64), (5378, 64), (5378, 128)])
def test_metal_kernel_matches_the_reference(seq, block):
    kernel = _kernel_or_skip()
    heads, dim = 4, 128
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=23)
    scale = 1.0 / math.sqrt(dim)
    q_mean, _ = sparse.block_means(q, block, seq)
    k_mean, _ = sparse.block_means(k, block, seq)
    selected = sparse.route_blocks_diag_reference(q_mean.float(), k_mean.float(),
                                                  scale, tau=1.0, sink_blocks=2)
    got = _run_kernel(kernel, q, k, v, selected, scale, block, correct=True)
    want = sparse.sparse_attention_reference(q, k, v, selected, scale, block)
    assert not torch.isnan(got).any()
    error = (got.float() - want.float()).norm() / want.float().norm()
    assert error < 2e-2, error


@requires_mps
def test_padded_rows_cannot_leak_into_the_output():
    """A poisoned pad region must not perturb any valid row."""
    kernel = _kernel_or_skip()
    heads, seq, dim, block = 4, 300, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=29)
    scale = 1.0 / math.sqrt(dim)
    n_blocks = (seq + block - 1) // block
    selected = torch.ones(heads, n_blocks, n_blocks, dtype=torch.bool, device="mps")
    indices, counts = sparse.selection_to_lists(selected)
    pad = n_blocks * block - seq
    assert pad > 0
    clean = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]
    dirty = [t.clone() for t in clean]
    for t in dirty:
        t[:, seq:] = 1234.0
    outputs = []
    for parts in (clean, dirty):
        out, _ = kernel.sol_sparse_fwd(
            parts[0].unsqueeze(0), parts[1].unsqueeze(0), parts[2].unsqueeze(0),
            indices, counts, scale, seq, seq, block, 0, False)
        outputs.append(out[0, :, :seq].float())
    assert torch.equal(outputs[0], outputs[1])


@requires_mps
def test_kernel_lse_matches_a_dense_logsumexp():
    kernel = _kernel_or_skip()
    heads, seq, dim, block = 2, 512, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=31)
    scale = 1.0 / math.sqrt(dim)
    n_blocks = seq // block
    selected = torch.zeros(heads, n_blocks, n_blocks, dtype=torch.bool, device="mps")
    selected[:, :, :3] = True
    indices, counts = sparse.selection_to_lists(selected)
    _, lse = kernel.sol_sparse_fwd(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), indices, counts,
        scale, seq, seq, block, 0, False)
    scores = scale * (q.float() @ k.float().transpose(-1, -2))
    keep = torch.zeros(seq, dtype=torch.bool, device="mps")
    keep[:3 * block] = True
    want = torch.logsumexp(scores.masked_fill(~keep.view(1, 1, -1), float("-inf")), -1)
    assert torch.allclose(lse, want, atol=2e-2)


# --- integration through ComfyUI's real attention dispatch ------------------

@requires_mps
def test_override_fires_once_through_comfy_dispatch_and_declining_reaches_flash41():
    """The seam this whole node depends on.

    comfy's wrap_attn fires `optimized_attention_override`; declining inside it
    must produce the same numbers as a plain dense call, and must not recurse.
    """
    # imported here, not at module scope: without ComfyUI on sys.path only this
    # test should skip, not the whole file
    comfy_attention = pytest.importorskip(
        "comfy.ldm.modules.attention",
        reason="needs ComfyUI on sys.path (run from a ComfyUI checkout)")
    from _patches import attn_router_mps

    fired = []

    def override(func, q, k, v, heads, **kwargs):
        fired.append(1)
        return attn_router_mps.dense_without_override(func, q, k, v, heads, **kwargs)

    heads, seq, dim = 8, 4096, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=37)
    q, k, v = (t.unsqueeze(0) for t in (q, k, v))

    plain = comfy_attention.optimized_attention(
        q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True,
        transformer_options={})
    routed = comfy_attention.optimized_attention(
        q, k, v, heads, mask=None, skip_reshape=True, skip_output_reshape=True,
        transformer_options={"optimized_attention_override": override})

    assert len(fired) == 1, "override must fire exactly once per attention call"
    assert plain.shape == routed.shape
    assert torch.equal(plain, routed), "declining changed the dense result"


# --- Phase E: the fused kernel ---------------------------------------------

def test_route_tiles_forces_prefix_audio_and_local_band():
    heads, block, tile, n_blocks = 2, 8, 8, 6
    query = torch.zeros(heads, n_blocks * block, 4)
    k_mean = torch.zeros(heads, n_blocks, 4)
    # zero proxy + huge tau selects nothing on merit
    selected = sparse.route_tiles(query, k_mean, scale=1.0, tau=100.0,
                                  block=block, tile=tile, prefix_rows=0,
                                  audio_rows=(0, 0), prefix_blocks=2,
                                  audio_blocks=(4, 5))
    assert selected[:, :, :2].all()          # prefix keys are sinks
    assert selected[:, :, 4:5].all()         # audio keys are sinks
    for t in range(n_blocks):
        assert selected[0, t, t]             # local band always present


def test_route_tiles_marks_prefix_and_audio_query_tiles_dense():
    heads, block, tile, n_blocks = 1, 8, 8, 6
    query = torch.zeros(heads, n_blocks * block, 4)
    k_mean = torch.zeros(heads, n_blocks, 4)
    selected = sparse.route_tiles(query, k_mean, scale=1.0, tau=100.0,
                                  block=block, tile=tile, prefix_rows=8,
                                  audio_rows=(16, 24), prefix_blocks=0,
                                  audio_blocks=(0, 0))
    assert selected[0, 0].all()              # tile 0 holds prefix rows
    assert selected[0, 2].all()              # tile 2 holds audio rows
    assert not selected[0, 4].all()          # an ordinary tile is not dense


def test_fused_reference_reduces_to_dense_when_everything_is_selected():
    heads, seq, dim, block = 2, 32, 8, 8
    q, k, v = _qkv(heads, seq, dim, seed=41)
    scale = 1.0 / math.sqrt(dim)
    got = sparse.sparse_attention_fused_reference(
        q, k, v, scale, tau=-1e9, block=block, tile=8)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), scale=scale)[0]
    assert torch.allclose(got, want, atol=1e-5)


def _fused_or_skip():
    kernel = _kernel_or_skip()
    if not hasattr(kernel, "sol_fused_fwd"):
        pytest.skip("fused kernel unavailable")
    return kernel


def _run_fused_kernel(kernel, q, k, v, scale, tau, block,
                      prefix_rows=0, audio=(0, 0), rows_per_frame=0, taps=0):
    heads, seq, dim = q.shape
    k_mean, lengths = sparse.block_means(k, block, seq)
    v_mean, _ = sparse.block_means(v, block, seq)
    n_blocks = k_mean.shape[1]
    prefix_blocks = (prefix_rows + block - 1) // block
    a_b = ((audio[0] // block, (audio[1] + block - 1) // block)
           if audio[1] > audio[0] else (0, 0))
    pad = n_blocks * block - seq
    parts = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]
    out, stat = kernel.sol_fused_fwd(
        parts[0].unsqueeze(0), parts[1].unsqueeze(0), parts[2].unsqueeze(0),
        k_mean.contiguous(), v_mean.contiguous(), lengths.log().contiguous(),
        scale, tau, seq, seq, block, 0, False,
        prefix_rows, audio[0], audio[1], prefix_blocks, a_b[0], a_b[1], 1,
        audio[1], rows_per_frame, taps)
    return out[0, :, :seq], stat, n_blocks, a_b, prefix_blocks


@requires_mps
@pytest.mark.parametrize("seq,block,tau", [
    (1024, 64, 1.0), (5378, 64, 1.0), (5378, 128, 1.0), (1024, 64, 1.5),
    (2048, 32, 1.0), (5378, 32, 1.5),
])
def test_fused_kernel_matches_the_reference(seq, block, tau):
    kernel = _fused_or_skip()
    heads, dim = 4, 128
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=43)
    scale = 1.0 / math.sqrt(dim)
    got, _, _, a_b, prefix_blocks = _run_fused_kernel(kernel, q, k, v, scale, tau, block)
    want = sparse.sparse_attention_fused_reference(
        q, k, v, scale, tau, block, 32, 0, (0, 0), prefix_blocks, a_b, 1)
    assert not torch.isnan(got).any()
    error = (got.float() - want.float()).norm() / want.float().norm()
    # Looser than the two-stage kernel's 2e-2 because this compares two
    # different roundings of the same approximation: the kernel is handed
    # bf16 pooled keys and runs the centroid columns through the online softmax
    # before the exact blocks, while the reference recomputes the pooled keys in
    # fp32 and does one softmax over concatenated scores. The exact assertion
    # on the routing itself is
    # test_fused_kernel_selection_matches_the_reference.
    assert error < 5e-2, error


@requires_mps
def test_fused_kernel_honours_prefix_and_audio_spans():
    kernel = _fused_or_skip()
    heads, seq, dim, block = 4, 2048, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=47)
    scale = 1.0 / math.sqrt(dim)
    got, _, _, a_b, prefix_blocks = _run_fused_kernel(
        kernel, q, k, v, scale, 1.0, block, prefix_rows=200, audio=(200, 280))
    want = sparse.sparse_attention_fused_reference(
        q, k, v, scale, 1.0, block, 32, 200, (200, 280), prefix_blocks, a_b, 1)
    error = (got.float() - want.float()).norm() / want.float().norm()
    assert error < 5e-2, error


@requires_mps
@pytest.mark.parametrize("seq,block", [(1024, 64), (4096, 64), (5378, 64)])
def test_fused_kernel_selection_matches_the_reference(seq, block):
    """The routing decision must agree with the spec EXACTLY, tile by tile.

    Output L2 mixes routing with arithmetic; this isolates the routing. The
    kernel reports how many blocks it actually computed per (head, tile), which
    must equal the reference's selection count everywhere.
    """
    kernel = _fused_or_skip()
    heads, dim = 4, 128
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=61)
    scale = 1.0 / math.sqrt(dim)
    _, stat, _, _, _ = _run_fused_kernel(kernel, q, k, v, scale, 1.0, block)

    k_mean, _ = sparse.block_means(k, block, seq)
    n_blocks = k_mean.shape[1]
    q_pad = torch.nn.functional.pad(q, (0, 0, 0, n_blocks * block - seq))
    selected = sparse.route_tiles(q_pad, k_mean, scale, 1.0, block, 32,
                                  0, (0, 0), 0, (0, 0), 1)
    assert torch.equal(selected.sum(-1).int().reshape(-1),
                       stat.int().reshape(-1))


@requires_mps
def test_fused_kernel_is_immune_to_a_dirty_allocator():
    """A partial pooled-key tile must not read poisoned memory.

    The pooled-key tiles are BC wide; a trailing partial tile used to read past
    kbar/vbar, and although those columns are masked to -inf, the PV matmul then
    evaluated 0 * garbage — and 0 * NaN is NaN. Zero-padding fixed it; this
    pins the behaviour with a deliberately poisoned allocator.
    """
    kernel = _fused_or_skip()
    heads, seq, dim, block = 4, 1024, 128, 64      # nB = 16, a partial 64 tile
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=53)
    scale = 1.0 / math.sqrt(dim)
    clean, _, _, _, _ = _run_fused_kernel(kernel, q, k, v, scale, 1.5, block)
    # churn the allocator with NaN-filled buffers, then repeat
    junk = [torch.full((512, 512), float("nan"), device="mps") for _ in range(8)]
    del junk
    dirty, _, _, _, _ = _run_fused_kernel(kernel, q, k, v, scale, 1.5, block)
    assert not torch.isnan(clean).any()
    assert not torch.isnan(dirty).any()
    assert torch.equal(clean, dirty)


@requires_mps
def test_fused_density_falls_as_tau_rises():
    kernel = _fused_or_skip()
    heads, seq, dim, block = 8, 4096, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=59)
    scale = 1.0 / math.sqrt(dim)
    densities = []
    for tau in (0.0, 0.5, 1.0, 1.5):
        _, stat, n_blocks, _, _ = _run_fused_kernel(kernel, q, k, v, scale, tau, block)
        densities.append(stat.float().mean().item() / n_blocks)
    assert densities == sorted(densities, reverse=True)
    assert densities[0] > densities[-1]


# --- temporal band + end_percent -------------------------------------------

def test_temporal_band_forces_the_co_located_block_in_adjacent_frames():
    """The spatial band is local WITHIN a frame and never reaches another one.

    One latent frame is `rows_per_frame` rows (1620 at 864x480, ~25 blocks), so
    without an explicit temporal band a static region has no guaranteed
    reference in any other frame — which is what shows up as background flicker.
    """
    heads, block, tile = 1, 8, 8
    rows_per_frame, n_frames = 32, 5           # 4 blocks per frame
    seq = rows_per_frame * n_frames
    query = torch.zeros(heads, seq, 4)
    k_mean = torch.zeros(heads, seq // block, 4)
    common = dict(scale=1.0, tau=100.0, block=block, tile=tile, prefix_rows=0,
                  audio_rows=(0, 0), prefix_blocks=0, audio_blocks=(0, 0),
                  local_radius=1)
    without = sparse.route_tiles(query, k_mean, **common)
    with_band = sparse.route_tiles(query, k_mean, video_row_start=0,
                                   rows_per_frame=rows_per_frame, taps=1,
                                   valid_rows=seq, **common)

    # a tile in the middle frame: its co-located block one frame away
    tile_index = (2 * rows_per_frame) // tile          # inside frame 2
    centre = tile_index * tile + tile // 2
    target = (centre + rows_per_frame) // block
    assert not without[0, tile_index, target], "spatial band should not reach it"
    assert with_band[0, tile_index, target], "temporal band must force it"
    # and it only ever adds blocks
    assert (with_band | without).equal(with_band)


def test_temporal_band_is_not_applied_to_prefix_rows():
    heads, block, tile, rows_per_frame = 1, 8, 8, 32
    seq = rows_per_frame * 4
    query = torch.zeros(heads, seq, 4)
    k_mean = torch.zeros(heads, seq // block, 4)
    selected = sparse.route_tiles(
        query, k_mean, scale=1.0, tau=100.0, block=block, tile=tile,
        prefix_rows=0, audio_rows=(0, 0), prefix_blocks=0, audio_blocks=(0, 0),
        local_radius=1, video_row_start=rows_per_frame,
        rows_per_frame=rows_per_frame, taps=1, valid_rows=seq)
    # tile 0 sits before video_row_start, so only its own local band is forced
    assert int(selected[0, 0].sum()) <= 3


def test_sparse_window_protects_both_ends():
    holder = sparse.H3SparseHolder(start_percent=0.20, end_percent=0.85)
    holder.start_sigma, holder.end_sigma = 0.8, 0.2
    for sigma, expected in ((0.9, False), (0.8, True), (0.5, True),
                            (0.2, True), (0.1, False)):
        holder.sigma = sigma
        assert holder.in_sparse_window() is expected, sigma


def test_end_percent_must_not_precede_start_percent():
    with pytest.raises(ValueError, match="0 <= start <= end <= 1"):
        sparse.H3SparseHolder(start_percent=0.9, end_percent=0.2)


def test_begin_evaluation_records_rows_per_frame():
    holder = sparse.H3SparseHolder()
    video = torch.zeros(1, 24, 3, 8, 8)       # 3 latent frames, 4x4 patch rows
    audio = torch.zeros(1, 32, 2, 10)
    holder.begin_evaluation(video, audio, sigma=0.5)
    assert holder.rows_per_frame == 16        # patch_h * patch_w = 4 * 4


@requires_mps
def test_fused_kernel_matches_the_reference_with_a_temporal_band():
    kernel = _fused_or_skip()
    heads, seq, dim, block = 4, 2048, 128, 64
    rows_per_frame = 256
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=67)
    scale = 1.0 / math.sqrt(dim)
    got, stat, n_blocks, a_b, prefix_blocks = _run_fused_kernel(
        kernel, q, k, v, scale, 1.5, block, rows_per_frame=rows_per_frame, taps=2)
    want = sparse.sparse_attention_fused_reference(
        q, k, v, scale, 1.5, block, 32, 0, (0, 0), prefix_blocks, a_b, 1,
        video_row_start=0, rows_per_frame=rows_per_frame, taps=2)
    assert not torch.isnan(got).any()
    error = (got.float() - want.float()).norm() / want.float().norm()
    assert error < 5e-2, error


@requires_mps
def test_temporal_band_costs_little_density():
    """It must buy consistency without undoing the sparsity."""
    kernel = _fused_or_skip()
    heads, seq, dim, block = 4, 8192, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=71)
    scale = 1.0 / math.sqrt(dim)
    _, plain, n_blocks, _, _ = _run_fused_kernel(kernel, q, k, v, scale, 1.5, block)
    _, banded, _, _, _ = _run_fused_kernel(
        kernel, q, k, v, scale, 1.5, block, rows_per_frame=1620, taps=2)
    d0 = plain.float().mean().item() / n_blocks
    d1 = banded.float().mean().item() / n_blocks
    assert d1 >= d0                       # strictly adds blocks
    # 2 taps x 2 directions x (2*radius+1) blocks = up to 12 blocks; at
    # nB=128 that is ~9% and it shrinks as the clip lengthens (~4.5% at nB=266)
    assert d1 - d0 < 0.10, (d0, d1)


# --- diag64: the Metal route against the Python reference -------------------

def _route_or_skip():
    kernel = _kernel_or_skip()
    if not hasattr(kernel, "sol_diag_route"):
        pytest.skip("sol_ext predates the diag64 router")
    return kernel


def _metal_route(kernel, q, k, scale, tau, block, sink_rows):
    """Run the production route and return (metal selection, reference, gap)."""
    heads, seq, dim = q.shape
    k_mean, _ = sparse.block_means(k, block, seq)
    n_blocks = k_mean.shape[1]
    q_pad, = sparse._pad_to_blocks((q,), n_blocks * block - seq)
    k_cent = k_mean.float().contiguous()
    cent_mean, cent_var = sparse.key_centroid_stats(k_cent)
    sink_blocks = sparse.sink_block_count(sink_rows, block)

    mask, counts = kernel.sol_diag_route(
        q_pad.unsqueeze(0), k_cent, cent_mean.contiguous(),
        cent_var.contiguous(), scale, tau, seq, block, sink_blocks,
        sparse.LOCAL_RADIUS)
    got = sparse.mask_to_selection(mask, n_blocks)

    q_cent, _ = sparse.block_means(q.float(), block, seq)
    want = sparse.route_blocks_diag_reference(q_cent, k_cent, scale, tau,
                                              sink_blocks, sparse.LOCAL_RADIUS)
    s = scale * sparse.LOG2E
    proxy = s * (q_cent @ k_cent.transpose(-1, -2))
    threshold = _diag_threshold(q_cent.cpu(), k_cent.cpu(), scale, tau).to(q.device)
    gap = (proxy - threshold.unsqueeze(-1)).abs()
    return got, want, gap, counts, n_blocks


@requires_mps
@pytest.mark.parametrize("seq,tau,sink_rows", [
    (1024, 1.0, 0),        # aligned, no sink
    (5378, 1.0, 320),      # ragged query AND kv tail, sink on a boundary
    (8192, 0.5, 130),      # sink straddling a block boundary (rounds to 3)
    (4096, 0.0, 64),       # tau=0, the densest configuration we ever ship
    (300, 1.5, 40),        # shorter than one block-aligned frame
])
def test_metal_route_matches_the_reference(seq, tau, sink_rows):
    """The gate the whole calibration depends on: same route, same blocks.

    The plan's tolerance allows bf16 boundary disagreement. In practice the
    route runs entirely in fp32 — pooled query centroids included — so it comes
    out bit-identical, and that is what is asserted here. The looser bounds are
    kept as explicit companions so a future arithmetic change degrades to
    "still acceptable" rather than silently to "wrong".
    """
    kernel = _route_or_skip()
    heads, dim, block = 4, 128, 64
    q, k, _ = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=101)
    scale = 1.0 / math.sqrt(dim)
    got, want, gap, counts, n_blocks = _metal_route(
        kernel, q, k, scale, tau, block, sink_rows)

    agreement = (got == want).float().mean().item()
    assert agreement >= 0.9999, agreement
    disagreed = got != want
    far = (disagreed & (gap > 2e-4)).sum().item()
    assert far == 0, f"{far} entries disagree far from the threshold"
    delta = abs(got.float().mean().item() - want.float().mean().item())
    assert delta <= 0.001, f"density differs by {delta * 100:.3f}pp"
    assert torch.equal(got, want), "expected bit-identical fp32 routing"
    assert torch.equal(counts.cpu(), got.sum(-1).int().cpu())


@requires_mps
def test_metal_route_pools_a_ragged_query_block_by_its_real_length():
    """A tail block must not average in padding rows.

    Padding is zeros, so dividing by 64 instead of the real length pulls the
    centroid towards the origin and shifts that block's whole threshold. The
    check is that poisoned padding cannot change the route at all.
    """
    kernel = _route_or_skip()
    heads, seq, dim, block = 4, 1000, 128, 64      # 1000 = 15 blocks + 40 rows
    q, k, _ = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=103)
    scale = 1.0 / math.sqrt(dim)
    n_blocks = (seq + block - 1) // block
    k_mean, _ = sparse.block_means(k, block, seq)
    k_cent = k_mean.float().contiguous()
    cent_mean, cent_var = sparse.key_centroid_stats(k_cent)

    clean = torch.nn.functional.pad(q, (0, 0, 0, n_blocks * block - seq))
    dirty = clean.clone()
    dirty[:, seq:] = 987.0
    routes = []
    for tensor in (clean, dirty):
        mask, _ = kernel.sol_diag_route(
            tensor.unsqueeze(0), k_cent, cent_mean.contiguous(),
            cent_var.contiguous(), scale, 1.0, seq, block, 2,
            sparse.LOCAL_RADIUS)
        routes.append(sparse.mask_to_selection(mask, n_blocks))
    assert torch.equal(routes[0], routes[1])


@requires_mps
def test_metal_route_density_falls_as_tau_rises():
    kernel = _route_or_skip()
    heads, seq, dim, block = 8, 4096, 128, 64
    q, k, _ = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=107)
    scale = 1.0 / math.sqrt(dim)
    densities = []
    for tau in (0.0, 0.5, 1.0, 1.5):
        got, _, _, _, _ = _metal_route(kernel, q, k, scale, tau, block, 128)
        densities.append(got.float().mean().item())
    assert densities == sorted(densities, reverse=True)
    assert densities[0] > densities[-1]


# --- diag64: corrected attention under that route ---------------------------

def _masked_or_skip():
    kernel = _kernel_or_skip()
    if not hasattr(kernel, "sol_masked_corrected_fwd"):
        pytest.skip("sol_ext predates the diag64 attention kernel")
    return kernel


def _run_masked(kernel, q, k, v, scale, tau, block, sink_rows, correct=True):
    heads, seq, dim = q.shape
    k_mean, lengths = sparse.block_means(k, block, seq)
    v_mean, _ = sparse.block_means(v, block, seq)
    n_blocks = k_mean.shape[1]
    q_p, k_p, v_p = sparse._pad_to_blocks((q, k, v), n_blocks * block - seq)
    k_cent = k_mean.float().contiguous()
    cent_mean, cent_var = sparse.key_centroid_stats(k_cent)
    mask, counts = kernel.sol_diag_route(
        q_p.unsqueeze(0), k_cent, cent_mean.contiguous(), cent_var.contiguous(),
        scale, tau, seq, block, sparse.sink_block_count(sink_rows, block),
        sparse.LOCAL_RADIUS)
    out, stat = kernel.sol_masked_corrected_fwd(
        q_p.unsqueeze(0), k_p.unsqueeze(0), v_p.unsqueeze(0),
        k_mean.contiguous(), v_mean.contiguous(), lengths.log().contiguous(),
        mask, scale, seq, seq, block, 0, False, correct)
    selection = sparse.mask_to_selection(mask, n_blocks)
    return out[0, :, :seq], stat, counts, selection


@requires_mps
@pytest.mark.parametrize("seq,tau,sink_rows", [
    (1024, 1.0, 0), (5378, 1.0, 320), (8192, 0.5, 130), (300, 1.5, 40),
])
def test_masked_kernel_matches_the_fp32_reference(seq, tau, sink_rows):
    kernel = _masked_or_skip()
    heads, dim, block = 4, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=109)
    scale = 1.0 / math.sqrt(dim)
    got, _, _, selection = _run_masked(kernel, q, k, v, scale, tau, block, sink_rows)
    want = sparse.sparse_attention_reference(q, k, v, selection, scale, block)
    assert not torch.isnan(got).any()
    error = ((got.float() - want.float()).norm() / want.float().norm()).item()
    # Tight, unlike the tile-routed fused kernel's 5e-2: the route is now
    # identical to the reference's, so the only difference left is bf16
    # rounding of the same arithmetic.
    assert error < 5e-3, error


@requires_mps
def test_masked_kernel_hard_drop_matches_the_uncorrected_reference():
    kernel = _masked_or_skip()
    heads, seq, dim, block = 4, 2048, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=113)
    scale = 1.0 / math.sqrt(dim)
    got, _, _, selection = _run_masked(kernel, q, k, v, scale, 1.0, block, 64,
                                       correct=False)
    want = sparse.sparse_attention_reference(q, k, v, selection, scale, block,
                                             correct=False)
    error = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert error < 5e-3, error


@requires_mps
def test_masked_kernel_reduces_to_dense_when_everything_is_selected():
    kernel = _masked_or_skip()
    heads, seq, dim, block = 4, 2048, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=127)
    scale = 1.0 / math.sqrt(dim)
    got, _, counts, selection = _run_masked(kernel, q, k, v, scale, -1e9, block, 0)
    assert selection.all(), "tau below every proxy must select every block"
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).float(), k.unsqueeze(0).float(), v.unsqueeze(0).float(),
        scale=scale)[0]
    error = ((got.float() - dense).norm() / dense.norm()).item()
    assert error < 5e-3, error


@requires_mps
def test_both_compute_tiles_consume_the_same_64_row_route():
    """BR=32 but the policy is 64-row. The halves must not route separately.

    The kernel reports blocks computed per 32-row tile; each must equal the
    selected count of the 64-row query block it belongs to, so the two halves
    are provably reading one decision rather than two.
    """
    kernel = _masked_or_skip()
    heads, seq, dim, block = 4, 4096, 128, 64
    q, k, v = _qkv(heads, seq, dim, device="mps", dtype=torch.bfloat16, seed=131)
    scale = 1.0 / math.sqrt(dim)
    _, stat, counts, _ = _run_masked(kernel, q, k, v, scale, 1.0, block, 128)
    per_tile = counts.repeat_interleave(block // 32, dim=1).reshape(-1)
    assert torch.equal(per_tile.int().cpu(), stat.reshape(-1).int().cpu())


# --- the 20-step policy -----------------------------------------------------

# MiniMax H3 samples a flow model with shift=12.0, whose sigma schedule is
# strongly non-linear; these are the real values ComfyUI produces for 20 steps.
H3_SIGMAS_20 = (
    1.0, 0.9954, 0.9904, 0.9848, 0.9786, 0.9715, 0.9636, 0.9545, 0.944, 0.9317,
    0.9172, 0.8998, 0.8785, 0.8518, 0.8176, 0.7718, 0.7077, 0.6113, 0.4502, 0.126,
)
H3_SIGMA_AT_50_PERCENT = 0.92308
H3_SIGMA_AT_85_PERCENT = 0.67925


def test_twenty_step_window_protects_ten_leading_and_three_trailing():
    holder = sparse.H3SparseHolder(start_percent=0.50, end_percent=0.85)
    holder.start_sigma = H3_SIGMA_AT_50_PERCENT
    holder.end_sigma = H3_SIGMA_AT_85_PERCENT
    sparse_evals = []
    for index, sigma in enumerate(H3_SIGMAS_20):
        holder.sigma = sigma
        if holder.in_sparse_window():
            sparse_evals.append(index)
    assert sparse_evals == [10, 11, 12, 13, 14, 15, 16]


def test_twenty_step_evaluation_counters_split_sparse_from_dense():
    holder = sparse.H3SparseHolder(start_percent=0.50, end_percent=0.85)
    holder.start_sigma = H3_SIGMA_AT_50_PERCENT
    holder.end_sigma = H3_SIGMA_AT_85_PERCENT
    video = torch.zeros(1, 24, 3, 8, 8)
    audio = torch.zeros(1, 32, 2, 10)
    for sigma in H3_SIGMAS_20:
        holder.begin_evaluation(video, audio, sigma)
    assert holder.stats.evaluations == 20
    assert holder.stats.sparse_evaluations == 7
    assert holder.stats.dense_evaluations == 13


def test_diag64_defaults_are_the_twenty_step_policy():
    holder = sparse.H3SparseHolder()
    assert holder.router == sparse.ROUTER_DIAG64
    assert holder.tau == 1.0
    assert holder.start_percent == 0.50
    assert holder.end_percent == 0.85
    assert holder.dense_leading_blocks == 2
    assert holder.min_sequence_tokens == 8192
    assert holder.block_size == 64
    assert holder.temporal_taps == 0


def test_diag64_refuses_a_non_64_block_size():
    with pytest.raises(ValueError, match="block_size must be 64"):
        sparse.H3SparseHolder(block_size=32)
    with pytest.raises(ValueError, match="block_size must be 64"):
        sparse.H3SparseHolder(block_size=128)


def test_diag64_refuses_a_temporal_band():
    with pytest.raises(ValueError, match="temporal_taps must be 0"):
        sparse.H3SparseHolder(temporal_taps=1)


def test_legacy_router_is_reachable_only_through_the_environment(monkeypatch):
    monkeypatch.setenv("ASFP8_SOL_ROUTER", sparse.ROUTER_LEGACY)
    holder = sparse.H3SparseHolder(block_size=32, temporal_taps=2)
    assert holder.router == sparse.ROUTER_LEGACY
    assert holder.block_size == 32
    # and no node widget offers it
    widgets = sparse.AppleSiliconH3SparseAttention.INPUT_TYPES()["required"]
    assert "router" not in widgets
    assert sparse.ROUTER_LEGACY not in str(widgets)


def test_an_unrecognised_router_name_falls_back_to_diag64(monkeypatch):
    monkeypatch.setenv("ASFP8_SOL_ROUTER", "morton_reorder")
    assert sparse.router_name() == sparse.ROUTER_DIAG64


# --- no synchronization on the hot path -------------------------------------

class _StubKernel:
    """Stands in for sol_ext so the sync policy is testable without a GPU."""

    def sol_diag_route(self, q, kcent, kmean, kvar, scale, tau, lq_valid,
                       blk, sink_blocks, local_radius):
        heads, n_q = q.shape[1], q.shape[2] // blk
        return (torch.zeros(heads, n_q, 32, dtype=torch.int32),
                torch.full((heads, n_q), 3, dtype=torch.int32))

    def sol_masked_corrected_fwd(self, q, k, v, kbar, vbar, loglen, mask,
                                 scale, lq, lk, blk, br, half_s, correct):
        return (torch.zeros_like(q),
                torch.zeros(q.shape[1], q.shape[2] // 32, dtype=torch.int32))


def _count_syncs(monkeypatch):
    """Count every call that would drain the MPS queue."""
    counts = {"item": 0, "tolist": 0, "cpu": 0, "numpy": 0}
    for name in counts:
        original = getattr(torch.Tensor, name)

        def wrapper(self, *args, _n=name, _f=original, **kwargs):
            counts[_n] += 1
            return _f(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, wrapper)
    return counts


def test_no_synchronization_inside_a_sparse_attention_call(monkeypatch):
    holder = _holder(min_sequence_tokens=0, dense_leading_blocks=0)
    holder.target_rows = 100
    holder.audio_rows = 40
    monkeypatch.setattr(sparse, "_kernel", lambda: _StubKernel())
    monkeypatch.setattr(sparse, "_eligible", lambda *a: None)
    options = {sparse.STATE_KEY: holder}
    x = torch.zeros(1, 2, 1024, 128, dtype=torch.bfloat16)

    counts = _count_syncs(monkeypatch)
    for _ in range(8):
        sparse.sparse_attention_override(
            None, x, x, x, 2, skip_reshape=True, skip_output_reshape=True,
            scale=0.1, transformer_options=options)
    assert holder.stats.routed_calls == 8
    assert counts == {"item": 0, "tolist": 0, "cpu": 0, "numpy": 0}, counts


def test_the_density_costs_exactly_one_synchronization_per_run(monkeypatch):
    holder = _holder(min_sequence_tokens=0, dense_leading_blocks=0)
    holder.target_rows = 100
    holder.audio_rows = 40
    monkeypatch.setattr(sparse, "_kernel", lambda: _StubKernel())
    monkeypatch.setattr(sparse, "_eligible", lambda *a: None)
    options = {sparse.STATE_KEY: holder}
    x = torch.zeros(1, 2, 1024, 128, dtype=torch.bfloat16)
    for _ in range(8):
        sparse.sparse_attention_override(
            None, x, x, x, 2, skip_reshape=True, skip_output_reshape=True,
            scale=0.1, transformer_options=options)

    counts = _count_syncs(monkeypatch)
    summary = holder.summary()
    assert counts["item"] == 1, counts
    # 8 calls x 16 query blocks x 3 selected, over 16 query blocks x 16 blocks
    assert "effective_density=18.8%" in summary
    assert "router=diag64" in summary


def test_density_accumulates_across_differently_shaped_calls(monkeypatch):
    """A run mixes sequence lengths; the density must weight by block slots."""
    stats = sparse._RunStats()
    stats.add_selected(torch.full((2, 4), 2, dtype=torch.int32), 2 * 4 * 8)
    stats.add_selected(torch.full((2, 16), 12, dtype=torch.int32), 2 * 16 * 16)
    total_selected = 2 * 4 * 2 + 2 * 16 * 12
    total_slots = 2 * 4 * 8 + 2 * 16 * 16
    assert stats.mean_density() == pytest.approx(total_selected / total_slots)
