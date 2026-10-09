"""Optional fixed-budget routing: protected globals, deterministic ties, cloning."""
import math
import os
import pytest
import torch
from _patches import h3_sol_mps as sol, minimax_h3_sparse as sparse


@pytest.mark.parametrize("ratio", [-.1, 1, float("nan"), float("inf")])
def test_invalid_budget_rejected(ratio):
    with pytest.raises(ValueError, match="topk_ratio"):
        sparse.H3SparseHolder(backend="int8", topk_ratio=ratio)


def test_budget_is_independent_and_cloned():
    holder = sparse.H3SparseHolder(backend="int8", topk_ratio=.15, tau=.7)
    clone = holder.clone()
    assert clone is not holder and clone.topk_ratio == .15 and clone.tau == .7
    assert sparse.H3SparseHolder().topk_ratio == 0
    with pytest.raises(ValueError, match="INT8"):
        sparse.H3SparseHolder(backend="bf16", topk_ratio=.15)


@pytest.mark.skipif(os.environ.get("H3_RUN_SOL_ROUTER_TESTS") != "1",
                    reason="explicit opt-in GPU test; avoid interference with benchmark runs")
@pytest.mark.parametrize("tied", [False, True])
def test_metal_routes_match_fixed_budget_reference(tied):
    if not sol.supported():
        pytest.skip("M5 Metal required")
    h, tokens, prefix, ratio = 2, 333, 70, .25
    blocks = math.ceil(tokens / 64)
    generator = torch.Generator().manual_seed(712)
    qc = torch.randint(-3, 4, (h, blocks, 128), generator=generator).float()
    kc = torch.randint(-3, 4, (h, 64, 128), generator=generator).half()
    if tied:
        kc.zero_()
    expected = torch.zeros(h, blocks, blocks, dtype=torch.bool)
    sinks = math.ceil(prefix / 64)
    keep = math.ceil((blocks - sinks) * ratio)
    for head in range(h):
        scores = qc[head] @ kc[head, :blocks].float().T
        for row in range(blocks):
            rank = sorted(range(sinks, blocks), key=lambda col: (-scores[row, col], col))
            expected[head, row, rank[:keep]] = True
            expected[head, row, :sinks] = True
            expected[head, row, max(0, row-1):row+2] = True
            if row * 64 < prefix:
                expected[head, row] = True
    qc, kc = qc.to("mps"), kc.to("mps")
    routes = torch.empty(h, blocks, blocks, device="mps", dtype=torch.uint8)
    bits = torch.empty(h, blocks, 1, device="mps", dtype=torch.int32)
    params = sol._parameters(tokens, h, prefix, 128**-.5, 1.3)
    library, _ = sol._library(tokens, h, True, "linear", ratio)
    stats = torch.empty(h, 128, 2, device="mps")
    for _ in range(2):
        library.sol_route_topk(qc, kc, stats, routes, params, bits,
            threads=(blocks*128,h,1), group_size=(128,1,1))
        assert torch.equal(routes.cpu().bool(), expected)
        encoded = sum(expected[..., col].int() << col for col in range(blocks))
        assert torch.equal(bits.cpu().squeeze(-1), encoded)


@pytest.mark.skipif(os.environ.get("H3_RUN_SOL_ROUTER_TESTS") != "1", reason="opt-in GPU check")
def test_maximum_sort_size_fits_metal_shared_memory():
    if not sol.supported():
        pytest.skip("M5 Metal required")
    tokens, blocks = 262144, 4096
    qc = torch.zeros(1, blocks, 128, device="mps")
    kc = torch.zeros_like(qc, dtype=torch.float16)
    routes = torch.empty(1, blocks, blocks, device="mps", dtype=torch.uint8)
    bits = torch.empty(1, blocks, blocks//32, device="mps", dtype=torch.int32)
    stats = torch.empty(1,128,2,device="mps")
    library, _ = sol._library(tokens,1,True,"linear",.10)
    # One row proves the maximum workgroup allocation fits without launching
    # a huge clip or allocating Q/K/V at the full token count.
    library.sol_route_topk(qc,kc,stats,routes,sol._parameters(tokens,1,0,128**-.5,1.3),bits,
        threads=(128,1,1),group_size=(128,1,1))
    expected = torch.arange(blocks) < math.ceil(blocks*.10)
    assert torch.equal(routes[0,0].cpu().bool(),expected)
