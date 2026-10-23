"""VSA geometry, trained routing rules and coarse-branch regression checks."""
import math
import pytest
import torch

from _patches import h3_vsa_mps as vsa


def inputs(plan, heads=2, dtype=torch.float32):
    rng = torch.Generator().manual_seed(127)
    q, k, v, gate = torch.randn(4, plan.seq_len, heads, 128, generator=rng).to(dtype)
    return q, k, v, gate * .2


@pytest.mark.parametrize("segments", [
    (33, 7, (3, 5, 9)),
    (17, (1, 5, 9), 3, (5, 3, 9)),
    (0, (1, 1, 1)),
    (129, (5, 5, 9), 1, (3, 3, 3), 5, (4, 4, 8)),
])
def test_plan_is_bijective_and_padding_is_a_valid_prefix(segments):
    plan = vsa.build_plan(segments)
    rows = plan.perm.reshape(-1, 128)
    assert torch.equal(rows >= 0, torch.arange(128)[None] < plan.valid_count[:, None])
    real = plan.perm[plan.perm >= 0]
    assert torch.equal(real.sort().values, torch.arange(plan.seq_len))
    assert torch.equal(plan.perm[plan.inverse], torch.arange(plan.seq_len))
    x = torch.arange(plan.seq_len)[:, None, None].expand(-1, 2, 128).float()
    got = vsa.gather(x, plan)
    assert not got[plan.perm < 0].any()
    assert torch.equal(got[plan.inverse], x)


def test_dense_segments_do_not_share_a_tile_and_cube_order_is_t_h_w():
    plan = vsa.build_plan((3, 2, (5, 5, 9)))
    rows = plan.perm.reshape(-1, 128)
    assert torch.equal(rows[-2, :3], torch.arange(3))
    assert torch.equal(rows[-1, :2], torch.arange(3, 5))
    grid = torch.arange(5, plan.seq_len).reshape(5, 5, 9)
    assert torch.equal(rows[0], grid[:4, :4, :8].reshape(-1))
    assert torch.equal(rows[1, :16], grid[:4, :4, 8:].reshape(-1))
    assert torch.equal(rows[2, :32], grid[:4, 4:, :8].reshape(-1))


def test_each_reference_region_gets_its_own_budget_without_diagonal_rescue():
    plan = vsa.build_plan(((8, 4, 8), 17, (4, 4, 16)))
    scores = torch.zeros(1, plan.n_tiles, plan.n_tiles)
    scores[..., 0] = 8
    scores[..., 2] = 9
    index, keep = vsa.select(scores, plan, .9, .1)
    assert index.shape == (1, 4, 2)
    assert torch.equal(index, torch.tensor([[[0, 2]]]).expand(1, 4, 2))
    assert keep.all()
    dense, _ = vsa.select(scores, plan, 0, .1)
    assert dense.shape[-1] == plan.n_video_tiles


@torch.inference_mode()
def test_zero_gate_dense_budget_recovers_dense_attention():
    plan = vsa.build_plan((7, (3, 3, 5)))
    q, k, v, gate = inputs(plan)
    gate.zero_()
    got = vsa.attention(q, k, v, gate, plan, backend="reference", sparsity=0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(got, expected, atol=3e-7, rtol=3e-6)


@torch.inference_mode()
def test_nonzero_raw_gate_adds_pooled_branch_even_on_dense_queries():
    plan = vsa.build_plan((3, (2, 3, 5)))
    q, k, v, gate = inputs(plan)
    q.zero_()
    k.zero_()
    gate.fill_(-.5)  # Must not pass through a sigmoid or clamp.
    no_gate = vsa.attention(q, k, v, gate*0, plan, backend="reference")
    got = vsa.attention(q, k, v, gate, plan, backend="reference")
    # Uniform pooled attention weights every TILE equally, not every real row.
    # The 3-row dense tile and the 30-row video tile therefore have equal weight.
    mean = (v[:3].mean(0) + v[3:].mean(0))/2
    torch.testing.assert_close(got, no_gate - .5*mean, atol=2e-7, rtol=3e-6)
    assert not torch.equal(got[:3], no_gate[:3])


@torch.inference_mode()
def test_head_chunking_does_not_change_fp32_reference():
    plan = vsa.build_plan((5, (1, 4, 17), 3, (2, 3, 9)))
    args = inputs(plan, heads=3)
    one = vsa.attention(*args, plan, backend="reference", head_chunk=1)
    all_heads = vsa.attention(*args, plan, backend="reference", head_chunk=3)
    torch.testing.assert_close(one, all_heads, atol=2e-7, rtol=3e-6)


@pytest.mark.parametrize("segments", [(), (5,), ((0, 1, 1),), ((1, 2),), (True, (1, 1, 1)), (-1, (1, 1, 1))])
def test_invalid_geometry_is_rejected(segments):
    with pytest.raises(ValueError):
        vsa.build_plan(segments)


@torch.inference_mode()
@pytest.mark.parametrize("kwargs", [dict(sparsity=math.nan), dict(sparsity=1),
    dict(reference_keep=0), dict(reference_keep=math.inf), dict(head_chunk=0),
    dict(workspace_mib=.0001), dict(backend="dense")])
def test_invalid_dispatch_is_rejected(kwargs):
    plan = vsa.build_plan((1, (1, 1, 1)))
    with pytest.raises(ValueError):
        vsa.attention(*inputs(plan), plan, **(dict(backend="reference") | kwargs))


def test_autograd_is_explicitly_rejected():
    plan = vsa.build_plan(((1, 1, 1),))
    # Other suites import ComfyUI, which can leave autograd disabled globally.
    with torch.enable_grad(), pytest.raises(ValueError, match="inference-only"):
        vsa.attention(*inputs(plan), plan, backend="reference")


@torch.inference_mode()
def test_metal_never_silently_uses_cpu_or_dense():
    plan = vsa.build_plan(((1, 1, 1),))
    with pytest.raises(RuntimeError, match="Metal VSA currently requires"):
        vsa.attention(*inputs(plan), plan)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("segments", [(33, 7, (5, 5, 9)),
    (7, (1, 5, 9), 5, (3, 3, 17)), ((1, 1, 1),)])
@torch.inference_mode()
def test_metal_matches_masked_fp32_oracle_with_learned_branch(dtype, tile_size, segments):
    from _patches import torch_mpp_qcache
    if not torch_mpp_qcache._hardware():
        pytest.skip("requires M5 Ultra")
    cpu_plan = vsa.build_plan(segments, tile_size=tile_size)
    args = inputs(cpu_plan, heads=3, dtype=dtype)
    expected = vsa.attention(*args, cpu_plan, backend="reference", head_chunk=3)
    gpu_plan = vsa.build_plan(segments, "mps", tile_size)
    got = vsa.attention(*(t.to("mps") for t in args), gpu_plan, head_chunk=2).cpu()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.float(), expected.float(), atol=.008, rtol=.025)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_staging_preserves_strided_values_and_valid_row_pooling(dtype):
    plan = vsa.build_plan((17, (3, 5, 9)), "mps")
    rng = torch.Generator().manual_seed(17)
    # Exercise offset head slices and non-unit channel strides.
    args = [t[:, 1:3, ::2] for t in torch.randn(4, plan.seq_len, 4, 256, generator=rng).to("mps", dtype)]
    qt, kt, vt, gt, pq, pk, pv = vsa._metal_gather_pool(*args, plan)
    for source, staged in zip(args, (qt, kt, vt, gt)):
        torch.testing.assert_close(staged, vsa.gather(source, plan), atol=0, rtol=0)
    for staged, pooled in zip((qt, kt, vt), (pq, pk, pv)):
        torch.testing.assert_close(pooled, vsa.pool(staged, plan), atol=2e-7, rtol=2e-5)
