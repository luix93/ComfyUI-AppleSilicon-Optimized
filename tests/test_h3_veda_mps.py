"""Checks sparse tile-list execution, including padding and global rows."""

import pytest
import torch

from _patches import torch_mpp_qcache
from _patches.h3_veda_mps import MetalBackend, gather_pool
from _patches.h3_runtime.veda.backends import base
from _patches.h3_runtime.veda.core import reference, selection, tiling, predictor


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("key_block", [32, 64, 128])
def test_native_sparse_attention_selftest(dtype, key_block):
    if not torch_mpp_qcache._hardware():
        pytest.skip("requires M5 Ultra")
    with torch.inference_mode():
        base.self_test(MetalBackend(key_block), torch.device("mps"), dtype)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("references", [False, True])
def test_tile_lists_with_empty_video_tiles_and_variable_budgets(references):
    if not torch_mpp_qcache._hardware():
        pytest.skip("requires M5 Ultra")
    with torch.inference_mode():
        spans = [tiling.TiledSpan(33, (3, 3, 3), tiling.TileShape(4, 4, 8))]
        if references:
            spans.insert(0, tiling.TiledSpan(0, (1, 3, 3), tiling.TileShape(1, 8, 16)))
        layout = tiling.build_tile_layout(spans, 70, 'mps')
        gen = torch.Generator().manual_seed(33)
        q, k, v = [tiling.gather_tiles(t.to('mps', torch.bfloat16), layout, torch.arange(3, device='mps'))
                   for t in torch.randn(3, 70, 3, 128, generator=gen)]
        scores = torch.randn(3, layout.n_video_tiles, layout.n_video_tiles, generator=gen).to('mps')
        index, keep = selection.select(scores, layout,
            selection.column_blocks(layout, selection.Budget.sparsity(90), selection.Budget.sparsity(50)))
        mask = selection.block_mask(index, keep, layout)
        out = MetalBackend().attend_selected(q, k, v, index, keep, layout)
        wanted = reference.block_sparse_attention(q, k, v, mask, layout)
        valid = layout.slot_valid.bool()
        assert torch.isfinite(out[valid]).all()
        torch.testing.assert_close(out[valid].float(), wanted[valid].float(), atol=0.02, rtol=0.02)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_gather_and_pool_matches_reference_with_strided_heads(dtype):
    with torch.inference_mode():
        layout = tiling.build_tile_layout([
            tiling.TiledSpan(35, (5, 5, 5), tiling.TileShape(4, 4, 8))], 220, 'mps')
        hs = torch.tensor([3, 0, 2], dtype=torch.int64, device='mps')
        source = torch.randn(3, 4, 220, 128, generator=torch.Generator().manual_seed(19)).to('mps', dtype)
        q, k, v = [t.transpose(0, 1) for t in source]
        qt, kt, vt, fq, fk = gather_pool(q, k, v, layout, hs)
        for original, got in zip((q, k, v), (qt, kt, vt)):
            torch.testing.assert_close(got, tiling.gather_tiles(original, layout, hs), rtol=0, atol=0)
        torch.testing.assert_close(fq, predictor.pool_video_tiles(qt, layout), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fk, predictor.pool_video_tiles(kt, layout), rtol=1e-5, atol=1e-6)
