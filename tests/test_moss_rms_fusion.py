"""Exactness checks for the MOSS composite-RMS memory-pass fusion."""
import pytest
import torch
from _patches import moss_fast as p


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')
@pytest.mark.parametrize('batch,rows', [(1, 1), (1, 31), (1, 257), (1, 4096), (1, 6197), (2, 127)])
def test_same_reduction_rms_mod_bytes(batch, rows):
    rng = torch.Generator().manual_seed(731)
    with torch.inference_mode():
        assert p._ready()
        x = torch.randn(batch, rows, 3072, generator=rng).to(device='mps', dtype=torch.bfloat16)
        scale = torch.randn(batch, 1, 3072, generator=rng).to(device='mps', dtype=torch.bfloat16)
        shift = torch.randn(batch, 1, 3072, generator=rng).to(device='mps', dtype=torch.bfloat16)
        ref = p._mod_tensor(torch.nn.functional.rms_norm(x, (3072,), eps=1e-6), scale, shift)
        got = p._rms_mod_tensor(x, scale, shift, 1e-6)
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
