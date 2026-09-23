"""Exact NA attention tails and Ultra long-sequence scheduling."""
import os

import pytest
import torch
import torch.nn.functional as F

requires_flash = pytest.mark.skipif(
    os.environ.get('ASFP8_ATTN_ROUTER') != '1'
    or not torch.backends.mps.is_available(),
    reason='set ASFP8_ATTN_ROUTER=1 on an MPS device',
)


@requires_flash
@pytest.mark.parametrize('length', [15, 16, 17, 255, 256, 257])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_query_tail_matches_explicit_threadgroup_padding(length, dtype):
    from _patches.mfa4_ext import loader

    kernel = loader.module()
    assert kernel is not None
    generator = torch.Generator().manual_seed(83)
    q = torch.randn(1, 2, length, 128, generator=generator).to('mps', dtype)
    k, v = [torch.randn(1, 2, 131, 128, generator=generator).to('mps', dtype)
            for _ in range(2)]
    padded = ((length + 255) // 256) * 256
    reference = kernel.flash_fwd_na(
        F.pad(q, (0, 0, 0, padded - length)), k, v, 128 ** -0.5)
    output = kernel.flash_fwd_na(q, k, v, 128 ** -0.5)
    torch.mps.synchronize()
    assert torch.equal(output, reference[:, :, :length]), (
        (output.float() - reference[:, :, :length].float()).abs().max().item())


@requires_flash
@pytest.mark.parametrize('batch,length', [
    (1, 8191), (1, 8192), (2, 8193), (1, 8256), (1, 8320),
    (1, 8321), (1, 12287), (1, 12288), (1, 12289), (2, 12289),
    (1, 15456), (1, 15458), (1, 15552), (1, 16384), (1, 16385),
])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_long_h3_schedule_matches_original(batch, length, dtype):
    from _patches.mfa4_ext import loader

    kernel = loader.module()
    assert kernel is not None
    generator = torch.Generator().manual_seed(107)
    q, k, v = [torch.randn(batch, 56, length, 128, generator=generator).to('mps', dtype)
               for _ in range(3)]
    reference = kernel.flash_fwd_na(q, k, v, 128 ** -0.5, long_schedule=False)
    output = kernel.flash_fwd_na(q, k, v, 128 ** -0.5)
    torch.mps.synchronize()
    assert torch.equal(output.cpu().view(torch.uint8), reference.cpu().view(torch.uint8)), (
        (output.float() - reference.float()).abs().max().item())
