"""Native byte checks for the M5 Ultra Krea2 1MP output-tile schedule."""
import os
import subprocess

import pytest
import torch

from _patches.int8_ext import loader

pytestmark = pytest.mark.skipif(
    os.environ.get('ASFP8_INT8_EXT') != '1' or not torch.backends.mps.is_available(),
    reason='requires ASFP8_INT8_EXT=1 and MPS')


@pytest.fixture(scope='module')
def kernel():
    if 'M5 Ultra' not in subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True):
        pytest.skip('the specialization is enabled only on M5 Ultra')
    mod = loader.module()
    assert mod is not None, 'native extension failed to build'
    return mod


@pytest.mark.parametrize('rows', [4095, 4096, 4109, 4352, 4353])
@pytest.mark.parametrize('columns', [1536, 6144])
@pytest.mark.parametrize('with_bias', [False, True])
def test_fused_outputs_match_previous_schedule(kernel, monkeypatch, rows, columns, with_bias):
    # Non-aligned rows exercise both tile tails; endpoints also exercise the
    # unchanged outside-range route. Include negative values and scale zeros.
    generator = torch.Generator().manual_seed(rows + columns)
    a = torch.randint(-128, 128, (rows, 6144), generator=generator, dtype=torch.int8).to('mps')
    b = torch.randint(-128, 128, (columns, 6144), generator=generator, dtype=torch.int8).to('mps')
    rs = (torch.rand(rows, generator=generator) * .01).to('mps')
    cs = (torch.rand(columns, generator=generator) * .02).to('mps')
    rs[0] = 0
    cs[0] = 0
    bias = torch.randn(columns, generator=generator).bfloat16().to('mps') if with_bias else None
    with torch.inference_mode():
        monkeypatch.setenv('ASFP8_KREA2_GEMM_TILE', '0')
        reference = kernel.i8_matmul2d_nt_fused_cs(a, b, rs, cs, bias)
        monkeypatch.setenv('ASFP8_KREA2_GEMM_TILE', '1')
        output = kernel.i8_matmul2d_nt_fused_cs(a, b, rs, cs, bias)
        torch.mps.synchronize()
        assert torch.equal(output.cpu().view(torch.uint8), reference.cpu().view(torch.uint8))
