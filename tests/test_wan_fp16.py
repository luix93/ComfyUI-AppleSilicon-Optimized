"""Wan's FP16 rescale/bias rounding and batched attention layouts."""
import pytest
import torch
import torch.nn.functional as F

from _patches import int8_linear_kernel_mps as native
from _patches import torch_mpp_qcache as attention
from _patches import wan_fp16


def bits(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),
                       b.cpu().contiguous().view(torch.uint8))


@pytest.fixture
def runtime(monkeypatch):
    if not (torch.backends.mps.is_available() and attention._hardware()
            and torch.__version__.split('+')[0] == '2.14.0'
            and torch.version.git_version == attention._TORCH_GIT):
        pytest.skip('validated M5 Ultra / PyTorch 2.14 required')
    for flag in ('ASFP8_INT8_EXT', 'ASFP8_DIT_EXT', 'ASFP8_WAN_FAST',
                 'ASFP8_WAN_EXACT', 'ASFP8_WAN_FP16_GEMM', 'ASFP8_WAN_ATTN',
                 'ASFP8_MPP_QCACHE'):
        monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')


@pytest.mark.parametrize('rows,width,inner', [(1,5120,5120),(65,5120,5120),
                                            (1024,5120,5120),(10240,5120,5120),
                                            (257,13824,5120),(257,5120,13824)])
@pytest.mark.parametrize('with_bias', [False, True])
def test_scaled_integer_output(runtime, monkeypatch, rows, width, inner, with_bias):
    from _patches.int8_ext import loader
    monkeypatch.setattr(native, '_kernel', loader.module())
    assert native._kernel is not None
    with torch.inference_mode():
        torch.manual_seed(rows)
        # Nonzero offsets exercise the extension's storage-offset handling.
        x = torch.randint(-127,128,(rows+1,inner),device='mps',dtype=torch.int8)[1:]
        w = torch.randint(-127,128,(width+1,inner),device='mps',dtype=torch.int8)[1:]
        rs = torch.rand(rows+1,1,device='mps')[1:] * .02
        cs = torch.rand(width+1,device='mps')[1:] * .03
        rs[0] = 0
        if rows >= 65:
            rs[1:5,0] = torch.tensor([1e-10,1e-7,.01,1.],device='mps')
        bias = (torch.nn.Parameter(torch.randn(width,device='mps',dtype=torch.float16),
                                   requires_grad=False) if with_bias else None)
        shape = (1,rows,inner)
        monkeypatch.setenv('ASFP8_WAN_EXACT', '0')
        reference = native._int8_linear_prequant(x,rs,w,cs,bias,torch.float16,shape)
        monkeypatch.setenv('ASFP8_WAN_EXACT', '1')
        before = wan_fp16._calls
        got = native._int8_linear_prequant(x,rs,w,cs,bias,torch.float16,shape)
        assert wan_fp16._calls == before + 1 and not wan_fp16._dead
        bits(got, reference)
        monkeypatch.setenv('ASFP8_WAN_FP16_GEMM', '0')
        assert wan_fp16.maybe_prequant(native._kernel,x,rs,w,cs,bias,shape) is None


@pytest.mark.parametrize('batch,seq,scale', [(1,2049,None),(2,5120,None),
                                          (2,2111,0.),(2,21504,None)])
def test_batched_attention_layout(runtime, batch, seq, scale):
    with torch.inference_mode():
        torch.manual_seed(seq)
        q,k,v = [torch.randn(batch,seq+2,40,128,device='mps',dtype=torch.float16)
                 [:,1:-1].transpose(1,2) for _ in range(3)]
        expected = F.scaled_dot_product_attention(q,k,v,scale=scale)
        for layout in (False,True):
            got = attention.maybe_attention(q,k,v,scale=scale,output_bshd=layout)
            assert got is not None and not attention._dead
            bits(got,expected)
            if layout:
                assert got.transpose(1,2).is_contiguous()


def test_attention_fallbacks(runtime, monkeypatch):
    with torch.inference_mode():
        x = torch.zeros(2,40,2048,128,device='mps',dtype=torch.float16)
        for kwargs in ({'is_causal':True},{'dropout_p':.1},
                       {'attn_mask':torch.ones(2048,device='mps',dtype=torch.bool)}):
            assert attention.maybe_attention(x,x,x,**kwargs) is None
        assert attention.maybe_attention(x,x[:,:,:256],x[:,:,:256]) is None
        assert attention.maybe_attention(x[:,:32],x[:,:32],x[:,:32]) is None
        for flag in ('ASFP8_WAN_EXACT','ASFP8_WAN_ATTN','ASFP8_WAN_FAST'):
            monkeypatch.setenv(flag,'0')
            assert attention.maybe_attention(x,x,x) is None
            monkeypatch.setenv(flag,'1')
        monkeypatch.setenv('PYTORCH_MPS_FAST_MATH','1')
        assert attention.maybe_attention(x,x,x) is None


def test_prequant_unsupported(runtime):
    x = torch.zeros(1,5120,dtype=torch.int8)
    w = torch.zeros(5120,5120,dtype=torch.int8)
    with torch.inference_mode():
        assert wan_fp16.maybe_prequant(None,x,torch.ones(1),w,
                                       torch.ones(5120),None,(1,1,5120)) is None


@pytest.mark.parametrize('batch,seq', [(1,2051),(2,21504)])
def test_text_cross_attention(runtime, monkeypatch, batch, seq):
    monkeypatch.setenv('ASFP8_WAN_CROSS_ATTN', '1')
    with torch.inference_mode():
        torch.manual_seed(804)
        q = torch.randn(batch,seq+2,40,128,device='mps',dtype=torch.float16)[:,1:-1].transpose(1,2)
        k,v = [torch.randn(batch,514,40,128,device='mps',dtype=torch.float16)[:,1:-1].transpose(1,2)
               for _ in range(2)]
        expected = F.scaled_dot_product_attention(q,k,v)
        got = attention.maybe_attention(q,k,v,output_bshd=True)
        assert got is not None and got.transpose(1,2).is_contiguous()
        bits(got,expected)
        monkeypatch.setenv('ASFP8_WAN_CROSS_ATTN','0')
        assert attention.maybe_attention(q,k,v) is None
