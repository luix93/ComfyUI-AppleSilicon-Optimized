"""Current-API Ideogram preparation, SDPA layout, offsets and safe fallbacks."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from comfy_kitchen.backends.eager.rope import rms_rope_split_half
from _patches import ideogram4_exact as patch


def bits(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),
                       b.cpu().contiguous().view(torch.uint8))


@pytest.fixture
def runtime(monkeypatch):
    if not (torch.backends.mps.is_available() and patch.torch_mpp_qcache._hardware()
            and torch.__version__.split('+')[0] == '2.14.0'
            and torch.version.git_version == patch.torch_mpp_qcache._TORCH_GIT):
        pytest.skip('validated M5 Ultra / PyTorch 2.14 required')
    for flag in ('ASFP8_DIT_EXT','ASFP8_IDEOGRAM_FAST','ASFP8_IDEOGRAM_EXACT',
                 'ASFP8_IDEOGRAM_PREP','ASFP8_IDEOGRAM_ATTN_LAYOUT'):
        monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')
    monkeypatch.setattr(patch, '_model', SimpleNamespace(comfy=SimpleNamespace(
        quant_ops=SimpleNamespace(ck=SimpleNamespace(rms_rope_split_half=rms_rope_split_half)),
        model_management=SimpleNamespace(in_training=False))))


@pytest.mark.parametrize('seq,seed,eps', [(1,0,1e-5),(79,11,1e-6),
                                         (4414,42,1e-5),(4096,719,1e-5)])
def test_preparation(runtime, monkeypatch, seq, seed, eps):
    with torch.inference_mode():
        torch.manual_seed(seed)
        # Offsets test the shader bindings, not just zero-offset allocations.
        qkv = torch.randn(1,seq+2,3*18*256,device='mps',dtype=torch.bfloat16)[:,1:-1]
        pe = torch.randn(1,seq+2,1,128,2,2,device='mps')[:,1:-1]
        qw,kw = [torch.nn.Parameter(torch.randn(257,device='mps',dtype=torch.bfloat16)[1:],
                                    requires_grad=False) for _ in range(2)]
        q,k,v = qkv.view(1,seq,3,18,256).unbind(2)
        eq,ek = rms_rope_split_half(q,k,pe,qw,kw,eps)
        expected = (eq.transpose(1,2),ek.transpose(1,2),v.transpose(1,2))
        for a,b in zip(patch._prep(qkv,pe,qw,kw,18,eps),expected):bits(a,b)
        monkeypatch.setenv('ASFP8_IDEOGRAM_PREP','0')
        for a,b in zip(patch._prep(qkv,pe,qw,kw,18,eps),expected):bits(a,b)


@pytest.mark.parametrize('seq', [2049,4096,4414,4609])
def test_attention_layout(runtime, seq):
    with torch.inference_mode():
        torch.manual_seed(seq)
        q,k,v = [torch.randn(1,seq+2,18,256,device='mps',dtype=torch.bfloat16)
                 [:,1:-1].transpose(1,2) for _ in range(3)]
        ref = F.scaled_dot_product_attention(q,k,v)
        token=patch._scope.set(True)
        try:out=patch.maybe_attention(q,k,v,output_bshd=True)
        finally:patch._scope.reset(token)
        assert out is not None and not patch._attention_dead
        assert out.transpose(1,2).is_contiguous()
        bits(out,ref)


def test_attention_fallbacks(runtime, monkeypatch):
    with torch.inference_mode():
        q=torch.zeros(1,18,2048,256,device='mps',dtype=torch.bfloat16)
        assert patch.maybe_attention(q,q,q,output_bshd=True) is None  # outside Ideogram
        token=patch._scope.set(True)
        try:
            for kw in ({'is_causal':True},{'dropout_p':.1},{'scale':.25},
                       {'attn_mask':torch.zeros(1,1,2048,2048,device='mps',dtype=q.dtype)}):
                assert patch.maybe_attention(q,q,q,output_bshd=True,**kw) is None
            for flag in ('ASFP8_IDEOGRAM_FAST','ASFP8_IDEOGRAM_EXACT','ASFP8_IDEOGRAM_ATTN_LAYOUT'):
                monkeypatch.setenv(flag,'0')
                assert patch.maybe_attention(q,q,q,output_bshd=True) is None
                monkeypatch.setenv(flag,'1')
            assert patch.maybe_attention(q,q,q) is None
            assert patch.maybe_attention(q[:,:17],q[:,:17],q[:,:17],output_bshd=True) is None
        finally:patch._scope.reset(token)


def test_cpu_keeps_original(monkeypatch):
    calls=[]
    monkeypatch.setattr(patch,'_original',lambda *a:calls.append(a) or 'original')
    monkeypatch.setattr(patch,'_runtime',lambda:True)
    monkeypatch.setattr(patch,'_model',SimpleNamespace(comfy=SimpleNamespace(model_management=SimpleNamespace(in_training=False))))
    with torch.inference_mode():
        args=(object(),torch.ones(1,3,4608,dtype=torch.bfloat16),None,object(),{})
        assert patch._attention(*args)=='original'
        assert calls==[args]
