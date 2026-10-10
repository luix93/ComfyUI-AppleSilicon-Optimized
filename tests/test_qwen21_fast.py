"""Qwen-Image 2.1 fusions retain all BF16 rounding and view offsets."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from comfy_kitchen.backends.eager.rope import rms_rope

from _patches import qwen21_fast as patch


def bits(a,b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),b.cpu().contiguous().view(torch.uint8))


@pytest.fixture
def kernel(monkeypatch):
    if not (torch.backends.mps.is_available() and patch.torch_mpp_qcache._hardware()
            and patch.torch_mpp_qcache._torch_build_supported()):
        pytest.skip('validated M5 Ultra / PyTorch 2.14 required')
    monkeypatch.setattr(patch,'_model',SimpleNamespace(comfy=SimpleNamespace(model_management=SimpleNamespace(in_training=False))))
    assert patch._lib() is not None
    return patch._library


@pytest.mark.parametrize('seq,prefix',[(1,0),(37,37),(335,71),(4096,0)])
@pytest.mark.parametrize('seed',[0,19,271828])
def test_modulation_and_inplace_residual(kernel,seq,prefix,seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        x=torch.randn(1,seq+2,4096,device='mps',dtype=torch.bfloat16)[:,1:-1]
        y=torch.randn_like(x)
        scale=[torch.randn(1,1,4096,device='mps',dtype=torch.bfloat16) for _ in range(2)]
        zero=torch.zeros_like(scale[0]);norm=SimpleNamespace(eps=1e-6)
        ref=F.layer_norm(x,(4096,),eps=norm.eps)*(1+scale[1])+zero
        ref[:,:prefix]=F.layer_norm(x[:,:prefix],(4096,),eps=norm.eps)*(1+scale[0])+zero
        bits(patch._modulate(norm,x,scale,prefix,zero),ref)
        ref=x.clone();ref[:,prefix:].addcmul_(y[:,prefix:],scale[1]);ref[:,:prefix].addcmul_(y[:,:prefix],scale[0])
        got=patch._residual(x,y,scale,prefix)
        assert got is x
        bits(got,ref)


@pytest.mark.parametrize('seq,eps',[(1,1e-6),(335,1e-6),(4096,1e-6),(73,1e-5)])
@pytest.mark.parametrize('seed',[0,19,271828])
def test_qk_norm_and_rope(kernel,seq,eps,seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        q,k=[torch.randn(1,seq+2,32,128,device='mps',dtype=torch.bfloat16)[:,1:-1] for _ in range(2)]
        qw,kw=[torch.nn.Parameter(torch.randn(128,device='mps',dtype=torch.bfloat16),requires_grad=False) for _ in range(2)]
        pe=torch.randn(1,seq,1,64,2,2,device='mps')
        for a,b in zip(patch._rope(q,k,pe,qw,kw,eps),rms_rope(q,k,pe,qw,kw,eps)):bits(a,b)


def test_cpu_and_disabled_route_fall_back(monkeypatch):
    sentinel=object();calls=[]
    monkeypatch.setattr(patch,'_original_norm',lambda *a:calls.append(a) or sentinel)
    x=torch.ones(1,2,4096,dtype=torch.bfloat16)
    args=(object(),x,[object(),object()],0,object())
    assert patch._modulate(*args) is sentinel
    assert calls==[args]
    monkeypatch.setenv('ASFP8_QWEN21_FAST','0')
    assert not patch._enabled('ROPE',x)


def test_attention_hooks_keep_original_forward(monkeypatch):
    monkeypatch.setattr(patch,'_enabled',lambda *a:True)
    monkeypatch.setattr(patch,'_model',SimpleNamespace(comfy=SimpleNamespace(model_management=SimpleNamespace(in_training=False))))
    calls=[]
    monkeypatch.setattr(patch,'_original_attention',lambda *a:calls.append(a) or 'original')
    layer=object();x=object();pe=object();attn=object();opts={'patches':{'attn1_patch':[object()]}}
    assert patch._attention(layer,x,pe,attn,42,opts)=='original'
    assert calls==[(layer,x,pe,attn,42,opts)]


@pytest.mark.parametrize('containers', [False, True])
def test_attention_preserves_current_and_legacy_callback_contract(kernel, monkeypatch, containers):
    from contextlib import contextmanager
    from comfy.ldm.modules.attention import AttentionTensorContainer
    model = SimpleNamespace(comfy=SimpleNamespace(model_management=SimpleNamespace(in_training=False)))
    if containers:
        model.AttentionTensorContainer = AttentionTensorContainer
    monkeypatch.setattr(patch, '_model', model)
    @contextmanager
    def cast(norm, x, **kwargs):
        yield norm.weight, None
    monkeypatch.setattr(patch, '_ops', SimpleNamespace(CastBiasWeightContext=cast))
    with torch.inference_mode():
        torch.manual_seed(114)
        x = torch.randn(1, 3, 4096, device='mps', dtype=torch.bfloat16)
        pe = torch.randn(1, 3, 1, 64, 2, 2, device='mps')
        q, k, v = [torch.randn_like(x) for _ in range(3)]
        qw, kw = [torch.ones(128, device='mps', dtype=x.dtype) for _ in range(2)]
        expected_q, expected_k = rms_rope(q.view(1, 3, 32, 128), k.view(1, 3, 32, 128), pe, qw, kw, 1e-6)
        preferred = object()
        layer = SimpleNamespace(heads=32, norm_q=SimpleNamespace(eps=1e-6, weight=qw),
            norm_k=SimpleNamespace(eps=1e-6, weight=kw), to_q=lambda x:q,
            to_k=lambda x:k, to_v=lambda x:v, to_out=[lambda x:x], comfy_attention=preferred)
        def attention(a, b, c, heads, preferred_attention):
            assert heads == 32 and preferred_attention is preferred
            if containers:
                assert all(isinstance(t, AttentionTensorContainer) for t in (a, b, c))
                a, b, c = [t.take() for t in (a, b, c)]
            bits(a, expected_q)
            bits(b, expected_k)
            bits(c, v.view(1, 3, 32, 128))
            return c.flatten(2)
        bits(patch._attention(layer, x, pe, attention, 0), v)


@pytest.fixture(scope='module')
def original_vae_kernel():
    if not torch.backends.mps.is_available():
        pytest.skip('MPS required')
    source = (Path(patch.__file__).parent / 'wan_ext/wan_kernels.mm').read_text()
    header = source[source.index('#include <metal_stdlib>'):source.index('// LayerNorm (no affine)')]
    body = source[source.index('#define CR_ITERS'):source.index('// Build the folded conv2d')]
    return torch.mps.compile_shader((header + body).replace('@DT@','bfloat').replace('@DTN@','bfloat'))


@pytest.mark.parametrize('batch,channels,spatial', [(1,1152,4096),(1,576,8448),
                                                  (1,144,120832),(2,37,259)])
@pytest.mark.parametrize('silu', [False, True])
def test_vae_spatial_scheduling(kernel, original_vae_kernel, batch, channels, spatial, silu):
    torch.manual_seed(spatial)
    with torch.inference_mode():
        # Contiguous tensors can still have a nonzero storage offset.
        x = torch.randn(batch*channels*spatial+1,device='mps',dtype=torch.bfloat16)[1:].view(batch,channels,spatial)
        gamma = torch.randn(channels+1,device='mps',dtype=torch.bfloat16)[1:]
        scale,eps = channels**.5,1e-12
        ref = torch.empty_like(x)
        original_vae_kernel.chan_rms(x,gamma,ref,channels,spatial,batch,float(scale),float(eps),int(silu),
                                    threads=batch*((spatial+8191)//8192)*256,group_size=256)
        bits(patch._vae_rms(x,gamma,scale,eps,silu),ref)
