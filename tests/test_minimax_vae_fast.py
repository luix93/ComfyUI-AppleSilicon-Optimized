"""FP16 VAE kernel byte checks; GPU cases require ASFP8_DIT_EXT=1 and MPS."""
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from _patches import minimax_vae_fast as patch

gpu = pytest.mark.skipif(
    os.environ.get('ASFP8_DIT_EXT') != '1' or not torch.backends.mps.is_available(),
    reason='requires ASFP8_DIT_EXT=1 and MPS')

def assert_bytes(a,b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),
                       b.cpu().contiguous().view(torch.uint8))

@pytest.fixture(scope='module')
def kernel():
    from _patches.minimax_vae_ext.loader import module
    result=module()
    assert result is not None
    return result

def test_disabled_install_does_not_import_comfy(monkeypatch):
    monkeypatch.setattr(patch,'_installed',False)
    monkeypatch.setenv('ASFP8_DIT_EXT','1')
    monkeypatch.setenv('ASFP8_MINIMAX_VAE_FAST','0')
    patch.install()
    assert not patch._installed

def test_cpu_and_training_keep_original_calls(monkeypatch):
    seen=[]
    def original(*args):
        seen.append(args)
        return 'original'
    monkeypatch.setattr(patch,'_original_attention',original)
    monkeypatch.setattr(patch,'_original_ff',original)
    monkeypatch.setattr(patch,'_kernel',object())
    x=torch.zeros(1,5,2048,dtype=torch.float16)
    args=(object(),x,None,object(),x,object())
    with torch.inference_mode():
        assert patch._attention(*args)=='original'
        assert patch._feed_forward(*args[:2],*args[3:])=='original'
    assert len(seen)==2
    with torch.enable_grad():
        assert not patch._input_ok(x)

def test_unknown_and_old_rotary_apis_rejected():
    class Attention:
        def forward(self,x):return x
    assert not patch._compatible(SimpleNamespace(Attention=Attention),SimpleNamespace())

def test_dense_attention_disabled_and_failure_use_exact_backend(monkeypatch):
    calls=[]
    q=torch.zeros(1,32,5,64,dtype=torch.float16)
    def exact(*args,**kwargs):
        calls.append(('exact',kwargs))
        return q
    def fail(*args):
        calls.append(('dense',{}))
        raise RuntimeError('intentional compiler failure')
    monkeypatch.setattr(patch,'_vm',SimpleNamespace(optimized_attention=exact))
    monkeypatch.setattr(patch,'_dense_kernel',SimpleNamespace(flash_fwd_na=fail))
    monkeypatch.setattr(patch,'_merge',True)
    monkeypatch.setattr(patch,'_dense_dead',False)
    monkeypatch.setattr(patch,'_dense',False)
    assert patch._run_attention(q,q,q,32) is q
    assert [c[0] for c in calls]==['exact']
    monkeypatch.setattr(patch,'_dense',True)
    assert patch._run_attention(q,q,q,32) is q
    assert patch._dense_dead
    assert patch._run_attention(q,q,q,32) is q
    assert [c[0] for c in calls]==['exact','dense','exact','exact']
    assert calls[-1][1]=={'skip_reshape':True,'skip_output_reshape':True}

@pytest.mark.parametrize('merge',[False,True])
def test_dense_attention_preserves_requested_output_layout(monkeypatch,merge):
    q=torch.arange(32*5*64,dtype=torch.float32).reshape(1,32,5,64)
    monkeypatch.setattr(patch,'_dense',True)
    monkeypatch.setattr(patch,'_dense_dead',False)
    monkeypatch.setattr(patch,'_merge',merge)
    def dense(a,b,c,scale):
        assert scale==64**-.5
        return q
    monkeypatch.setattr(patch,'_dense_kernel',SimpleNamespace(flash_fwd_na=dense))
    actual=patch._run_attention(q,q,q,32)
    expected=q if merge else q.transpose(1,2).reshape(1,5,2048)
    assert torch.equal(actual,expected)

@gpu
@pytest.mark.parametrize('batch,length,rot',[(1,1,16),(1,261,24),(2,517,24),(4,1797,24),(3,37,32)])
def test_qkv_strided_reference_and_input_ownership(kernel,batch,length,rot):
    torch.manual_seed(572)
    with torch.inference_mode():
        x=torch.randn(batch,length,32*192,dtype=torch.float16,device='mps')
        before=x.clone()
        # Deliberately include a nonzero storage offset.
        storage=torch.empty(x.numel()+8,dtype=x.dtype,device=x.device)
        view=storage[8:].view_as(x);view.copy_(x);x=view
        weight=torch.linspace(.25,1.75,64,dtype=torch.float16,device='mps')
        rope=torch.randn(batch,length,1,rot,2,2,dtype=torch.float16,device='mps')
        q,k,v=x.view(batch,length,32,192).chunk(3,-1)
        def reference(t):
            norm=F.rms_norm(t,(64,),weight,1e-5)
            a,b=norm[...,:rot],norm[...,rot:2*rot]
            left=a*rope[...,0,0];left.addcmul_(b,rope[...,0,1])
            right=b*rope[...,1,1];right.addcmul_(a,rope[...,1,0])
            return torch.cat((left,right,norm[...,2*rot:]),-1).transpose(1,2).contiguous()
        expected=(reference(q),reference(k),v.transpose(1,2).contiguous())
        actual=kernel.qkv(x,rope,weight,32,1e-5)
        for a,b in zip(expected,actual):assert_bytes(a,b)
        assert_bytes(x,before)

@gpu
@pytest.mark.parametrize('factor',[0.0,-1.0,.125,1.0,3.25])
def test_swiglu_all_finite_half_gates(kernel,factor):
    with torch.inference_mode():
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
        gate=bits[torch.isfinite(bits)].to('mps')
        up=torch.full_like(gate,factor)
        x=torch.cat((gate,up)).view(1,1,-1)
        assert_bytes(F.silu(gate)*up,kernel.swiglu(x).flatten())

@gpu
def test_sanitize_all_half_bits_and_signed_zero(kernel):
    with torch.inference_mode():
        x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
        x=x.view(2,4,128,64).to('mps')
        expected=torch.nan_to_num(x.transpose(1,2).reshape(2,128,256))
        assert_bytes(expected,kernel.merge_sanitize(x))

@gpu
def test_invalid_kernel_inputs_raise_before_dispatch(kernel):
    x=torch.zeros(1,3,2048,dtype=torch.float16,device='mps')
    with pytest.raises(RuntimeError):kernel.qkv(x,x,x,32,1e-5)
    with pytest.raises(RuntimeError):kernel.swiglu(x[...,::2])
    with pytest.raises(RuntimeError):kernel.swiglu(x[:,:0])
    with pytest.raises(RuntimeError):kernel.merge_sanitize(x)
    with pytest.raises(RuntimeError):kernel.swiglu(x.float())
