"""Native RMS/RoPE parity including packed inputs and fallback boundaries."""
import pytest
import torch
from _patches import moss_fast as p

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')

@pytest.mark.parametrize('rows,packed,magnitude',[(1,False,1),(31,True,1),(257,False,1e-12),(257,True,1e12),(4096,False,1),(6197,True,1),(6656,True,1),(31,True,0)])
def test_native_qk_rope_bytes(rows,packed,magnitude):
    gen=torch.Generator().manual_seed(374)
    with torch.inference_mode():
        assert p._ready()
        if packed:
            qkv=(torch.randn(1,rows,9216,generator=gen)*magnitude).to(device='mps',dtype=torch.bfloat16)
            q,k,_=[t.view(1,rows,24,128) for t in qkv.chunk(3,dim=-1)]
        else:
            q,k=[(torch.randn(1,rows,24,128,generator=gen)*magnitude).to(device='mps',dtype=torch.bfloat16) for _ in range(2)]
        qw,kw=[torch.randn(128,generator=gen).to(device='mps',dtype=torch.bfloat16) for _ in range(2)]
        if packed:qw,kw=(torch.nn.Parameter(t,requires_grad=False) for t in (qw,kw))
        pe=torch.randn(1,rows,1,64,2,2,generator=gen).to('mps')
        ref=p._rope_tensors(torch.nn.functional.rms_norm(q,(128,),qw,1e-6),torch.nn.functional.rms_norm(k,(128,),kw,1e-6),pe)
        token=p._scope.set({})
        try:got=p._qk_rope_tensors(q,k,qw,kw,pe,1e-6)
        finally:p._scope.reset(token)
        assert got is not None
        for a,b in zip(ref,got):assert torch.equal(a.view(torch.int16),b.view(torch.int16))

@pytest.mark.parametrize('case',['disabled','batch','long','fp32_weights','noncontiguous_weights','grad','no_scope'])
def test_qk_fallback(case,monkeypatch):
    b,n=(2,31) if case=='batch' else (1,6657) if case=='long' else (1,31)
    q=torch.zeros(b,n,24,128,dtype=torch.bfloat16,device='mps');k=torch.zeros_like(q)
    w=torch.ones(128,dtype=torch.bfloat16,device='mps');pe=torch.ones(1,n,1,64,2,2,device='mps')
    if case=='fp32_weights':w=w.float()
    if case=='noncontiguous_weights':w=torch.ones(256,dtype=w.dtype,device=w.device)[::2]
    if case=='disabled':monkeypatch.setenv('ASFP8_MOSS_QK_FUSION','0')
    with torch.set_grad_enabled(case=='grad'):
        token=p._scope.set(None if case=='no_scope' else {})
        try:assert p._qk_rope_tensors(q,k,w,w,pe,1e-6) is None
        finally:p._scope.reset(token)
