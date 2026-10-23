"""Native parity and fallback checks for the narrowly scoped Krea query tile."""
from types import SimpleNamespace as NS
import pytest
import torch
from _patches import krea2_attn_fast as p, torch_mpp_qcache as mpp

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')

def model():
    return NS(**{n:NS(weight=torch.nn.Parameter(torch.zeros(1))) for n in ('wq','wk','wv','wo','gate')})

def inputs(n=6192, device='mps', dtype=torch.bfloat16):
    gen=torch.Generator().manual_seed(913)
    return tuple(torch.randn(1,h,n,128,generator=gen).to(device=device,dtype=dtype) for h in (48,12,12))

@pytest.mark.parametrize('n',[6144,6192,8192,13872,14336])
def test_native_bytes_and_tail(n, monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND','pytorch')
    with torch.inference_mode():
        q,k,v=inputs(n)
        ref=mpp.maybe_attention(q,k,v,scale=128**-.5,enable_gqa=True)
        got=p._query_tile32(model(),q,k,v)
        assert ref is not None and got is not None
        assert torch.equal(ref.view(torch.int16),got.view(torch.int16))

@pytest.mark.parametrize('case',['cpu','fp16','batch','too_short','too_long','subclass','missing_weight','grad','disabled','fast_math','version'])
def test_fallback(case,monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND','pytorch')
    q,k,v=inputs(6143 if case=='too_short' else 14337 if case=='too_long' else 6192,
                 device='cpu' if case=='cpu' else 'mps',dtype=torch.float16 if case=='fp16' else torch.bfloat16)
    owner=model()
    if case=='batch':q,k,v=(t.expand(2,-1,-1,-1) for t in (q,k,v))
    if case=='subclass':
        class Weight(torch.Tensor):pass
        owner.wq.weight=torch.zeros(1).as_subclass(Weight)
    if case=='missing_weight':del owner.gate.weight
    if case=='disabled':monkeypatch.setenv('ASFP8_KREA2_TILE32','0')
    if case=='fast_math':monkeypatch.setenv('PYTORCH_MPS_FAST_MATH','1')
    if case=='version':monkeypatch.setattr(torch,'__version__','2.14.0')
    with torch.set_grad_enabled(case=='grad'):
        assert p._query_tile32(owner,q,k,v) is None

def test_compile_failure_latches_only_new_route(monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND','pytorch')
    monkeypatch.setattr(p,'_tile32_library',None)
    monkeypatch.setattr(p,'_tile32_dead',False)
    def fail(*args,**kwargs):raise RuntimeError('simulated compiler failure')
    monkeypatch.setattr(torch.mps,'compile_shader',fail)
    old_dead=mpp._dead
    with torch.inference_mode():
        q,k,v=inputs()
        assert p._query_tile32(model(),q,k,v) is None
        assert p._tile32_dead and mpp._dead==old_dead
        assert p._query_tile32(model(),q,k,v) is None
