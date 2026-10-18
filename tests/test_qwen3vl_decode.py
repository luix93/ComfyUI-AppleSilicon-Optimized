import inspect
from types import SimpleNamespace

import pytest
import torch
from _patches import qwen3vl_decode as patch


class Quantized:
    _version = 0
    def __init__(self, data):
        self._qdata = data
        self._params = SimpleNamespace(scale=torch.ones(1), convrot=True, convrot_groupsize=256)
        self.calls = 0
    def dequantize(self):
        self.calls += 1
        return self._qdata.float() * self._params.scale


def fixture(monkeypatch, budget=4096):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 2**30)
    module = SimpleNamespace(weight=Quantized(torch.ones(2,3,dtype=torch.int8)),bias=None,
                             _full_precision_mm=True,out_features=2,weight_function=[],bias_function=[])
    state = dict(modules={id(module)},cache={},bytes=0,budget=budget,hits=0,misses=0)
    calls=[]
    def original(s,input=None,dtype=None,device=None,bias_dtype=None,offloadable=False,compute_dtype=None,want_requant=False):
        calls.append(s)
        return (s.weight,None,'lease') if offloadable else (s.weight,None)
    def invoke(**kw):
        return patch._cast(original,inspect.signature(original),Quantized,module,
                           device=torch.device('mps'),dtype=torch.float32,**kw)
    return module,state,calls,invoke


def test_exact_owned_cache_and_scale_revision(monkeypatch):
    module,state,calls,invoke=fixture(monkeypatch)
    token=patch._scope.set(state)
    try:
        first=invoke(offloadable=True)
        hit=invoke(offloadable=True)
        assert first[2]=='lease' and hit[2] is None
        assert len(calls)==1 and module.weight.calls==1
        first[0].fill_(77)
        assert torch.equal(hit[0],torch.ones(2,3))
        module.weight._params.scale.mul_(2)
        assert torch.equal(invoke()[0],torch.full((2,3),2.))
        assert len(calls)==2
        module.weight._qdata.add_(1)
        assert torch.equal(invoke()[0],torch.full((2,3),4.))
    finally:patch._scope.reset(token)
    assert invoke()[0] is module.weight  # no conversion outside generation


@pytest.mark.parametrize('reason',['budget','pressure','training','weights','bias','lowvram','not_full','requant','foreign'])
def test_decline_cache_without_changing_lease(monkeypatch,reason):
    module,state,calls,invoke=fixture(monkeypatch,0 if reason=='budget' else 4096)
    if reason=='pressure':monkeypatch.setattr(torch.mps,'driver_allocated_memory',lambda:2**30)
    if reason=='training':monkeypatch.setattr(torch,'is_grad_enabled',lambda:True)
    if reason=='weights':module.weight_function=[object()]
    if reason=='bias':module.bias_function=[object()]
    if reason=='lowvram':module.weight_lowvram_function=object()
    if reason=='not_full':module._full_precision_mm=False
    if reason=='foreign':state['modules']=set()
    token=patch._scope.set(state)
    try:
        for _ in range(2):assert invoke(offloadable=True,want_requant=reason=='requant')[2]=='lease'
        assert len(calls)==2 and not state['cache']
    finally:patch._scope.reset(token)


def test_release_on_cancellation(monkeypatch):
    _,state,_,invoke=fixture(monkeypatch)
    def original(*a):
        invoke()
        assert state['cache']
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):patch._run_scoped(state,original,None)
    assert patch._scope.get() is None and not state['cache']
    assert patch._last_stats['misses']==1


def test_budget_caps_invalid_and_small_machines(monkeypatch):
    monkeypatch.setattr(torch.mps,'recommended_max_memory',lambda:16*2**30)
    monkeypatch.setenv('ASFP8_QWEN3VL_CACHE_GIB','8')
    assert patch._budget()==int(1.6*2**30)
    for value in ('0','-1','nan','inf','bad'):
        monkeypatch.setenv('ASFP8_QWEN3VL_CACHE_GIB',value)
        assert patch._budget()==0


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
@pytest.mark.parametrize('n,k,scalar',[(1,1,True),(17,129,False),(17,2560,False),(129,2560,True)])
def test_integer_gemv_bytes(n,k,scalar):
    g=torch.Generator().manual_seed(k)
    x=torch.randint(-128,128,(1,k),dtype=torch.int8,generator=g)
    w=torch.randint(-128,128,(n,k),dtype=torch.int8,generator=g)
    x[0,:]=127;w[0,:]=-128
    xs=torch.tensor([.03125],dtype=torch.float32)
    ws=torch.rand((1 if scalar else n,),generator=g)*.1
    ref=((x.long()@w.long().T).float()*(xs*ws)).bfloat16()
    with torch.inference_mode():out=patch._gemv(x.to('mps'),w.to('mps'),xs.to('mps'),ws.to('mps'))
    torch.mps.synchronize()
    assert torch.equal(out.cpu().view(torch.uint8),ref.view(torch.uint8))

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
@pytest.mark.parametrize('seed',[7,281])
def test_rope_matches_native_kitchen_bytes(seed):
    from pathlib import Path
    from comfy_kitchen.backends.eager.rope import apply_rope_split_half
    with torch.inference_mode():
        g=torch.Generator().manual_seed(seed)
        q=torch.randn((1,32,1,128),generator=g).bfloat16().to('mps')
        k=torch.randn((1,8,1,128),generator=g).bfloat16().to('mps')
        matrix=torch.randn((1,1,1,64,2,2),generator=g).to('mps')
        reference=apply_rope_split_half(q,k,matrix)
        if patch._library is None:
            patch._library=torch.mps.compile_shader(Path(patch.__file__).with_suffix('.metal').read_text())
        oq,ok=torch.empty_like(q),torch.empty_like(k)
        patch._library.qwen_rope(q,k,matrix,oq,ok,threads=2560,group_size=256)
        torch.mps.synchronize()
        for actual,expected in zip((oq,ok),reference):
            assert torch.equal(actual.cpu().view(torch.uint8),expected.cpu().view(torch.uint8))


def test_quantized_embedding_stays_quantized(monkeypatch):
    module,state,calls,invoke=fixture(monkeypatch)
    del module.out_features
    module.num_embeddings=151936
    token=patch._scope.set(state)
    try:
        for _ in range(2):assert invoke()[0] is module.weight
        assert module.weight.calls==0 and not state['cache']
    finally:patch._scope.reset(token)


def test_floating_tied_table_reuses_original_conversion(monkeypatch):
    module,state,_,_=fixture(monkeypatch)
    del module.out_features
    module.num_embeddings=151936
    module._full_precision_mm=False
    def original(s,input=None,dtype=None,device=None,bias_dtype=None,offloadable=False,compute_dtype=None,want_requant=False):
        result=(s.weight.dequantize(),None)
        return (*result,'lease') if offloadable else result
    def call():
        return patch._cast(original,inspect.signature(original),Quantized,module,
                           device=torch.device('mps'),dtype=torch.float32,offloadable=True)
    token=patch._scope.set(state)
    try:
        a,b=call(),call()
        assert a[2]=='lease' and b[2] is None and module.weight.calls==1
        assert torch.equal(a[0],b[0]) and state['hits']==1
    finally:patch._scope.reset(token)


def test_group_cache_replaces_owned_storage_and_tracks_revision(monkeypatch):
    monkeypatch.setattr(torch.mps,'driver_allocated_memory',lambda:0)
    monkeypatch.setattr(torch.mps,'recommended_max_memory',lambda:2**30)
    modules=[]
    state=dict(modules=set(),cache={},bytes=0,budget=4096,hits=0,misses=0)
    x=torch.randn(1,1,8)
    p=dict(input=x,dtype=None,device=None,bias_dtype=None,compute_dtype=x.dtype,want_requant=False)
    for n in (8,4,4):
        m=torch.nn.Linear(8,n,bias=False)
        m._full_precision_mm=True;m.comfy_force_cast_weights=True
        modules.append(m);state['modules'].add(id(m))
        state['cache'][patch._key(m,p)]=(m.weight.detach().clone(),None)
    ref=[torch.nn.functional.linear(x,m.weight) for m in modules]
    out=patch._grouped_projection(state,modules,x)
    assert all(torch.equal(a,b) for a,b in zip(out,ref))
    packed=next(iter(state['projection_groups'].values()))[1]
    assert all(e[0].untyped_storage().data_ptr()==packed.untyped_storage().data_ptr() for e in state['cache'].values())
    assert all(torch.equal(a,b) for a,b in zip(out,patch._grouped_projection(state,modules,x)))
    with torch.no_grad():modules[0].weight.add_(1)
    assert patch._grouped_projection(state,modules,x) is None  # new revision not cast yet
    modules[1].weight_function=[object()]
    assert patch._grouped_projection(state,modules,x) is None


def test_generation_restores_group_overrides_on_error(monkeypatch):
    # Even a cancelled generation must not leave model.forward wrappers behind.
    state=dict(modules=set(),cache={},bytes=0,budget=1,hits=0,misses=0)
    modules=[torch.nn.Linear(2,2,bias=False) for _ in range(3)]
    override=lambda x:x+1
    modules[1].forward=override
    def prep(s,m):patch._wrap_projection_group(s,modules)
    monkeypatch.setattr(patch,'_prepare_projection_groups',prep)
    monkeypatch.setattr(patch,'_prepare_qk_norms',lambda *a:None)
    def fail(*a):
        assert all('forward' in m.__dict__ for m in modules)
        raise RuntimeError('cancel')
    with pytest.raises(RuntimeError,match='cancel'):patch._run_scoped(state,fail,None)
    assert 'forward' not in modules[0].__dict__ and 'forward' not in modules[2].__dict__
    assert modules[1].forward is override and patch._scope.get() is None


def test_pending_norm_is_restored_when_kernel_disabled(monkeypatch):
    monkeypatch.setattr(patch,'_kernel_failed',True)
    q,k=torch.randn(1,32,1,128),torch.randn(1,8,1,128)
    state=dict(norm_pending={id(q):(q,lambda x:x*2,None,1e-6),id(k):(k,lambda x:x*3,None,1e-6)})
    token=patch._scope.set(state)
    try:
        a,b=patch._rope(lambda a,b,f:(a,b),q,k,None)
        assert torch.equal(a,q*2) and torch.equal(b,k*3) and not state['norm_pending']
    finally:patch._scope.reset(token)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
@pytest.mark.parametrize('seed,eps',[(0,1e-6),(7,1e-5),(23,1e-6),(521,1e-6)])
def test_fused_qk_norm_rope_bytes(seed,eps):
    from pathlib import Path
    from comfy_kitchen.backends.eager.rope import apply_rope_split_half
    with torch.inference_mode():
        g=torch.Generator().manual_seed(seed)
        q=torch.randn((1,32,1,128),generator=g).bfloat16().to('mps')
        k=torch.randn((1,8,1,128),generator=g).bfloat16().to('mps')
        qw=torch.randn((128,),generator=g).bfloat16().to('mps')
        kw=torch.randn((128,),generator=g).bfloat16().to('mps')
        matrix=torch.randn((1,1,1,64,2,2),generator=g).to('mps')
        ref=apply_rope_split_half(torch.nn.functional.rms_norm(q,(128,),qw,eps),torch.nn.functional.rms_norm(k,(128,),kw,eps),matrix)
        if patch._library is None:patch._library=torch.mps.compile_shader(Path(patch.__file__).with_suffix('.metal').read_text())
        oq,ok=torch.empty_like(q),torch.empty_like(k)
        patch._library.qk_norm_rope(q,k,qw,kw,matrix,oq,ok,eps,threads=1280,group_size=32)
        for actual,expected in zip((oq,ok),ref):assert torch.equal(actual.cpu().view(torch.uint8),expected.cpu().view(torch.uint8))
