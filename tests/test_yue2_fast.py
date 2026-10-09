import inspect
from types import SimpleNamespace
import pytest
import torch
from _patches import yue2_fast as patch

def setup_cache(monkeypatch, budget=4096):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 1024**3)
    module = SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(2, 3)), bias=None,weight_function=[], bias_function=[])
    state = dict(modules={id(module)}, cache={}, budget=budget,hits=0, misses=0, bytes=0)
    calls = []
    def cast(s, input=None, dtype=None, device=None, bias_dtype=None,offloadable=False, compute_dtype=None, want_requant=False):
        calls.append(s);w = s.weight.detach().clone()
        return (w, None, 'original lease') if offloadable else (w, None)
    def invoke(**kw):
        return patch._cast_cached(cast, inspect.signature(cast), module,device=torch.device('mps'), dtype=torch.float32, **kw)
    return module, state, calls, invoke

def test_owned_copy_hits_and_revision_invalidation(monkeypatch):
    module, state, calls, invoke = setup_cache(monkeypatch);token = patch._scope.set(state)
    try:
        first = invoke(offloadable=True);assert first[2] == 'original lease'
        hit = invoke(offloadable=True);assert hit[2] is None and hit[0].data_ptr() != first[0].data_ptr()
        assert len(calls) == 1 and state['hits'] == 1
        first[0].fill_(9);assert torch.equal(hit[0], torch.zeros(2, 3))
        with torch.no_grad():module.weight.add_(1)
        assert torch.equal(invoke(offloadable=True)[0], torch.ones(2, 3));assert len(calls) == 2
    finally:patch._scope.reset(token)
    invoke();assert len(calls) == 3

@pytest.mark.parametrize('reason', ['budget', 'functions', 'lowvram', 'disabled', 'training'])
def test_fallbacks_preserve_original_lease(monkeypatch, reason):
    module, state, calls, invoke = setup_cache(monkeypatch, budget=0 if reason == 'budget' else 4096)
    if reason == 'functions':module.weight_function = [lambda x: x]
    if reason == 'lowvram':module.weight_lowvram_function = object()
    if reason == 'disabled':monkeypatch.setenv('ASFP8_YUE2_WEIGHT_CACHE', '0')
    if reason == 'training':monkeypatch.setattr(torch, 'is_grad_enabled', lambda: True)
    token = patch._scope.set(state)
    try:
        for _ in range(2):assert invoke(offloadable=True)[2] == 'original lease'
        assert len(calls) == 2 and not state['cache']
    finally:patch._scope.reset(token)

def test_scope_is_cleared_on_exception_and_head_restored(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(patch, '_budget', lambda: 4096)
    monkeypatch.setattr(patch, '_head_supported', lambda *a: True)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    class Head:
        def _forward(self, x, w, b):return x
    head = Head();model = SimpleNamespace(lm_head=head, modules=lambda: [head])
    encoder = SimpleNamespace(model=model, execution_device=torch.device('mps'),config=SimpleNamespace(vocab_size=184704))
    def generate(*a, **kw):
        assert patch._scope.get() is not None and '_forward' in head.__dict__;raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):patch._generate(generate, encoder, [], 0, 128, 'abc', torch.bfloat16)
    assert patch._scope.get() is None and '_forward' not in head.__dict__

def test_head_phase_bounds_and_full_vocab_mask():
    assert patch._head_bounds('abc', 184704) == (0, 151849)
    assert patch._head_bounds('semantic', 184704) == (151852, 184621)
    assert patch._head_bounds('other', 184704) is None
    assert patch._head_bounds('abc', 32) is None
    x=torch.tensor([[1.,2.]]);w=torch.arange(20,dtype=torch.float32).reshape(10,2)
    original=lambda x,w,b:torch.nn.functional.linear(x,w,b)
    out=patch._slice_head(original,3,7,x,w,None);assert out.shape==(1,10)
    assert torch.equal(out[:,3:7],original(x,w,None)[:,3:7]);assert torch.equal(out[:,:3],torch.zeros(1,3))

def test_cache_budget_is_bounded(monkeypatch):
    monkeypatch.setattr(torch.mps,'recommended_max_memory',lambda:10*2**30)
    monkeypatch.setenv('ASFP8_YUE2_CACHE_GIB','8');assert patch._budget()==2*2**30
    for value in ['0','-1','nan','inf','invalid']:
        monkeypatch.setenv('ASFP8_YUE2_CACHE_GIB',value);assert patch._budget()==0


def test_inference_revision_avoids_version_access():
    with torch.inference_mode():
        param=torch.nn.Parameter(torch.ones(2))
    assert patch._revision(param)==(id(param),None)

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
@pytest.mark.parametrize('queries,keys',[(2082,4731),(3002,6601),(4096,8192)])
def test_scoped_acoustic_attention_parity(monkeypatch,queries,keys):
    from _patches import torch_mpp_qcache as qc
    if torch.__version__.split('+')[0]!='2.14.0' or torch.version.git_version!=qc._TORCH_GIT or not qc._hardware():pytest.skip('requires validated M5 Ultra / 2.14')
    for key,value in [('ASFP8_ATTN_BACKEND','pytorch'),('ASFP8_MPP_QCACHE','1'),('ASFP8_YUE2_FAST','1'),('ASFP8_YUE2_ATTN','1')]:monkeypatch.setenv(key,value)
    with torch.inference_mode():
        torch.manual_seed(queries)
        q=torch.randn(1,queries+2,16,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2)
        k,v=[torch.randn(1,keys+2,8,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2) for _ in range(2)]
        assert qc.maybe_attention(q,k,v,enable_gqa=True) is None
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,enable_gqa=True)
        token=patch._acoustic_scope.set(True)
        try:
            got=qc.maybe_attention(q,k,v,enable_gqa=True,output_bshd=True)
            assert got is not None and not qc._dead
            assert torch.equal(got.cpu().contiguous().view(torch.uint8),expected.cpu().contiguous().view(torch.uint8))
            assert got.transpose(1,2).is_contiguous()
            monkeypatch.setenv('ASFP8_YUE2_ATTN','0')
            assert qc.maybe_attention(q,k,v,enable_gqa=True) is None
        finally:patch._acoustic_scope.reset(token)
