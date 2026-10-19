from types import SimpleNamespace as NS
import pytest
import torch
from _patches import moss_fast as p


def test_source_guard_missing_changed(tmp_path):
    f=tmp_path/'source.py';f.write_text('different code')
    assert not p._matches(NS(__file__=str(f)),p._MODEL_HASH)
    assert not p._matches(NS(),p._MODEL_HASH)
    assert not p._function_matches(None, p._SITU_HASH)
    assert not p._function_matches(len, p._SITU_HASH)
    assert not p._function_matches(test_source_guard_missing_changed, p._SITU_HASH)


@pytest.mark.parametrize('style', ['old', 'positional', 'keyword'])
def test_position_cache_accepts_both_core_signatures(style):
    calls = []
    value = torch.zeros(1, 3)
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return value
    args = ({'cond_or_uncond': [0]},) if style == 'positional' else ()
    kwargs = {'transformer_options': {'cond_or_uncond': [1]}} if style == 'keyword' else {}
    state = dict(positions={}, reuse_bytes=0, position_hits=0)
    with torch.inference_mode():
        token = p._scope.set(state)
        try:
            for _ in range(2):
                assert p._position(original, None, 64, 64, 83, 'mps', *args, **kwargs) is value
        finally:
            p._scope.reset(token)
    assert len(calls) == 1 and state['position_hits'] == 1
    assert calls[0][0][5:] == args and calls[0][1] == kwargs


def test_position_controls_bypass_cache_and_keep_options():
    calls = []
    def original(module, h, w, length, device, transformer_options):
        calls.append(transformer_options)
        return torch.tensor(transformer_options['rope_options']['shift_x'])
    state = dict(positions={}, reuse_bytes=0, position_hits=0)
    with torch.inference_mode():
        token = p._scope.set(state)
        try:
            for shift in (1., 2., 1.):
                options = {'rope_options': {'shift_x': shift}}
                assert p._position(original, None, 64, 64, 83, 'mps', options).item() == shift
                assert calls[-1] is options
        finally:
            p._scope.reset(token)
    assert len(calls) == 3 and not state['positions']


def test_scope_and_grad_required(monkeypatch):
    with torch.inference_mode():
        assert not p._enabled('SITU')
        token=p._scope.set({})
        try:
            assert p._enabled('SITU')
            monkeypatch.setenv('ASFP8_MOSS_SITU','0')
            assert not p._enabled('SITU')
        finally:p._scope.reset(token)


def test_foreign_cpu_activation_keeps_original():
    x=torch.randn(1,3,16384,dtype=torch.bfloat16)
    with torch.inference_mode():
        token=p._scope.set({'situ_calls':0})
        try:
            assert p._situ(lambda value:value,x) is x
            assert p._scope.get()['situ_calls']==0
        finally:p._scope.reset(token)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([])


def guider():
    return NS(inner_model=NS(diffusion_model=Model()),model_patcher=NS(load_device=torch.device('mps')),model_options={})


@pytest.mark.parametrize('field',['patches','patches_replace','wrappers','optimized_attention_override'])
def test_custom_transformer_paths_fall_back(field):
    g=guider();g.model_options={'transformer_options':{field:{'custom':object()}}}
    with torch.inference_mode():assert not p._eligible(g,Model)


def test_forward_hooks_fall_back():
    g=guider();g.inner_model.diffusion_model.register_forward_hook(lambda *a:None)
    with torch.inference_mode():assert not p._eligible(g,Model)


@pytest.mark.parametrize('fail',[False,True])
def test_scope_released_after_sampler_completion_or_failure(monkeypatch,fail):
    monkeypatch.setattr(p,'_cache_budget',lambda:0)
    g=guider()
    def original(*a,**kw):
        assert p._scope.get() is not None
        if fail:raise RuntimeError('cancelled')
        return 123
    with torch.inference_mode():
        if fail:
            with pytest.raises(RuntimeError,match='cancelled'):p._sample(original,Model,g)
        else:assert p._sample(original,Model,g)==123
    assert p._scope.get() is None
    assert 'cache' not in p._stats


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_bf16_situ_all_finite_codes_and_modulation_exact():
    with torch.inference_mode():
        assert p._ready()
        codes=torch.arange(65536,dtype=torch.int32,device='mps').to(torch.int16).view(torch.bfloat16)
        # Exercise the complete finite input domain, with several different pairings.
        codes=torch.where(torch.isfinite(codes),codes,torch.zeros_like(codes))
        for delta in (0,1,32768,12345):
            h=codes.reshape(1,8,8192);g=codes.roll(delta).reshape_as(h)
            x=torch.cat((h,g),-1)
            ref=(4.*torch.tanh(g/4.)*torch.sigmoid(g))*(25.*torch.tanh(h/25.))
            got=p._situ_tensor(x)
            assert torch.equal(got.view(torch.int16),ref.view(torch.int16))
        for batch in (1,2):
            x=torch.randn(batch,83,3072,device='mps',dtype=torch.bfloat16)
            scale=torch.randn(batch,1,3072,device='mps',dtype=x.dtype);shift=torch.randn_like(scale)
            assert torch.equal(p._mod_tensor(x,scale,shift).view(torch.int16),(x*(1+scale)+shift).view(torch.int16))


def test_explicit_attention_backend_is_preserved():
    untouched=object()
    with torch.inference_mode():
        token=p._scope.set({'attention_calls':0})
        try:
            assert p._optimized_attention(lambda *a,**k:untouched,None,None,None,24,preferred_attention=NS(function=object())) is untouched
            assert p._scope.get()['attention_calls']==0
        finally:p._scope.reset(token)


@pytest.mark.parametrize('foreign_hook', [False, True])
def test_temporary_cast_hook_nested_lifetime(monkeypatch, foreign_hook):
    import sys
    from types import ModuleType
    comfy = ModuleType('comfy')
    ops = ModuleType('comfy.ops')
    original = lambda s, input=None: (s, input)
    ops.cast_bias_weight = original
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.ops', ops)
    release_outer = p._acquire_cast()
    wrapped = ops.cast_bias_weight
    release_inner = p._acquire_cast()
    assert ops.cast_bias_weight is wrapped
    # Other work using the process-wide function remains a direct pass-through.
    assert ops.cast_bias_weight(1, 2) == (1, 2)
    release_inner()
    assert ops.cast_bias_weight is wrapped
    foreign = lambda *a: None
    if foreign_hook:
        ops.cast_bias_weight = foreign
    release_outer()
    assert ops.cast_bias_weight is (foreign if foreign_hook else original)
    assert p._cast_users == 0 and p._active_cast is None


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_cached_weights_owned_and_invalidated():
    import inspect
    layer=torch.nn.Linear(4,4,bias=False,device='mps',dtype=torch.bfloat16)
    calls=[]
    def original(s,input=None,dtype=None,device=None,bias_dtype=None,offloadable=False,compute_dtype=None,want_requant=False):
        calls.append(1);w=s.weight*2
        return (w,None,'lease') if offloadable else (w,None)
    sig=inspect.signature(original)
    state=dict(modules={id(layer)},budget=2**20,cache={},cache_bytes=0,cache_hits=0,cache_misses=0)
    x=torch.ones(1,4,device='mps',dtype=torch.bfloat16)
    with torch.inference_mode():
        token=p._scope.set(state)
        try:
            a=p._cached_cast(original,sig,layer,x,offloadable=True)
            b=p._cached_cast(original,sig,layer,x,offloadable=True)
            assert a[2]=='lease' and b[2] is None
            assert torch.equal(a[0],b[0]) and a[0].data_ptr()!=b[0].data_ptr()
            assert len(calls)==1
            layer.weight.add_(1)
            c=p._cached_cast(original,sig,layer,x,offloadable=True)
            assert len(calls)==2 and torch.equal(c[0],layer.weight*2)
            layer.weight_function=[lambda w:w]
            p._cached_cast(original,sig,layer,x,offloadable=True)
            assert len(calls)==3
        finally:p._scope.reset(token)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_split_half_rope_strides_and_tail_exact():
    with torch.inference_mode():
        assert p._ready()
        for b,n in ((1,17),(2,83),(1,4127)):
            qkv=torch.randn(b,n,3*3072,device='mps',dtype=torch.bfloat16)
            q,k,_=[x.view(b,n,24,128) for x in qkv.chunk(3,-1)]
            pe=torch.randn(1,n,1,64,2,2,device='mps')
            def reference(x):
                diagonal=pe.diagonal(dim1=-2,dim2=-1).movedim(-1,-2)
                out=x.unflatten(-1,(2,-1))*diagonal
                out[...,0,:].addcmul_(x[...,64:],pe[...,0,1])
                out[...,1,:].addcmul_(x[...,:64],pe[...,1,0])
                return out.reshape(x.shape).to(x.dtype)
            got=p._rope_tensors(q,k,pe)
            assert all(torch.equal(a.view(torch.int16),reference(x).view(torch.int16)) for a,x in zip(got,(q,k)))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')
@pytest.mark.parametrize('tokens', [4609, 6107, 6167, 6656])
def test_larger_query_reuse_matches_native_sdpa(tokens):
    # Exercise non-aligned tails, both real CFG branch lengths and the new ceiling.
    # Packed QKV views also validate the non-contiguous input strides.
    if not p.torch_mpp_qcache._torch_build_supported() or not p.torch_mpp_qcache._hardware():
        pytest.skip('requires reviewed PyTorch build and M5 Ultra')
    with torch.inference_mode():
        torch.manual_seed(37)
        packed = torch.randn(1, tokens, 9216, device='mps', dtype=torch.bfloat16)
        q, k, v = [t.unflatten(-1, (24, 128)).transpose(1, 2) for t in packed.chunk(3, -1)]
        reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if p._attention_library is None:
            p._attention_library = torch.mps.compile_shader(p.torch_mpp_qcache._source())
        params = p.torch_mpp_qcache._parameters(
            tuple(q.shape), tuple(k.shape), q.stride(), k.stride(), v.stride(), 128**-.5)
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        p._attention_library.custom_attn(
            q, k, v, out, params, params, params,
            threads=(((tokens + 63) // 64) * 128, 24, 1), group_size=(128, 1, 1))
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')
@pytest.mark.parametrize('cast_kind', ['owned', 'cache_owned', 'foreign', 'cache_foreign', 'disabled', 'changed_core', 'hook'])
def test_direct_ffn_requires_owned_cast(monkeypatch, cast_kind):
    import sys
    from types import ModuleType
    import comfy_kitchen.tensor as kitchen_tensor
    from _patches import ops_bias_fp8
    x = torch.zeros(1, 1, 16384, device='mps', dtype=torch.bfloat16)
    events = []
    class Weight:
        _layout_cls = 'TensorWiseINT8Layout'
        dtype = torch.bfloat16
        device = x.device
        shape = (3072, 8192)
    class Layer:
        weight = Weight()
        bias = None
        def __call__(self, activation):
            events.append('module')
            return activation
    layer = Layer()
    monkeypatch.setattr(kitchen_tensor, 'QuantizedTensor', Weight)
    comfy = ModuleType('comfy')
    ops = ModuleType('comfy.ops')
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.ops', ops)
    owned = lambda *a, **k: None
    foreign = lambda *a, **k: None
    monkeypatch.setattr(ops_bias_fp8, '_installed_cast', owned)
    monkeypatch.setattr(p, '_direct_ffn_supported', True)
    monkeypatch.setenv('ASFP8_MOSS_FAST', '1')
    monkeypatch.setenv('ASFP8_MOSS_FFN_DISPATCH', '1')
    cache = lambda *a, **k: None
    cache.__wrapped__ = owned if cast_kind == 'cache_owned' else foreign
    monkeypatch.setattr(p, '_active_cast', cache)
    ops.cast_bias_weight = cache if cast_kind.startswith('cache_') else foreign if cast_kind == 'foreign' else owned
    if cast_kind == 'disabled':
        monkeypatch.setenv('ASFP8_MOSS_FFN_DISPATCH', '0')
    if cast_kind == 'changed_core':
        monkeypatch.setattr(p, '_direct_ffn_supported', False)
    if cast_kind == 'hook':
        layer._forward_hooks = {1: object()}
    ops.run_every_op = lambda: events.append('interrupt_check')
    ops._eager_input_act = lambda value, act: value
    def original(*a, **k):
        events.append('original')
        return x
    state = {'modules': {id(layer)}}
    with torch.inference_mode():
        token = p._scope.set(state)
        try:
            assert p._linear_input_act(original, layer, x, 'situ') is x
        finally:
            p._scope.reset(token)
    expected = cast_kind in ('owned', 'cache_owned')
    assert events == (['interrupt_check', 'module'] if expected else ['original'])
    assert state.get('direct_ffn_calls', 0) == int(expected)


@pytest.mark.parametrize('kind', ['transparent', 'active_qwen', 'active_yue', 'foreign', 'cycle'])
def test_ffn_cast_chain_is_owned_and_inactive(monkeypatch, kind):
    from _patches import ops_bias_fp8, qwen3vl_decode, yue2_fast
    owned = lambda: None
    yue = lambda: None
    qwen = lambda: None
    moss = lambda: None
    yue.__wrapped__ = owned
    qwen.__wrapped__ = yue
    moss.__wrapped__ = qwen
    monkeypatch.setattr(ops_bias_fp8, '_installed_cast', owned)
    monkeypatch.setattr(yue2_fast, '_installed_cast', yue)
    monkeypatch.setattr(qwen3vl_decode, '_installed_cast', qwen)
    monkeypatch.setattr(p, '_active_cast', moss)
    if kind == 'foreign':
        foreign = lambda: None
        foreign.__wrapped__ = yue
        qwen.__wrapped__ = foreign
    elif kind == 'cycle':
        yue.__wrapped__ = qwen
    qt = qwen3vl_decode._scope.set({} if kind == 'active_qwen' else None)
    yt = yue2_fast._scope.set({} if kind == 'active_yue' else None)
    try:
        assert p._owned_dequantizing_cast(moss) == (kind == 'transparent')
    finally:
        qwen3vl_decode._scope.reset(qt)
        yue2_fast._scope.reset(yt)
