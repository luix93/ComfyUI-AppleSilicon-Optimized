import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from _patches import qwen_vl_vision as patch


def test_budget_validation(monkeypatch):
    for value, expected in [('nan',512),('bad',512),('-1',64),('8192',1024),('128',128)]:
        monkeypatch.setenv('ASFP8_QWEN_VISION_MEMORY_MB',value)
        assert patch._budget() == expected * 2**20


def test_crash_geometry_without_allocating():
    q = torch.empty(1,16,47776,64,device='meta')
    rows = patch._rows(q,q,16,None,512*2**20)
    assert 0 < rows < q.shape[-2]
    assert 4*4*16*rows*47776 <= 512*2**20
    assert 4*16*47776**2 > 136*2**30


@pytest.fixture
def core(monkeypatch):
    # CPU unit tests need no ComfyUI model imports.
    calls=[]
    def basic(q,k,v,heads,**kwargs):
        calls.append((q.shape,k.shape,kwargs.copy()))
        y=((q@k.transpose(-1,-2))*q.shape[-1]**-.5).softmax(-1)@v
        if kwargs.get('skip_output_reshape'):return y
        return y.transpose(1,2).reshape(q.shape[0],q.shape[-2],-1)
    a=SimpleNamespace(get_attn_precision=lambda p,d:p,attention_basic=basic,
                      wrap_attn=lambda fn:fn)
    qwen=ModuleType('comfy.text_encoders.qwen35')
    native=lambda device,mask=False,small_input=False: basic if small_input else 'native'
    qwen.optimized_attention_for_device=native
    mm=SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
    entries={'comfy':SimpleNamespace(model_management=mm),
             'comfy.ldm.modules':SimpleNamespace(attention=a),
             'comfy.ldm.modules.attention':a,
             'comfy.text_encoders':SimpleNamespace(qwen35=qwen)}
    for name,value in entries.items():monkeypatch.setitem(sys.modules,name,value)
    return basic,calls,qwen,mm


def test_install_scoped_idempotent(monkeypatch,core):
    basic,calls,qwen,_=core
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(patch,'_installed',False)
    patch.install()
    first=qwen.optimized_attention_for_device
    patch.install()
    assert first is qwen.optimized_attention_for_device
    assert first('mps',small_input=True) is not basic
    assert first('cpu',small_input=True) is basic
    assert first('mps',small_input=False)=='native'
    assert first('mps',mask=True,small_input=True) is basic


def test_keep_explicit_nonbasic_backend(monkeypatch,core):
    _,_,qwen,_=core
    selected=object()
    qwen.optimized_attention_for_device=lambda *a,**k:selected
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(patch,'_installed',False)
    patch.install()
    assert qwen.optimized_attention_for_device('mps',small_input=True) is selected


def test_cpu_unchanged(core):
    basic,calls,_,_=core
    x=torch.randn(1,2,13,8)
    assert torch.equal(patch._bounded(basic,x,x,x,2,skip_reshape=True),basic(x,x,x,2,skip_reshape=True))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
@pytest.mark.parametrize('output_heads',[False,True])
def test_chunks_preserve_all_keys_and_output_layout(monkeypatch,core,output_heads):
    basic,calls,_,_=core
    monkeypatch.setattr(patch,'_budget',lambda:32*2**20)
    torch.manual_seed(84)
    with torch.inference_mode():
        q,k,v=[torch.randn(1,16,513,64,device='mps') for _ in range(3)]
        expected=basic(q,k,v,16,skip_reshape=True,skip_output_reshape=output_heads)
        calls.clear()
        out=patch._bounded(basic,q,k,v,16,skip_reshape=True,skip_output_reshape=output_heads)
        assert len(calls)>1
        assert sum(s[0][-2] for s in calls)==513
        assert all(s[1][-2]==513 for s in calls)
        torch.testing.assert_close(out,expected,atol=2e-6,rtol=2e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_small_input_and_opt_out_use_original(monkeypatch,core):
    basic,calls,_,_=core
    with torch.inference_mode():
        x=torch.randn(1,2,16,8,device='mps')
        expected=basic(x,x,x,2,skip_reshape=True)
        calls.clear()
        out=patch._bounded(basic,x,x,x,2,skip_reshape=True)
        assert len(calls)==1
        assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8))
        monkeypatch.setenv('ASFP8_QWEN_VISION_SAFE','0')
        monkeypatch.setattr(patch,'_rows',lambda *a:pytest.fail('opt-out must bypass'))
        patch._bounded(basic,x,x,x,2,skip_reshape=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_memory_pressure_and_cancellation(monkeypatch,core):
    basic,calls,_,mm=core
    monkeypatch.setattr(patch,'_budget',lambda:32*2**20)
    with torch.inference_mode():
        x=torch.randn(1,16,513,64,device='mps')
        with monkeypatch.context() as m:
            m.setattr(torch.mps,'recommended_max_memory',lambda:2**20)
            with pytest.raises(RuntimeError,match='headroom'):
                patch._bounded(basic,x,x,x,16,skip_reshape=True)
            assert not calls
        def cancel():raise RuntimeError('cancelled')
        mm.throw_exception_if_processing_interrupted=cancel
        with pytest.raises(RuntimeError,match='cancelled'):
            patch._bounded(basic,x,x,x,16,skip_reshape=True)
        assert not calls


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_changed_layout_fails_before_allocation(core):
    basic,calls,_,_=core
    x=torch.zeros(1,2,16,8,device='mps')
    with pytest.raises(RuntimeError,match='Unsupported'):
        patch._bounded(basic,x,x,x,2,skip_reshape=True,mask=torch.ones(16,16,device='mps'))
    assert not calls
