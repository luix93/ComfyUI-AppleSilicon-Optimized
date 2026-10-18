import importlib.util
from pathlib import Path
import pytest
import torch
from _patches import kitchen_attention_mps as k
from _patches import moss_situ_quant_mps as ffn


def test_metadata_is_explicit_and_does_not_infer_from_weight_type():
    assert k.requested({'attention':'comfy_kitchen_int8'})
    assert k.requested([{'attention':'pytorch'},{'attention':'comfy_kitchen_int8'}])
    for config in (None,{},'comfy_kitchen_int8',{'format':'int8_tensorwise'}, {'attention':'comfy_kitchen_sol'}):
        assert not k.requested(config)


def test_fused_ffn_rejects_cpu_and_wrong_shape():
    with pytest.raises(ValueError):ffn.quantize(torch.zeros(1,1,16384,dtype=torch.bfloat16),None)


def test_signed_probability_correction_is_exact_including_tail():
    generator=torch.Generator().manual_seed(93)
    p=torch.randint(0,256,(17,128),generator=generator,dtype=torch.int32)
    v=torch.randint(-127,128,(128,128),generator=generator,dtype=torch.int32)
    p[:,93:]=0;v[93:]=0
    assert torch.equal(p@v, (p-128)@v+128*v.sum(0))


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires M5 Metal')
@pytest.mark.parametrize('nq,nk,case',[(17,1025,'random'),(65,1153,'biased'),(33,1025,'zero')])
def test_quantization_oracle_and_tail(nq,nk,case):
    from _patches.h3_sol_mps import supported
    if not supported():pytest.skip('M5 required')
    path=Path(__file__).parents[1]/'benchmarks/tools/kitchen_attention_oracle.py'
    spec=importlib.util.spec_from_file_location('oracle',path);oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    torch.manual_seed(1731)
    x={name:torch.randn(1,2,nq if name=='q' else nk,128).to(torch.bfloat16) for name in ('q','k','v')}
    if case=='zero':x={name:torch.zeros_like(t) for name,t in x.items()}
    if case=='biased':x['k']=(x['k'].float()+torch.randn(1,2,1,128)*3).to(torch.bfloat16)
    with torch.inference_mode():
        p=k.prequantize(**{name:t.to('mps') for name,t in x.items()})
        out=k.attend_prequantized(p).cpu()
    p={name:t.cpu() if isinstance(t,torch.Tensor) else t for name,t in p.items()}
    expected=oracle.quant_ref(x,p['anchors'])
    for name,t in expected.items():assert torch.equal(t,p[name]),name
    expected_out=oracle.attention_ref(p)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(),expected_out.float(),rtol=0,atol=.002)
    if case=='zero':assert not out.count_nonzero() and (p['anchors']==-1).all()


def test_no_global_routing_or_foreign_override(monkeypatch):
    from types import SimpleNamespace as NS
    from _patches import moss_fast
    marker=object()
    monkeypatch.setenv('ASFP8_MOSS_KITCHEN_INT8','1')
    with torch.inference_mode():
        # A flagged checkpoint outside a validated MOSS sampling scope is untouched.
        assert moss_fast._optimized_attention(lambda *a,**kw:marker,None,None,None,24,
            preferred_attention=NS(config={'attention':'comfy_kitchen_int8'},function=None)) is marker
        token=moss_fast._scope.set({})
        try:
            assert moss_fast._optimized_attention(lambda *a,**kw:marker,None,None,None,24,
                preferred_attention=NS(config={'attention':'comfy_kitchen_int8'},function=object())) is marker
            assert moss_fast._optimized_attention(lambda *a,**kw:marker,None,None,None,24,
                transformer_options={'optimized_attention_override':object()}) is marker
        finally:moss_fast._scope.reset(token)
