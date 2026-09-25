"""FP16 table coverage, final multiply rounding and inference fallbacks."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from _patches import sdxl_geglu as patch

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_every_finite_fp16_gate_and_ragged_rows():
    with torch.inference_mode():
        values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
        values=values[torch.isfinite(values)].to('mps')
        a=torch.randn_like(values)
        x=torch.stack((a,values),dim=-1).contiguous()
        expected=(a*F.gelu(values)).reshape(-1,1)
        actual=patch._fused(x)
        assert torch.equal(actual.cpu().view(torch.uint8),expected.cpu().view(torch.uint8))
        x=torch.randn(2,113,10240,device='mps',dtype=torch.float16)
        a,g=x.chunk(2,-1)
        assert torch.equal(patch._fused(x).cpu().view(torch.uint8),(a*F.gelu(g)).cpu().view(torch.uint8))

def test_cpu_does_not_call_projection(monkeypatch):
    monkeypatch.setattr(patch,'_original',lambda *a:'eager')
    layer=SimpleNamespace(proj=lambda x:pytest.fail('unsupported projection'))
    assert patch._forward(layer,torch.ones(1,1024))=='eager'

def test_kill_switch(monkeypatch):
    monkeypatch.setenv('ASFP8_SDXL_GEGLU','0')
    monkeypatch.setattr(patch,'_original',lambda *a:'disabled')
    assert patch._forward(None,None)=='disabled'
