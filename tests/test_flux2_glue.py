"""Klein activation rounding, strided inputs, layout and hook fallbacks."""
from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.flux.layers import SiLUActivation
from _patches import flux2_glue as patch


def bits(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


@pytest.fixture
def kernel(monkeypatch):
    if not (torch.backends.mps.is_available() and patch.torch_mpp_qcache._hardware()
            and patch.torch_mpp_qcache._torch_build_supported()):
        pytest.skip('validated M5 Ultra / PyTorch 2.14 required')
    monkeypatch.setattr(patch, '_activation_type', SiLUActivation)
    monkeypatch.setattr(patch, '_original', SiLUActivation.forward)
    monkeypatch.setattr(patch, '_installed', True)
    monkeypatch.setenv('ASFP8_FLUX2_GLUE', '1')
    assert patch._ready()


@pytest.mark.parametrize('seq,stride', [(1,24576),(7,36864),(65,36864),(512,24576)])
def test_strided_swiglu(kernel, seq, stride):
    torch.manual_seed(seq)
    with torch.inference_mode():
        storage = torch.randn(1,seq+2,stride,device='mps',dtype=torch.bfloat16)
        x = storage[:,1:-1,-24576:]
        layer = SiLUActivation()
        bits(patch._activation(layer,x), layer(x))


def test_all_finite_gate_values(kernel):
    with torch.inference_mode():
        values = torch.arange(65536,device='mps',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        values = values[values.isfinite()]
        x = torch.ones(1,6,24576,device='mps',dtype=torch.bfloat16)
        gate = torch.zeros(6*12288,device='mps',dtype=torch.bfloat16)
        gate[:values.numel()] = values
        x[...,:12288] = gate.reshape(1,6,12288)
        torch.manual_seed(913)
        x[...,12288:] = torch.randn(1,6,12288,device='mps',dtype=torch.bfloat16)
        layer = SiLUActivation()
        bits(patch._activation(layer,x), layer(x))


@pytest.mark.parametrize('seq', [1,7,127])
def test_merged_attention_and_swiglu(kernel, monkeypatch, seq):
    layer = SiLUActivation()
    monkeypatch.setattr(SiLUActivation, 'forward', patch._activation)
    block = SimpleNamespace(yak_mlp=False,hidden_size=4096,mlp_hidden_dim=12288,
                            mlp_hidden_dim_first=24576,mlp_act=layer)
    with torch.inference_mode():
        projected = torch.randn(1,seq+2,36864,device='mps',dtype=torch.bfloat16)[:,1:-1]
        attention = torch.randn(1,seq,4096,device='mps',dtype=torch.bfloat16)
        expected = torch.cat((attention,patch._original(layer,projected[...,12288:])),2)
        bits(patch.single_mlp_input(block,projected,attention),expected)
        monkeypatch.setenv('ASFP8_FLUX2_MLP_LAYOUT','0')
        assert patch.single_mlp_input(block,projected,attention) is None


def test_disabled_and_hooked_activation(kernel, monkeypatch):
    layer = SiLUActivation()
    with torch.inference_mode():
        x = torch.randn(1,3,24576,device='mps',dtype=torch.bfloat16)
        hook = layer.gate_fn.register_forward_hook(lambda module,args,value:value+1)
        try:bits(patch._activation(layer,x),layer(x))
        finally:hook.remove()
        monkeypatch.setenv('ASFP8_FLUX2_SWIGLU','0')
        bits(patch._activation(layer,x),layer(x))


def test_unsupported_inputs_fall_back(kernel):
    layer = SiLUActivation()
    for x in [torch.randn(1,3,24576),torch.randn(2,3,128),torch.randn(1,3,24576,device='mps',dtype=torch.float16)]:
        with torch.inference_mode():bits(patch._activation(layer,x),layer(x))
