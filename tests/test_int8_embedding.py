"""Embedding casts must not apply INT8 scale/ConvRot twice."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

from _patches import int8_embedding as patch


def build(cast, *, hooks=False, fmt='int8_tensorwise'):
    events=[]
    raw=torch.arange(32,dtype=torch.int8).reshape(4,8)
    params=TensorWiseINT8Layout.Params(scale=torch.tensor(.25),orig_dtype=torch.bfloat16,orig_shape=(4,8))
    weight=QuantizedTensor(raw,'TensorWiseINT8Layout',params)
    @contextmanager
    def context(layer,**kwargs):
        events.append(('enter',kwargs));yield (weight if isinstance(cast,str) else cast), None;events.append(('exit',))
    def decode(q,p,indices):
        assert q.dtype==torch.int8
        events.append(('decode',))
        return (q[indices].float()*p.scale).to(p.orig_dtype)
    class Embedding:
        def forward_comfy_cast_weights(self,input,out_dtype=None):events.append(('original',));return 'original'
    ops=SimpleNamespace(QuantizedTensor=QuantizedTensor,CastBiasWeightContext=context,get_layout_class=lambda _:SimpleNamespace(dequantize_embedding=decode))
    patch._patch_embedding(ops,Embedding)
    layer=Embedding();layer.weight=weight;layer.quant_format=fmt;layer.layout_type='TensorWiseINT8Layout';layer.weight_function=[object()] if hooks else []
    layer.padding_idx=None;layer.max_norm=None;layer.norm_type=2.;layer.scale_grad_by_freq=False;layer.sparse=False
    return layer,events


@pytest.mark.parametrize('out_dtype',[None,torch.float32,torch.float16])
def test_dequantized_cast_is_gathered_once(out_dtype):
    dense=(torch.arange(32).reshape(4,8).float()/7).bfloat16()
    layer,events=build(dense)
    index=torch.tensor([[3,1,3]])
    actual=layer.forward_comfy_cast_weights(index,out_dtype)
    expected=dense[index].to(out_dtype)
    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    assert [x[0] for x in events]==['enter','exit']


def test_quantized_cast_keeps_per_row_decoder():
    layer,events=build('quantized');index=torch.tensor([[1,0,1]])
    out=layer.forward_comfy_cast_weights(index,torch.float32)
    assert torch.equal(out,layer.weight._qdata[index].float()*.25)
    assert [x[0] for x in events]==['enter','decode','exit']


@pytest.mark.parametrize('hooks,fmt',[(True,'int8_tensorwise'),(False,'float8_e4m3fn')])
def test_other_formats_and_weight_patches_keep_original(hooks,fmt):
    layer,events=build('quantized',hooks=hooks,fmt=fmt)
    assert layer.forward_comfy_cast_weights(torch.tensor([0]))=='original'
    assert events==[('original',)]
