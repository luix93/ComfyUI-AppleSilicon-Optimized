"""Exact SeedVR2 transforms, cache ownership, and guarded GPU operations."""
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from _patches import conv3d_mps as conv
from _patches import seedvr2_fast as fast

gpu = pytest.mark.skipif(
    os.environ.get('ASFP8_DIT_EXT') != '1' or not torch.backends.mps.is_available(),
    reason='requires native MPS kernels')


def assert_bytes(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),
                       b.cpu().contiguous().view(torch.uint8))


def test_packing_cache_reuses_invalidates_and_releases():
    weight = torch.arange(2*3*3*3*3.).reshape(2,3,3,3,3)
    with torch.no_grad():
        with conv.packing_scope():
            first = conv._pack_weight(weight,2,3,3,1,3,3)
            assert conv._pack_weight(weight,2,3,3,1,3,3) is first
            weight.add_(1)
            changed = conv._pack_weight(weight,2,3,3,1,3,3)
            assert changed is not first
            assert_bytes(changed,weight.transpose(1,2).reshape(2,9,3,3).contiguous())
            outer = conv._packing_state.get()
            with pytest.raises(RuntimeError), conv.packing_scope():
                raise RuntimeError('intentional')
            assert conv._packing_state.get() is outer
        assert conv._packing_state.get() is None
        with conv.packing_scope():
            assert conv._pack_weight(weight,2,3,3,1,3,3) is not changed


def test_packing_grad_path_does_not_cache():
    weight = torch.randn(2,3,3,3,3,requires_grad=True)
    with conv.packing_scope():
        out = conv._pack_weight(weight,2,3,3,1,3,3)
        assert conv._packing_state.get() == {}
        out.sum().backward()
        assert_bytes(weight.grad, torch.ones_like(weight))


def test_cpu_attention_and_training_keep_original(monkeypatch):
    calls=[]
    monkeypatch.setattr(fast,'_original_attention',lambda *a,**kw:calls.append((a,kw)) or 'original')
    x=torch.zeros(7,2,4)
    assert fast._bounded_attention(x,x,x,2,[0,7],[0,7],skip_reshape=True)=='original'
    assert calls[0][1]['skip_reshape'] is True
    assert not fast._bf16(x)


@pytest.mark.parametrize('value,expected',[('garbage',1),('0',1),('-8',1),('999',16),('4',4)])
def test_attention_batch_is_bounded(monkeypatch,value,expected):
    monkeypatch.setenv('ASFP8_SEEDVR2_ATTN_BATCH',value)
    assert fast._batch_size()==expected


@pytest.fixture(scope='module')
def kernel():
    with torch.inference_mode():
        torch.zeros(1,device='mps');torch.mps.synchronize()
    from _patches.seedvr2_ext.loader import module
    result=module()
    assert result is not None
    return result


@gpu
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_ragged_attention_batched_matches_original(monkeypatch,dtype):
    import comfy.ldm.seedvr.attention as attention
    monkeypatch.setattr(fast,'_original_attention',attention.var_attention_optimized_split)
    monkeypatch.setenv('ASFP8_SEEDVR2_ATTN_BATCH','8')
    lengths=[483]*36+[118,208,483,1]
    cumulative=[0]
    for n in lengths:cumulative.append(cumulative[-1]+n)
    with torch.inference_mode():
        torch.manual_seed(574)
        q,k,v=[torch.randn(sum(lengths),24,128,dtype=dtype).to('mps') for _ in range(3)]
        reference=attention.var_attention_optimized_split(q,k,v,24,cumulative,cumulative,skip_reshape=True)
        result=fast._bounded_attention(q,k,v,24,cumulative,cumulative,skip_reshape=True)
        assert_bytes(result,reference)


@gpu
@pytest.mark.parametrize('channels,side,taps',[(128,64,1),(128,128,3),(512,17,3)])
def test_groupnorm_silu_fold_matches_rounding(kernel,channels,side,taps):
    with torch.inference_mode():
        torch.manual_seed(side)
        x=torch.randn(2,channels,side,side,dtype=torch.float16).to('mps')
        w=torch.randn(channels,dtype=torch.float16).to('mps');b=torch.randn_like(w)
        reference=F.silu(F.group_norm(x,32,w,b,1e-6))
        result=kernel.groupnorm_fold(x,w,b,32,1e-6,taps)
        assert_bytes(result,torch.cat([reference]*taps,dim=1))


@gpu
def test_silu_fold_all_finite_half_inputs(kernel):
    with torch.inference_mode():
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.float16)
        x=bits[torch.isfinite(bits)].reshape(1,1,1,-1).to('mps')
        assert_bytes(kernel.silu_fold(x,3),torch.cat([F.silu(x)]*3,dim=1))


@gpu
def test_composite_rms_modulation_gate_and_rope(kernel):
    with torch.inference_mode():
        torch.manual_seed(391)
        x=torch.randn(523,3072).to('mps',torch.bfloat16)
        scale=torch.randn(3072).to('mps',torch.bfloat16)
        shift=torch.randn_like(scale)
        inv=kernel.rms_square(x).mean(-1,keepdim=True).add_(1e-6).rsqrt_()
        expected=F.rms_norm(x,(3072,),None,1e-6)*scale+shift
        assert_bytes(kernel.rms_apply(x,inv,scale,shift),expected)
        assert_bytes(kernel.elementwise(x,scale,shift,0,0.),x*scale+shift)
        assert_bytes(kernel.elementwise(x,scale,x,1,0.),x*scale+x)
        q=x.reshape(523,24,128);k=q.roll(1,0).contiguous()
        freqs=torch.randn(523,126,device='mps');cs,sn=freqs.cos(),freqs.sin()
        def reference(t):
            front=t.float()[...,:126];pairs=front.reshape(523,24,63,2)
            rotated=torch.stack((-pairs[...,1],pairs[...,0]),-1).flatten(-2)
            return torch.cat((front*cs[:,None]+rotated*sn[:,None],t.float()[...,126:]),-1).to(t.dtype)
        aq,ak=kernel.rope(q,k,cs,sn)
        assert_bytes(aq,reference(q));assert_bytes(ak,reference(k))


@gpu
def test_bounded_int8_projection_preserves_row_quantization(monkeypatch):
    from _patches import int8_linear_kernel_mps as linear
    from _patches.int8_ext.loader import module
    monkeypatch.setattr(linear, '_kernel', module())
    monkeypatch.setenv('ASFP8_INT8_ACT_CACHE', '0')
    with torch.inference_mode():
        torch.manual_seed(845)
        x=torch.randn(9001,512,dtype=torch.bfloat16).to('mps')
        weight=torch.randint(-127,128,(512,512),dtype=torch.int8).to('mps')
        scale=torch.rand(512,device='mps')*.01
        bias=torch.randn(512,device='mps',dtype=torch.bfloat16)
        expected=linear._int8_linear_kernel(x,weight,scale,bias,x.dtype,True,256)
        actual=linear._int8_linear_bounded(x,weight,scale,bias,x.dtype,True,256)
        assert_bytes(actual,expected)
        # Force the automatic size guard at a small, safe allocation size.
        monkeypatch.setattr(linear, '_LINEAR_MAX_ELEMS', 8192*512)
        assert_bytes(linear._int8_linear_kernel(x,weight,scale,bias,x.dtype,True,256),expected)


@gpu
def test_gather_preserves_strided_rows_and_virtual_concatenation(kernel):
    with torch.inference_mode():
        torch.manual_seed(924)
        full=torch.randn(519,3,24,128,dtype=torch.bfloat16).to('mps')
        a=full[:,1]
        b=torch.randn(58,24,128,dtype=torch.bfloat16).to('mps')
        ix=torch.randperm(577,device='mps')
        assert_bytes(kernel.gather_rows(a,b,ix),torch.cat((a,b))[ix])
        ix=torch.randperm(519,device='mps')
        assert_bytes(kernel.gather_rows(a,a,ix),a[ix])


@gpu
@pytest.mark.parametrize('scale', [1.0, 2.5])
def test_resnet_unit_scale_preserves_half_bytes(monkeypatch, scale):
    from comfy.ldm.seedvr.vae import MemoryState
    monkeypatch.setattr(fast, '_kernel', object())
    monkeypatch.setattr(fast, '_silu_conv', lambda conv, x, *args: x)
    block = SimpleNamespace(time_emb_proj=None, nonlinearity=torch.nn.SiLU(),
        conv1=None, conv2=None, norm1=None, norm2=None, conv_shortcut=None,
        output_scale_factor=scale)
    with torch.inference_mode():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        x = bits[torch.isfinite(bits)].reshape(1, 1, 1, 1, -1).to('mps')
        expected = (x + x) / scale
        actual = fast._resnet(block, x, None, MemoryState.DISABLED)
        assert_bytes(actual, expected)
