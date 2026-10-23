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
    # Other suites import ComfyUI, which can leave autograd disabled globally.
    with torch.enable_grad(), conv.packing_scope():
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


def test_current_window_partition_and_reverse_preserve_core_order():
    import comfy.ldm.seedvr.model as model
    shape = torch.tensor([[2, 3, 5], [1, 2, 4]])
    def windows(x):
        return [x[:, :, start:start + 2] for start in range(0, x.shape[2], 2)]
    expected = model.window_idx(shape, windows)
    got = fast._window_idx(shape, windows)
    assert len(got) == len(expected)
    x = torch.arange(shape.prod(-1).sum() * 3.).reshape(-1, 3)
    assert_bytes(got[0](x), expected[0](x))
    assert_bytes(got[1](got[0](x)), expected[1](expected[0](x)))
    assert_bytes(got[2], expected[2])
    for actual, reference in zip(got[3:], expected[3:]):
        if isinstance(actual, torch.Tensor):
            assert_bytes(actual, reference)
        else:
            assert actual == reference


def test_current_repeat_concat_preserves_sorted_lengths_and_text_coalescing():
    import comfy.ldm.seedvr.model as model
    vid_len, txt_len, repeats = torch.tensor([7, 3, 4]), torch.tensor([2, 1]), torch.tensor([2, 1])
    expected = model.repeat_concat_idx(vid_len, txt_len, repeats)
    got = fast._repeat_concat_idx(vid_len, txt_len, repeats)
    assert len(got) == len(expected)
    v, t = torch.arange(42.).reshape(14, 3), torch.arange(9.).reshape(3, 3) + 100
    assert_bytes(got[0](v, t), expected[0](v, t))
    rows = torch.randn_like(got[0](v, t))
    for actual, reference in zip(got[1](rows), expected[1](rows)):
        assert_bytes(actual, reference)
    assert got[2:] == expected[2:]


def test_current_attention_containers_are_consumed_before_cpu_fallback(monkeypatch):
    from comfy.ldm.modules.attention import AttentionTensorContainer
    q, k, v = [torch.randn(3, 2, 4) for _ in range(3)]
    packed = [AttentionTensorContainer(t) for t in (q, k, v)]
    preferred = object()
    def original(a, b, c, *args, **kwargs):
        assert a is q and b is k and c is v
        assert kwargs['preferred_attention'] is preferred
        return a
    monkeypatch.setattr(fast, '_original_attention', original)
    assert fast._bounded_attention(*packed, 2, [0, 3], [0, 3], preferred_attention=preferred) is q
    assert all(p.tensor is None for p in packed)


@pytest.mark.parametrize('version,build,hardware,expected',[
    ('2.14.1',True,True,True),('2.14.0',True,True,False),
    ('2.14.1',False,True,False),('2.14.1',True,False,False)])
def test_current_contract_guard_pins_build_hardware_and_sources(monkeypatch,tmp_path,version,build,hardware,expected):
    import hashlib
    from _patches import torch_mpp_qcache as qc
    m,v=tmp_path/'model.py',tmp_path/'vae.py'
    m.write_text('validated model');v.write_text('validated VAE')
    monkeypatch.setattr(fast,'_CURRENT_MODEL_SHA',hashlib.sha256(m.read_bytes()).hexdigest())
    monkeypatch.setattr(fast,'_CURRENT_VAE_SHA',hashlib.sha256(v.read_bytes()).hexdigest())
    monkeypatch.setattr(torch,'__version__',version)
    monkeypatch.setattr(qc,'_torch_build_supported',lambda:build)
    monkeypatch.setattr(qc,'_hardware',lambda:hardware)
    model,vae=SimpleNamespace(__file__=m),SimpleNamespace(__file__=v)
    assert fast._current_contract_supported(model,vae) is expected
    m.write_text('changed source')
    assert not fast._current_contract_supported(model,vae)
    m.unlink()
    assert not fast._current_contract_supported(model,vae)


@gpu
def test_current_matrix_rope_preserves_bytes_and_inplace_contract(monkeypatch, kernel):
    from comfy.ldm.seedvr import model
    monkeypatch.setattr(fast, '_kernel', kernel)
    with torch.inference_mode():
        torch.manual_seed(674)
        q,k=[torch.randn(533,24,128).to('mps',torch.bfloat16) for _ in range(2)]
        freqs=model._to_flux_freqs_cis(torch.randn(533,126,device='mps'))
        expected=[model._apply_rope1_partial(t.clone(),freqs) for t in (q,k)]
        storage={}
        def cache(key,fn):
            if key not in storage:storage[key]=fn()
            return storage[key]
        owner=SimpleNamespace(get_freqs=lambda shape:freqs)
        out=fast._rope(owner,q,k,None,cache)
        assert out[0] is q and out[1] is k
        for actual,reference in zip(out,expected):assert_bytes(actual,reference)


@gpu
def test_current_block_preserves_kitchen_scale_rounding(monkeypatch, kernel):
    from types import MethodType
    from comfy.ldm.seedvr import model
    monkeypatch.setattr(fast, '_kernel', kernel)
    with torch.inference_mode():
        torch.manual_seed(762)
        norm=torch.nn.RMSNorm(3072,eps=1e-6,elementwise_affine=False,device='mps',dtype=torch.bfloat16)
        norms=SimpleNamespace(shared_weights=True,all=norm,vid_only=False)
        ada=model.MMModule(model.AdaSingle,dim=model.MMArg(3072,3072),emb_dim=18432,
            layers=['attn','mlp'],shared_weights=True,device='mps',dtype=torch.bfloat16)
        for parameter in ada.parameters():parameter.copy_(torch.randn(parameter.shape).to('mps',torch.bfloat16)*.2)
        owner=SimpleNamespace(version=True,attn_norm=norms,mlp_norm=norms,ada=ada,norm_eps=1e-6,
            attn=lambda v,t,*args:(v*.2,t*.3),_chunked_mlp=lambda v,t:(v.sigmoid(),t.tanh()))
        owner._norm_ada_in=MethodType(model.NaMMSRTransformerBlock._norm_ada_in,owner)
        vid=torch.randn(533,3072).to('mps',torch.bfloat16)
        txt=torch.randn(17,3072).to('mps',torch.bfloat16)
        emb=torch.randn(1,18432).to('mps',torch.bfloat16)
        vshape=torch.tensor([[1,13,41]],device='mps');tshape=torch.tensor([[17]],device='mps')
        original=model.NaMMSRTransformerBlock.forward
        monkeypatch.setattr(fast,'_original_block',original)
        expected=original(owner,vid.clone(),txt.clone(),vshape,tshape,emb,model.Cache())
        actual=fast._block(owner,vid.clone(),txt.clone(),vshape,tshape,emb,model.Cache())
        for a,b in zip(actual[:2],expected[:2]):assert_bytes(a,b)


@gpu
@pytest.mark.parametrize('out_channels',[32,64])
def test_current_vae_resnet_preserves_complete_block(monkeypatch,kernel,out_channels):
    from comfy.ldm.seedvr import vae
    from _patches import conv3d_mps
    monkeypatch.setenv('ASFP8_CONV3D','1')
    conv3d_mps.install()
    assert conv3d_mps._installed
    monkeypatch.setattr(fast,'_kernel',kernel)
    original=vae.ResnetBlock3D.forward
    monkeypatch.setattr(fast,'_original_resnet',original)
    with torch.inference_mode():
        torch.manual_seed(156)
        block=vae.ResnetBlock3D(32,out_channels).to('mps',torch.float16)
        for parameter in block.parameters():parameter.copy_(torch.randn(parameter.shape).to('mps',torch.float16)*.1)
        x=torch.randn(1,32,1,32,32).to('mps',torch.float16)
        expected=original(block,x.clone(),vae.MemoryState.DISABLED,None)
        actual=fast._resnet_current(block,x.clone(),vae.MemoryState.DISABLED,None)
        assert_bytes(actual,expected)


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
@pytest.mark.parametrize('preferred',[False,True])
def test_ragged_attention_batched_matches_original(monkeypatch,dtype,preferred):
    import comfy.ldm.seedvr.attention as attention
    monkeypatch.setattr(fast,'_original_attention',attention.var_attention_optimized_split)
    monkeypatch.setenv('ASFP8_SEEDVR2_ATTN_BATCH','8')
    lengths=[483]*36+[118,208,483,1]
    cumulative=[0]
    for n in lengths:cumulative.append(cumulative[-1]+n)
    with torch.inference_mode():
        torch.manual_seed(574)
        q,k,v=[torch.randn(sum(lengths),24,128,dtype=dtype).to('mps') for _ in range(3)]
        kwargs={}
        if preferred:
            def callback(q,k,v,*args,**kwargs):
                assert isinstance(q, attention._attention.AttentionTensorContainer)
                return attention._attention.attention_pytorch(q.take(),k.take(),v.take(),*args,**kwargs)
            kwargs['preferred_attention']=SimpleNamespace(function=callback)
        reference=attention.var_attention_optimized_split(q,k,v,24,cumulative,cumulative,skip_reshape=True,**kwargs)
        result=fast._bounded_attention(q,k,v,24,cumulative,cumulative,skip_reshape=True,**kwargs)
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
    monkeypatch.setenv('ASFP8_INT8_EXT', '1')
    kernel = module()
    assert kernel is not None
    monkeypatch.setattr(linear, '_kernel', kernel)
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


@gpu
@pytest.mark.parametrize('channels,side,taps', [(128,64,1),(128,128,3),(512,17,3)])
def test_split_groupnorm_silu_fold_preserves_half_rounding(kernel,channels,side,taps):
    with torch.inference_mode():
        torch.manual_seed(side)
        x=torch.randn(2,channels,side,side,dtype=torch.float16).to('mps')
        w=torch.randn(channels,dtype=torch.float16).to('mps')
        b=torch.randn_like(w)
        expected=torch.cat([F.silu(F.group_norm(x,32,w,b,1e-6))]*taps,dim=1)
        assert_bytes(kernel.groupnorm_split_fold(x,w,b,32,1e-6,taps),expected)


@gpu
@pytest.mark.parametrize('out_channels,side', [(128,17),(256,64)])
def test_current_resnet_split_norm_preserves_complete_block(monkeypatch,kernel,out_channels,side):
    import comfy.ops as ops
    from comfy.ldm.seedvr import vae
    monkeypatch.setenv('ASFP8_CONV3D','1')
    monkeypatch.setenv('ASFP8_SEEDVR2_VAE_GN','0')
    monkeypatch.setenv('ASFP8_SEEDVR2_VAE_GN_SPLIT','1')
    monkeypatch.setattr(conv,'_installed',True)
    monkeypatch.setattr(conv,'_orig_conv_forward',ops.disable_weight_init.Conv3d._conv_forward)
    monkeypatch.setattr(ops.disable_weight_init.Conv3d,'_conv_forward',conv._patched_conv_forward)
    monkeypatch.setattr(fast,'_kernel',kernel)
    with torch.inference_mode():
        torch.manual_seed(side)
        block=vae.ResnetBlock3D(128,out_channels).to(device='mps',dtype=torch.float16)
        for parameter in block.parameters():
            parameter.copy_(torch.randn_like(parameter)*.02)
        x=torch.randn(1,128,1,side,side,device='mps',dtype=torch.float16)
        monkeypatch.setattr(fast,'_current_vae_fusions',False)
        reference=fast._resnet_current(block,x,vae.MemoryState.DISABLED,{})
        monkeypatch.setattr(fast,'_current_vae_fusions',True)
        actual=fast._resnet_current(block,x,vae.MemoryState.DISABLED,{})
        assert_bytes(actual,reference)
        monkeypatch.setenv('ASFP8_SEEDVR2_VAE_GN_SPLIT','0')
        assert_bytes(fast._resnet_current(block,x,vae.MemoryState.DISABLED,{}),reference)
