"""Current LTX split-RoPE: preserve eager store rounding and fallback."""
import math
import pytest
import torch
from comfy_kitchen.backends.eager.rope import apply_rope_split_half1
from _patches import ltx25_fast as patch
from _patches import ltx25_vae as vae_patch
from comfy_kitchen.backends.eager import na
from comfy_kitchen.backends.eager.rope import rms_rope_


def bits(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


@pytest.fixture
def kernel():
    if not (torch.backends.mps.is_available() and patch.torch_mpp_qcache._hardware()
            and torch.__version__.split('+')[0] == '2.14.0'):
        pytest.skip('validated M5 Ultra / PyTorch 2.14 required')
    assert patch._lib() is not None


@pytest.mark.parametrize('seq,dim', [(1,64), (26,64), (880,128), (3520,128)])
@pytest.mark.parametrize('seed', [0,42,901])
def test_current_split_rope(kernel, seq, dim, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        # Include nonzero storage offsets and independent Q/K values.
        q,k=[torch.randn(1,seq+2,32*dim,device='mps',dtype=torch.bfloat16)[:,1:-1] for _ in range(2)]
        pe=torch.randn(1,seq,32,dim//2,2,2,device='mps',dtype=torch.bfloat16)
        original=[apply_rope_split_half1(t.view(1,seq,32,dim),pe).view(t.shape) for t in (q,k)]
        assert patch._supports(q,(pe,True))
        for a,b in zip(patch._rope(q,k,pe),original):bits(a,b)
        bits(patch._rope(q,None,pe),original[0])


def test_unsupported_rotations_fall_back(kernel, monkeypatch):
    with torch.inference_mode():
        q=torch.ones(1,3,4096,device='mps',dtype=torch.bfloat16)
        pe=torch.ones(1,3,32,64,2,2,device='mps',dtype=torch.bfloat16)
        assert not patch._supports(q,(pe,False))
        assert not patch._supports(q,(pe,pe,True))
        assert not patch._supports(q,(pe.float(),True))
        assert not patch._supports(q.cpu(),(pe,True))
        monkeypatch.setenv('ASFP8_LTX25_ROPE','0')
        assert not patch._supports(q,(pe,True))


@pytest.mark.parametrize('seq,heads', [(1,4),(73,8),(1152,16),(5632,32)])
@pytest.mark.parametrize('seed', [0,42,901])
def test_vae_inplace_rope(kernel, seq, heads, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        q,k=[torch.randn(1,seq+2,heads,64,device='mps',dtype=torch.bfloat16)[:,1:-1] for _ in range(2)]
        qw,kw=[torch.randn(64,device='mps',dtype=torch.bfloat16) for _ in range(2)]
        pe=torch.randn(1,seq,1,32,2,2,device='mps')
        oq,ok=q.clone(),k.clone()
        rms_rope_(oq,ok,pe,qw,kw)
        token=vae_patch._scope.set(True)
        try:
            result=vae_patch._rms_rope(q,k,pe,qw,kw)
        finally:
            vae_patch._scope.reset(token)
        assert result[0] is q and result[1] is k
        bits(q,oq);bits(k,ok)


@pytest.mark.parametrize('dims,kernels', [((2,3,4),(3,7,7)),((13,32,64),(11,11,11)),((7,16,16),(3,7,7))])
def test_mask_and_batched_attention(kernel, monkeypatch, dims, kernels):
    monkeypatch.setattr(vae_patch,'_geometry',na)
    with torch.inference_mode():
        bounds=[na._window_bounds(d,k,False) for d,k in zip(dims,kernels)]
        rel=tuple((tuple(a),tuple(b)) for a,b in bounds)
        # Avoid allocating the untiled full mask at the large case.
        if math.prod(dims)**2 < 2**25:
            bits(vae_patch._group_mask(rel,torch.bfloat16,torch.device('mps')),
                 na._group_mask(rel,torch.bfloat16,torch.device('mps')))
        q,k,v=[torch.randn(1,*dims,4,64,device='mps',dtype=torch.bfloat16) for _ in range(3)]
        bits(vae_patch._batched_na(q,k,v,list(kernels),None,1.),na.na3d(q,k,v,list(kernels),None,1.))


@pytest.mark.parametrize('seq', [3520,3581,14080])
def test_ltx_query_cache(kernel,monkeypatch,seq):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND','pytorch')
    with torch.inference_mode():
        q,k,v=[torch.randn(1,seq,32,128,device='mps',dtype=torch.bfloat16).transpose(1,2) for _ in range(3)]
        got=patch.torch_mpp_qcache.maybe_attention(q,k,v,output_bshd=True)
        assert got is not None and got.transpose(1,2).is_contiguous()
        bits(got,torch.nn.functional.scaled_dot_product_attention(q,k,v))


def test_vae_scope_and_scalar_arguments_fall_back(monkeypatch):
    calls=[]
    monkeypatch.setattr(vae_patch,'_original_na',lambda *a:calls.append(a) or 'original')
    # Outside the LTX scope, preserve the public Kitchen scalar API.
    x=torch.zeros(1,2,3,4,4,64)
    assert vae_patch._na3d(x,x,x,3,False,1.)=='original'
    # Unsupported forms inside the scope must also reach Kitchen's normalizer.
    monkeypatch.setattr(vae_patch,'_tensor',lambda *a:True)
    with torch.inference_mode():
        token=vae_patch._scope.set(True)
        try:
            assert vae_patch._na3d(x,x,x,3,False,1.)=='original'
            assert vae_patch._na3d(x,x,x,[3,3,3],True,1.)=='original'
        finally:
            vae_patch._scope.reset(token)
    assert len(calls)==3


def test_vae_swiglu_all_finite_bfloat_inputs(kernel):
    with torch.inference_mode():
        values = torch.arange(65536, dtype=torch.int32, device='mps').to(torch.int16).view(torch.bfloat16)
        table = torch.nn.functional.silu(values)
        gate = values[values.isfinite()].contiguous()
        torch.manual_seed(801)
        up = torch.randn(gate.shape, dtype=gate.dtype, device=gate.device)
        got = torch.empty_like(gate)
        patch._library.vae_swiglu(gate, up, table, got, threads=gate.numel(), group_size=256)
        bits(got, torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize('channels', [256,512,1024,2048])
def test_vae_modulate_preserves_rounding(kernel, monkeypatch, channels):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as decoder
    monkeypatch.setattr(vae_patch, '_decoder', decoder)
    torch.manual_seed(channels)
    with torch.inference_mode():
        x = torch.randn(1,3,5,7,channels, device='mps', dtype=torch.bfloat16)
        scale, shift = [torch.randn(1,1,1,1,channels, device='mps', dtype=x.dtype) for _ in range(2)]
        bits(vae_patch._modulate(x, scale, shift), x * (1.0 + scale) + shift)


def test_vae_swiglu_residual_and_fallback(kernel, monkeypatch):
    from types import SimpleNamespace
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as decoder
    monkeypatch.setattr(vae_patch, '_decoder', decoder)
    original = decoder.SwiGLU.forward
    monkeypatch.setattr(vae_patch, '_original_swiglu', original)
    torch.manual_seed(902)
    layer = SimpleNamespace(**{name: torch.nn.Linear(a,b,bias=False,device='mps',dtype=torch.bfloat16)
                              for name,a,b in [('w_gate',256,1024),('w_up',256,1024),('w_down',1024,256)]})
    with torch.inference_mode():
        x = torch.randn(1,3,5,7,256,device='mps',dtype=torch.bfloat16)
        pre = lambda value: torch.nn.functional.rms_norm(value, (256,))
        expected_x, got_x = x.clone(), x.clone()
        expected = original(layer, expected_x, pre, expected_x)
        got = vae_patch._swiglu(layer, got_x, pre, got_x)
        assert got is got_x
        bits(got, expected)
        monkeypatch.setenv('ASFP8_LTX25_VAE_SWIGLU','0')
        bits(vae_patch._swiglu(layer,x,pre), original(layer,x,pre))


def test_vae_modulate_unsupported_calls_fall_back(monkeypatch):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as decoder
    monkeypatch.setattr(vae_patch, '_decoder', decoder)
    monkeypatch.setattr(vae_patch, '_original_modulate', lambda x,s,h: x*(1+s)+h)
    x = torch.randn(2,3,5,7,16)
    scale, shift = torch.randn(16), torch.randn(16)
    bits(vae_patch._modulate(x,scale,shift), x*(1+scale)+shift)


@pytest.mark.parametrize('dims,heads,kernels', [((2,3,4),4,(3,7,7)),((7,16,19),8,(3,5,5)),((13,32,64),4,(11,11,11))])
def test_vae_packed_layout_and_index_cache_bound(kernel, monkeypatch, dims, heads, kernels):
    monkeypatch.setattr(vae_patch,'_geometry',na)
    vae_patch._index_cache.clear()
    monkeypatch.setattr(vae_patch,'_index_bytes',0)
    monkeypatch.setattr(vae_patch,'_INDEX_BUDGET',8192)
    torch.manual_seed(501)
    with torch.inference_mode():
        q,k,v=[torch.randn(1,*dims,heads,64,device='mps',dtype=torch.bfloat16) for _ in range(3)]
        expected=vae_patch._batched_na(q,k,v,kernels,None,1.)
        bits(vae_patch._packed_na(q,k,v,kernels,1.),expected)
        assert vae_patch._index_bytes<=8192
        assert len(vae_patch._index_cache)<=128
    vae_patch._index_cache.clear()
