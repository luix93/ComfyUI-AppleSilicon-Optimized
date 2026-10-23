"""Visual guides may be approximated; text/audio and old workflow defaults survive."""
from types import SimpleNamespace
import pytest
from _patches import h3_sol_mps as sol, minimax_h3_sparse as sparse
from _patches import minimax_h3_presets as presets
from _patches import h3_dense_mps as dense, h3_floatpv_mps as fpv
import torch


def test_reference_layout_protects_every_non_visual_stream():
    segments=[(0,80,'text'),(80,1200,'cond'),(1200,1300,'cond_audio'),
              (1300,1700,'ref_img'),(1700,1750,'ref_audio'),(1750,1800,'audio'),
              (1800,3000,'video'),(3000,3008,'pad')]
    layout=SimpleNamespace(seq_len=3008,segments=segments)
    assert sparse.reference_protection(layout,3008)==((0,80),(1200,1300),(1700,1750),(1750,1800),(3000,3008))
    assert sol.normalize_protected_ranges(sparse.reference_protection(layout,3008),3008)==((0,128),(1152,1344),(1664,1856),(2944,3008))


@pytest.mark.parametrize('layout,seq',[
    (None,100),
    (SimpleNamespace(seq_len=100,segments=[(0,100,'cond')]),90),
    (SimpleNamespace(seq_len=100,segments=[(0,10,'text'),(20,100,'cond')]),100),
    (SimpleNamespace(seq_len=100,segments=[(0,10,'text'),(10,99,'cond')]),100),
    (SimpleNamespace(seq_len=100,segments=[(0,10,'text'),(10,100,'video')]),100),
])
def test_unknown_stale_or_unconditioned_layout_keeps_original_prefix(layout,seq):
    assert sparse.reference_protection(layout,seq) is None


def test_unknown_segment_kind_remains_exact():
    layout=SimpleNamespace(seq_len=128,segments=[(0,32,'future'),(32,64,'cond'),(64,128,'video')])
    assert sparse.reference_protection(layout,128)==((0,32),)


def test_defaults_clones_and_independent_switch():
    assert not sparse.H3SparseHolder().approximate_references
    h=sparse.H3SparseHolder(backend='int8',approximate_references=True)
    h.protected_ranges=((0,64),)
    copy=h.clone()
    assert copy.approximate_references and copy.protected_ranges is None
    for name in presets.SPARSE_PRESETS:
        old=presets.resolve_presets('Off',name,'Off')
        new=presets.resolve_presets('Off',name,'Off',approximate_references=True)
        assert not old['sparse_approximate_references']
        assert new['sparse_approximate_references']==(old.get('sparse_backend') in ('int8','int8_hybrid','dense_int8','dense_floatpv','dense_int8_v2','dense_int8_v3'))
        new['sparse_approximate_references']=False
        assert old==new


@pytest.mark.parametrize('ranges', [((-1,64),),((64,63),),((0,129),),((0,1.5),)])
def test_invalid_ranges_rejected_before_shader(ranges):
    with pytest.raises(ValueError):sol.normalize_protected_ranges(ranges,128)


@pytest.mark.parametrize('ratio',[0.,.25])
def test_generated_shader_replaces_every_prefix_guard(monkeypatch,ratio):
    source=[]
    monkeypatch.setattr(sol.torch.mps,'compile_shader',lambda s:source.append(s))
    sol._library.cache_clear()
    try:
        sol._library(601,2,True,'linear',ratio,((0,64),(256,320)))
        code=source[0]
        assert 'inline bool sol_protected' in code
        # Top-k's unused sink local is harmless; no routing tests may use it.
        assert ' < p.begin' not in code and ' > p.end' not in code
        if ratio:assert 'float(J - 2u)' in code
    finally:sol._library.cache_clear()


def test_dense_shader_skips_only_explicit_protected_rows(monkeypatch):
    source=[]
    monkeypatch.setattr(torch.mps,'compile_shader',lambda s:source.append(s))
    dense._library.cache_clear()
    try:
        dense._library(601,56,True,'linear',((0,64),(256,320)))
        assert 'tile.x * SOL_QB < p.begin' not in source[0]
        assert '(tile.x * SOL_QB >= 256u && tile.x * SOL_QB < 320u)' in source[0]
    finally:dense._library.cache_clear()


@pytest.mark.parametrize('backend,implementation,env', [('dense_int8',dense,None),('dense_floatpv',fpv,None),
                                                     ('int8_hybrid','i8',None),('int8_hybrid',dense,'v1')])
def test_dense_dispatch_uses_ranges_but_hybrid_protected_layers_keep_prefix(monkeypatch,backend,implementation,env):
    if implementation=='i8':
        from _patches import int8_attention_mps as implementation
        attribute='h3_attention'
    else:
        attribute='attention'
    if env:monkeypatch.setenv('ASFP8_H3_HYBRID_DENSE',env)
    holder=sparse.H3SparseHolder(backend=backend,approximate_references=True)
    holder.protected_ranges=((0,64),(256,320))
    got=[]
    x=torch.zeros(1,56,601,128)
    def attention(*args,**kwargs):
        got.append(kwargs['protected_ranges'])
        return x,None,0
    monkeypatch.setattr(implementation,attribute,attention)
    sparse._run_dense_int8(holder,x,x,x,None)
    assert got==[None if backend=='int8_hybrid' else holder.protected_ranges]
    assert holder.clone().approximate_references


@pytest.mark.parametrize('backend',['dense_int8','dense_floatpv'])
def test_dense_override_recomputes_layout_and_fails_closed(monkeypatch,backend):
    holder=sparse.H3SparseHolder(backend=backend,approximate_references=True,dense_leading_blocks=0,dense_layers='')
    x=torch.zeros(1,56,601,128)
    layout=SimpleNamespace(seq_len=601,segments=[(0,80,'text'),(80,256,'cond'),(256,320,'audio'),(320,601,'video')])
    monkeypatch.setattr(sparse,'_eligible',lambda *a:None)
    observed=[]
    def run(h,*args):
        observed.append(h.protected_ranges)
        return x[0],torch.ones(1),1
    monkeypatch.setattr(sparse,'_run_sparse',run)
    for current in (layout,None):
        sparse.sparse_attention_override(None,x,x,x,56,skip_reshape=True,skip_output_reshape=True,
            transformer_options={sparse.STATE_KEY:holder,'minimax_h3_layout':current})
    assert observed==[((0,80),(256,320)),None]
    assert holder.stats.reference_approximation_calls==1


@pytest.mark.skipif(not dense.supported(),reason='requires M5 Metal')
@pytest.mark.parametrize('implementation',[dense,fpv])
@pytest.mark.parametrize('tokens',[601,4217])
def test_dense_native_reference_ranges_preserve_protected_bytes(implementation,tokens):
    with torch.inference_mode():
        torch.manual_seed(tokens)
        # Model-style interleaved storage with nonzero offset.
        x=torch.randn(1,tokens+2,3,56,128,device='mps',dtype=torch.bfloat16)[:,1:-1]
        q,k,v=[x[:,:,i].transpose(1,2) for i in range(3)]
        ranges=((0,70),(tokens//2,tokens//2+50))
        args=dict(tau=0.,video_start=tokens//2+50)
        old,*_=implementation.attention(q,k,v,**args)
        off,*_=implementation.attention(q,k,v,**args,protected_ranges=None)
        assert torch.equal(old.cpu().contiguous().view(torch.uint8),off.cpu().contiguous().view(torch.uint8))
        on,*_=implementation.attention(q,k,v,**args,protected_ranges=ranges)
        repeat,*_=implementation.attention(q,k,v,**args,protected_ranges=ranges)
        assert torch.isfinite(on).all()
        assert torch.equal(on.cpu().contiguous().view(torch.uint8),repeat.cpu().contiguous().view(torch.uint8))
        for a,b in sol.normalize_protected_ranges(ranges,tokens):
            expected=torch.nn.functional.scaled_dot_product_attention(q[:,:,a:b],k,v)
            assert torch.equal(on[:,:,a:b].cpu().contiguous().view(torch.uint8),expected.cpu().contiguous().view(torch.uint8))
        # Reference approximation is actually active, rather than a silent no-op.
        assert not torch.equal(on[:,:,128:192].cpu(),off[:,:,128:192].cpu())
