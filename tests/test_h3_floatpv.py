"""Optional floating-PV routing and pre-allocation memory safety."""
import pytest
import torch
from _patches import h3_floatpv_mps as fpv, h3_dense_mps as dense
from _patches import minimax_h3_sparse as sparse
from _patches import minimax_h3_presets as presets
from test_h3_dense_attention import invoke


def test_removed_dense_presets_alias_to_v3_and_backends_stay_available():
    assert presets.SPARSE_PRESETS['Off'] == {}
    for old, new in [('Dense INT8 · protected', 'Dense Metal · protected'),
                     ('Dense QK INT8 / PV BF16 · protected', 'Dense Metal · protected'),
                     ('Dense QK INT8 / PV BF16 · full experimental', 'Dense Metal · full experimental')]:
        assert old not in presets.SPARSE_PRESETS and presets.SPARSE_ALIASES[old] == new
        assert presets.resolve_presets('Off', old, 'Off') == presets.resolve_presets('Off', new, 'Off')
    protected = presets.SPARSE_PRESETS['Dense Metal · protected']
    full = presets.SPARSE_PRESETS['Dense Metal · full experimental']
    assert protected['sparse_backend'] == full['sparse_backend'] == 'dense_int8_v3'
    assert (protected['sparse_start_percent'], protected['sparse_end_percent']) == (.2,.9)
    assert protected['sparse_dense_layers'] == '0,1,33,34,35'
    assert full['sparse_dense_layers'] == '' and full['sparse_dense_leading_blocks'] == 0
    assert (full['sparse_start_percent'], full['sparse_end_percent']) == (0,1)
    # The floating-PV backend itself remains selectable on the advanced node.
    from _patches import minimax_h3_unified as unified
    assert 'dense_floatpv' in unified.SPARSE_BACKENDS


def test_floatpv_clone_and_hidden_topk():
    holder=sparse.H3SparseHolder(backend='dense_floatpv',topk_ratio=.15)
    assert holder.topk_ratio==0 and holder.clone().backend=='dense_floatpv'


@pytest.mark.parametrize('reason',['protected_steps','protected_layer','short_sequence','masked_attention','not_mps'])
def test_floatpv_respects_exact_guards(monkeypatch,reason):
    holder=sparse.H3SparseHolder(backend='dense_floatpv')
    monkeypatch.setattr(sparse,'_run_sparse',lambda *a:pytest.fail('bypassed protection'))
    x,out=invoke(monkeypatch,holder,reason=reason)
    assert torch.equal(out,x+2)


def test_only_selected_floatpv_backend_calls_new_implementation(monkeypatch):
    from _patches import int8_attention_mps as i8
    x=torch.zeros(1,56,67,128)
    monkeypatch.setattr(fpv,'attention',lambda *a,**kw:(x+1,None,0))
    monkeypatch.setattr(dense,'attention',lambda *a,**kw:(x+2,None,0))
    monkeypatch.setattr(i8,'h3_attention',lambda *a,numerics,**kw:(x+(3 if numerics=='tiled' else 4),None,0))
    cases=[('dense_floatpv',1),('dense_int8',2),('int8_hybrid',3),('dense_int8_v3',3),('dense_int8_v2',4)]
    for backend,value in cases:
        holder=sparse.H3SparseHolder(backend=backend)
        got,_,_=sparse._run_dense_int8(holder,x,x,x,None)
        assert torch.equal(got,x[0]+value)
    # The original v1 kernel remains available to the hybrid for comparison.
    monkeypatch.setenv('ASFP8_H3_HYBRID_DENSE','v1')
    got,_,_=sparse._run_dense_int8(sparse.H3SparseHolder(backend='int8_hybrid'),x,x,x,None)
    assert torch.equal(got,x[0]+2)


def test_budget_rejection_precedes_shader_or_gpu_allocation(monkeypatch):
    monkeypatch.setattr(fpv,'workspace_budget',lambda *a:(1,2))
    monkeypatch.setattr(torch,'empty',lambda *a,**k:pytest.fail('allocated before budget'))
    with pytest.raises(fpv.WorkspaceBudgetExceeded):fpv.check_workspace(19508)
    assert fpv.scratch_bytes(65536)<4*fpv.scratch_bytes(19508)


def test_invalid_cpu_input_is_rejected_before_gpu_use(monkeypatch):
    monkeypatch.setattr(fpv,'supported',lambda:True)
    monkeypatch.setattr(fpv,'_library',lambda:pytest.fail('compiled invalid input'))
    x=torch.zeros(1,56,65,128,dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='MPS BF16'):fpv.attention(x,x,x)


@pytest.mark.parametrize('failure,latches', [
    (sparse._SparseBudgetExceeded, False), (sparse._SparseUnavailable, True)])
def test_floatpv_failure_falls_back_and_fresh_run_can_retry(monkeypatch, failure, latches):
    holder = sparse.H3SparseHolder(backend='dense_floatpv', dense_leading_blocks=0)
    def fail(*args):
        raise failure('test')
    monkeypatch.setattr(sparse, '_run_sparse', fail)
    x, got = invoke(monkeypatch, holder)
    assert torch.equal(got, x + 2)
    assert holder.disabled == latches
    assert not holder.clone().disabled


def test_floatpv_dense_reference_returns_exact_output(monkeypatch):
    holder = sparse.H3SparseHolder(backend='dense_floatpv', mode='dense_reference', dense_leading_blocks=0)
    monkeypatch.setattr(sparse, '_run_sparse', lambda h, q, k, v, scale:
                        (q[0]+10, torch.ones(2, 2), 8))
    x, got = invoke(monkeypatch, holder)
    assert torch.equal(got, x + 2)
    assert holder.stats.routed_calls == 1 and holder.stats.sparse_calls == 0


def test_shader_binding_resolves_all_function_constants(monkeypatch):
    compiled = []
    monkeypatch.setattr(torch.mps, 'compile_shader', lambda source: compiled.append(source) or object())
    fpv._library.cache_clear()
    try:
        _, bindings = fpv._library()
        assert 'function_constant(' not in compiled[0]
        assert {'compute_v_mean', 'finalize_v_mean', 'quantize_q', 'quantize_k', 'int8_attention'} <= bindings.keys()
    finally:
        fpv._library.cache_clear()
