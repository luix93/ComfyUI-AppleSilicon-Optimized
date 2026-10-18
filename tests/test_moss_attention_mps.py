"""Contracts for the optional MOSS dense attention modes."""
import pytest
import torch
from _patches import moss_attention_mps as m
from _patches import torch_mpp_qcache as cache


def test_default_invalid_and_explicit_exact_never_compile(monkeypatch):
    monkeypatch.setattr(m, '_run', lambda *a: pytest.fail('unexpected dispatch'))
    for mode in (None, 'exact', 'unknown'):
        if mode is None:
            monkeypatch.delenv('ASFP8_MOSS_ATTN_MODE', raising=False)
        else:
            monkeypatch.setenv('ASFP8_MOSS_ATTN_MODE', mode)
        assert m.selected_mode() == 'exact'
        assert m.maybe_attention(None, None, None) is None


def test_source_guards_and_bindings():
    with pytest.raises(RuntimeError, match='source changed'):
        m._replace('new source', 'old source', '')
    a, bindings = m._bindings(m._int8_source(4179))
    assert 'SOL_H = 24' in a
    assert 'qsum[r / SOL_QB]' not in a
    assert set(['sol_v_mean', 'sol_quantize', 'sol_attention']) <= bindings.keys()
    b, bindings = m._bindings(m._floatpv_source())
    assert 'constant uint Hq = 24' in b
    assert 'idx[1] * 128 * Hq]' in b
    assert 'int8_attention' in bindings


def test_changed_shader_is_rejected(monkeypatch):
    monkeypatch.setitem(m._HASHES, 'h3_dense_shaders/int8.metal', 'invalid')
    with pytest.raises(RuntimeError, match='reviewed source'):
        m._int8_source(4179)


@pytest.mark.parametrize('mode', m.MODES[1:])
def test_cpu_and_grad_are_untouched(monkeypatch, mode):
    monkeypatch.setenv('ASFP8_MOSS_ATTN_MODE', mode)
    monkeypatch.setattr(m, '_run', lambda *a: pytest.fail('unexpected dispatch'))
    x = torch.zeros(1, 24, 4, 128, dtype=torch.bfloat16)
    assert m.maybe_attention(x, x, x) is None
    with torch.inference_mode():
        assert m.maybe_attention(x, x, x) is None


GPU = torch.backends.mps.is_available() and cache._hardware() and cache._torch_build_supported()


@pytest.mark.skipif(not GPU, reason='requires the reviewed M5 Ultra/PyTorch MPS stack')
@pytest.mark.parametrize('tokens', [4096, 4179, 6167])
@pytest.mark.parametrize('interleaved_v', [False, True])
def test_kernel_bytes_against_each_previous_algorithm(monkeypatch, tokens, interleaved_v):
    from benchmarks.tools.moss_h3_dense_probe import load
    originals = load()
    with torch.inference_mode():
        generator = torch.Generator().manual_seed(tokens)
        q, k, v = [torch.randn(1, tokens, 24, 128, generator=generator, dtype=torch.bfloat16).to('mps').transpose(1, 2) for _ in range(3)]
        if interleaved_v:
            storage = torch.empty(1, tokens, 9216, device='mps', dtype=v.dtype)
            storage[:, :, 6144:].copy_(v.transpose(1, 2).flatten(2))
            v = storage[:, :, 6144:].unflatten(-1, (24, 128)).transpose(1, 2)
        lib = torch.mps.compile_shader(cache._source())
        p = cache._parameters(tuple(q.shape), tuple(k.shape), q.stride(), k.stride(), v.stride(), 128**-.5)
        exact = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lib.custom_attn(q, k, v, exact, p, p, p, threads=(((tokens+63)//64)*128, 24, 1), group_size=(128, 1, 1))
        refs = {'tile32':exact, 'dense_int8':originals['int8'](q, k, v), 'dense_int8_floatpv':originals['floatpv'](q, k, v)}
        for mode, ref in refs.items():
            monkeypatch.setenv('ASFP8_MOSS_ATTN_MODE', mode)
            out = m.maybe_attention(q, k, v)
            assert out is not None, mode
            assert out.transpose(1, 2).is_contiguous()
            assert torch.equal(ref.view(torch.int16), out.view(torch.int16)), mode
            assert out.isfinite().all()


@pytest.mark.skipif(not GPU, reason='requires M5 Ultra MPS')
def test_layout_shape_and_failure_fallback(monkeypatch):
    monkeypatch.setenv('ASFP8_MOSS_ATTN_MODE', 'tile32')
    with torch.inference_mode():
        x = torch.empty(1, 4096, 24, 128, dtype=torch.bfloat16, device='mps').transpose(1, 2)
        assert m.maybe_attention(x.contiguous(), x, x) is None
        assert m.maybe_attention(x[:, :, :100], x[:, :, :100], x[:, :, :100]) is None
        calls = []
        def failing(*args):
            calls.append(1)
            raise RuntimeError('simulated compiler failure')
        monkeypatch.setattr(m, '_run', failing)
        monkeypatch.setattr(m, '_dead_modes', set())
        assert m.maybe_attention(x, x, x) is None
        assert m.maybe_attention(x, x, x) is None
        assert calls == [1]
