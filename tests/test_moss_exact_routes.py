"""Exact MOSS routes: joined CFG rows, the INT8 tile schedule and the tile32 default."""
from types import SimpleNamespace as NS
import pytest
import torch
from _patches import moss_fast as p
from _patches import int8_linear_kernel_mps as i8


def _cond(tokens, **extra):
    fields = dict(input_x=torch.zeros(1, 4, 8, 8), control=None, patches=None, hooks=None, area=None,
                  conditioning={'c_crossattn': NS(cond=torch.zeros(1, tokens, 16))})
    fields.update(extra)
    c = NS(**fields)
    c._replace = lambda **k: NS(**{**fields, **k}, _replace=None)
    return c


class _Scope:
    def __enter__(self):
        self.inference = torch.inference_mode(); self.inference.__enter__()
        self.token = p._scope.set({})
        return self

    def __exit__(self, *exc):
        p._scope.reset(self.token); self.inference.__exit__(*exc)


def test_join_only_inside_an_exact_moss_sample(monkeypatch):
    seen = []
    original = lambda a, b: seen.append((a, b)) or False
    a, b = _cond(83), _cond(23)
    assert p._can_join(original, a, b) is False and seen          # no scope
    seen.clear()
    with _Scope():
        # Different prompt lengths pair once the remaining fields are concatable.
        assert p._can_join(lambda x, y: x.conditioning == y.conditioning == {}, a, b) is True
        assert p._can_join(original, a, b) is False
        monkeypatch.setenv('ASFP8_MOSS_CFG_JOIN', '0')
        assert p._can_join(lambda *x: 'original', a, b) == 'original'
        monkeypatch.setenv('ASFP8_MOSS_CFG_JOIN', '1')
        monkeypatch.setenv('ASFP8_MOSS_FFN_EPILOGUE', '1')
        assert p._can_join(lambda *x: 'original', a, b) == 'original'


@pytest.mark.parametrize('extra', [dict(control=object()), dict(patches=object()), dict(area=(1, 1, 0, 0)),
                                   dict(conditioning={'c_crossattn': NS(cond=torch.zeros(1, 9, 16)), 'ref': 1})])
def test_join_refuses_anything_beyond_plain_prompts(extra):
    with _Scope():
        assert p._can_join(lambda *x: 'original', _cond(83, **extra), _cond(23)) == 'original'


def test_join_pads_only_for_transport_and_records_lengths():
    texts = [torch.randn(1, 83, 16), torch.randn(1, 23, 16)]
    conds = [{'c_crossattn': NS(cond=t)} for t in texts]
    with _Scope():
        out = p._join(lambda *a: pytest.fail('cat'), conds)
        assert out['moss_cfg_lengths'] == [83, 23] and out['c_crossattn'].shape == (2, 83, 16)
        assert torch.equal(out['c_crossattn'][1, :23], texts[1][0]) and not out['c_crossattn'][1, 23:].any()
        assert p._join(lambda c, d: 'cat', conds[:1]) == 'cat'
    assert p._join(lambda c, d: 'cat', conds) == 'cat'


def test_unjoined_or_unsafe_calls_run_each_branch_separately(monkeypatch):
    calls = []
    def original(model, x, t, context, options, **kw):
        calls.append((x.shape[0], t.tolist(), context.shape[1], options.get('cond_or_uncond')))
        return x + 1
    model = NS(layers=[])
    x = torch.zeros(2, 4, 2, 2); context = torch.zeros(2, 83, 16)
    with _Scope():
        assert torch.equal(p._joined_forward(original, None, model, x, torch.ones(2), context, {}), x + 1)
        assert calls == [(2, [1., 1.], 83, None)]
        calls.clear()
        # Different timesteps (or non-INT8 projections) cannot share modulation/matmuls.
        out = p._joined_forward(original, None, model, x, torch.tensor([1., 2.]), context,
                                {'cond_or_uncond': [0, 1]}, moss_cfg_lengths=[83, 23])
        assert out.shape == x.shape and calls == [(1, [1.], 83, [0]), (1, [2.], 23, [1])]


def test_joined_rows_require_int8_projections():
    linear = lambda: NS(weight=torch.zeros(1))
    model = NS(layers=[NS(proj={k: linear() for k in ('qkv', 'o', 'ffn_up_gate', 'ffn_down')})])
    assert not p._int8_projections(model)


def test_gemm_schedule_flag_reaches_the_kernel(monkeypatch):
    seen = []
    class Kernel:
        def i8_matmul2d_nt_fused_cs(self, *args):
            seen.append(args[7:])
            return torch.zeros(4, 3, dtype=torch.bfloat16)
    monkeypatch.setattr(i8, '_kernel', Kernel())
    args = (torch.zeros(4, 8, dtype=torch.int8), torch.ones(4, 1), torch.zeros(3, 8, dtype=torch.int8),
            torch.ones(3), None, torch.bfloat16, (4, 8))
    i8._int8_linear_prequant(*args)
    token = i8.moss_gemm_schedule.set(True)
    try:
        i8._int8_linear_prequant(*args)
    finally:
        i8.moss_gemm_schedule.reset(token)
    assert seen == [(), (True,)]


def test_sample_scopes_the_gemm_schedule(monkeypatch):
    monkeypatch.setattr(p, '_cache_budget', lambda: 0)
    monkeypatch.setattr(p, '_eligible', lambda *a: True)
    model = NS(layers=[])
    g = NS(inner_model=NS(diffusion_model=model))
    with torch.inference_mode():
        assert p._sample(lambda *a, **k: i8.moss_gemm_schedule.get(), type(model), g) is True
        monkeypatch.setenv('ASFP8_MOSS_GEMM_SCHEDULE', '0')
        assert p._sample(lambda *a, **k: i8.moss_gemm_schedule.get(), type(model), g) is False
    assert i8.moss_gemm_schedule.get() is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')
@pytest.mark.parametrize('m,k,n', [(4179, 3072, 9216), (4119, 3072, 3072), (8298, 3072, 16384), (8298, 8192, 3072)])
def test_moss_tile_schedule_is_byte_identical(m, k, n):
    kernel = i8._load_kernel()
    if kernel is None:
        pytest.skip('INT8 extension unavailable')
    g = torch.Generator().manual_seed(m + n)
    a = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8).to('mps')
    b = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8).to('mps')
    rs, cs = torch.rand(m, generator=g).to('mps'), torch.rand(n, generator=g).to('mps')
    base = kernel.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None, False, True)
    moss = kernel.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None, False, True, True)
    assert torch.equal(base.view(torch.int16), moss.view(torch.int16))
