"""Exact Fibo 2.0 routes: joined CFG rows, fused norm/modulation/residual and tile32 attention."""
import math
from types import SimpleNamespace as NS
import pytest
import torch
import torch.nn.functional as F
from _patches import fibo2_fast as p
from _patches import torch_mpp_qcache as mpp


def _gist_count(text_len, max_gist):
    ratio = min(1.0, max(0.0, (text_len - 256) / (2048 - 256)))
    return max(256, min(max_gist, math.ceil((256 + ratio * (max_gist - 256)) / 64) * 64))


@pytest.fixture(autouse=True)
def _fibo(monkeypatch):
    monkeypatch.setattr(p, '_f2', NS(gist_count=_gist_count))


def _cond(tokens, **extra):
    fields = dict(input_x=torch.zeros(1, 128, 8, 8), control=None, patches=None, hooks=None, area=None,
                  conditioning={'c_crossattn': NS(cond=torch.zeros(1, tokens, 16))})
    fields.update(extra)
    c = NS(**fields)
    c._replace = lambda **k: NS(**{**fields, **k}, _replace=None)
    return c


class _Scope:
    def __init__(self, **state):
        self.state = {"max_gist": 512, "join_ok": True, **state}

    def __enter__(self):
        self.inference = torch.inference_mode(); self.inference.__enter__()
        self.token = p._scope.set(self.state)
        return self

    def __exit__(self, *exc):
        p._scope.reset(self.token); self.inference.__exit__(*exc)


def test_join_only_inside_a_fibo_sample_with_int8_projections(monkeypatch):
    concatable = lambda x, y: x.conditioning == y.conditioning == {}
    a, b = _cond(91), _cond(46)
    assert p._can_join(lambda *x: 'original', a, b) == 'original'          # no scope
    with _Scope():
        assert p._can_join(concatable, a, b) is True
        monkeypatch.setenv('ASFP8_FIBO2_CFG_JOIN', '0')
        assert p._can_join(lambda *x: 'original', a, b) == 'original'
    with _Scope(join_ok=False):
        assert p._can_join(lambda *x: 'original', a, b) == 'original'


def test_join_needs_equal_gist_counts_and_plain_prompts():
    with _Scope():
        # 91 tokens -> 256 gist tokens, 700 -> 320: the image stream would differ in length.
        assert p._can_join(lambda *x: 'original', _cond(700), _cond(46)) is False
        for extra in (dict(control=object()), dict(patches=object()), dict(area=(1, 1, 0, 0)),
                      dict(conditioning={'c_crossattn': NS(cond=torch.zeros(1, 9, 16)), 'ref_latents': 1})):
            assert p._can_join(lambda *x: 'original', _cond(91, **extra), _cond(46)) == 'original'


def test_join_pads_only_for_transport_and_records_lengths():
    texts = [torch.randn(1, 91, 16), torch.randn(1, 46, 16)]
    conds = [{'c_crossattn': NS(cond=t)} for t in texts]
    with _Scope():
        out = p._join(lambda *a: pytest.fail('cat'), conds)
        assert out['fibo2_cfg_lengths'] == [91, 46] and out['c_crossattn'].shape == (2, 91, 16)
        assert torch.equal(out['c_crossattn'][1, :46], texts[1][0]) and not out['c_crossattn'][1, 46:].any()
        assert p._join(lambda c, d: 'cat', conds[:1]) == 'cat'
    assert p._join(lambda c, d: 'cat', conds) == 'cat'


def test_unsafe_joined_calls_run_each_branch_separately(monkeypatch):
    calls = []
    def original(model, x, t, context, options, ref_latents=[], **kw):
        calls.append((x.shape[0], t.tolist(), context.shape[1], options.get('cond_or_uncond'), sorted(kw)))
        return x + 1
    monkeypatch.setattr(p, '_supported', lambda *a: True)
    model = NS(max_gist_tokens=512)
    x = torch.zeros(2, 128, 2, 2); context = torch.zeros(2, 91, 16)
    with _Scope():
        monkeypatch.setenv('ASFP8_FIBO2_GLUE', '0'); monkeypatch.setenv('ASFP8_FIBO2_ATTN', '0')
        # Different timesteps cannot share modulation; each branch gets its own trimmed prompt.
        out = p._forward(original, model, x, torch.tensor([1., 2.]), context, {'cond_or_uncond': [0, 1]},
                         fibo2_cfg_lengths=[91, 46], control=None)
        assert torch.equal(out, x + 1)
        assert calls == [(1, [1.], 91, [0], ['control']), (1, [2.], 46, [1], ['control'])]
        calls.clear()
        # An ordinary unjoined call with every part disabled is the upstream forward.
        p._forward(original, model, x[:1], torch.ones(1), context[:1], {})
        assert calls == [(1, [1.], 91, None, [])]


def test_join_requires_unpatched_int8_projections():
    linear = lambda: NS(weight=torch.zeros(1))
    block = NS(attention=NS(qkv=linear(), out=linear()), feed_forward=NS(w13=linear(), w2=linear(), fused=True))
    assert not p._int8_projections(NS(noise_refiner=[], layers=[block]))


GPU = torch.backends.mps.is_available()


def _upstream(x, w, factor, span, eps):
    # comfy.ldm.lumina.model.modulate after RMSNorm, with one image slice.
    out = F.rms_norm(x, (x.shape[-1],), w, eps)
    if span is None:
        return out * factor[:1].unsqueeze(1)
    s0, s1 = span
    out[:, s0:s1] *= factor[1:].unsqueeze(1)
    out[:, :s0] *= factor[:1].unsqueeze(1)
    out[:, s1:] *= factor[:1].unsqueeze(1)
    return out


@pytest.mark.skipif(not GPU, reason='requires MPS')
@pytest.mark.parametrize('batch,span', [(1, None), (1, (256, 4352)), (2, (256, 4352)), (1, (0, 4096))])
def test_fused_glue_matches_the_upstream_chain(monkeypatch, batch, span):
    if not p._ready():
        pytest.skip('Fibo 2 shader unavailable')
    g = torch.Generator().manual_seed(batch * 7 + (span or (0, 0))[0])
    rows = 4352 if span != (0, 4096) else 8192
    x, y = [(torch.randn(batch, rows, 3584, generator=g) * 3).to(torch.bfloat16).to('mps') for _ in range(2)]
    w = (torch.randn(3584, generator=g) * .3 + 1).to(torch.bfloat16).to('mps')
    scale, gate = [(torch.randn(2 if span else 1, 3584, generator=g) * .2).to(torch.bfloat16).to('mps') for _ in range(2)]
    norm = torch.nn.RMSNorm(3584, eps=1e-5, device='mps', dtype=torch.bfloat16)
    norm.weight.data.copy_(w)
    with _Scope(), torch.inference_mode():
        k = span or (0, 0)
        ref = _upstream(x.clone(), w, 1 + scale, span, 1e-5)
        assert torch.equal(p._modulated(x, norm, scale, k).view(torch.int16), ref.view(torch.int16))
        t = gate.tanh()
        ref = x + _upstream(y.clone(), w, t, span, 1e-5)
        out = p._gated_residual(x.clone(), y, norm, gate, k)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


TILE = GPU and mpp._hardware() and mpp._torch_build_supported()


@pytest.mark.skipif(not TILE, reason='requires the reviewed M5 Ultra/PyTorch MPS stack')
@pytest.mark.parametrize('batch,tokens', [(1, 4352), (2, 4352), (1, 8512)])
def test_tile32_attention_is_byte_identical_to_sdpa(monkeypatch, batch, tokens):
    g = torch.Generator().manual_seed(tokens + batch)
    q, k, v = [torch.randn(batch, 28, tokens, 128, generator=g).to(torch.bfloat16).to('mps') for _ in range(3)]
    with _Scope(), torch.inference_mode():
        out = p._attention_core(q, k, v, 28, {})
        assert p._scope.get()['tile32_calls'] == 1
        for j in range(batch):
            ref = F.scaled_dot_product_attention(q[j:j + 1], k[j:j + 1], v[j:j + 1]).transpose(1, 2).reshape(1, tokens, -1)
            assert torch.equal(out[j:j + 1].view(torch.int16), ref.view(torch.int16))
