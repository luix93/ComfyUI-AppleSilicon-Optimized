"""Optional fused projection: shape/layout correctness and latched fallback."""
import os
from types import SimpleNamespace

import pytest
import torch

from _patches import minimax_vae_fast as patch


def test_projection_failure_latches_without_changing_input(monkeypatch):
    calls = []
    def fail(*args):
        calls.append('kernel')
        raise RuntimeError('intentional compile failure')
    monkeypatch.setattr(patch, '_qkv_kernel', fail)
    monkeypatch.setattr(patch, '_qkv_dead', False)
    monkeypatch.setattr(patch, '_half_mps', lambda x: True)
    monkeypatch.setattr(patch, '_qkv_layer_ok', lambda x: True)
    monkeypatch.setattr(patch, '_ops', SimpleNamespace(_eager_input_act=lambda *a: a[0]))
    layer = SimpleNamespace(to_qkv=SimpleNamespace(
        weight=SimpleNamespace(shape=(6144, 2048)), bias=SimpleNamespace(shape=(6144,))),
        qk_norm_scale=None, norm_q=SimpleNamespace(eps=1e-6))
    norm = SimpleNamespace(weight=None, eps=1e-6)
    x = torch.arange(10)
    reference = x.clone()
    assert patch._project_qkv(layer, x, None, norm) is None
    assert patch._qkv_dead and torch.equal(x, reference)
    assert patch._project_qkv(layer, x, None, norm) is None
    assert calls == ['kernel']


def test_custom_projection_hooks_and_weight_transforms_keep_original(monkeypatch):
    monkeypatch.setattr(patch, '_ops', SimpleNamespace(
        disable_weight_init=SimpleNamespace(Linear=torch.nn.Linear),
        manual_cast=SimpleNamespace(Linear=torch.nn.Linear)))
    layer = torch.nn.Linear(4, 4)
    assert patch._qkv_layer_ok(layer)
    layer.weight_function = [lambda w: w + 1]
    assert not patch._qkv_layer_ok(layer)
    layer.weight_function = []
    handle = layer.register_forward_hook(lambda *a: None)
    assert not patch._qkv_layer_ok(layer)
    handle.remove()
    layer.forward = lambda x: x
    assert not patch._qkv_layer_ok(layer)


@pytest.mark.skipif(os.environ.get('ASFP8_DIT_EXT') != '1' or not torch.backends.mps.is_available(),
                    reason='requires ASFP8_DIT_EXT=1 and MPS')
@pytest.mark.parametrize('batch,tokens,rotary', [(1, 7, 24), (2, 33, 32)])
def test_fused_projection_ragged_rows_batches_and_rotary_layout(batch, tokens, rotary):
    from _patches.minimax_vae_ext.loader import module
    from _patches.minimax_vae_qkv import Qkv
    torch.manual_seed(2731)
    with torch.inference_mode():
        x = torch.randn(batch, tokens, 2048, dtype=torch.float16, device='mps')
        weight = torch.randn(6144, 2048, dtype=torch.float16, device='mps') * .01
        bias = torch.randn(6144, dtype=torch.float16, device='mps') * .01
        scale = torch.randn(64, dtype=torch.float16, device='mps') * .1 + 1
        angles = torch.randn(batch, tokens, 1, rotary, dtype=torch.float16, device='mps')
        rope = torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), -1)
        rope = rope.reshape(batch, tokens, 1, rotary, 2, 2).contiguous()
        expected = module().qkv(torch.nn.functional.linear(x, weight, bias), rope, scale, 32, 1e-6)
        actual = Qkv((64, 256, 4))(x, weight, bias, rope, scale, 1e-6)
        for a, b in zip(actual, expected):
            assert a.shape == (batch, 32, tokens, 64) and a.is_contiguous()
            assert a.dtype == torch.float16 and bool(a.isfinite().all())
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=.003, atol=.003)
