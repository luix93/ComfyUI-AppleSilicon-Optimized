"""Opt-in precedence, model-scoped routing and owned-input fallback."""
from types import SimpleNamespace

import pytest
import torch

from _patches import minimax_relaxed as patch
from _patches.mfa4_ext.relaxed_loader import tail_source


def test_exact_defaults_and_native_does_not_relax():
    for env in ({}, {"ASFP8_NATIVE": "1"}, {"ASFP8_MINIMAX_RELAXED": "0"}):
        assert patch.attention_mode(env) == "exact"
        assert not patch.vae_dense_enabled(env)
        assert not patch.vae_qkv_enabled(env)
        assert not patch.vae_tuning_enabled(env)


def test_preset_and_individual_overrides():
    env = {"ASFP8_MINIMAX_RELAXED": "1"}
    assert patch.attention_mode(env) == "tuned"
    assert patch.vae_dense_enabled(env)
    assert patch.vae_qkv_enabled(env)
    assert patch.vae_tuning_enabled(env)
    env.update(ASFP8_MINIMAX_ATTN_MODE="exact", ASFP8_MINIMAX_VAE_DENSE_ATTENTION="0")
    assert patch.attention_mode(env) == "exact"
    assert not patch.vae_dense_enabled(env)
    env['ASFP8_MINIMAX_VAE_QKV'] = '0'
    assert not patch.vae_qkv_enabled(env)
    assert patch.vae_qkv_enabled({'ASFP8_MINIMAX_VAE_QKV': '1'})
    env['ASFP8_MINIMAX_VAE_ATTN_TUNING'] = '0'
    assert not patch.vae_tuning_enabled(env)
    assert patch.vae_tuning_enabled({'ASFP8_MINIMAX_VAE_ATTN_TUNING': '1'})
    assert patch.attention_mode({"ASFP8_MINIMAX_ATTN_MODE": "tail"}) == "tail"
    assert patch.vae_dense_enabled({"ASFP8_MINIMAX_VAE_DENSE_ATTENTION": "1"})
    with pytest.raises(ValueError):
        patch.attention_mode({"ASFP8_MINIMAX_ATTN_MODE": "typo"})


@pytest.mark.parametrize("kwargs", [
    {"mask": object()}, {"attn_precision": torch.float32},
    {"is_causal": True}, {"dropout_p": .1},
    {"transformer_options": {"optimized_attention_override": object()}},
    {"transformer_options": {"optimized_attention_override": None}},
    {"transformer_options": {"asfp8_dense_attention_backend": object()}},
])
def test_declines_mask_precision_unknown_arguments_and_workflow_overrides(kwargs):
    kwargs = {"skip_reshape": True, **kwargs}
    assert not patch._eligible(None, None, None, 56, (), kwargs)


def test_cpu_short_and_training_calls_are_declined():
    q = torch.zeros(1, 56, 5, 128, dtype=torch.bfloat16)
    with torch.inference_mode():
        assert not patch._eligible(q, q, q, 56, (), {"skip_reshape": True})
    with torch.enable_grad():
        assert not patch._eligible(q, q, q, 56, (), {"skip_reshape": True})


class Container:
    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        return self.tensor

    def take(self):
        value, self.tensor = self.tensor, None
        return value


def test_declined_call_preserves_unconsumed_containers(monkeypatch):
    monkeypatch.setattr(patch, "_eligible", lambda *args: False)
    q, k, v = [Container(object()) for _ in range(3)]
    def original(a, b, c, heads, **kwargs):
        assert (a, b, c) == (q, k, v)
        assert all(t.peek() is not None for t in (a, b, c))
        return "original"
    wrapped = patch._make_attention(original, SimpleNamespace())
    assert wrapped(q, k, v, 56, skip_reshape=True) == "original"


def test_failed_kernel_latches_and_falls_back_with_intact_tensors(monkeypatch):
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    tensors = [torch.zeros(1, 56, 2, 128) for _ in range(3)]
    containers = [Container(t) for t in tensors]
    calls = []
    def failing(*args):
        calls.append("kernel")
        raise RuntimeError("intentional kernel failure")
    def original(q, k, v, heads, **kwargs):
        assert all(a is b for a, b in zip((q, k, v), tensors))
        calls.append("original")
        return q
    wrapped = patch._make_attention(original, SimpleNamespace(flash_fwd_na=failing))
    assert wrapped(*containers, 56, skip_reshape=True) is tensors[0]
    assert all(t.peek() is None for t in containers)
    assert wrapped(*tensors, 56, skip_reshape=True) is tensors[0]
    assert calls == ["kernel", "original", "original"]


@pytest.mark.parametrize("unmerged", [False, True])
def test_success_respects_zero_scale_and_output_layout(monkeypatch, unmerged):
    monkeypatch.setattr(patch, "_eligible", lambda *args: True)
    q = torch.arange(56 * 2 * 128).reshape(1, 56, 2, 128)
    def kernel(a, b, c, scale):
        assert scale == 0.0
        return q
    wrapped = patch._make_attention(None, SimpleNamespace(flash_fwd_na=kernel))
    out = wrapped(q, q, q, 56, scale=0.0, skip_output_reshape=unmerged)
    expected = q if unmerged else q.transpose(1, 2).reshape(1, 2, 56 * 128)
    assert torch.equal(out, expected)


def test_source_transform_refuses_unknown_kernel():
    with pytest.raises(ValueError):
        tail_source("unrecognized source")


def test_exact_install_returns_before_loading_extensions(monkeypatch):
    monkeypatch.setattr(patch, "_installed", False)
    monkeypatch.setattr(patch.sys, "platform", "darwin")
    monkeypatch.delenv("ASFP8_MINIMAX_ATTN_MODE", raising=False)
    monkeypatch.delenv("ASFP8_MINIMAX_RELAXED", raising=False)
    monkeypatch.setenv("ASFP8_DIT_EXT", "1")
    patch.install()
    assert not patch._installed
