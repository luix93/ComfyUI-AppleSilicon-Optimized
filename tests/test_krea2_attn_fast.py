"""Tests for patch #25: Krea2 GQA-native attention.

Unit tests (always run): install gating and fallback guards (they trip before
any comfy import). Parity against the eager forward runs in the standalone
harness (needs comfy + a flash41 build); the flash41 grouped-KV vs repeated-KV
equivalence check runs whenever MPS + the built kernel are available.
"""
import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

from _patches import krea2_attn_fast as patch

_dit_on = os.environ.get("ASFP8_DIT_EXT") == "1"


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_DIT_EXT", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


class TestFallbackGuards:
    def _self(self):
        return SimpleNamespace(heads=8, kvheads=2, headdim=64)

    def _sentinel(self, monkeypatch):
        calls = {}

        def orig(*a, **kw):
            calls["fell_back"] = True
            return "orig"

        monkeypatch.setattr(patch, "_orig_forward", orig, raising=False)
        monkeypatch.setattr(patch, "_kernel", object(), raising=False)
        monkeypatch.setattr(patch, "_fa41", object(), raising=False)
        return calls

    def test_falls_back_on_cpu(self, monkeypatch):
        calls = self._sentinel(monkeypatch)
        x = torch.zeros(1, 16, 512, dtype=torch.bfloat16)
        assert patch._fast_attn_forward(self._self(), x) == "orig"
        assert calls["fell_back"]

    def test_falls_back_on_mask(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch)
        x = torch.zeros(1, 16, 512, dtype=torch.bfloat16, device="mps")
        f = torch.zeros(1, 1, 16, 32, 2, 2, dtype=torch.float32, device="mps")
        out = patch._fast_attn_forward(self._self(), x, freqs=f,
                                       mask=torch.zeros(1))
        assert out == "orig" and calls["fell_back"]

    def test_falls_back_without_freqs(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch)
        x = torch.zeros(1, 16, 512, dtype=torch.bfloat16, device="mps")
        assert patch._fast_attn_forward(self._self(), x, freqs=None) == "orig"
        assert calls["fell_back"]

    def test_falls_back_on_ragged_heads(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch)
        s = SimpleNamespace(heads=8, kvheads=3, headdim=64)   # 8 % 3 != 0
        x = torch.zeros(1, 16, 512, dtype=torch.bfloat16, device="mps")
        f = torch.zeros(1, 1, 16, 32, 2, 2, dtype=torch.float32, device="mps")
        assert patch._fast_attn_forward(s, x, freqs=f) == "orig"
        assert calls["fell_back"]


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_preserves_current_comfy_attention_hooks(monkeypatch, hook):
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    test = TestFallbackGuards()
    calls = test._sentinel(monkeypatch)
    x = torch.zeros(1, 16, 512, dtype=torch.bfloat16, device="mps")
    options = {"block_index": 0, "patches": {hook: [object()]}}
    assert patch._fast_attn_forward(test._self(), x, transformer_options=options) == "orig"
    assert calls["fell_back"]


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() and _dit_on
         and os.environ.get("ASFP8_ATTN_ROUTER") == "1"),
    reason="needs MPS + ASFP8_DIT_EXT=1 + ASFP8_ATTN_ROUTER=1 (flash41 build)",
)
def test_flash41_grouped_kv_matches_repeated_kv():
    from _patches.mfa4_ext import loader
    mod = loader.module()
    if mod is None:
        pytest.skip("flash41 build unavailable")
    torch.manual_seed(0)
    B, Hq, Hkv, S, D = 1, 8, 2, 512, 128
    q = torch.randn(B, Hq, S, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, Hkv, S, D, dtype=torch.bfloat16, device="mps")
    sc = D ** -0.5
    o_gqa = mod.flash_fwd_bf16(q, k, v, sc, False, 0, 0, False)
    o_rep = mod.flash_fwd_bf16(q, k.repeat_interleave(4, dim=1),
                               v.repeat_interleave(4, dim=1), sc, False, 0, 0, False)
    assert (o_gqa.float() - o_rep.float()).abs().max().item() == 0.0
