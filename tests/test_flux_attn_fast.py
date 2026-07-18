"""Tests for patch #24: fused Flux double/single block attention prep.

Unit tests (always run): install gating, per-call fallback guards (they trip
before any comfy import, so no ComfyUI checkout is needed).

Parity (needs MPS + importable comfy + ASFP8_DIT_EXT=1): fast forwards match
the eager block forwards within 1 bf16 ulp at output magnitudes.
"""
import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

from _patches import flux_attn_fast as patch

_has_comfy = importlib.util.find_spec("comfy") is not None
_dit_on = os.environ.get("ASFP8_DIT_EXT") == "1"

requires_parity = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and _has_comfy and _dit_on),
    reason="needs MPS + comfy on sys.path + ASFP8_DIT_EXT=1",
)


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_DIT_EXT", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


class TestFallbackGuards:
    """Guards trip before the fused path (and before any comfy import)."""

    def _self(self):
        return SimpleNamespace(num_heads=4, hidden_size=4 * 64, modulation=True)

    def _sentinel(self, monkeypatch, name):
        calls = {}

        def orig(*a, **kw):
            calls["fell_back"] = True
            return name

        monkeypatch.setattr(patch, f"_orig_{name}", orig, raising=False)
        monkeypatch.setattr(patch, "_kernel", object(), raising=False)
        return calls

    def test_double_falls_back_on_cpu(self, monkeypatch):
        calls = self._sentinel(monkeypatch, "double")
        img = torch.zeros(1, 8, 256, dtype=torch.bfloat16)
        txt = torch.zeros(1, 4, 256, dtype=torch.bfloat16)
        out = patch._fast_double_forward(self._self(), img, txt, None, None)
        assert out == "double" and calls["fell_back"]

    def test_double_falls_back_on_mask(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch, "double")
        img = torch.zeros(1, 8, 256, dtype=torch.bfloat16, device="mps")
        txt = torch.zeros(1, 4, 256, dtype=torch.bfloat16, device="mps")
        pe = torch.zeros(1, 1, 12, 32, 2, 2, dtype=torch.float32, device="mps")
        out = patch._fast_double_forward(self._self(), img, txt, None, pe,
                                         attn_mask=torch.zeros(1))
        assert out == "double" and calls["fell_back"]

    def test_single_falls_back_on_bad_pe(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch, "single")
        x = torch.zeros(1, 12, 256, dtype=torch.bfloat16, device="mps")
        out = patch._fast_single_forward(self._self(), x, None, None)  # pe=None
        assert out == "single" and calls["fell_back"]

    def test_single_falls_back_on_attn1_patch(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("needs MPS")
        calls = self._sentinel(monkeypatch, "single")
        x = torch.zeros(1, 12, 256, dtype=torch.bfloat16, device="mps")
        pe = torch.zeros(1, 1, 12, 32, 2, 2, dtype=torch.float32, device="mps")
        out = patch._fast_single_forward(
            self._self(), x, None, pe,
            transformer_options={"patches": {"attn1_patch": [lambda *a, **k: None]}})
        assert out == "single" and calls["fell_back"]


@requires_parity
class TestParity:
    """Fast forwards vs eager on real blocks (small shapes, random weights)."""

    H, D = 4, 128
    S_txt, S_img = 16, 64

    @pytest.fixture()
    def blocks(self):
        import comfy.ops
        from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

        if patch._kernel is None:
            from _patches.dit_ext import loader
            patch._kernel = loader.module()
        if patch._kernel is None:
            pytest.skip("dit_ext kernel unavailable")
        if patch._orig_double is None:
            from comfy.ldm.flux.layers import DoubleStreamBlock as DB, SingleStreamBlock as SB
            patch._orig_double = DB.forward
            patch._orig_single = SB.forward

        torch.manual_seed(0)
        hidden = self.H * self.D
        ops = comfy.ops.disable_weight_init

        def init(m):
            for p in m.parameters():
                torch.nn.init.normal_(p, std=0.02)
            return m.to("mps", torch.bfloat16)

        dbl = init(DoubleStreamBlock(hidden, self.H, mlp_ratio=4.0, qkv_bias=True,
                                     dtype=torch.bfloat16, device="mps", operations=ops))
        sgl = init(SingleStreamBlock(hidden, self.H, mlp_ratio=4.0,
                                     dtype=torch.bfloat16, device="mps", operations=ops))
        return dbl, sgl

    def _pe(self, S):
        from comfy.ldm.flux.layers import EmbedND
        ids = torch.zeros(1, S, 3, device="mps")
        ids[:, :, 1] = torch.arange(S, device="mps")
        return EmbedND(dim=self.D, theta=10000, axes_dim=[16, 56, 56])(ids)

    def test_double_parity(self, blocks):
        dbl, _ = blocks
        hidden = self.H * self.D
        img = torch.randn(1, self.S_img, hidden, device="mps", dtype=torch.bfloat16)
        txt = torch.randn(1, self.S_txt, hidden, device="mps", dtype=torch.bfloat16)
        vec = torch.randn(1, hidden, device="mps", dtype=torch.bfloat16)
        pe = self._pe(self.S_txt + self.S_img)

        ei, et = patch._orig_double(dbl, img.clone(), txt.clone(), vec, pe)
        fi, ft = patch._fast_double_forward(dbl, img.clone(), txt.clone(), vec, pe)
        for f, e in ((fi, ei), (ft, et)):
            d = (f.float() - e.float()).abs()
            scale = e.float().abs().max().item()
            assert d.max().item() <= max(1e-2, scale) * 2 ** -7  # ~1 bf16 ulp
            assert d.mean().item() < 2e-3

    def test_single_parity(self, blocks):
        _, sgl = blocks
        hidden = self.H * self.D
        S = self.S_txt + self.S_img
        x = torch.randn(1, S, hidden, device="mps", dtype=torch.bfloat16)
        vec = torch.randn(1, hidden, device="mps", dtype=torch.bfloat16)
        pe = self._pe(S)

        e = patch._orig_single(sgl, x.clone(), vec, pe)
        f = patch._fast_single_forward(sgl, x.clone(), vec, pe)
        d = (f.float() - e.float()).abs()
        scale = e.float().abs().max().item()
        assert d.max().item() <= max(1e-2, scale) * 2 ** -7
        assert d.mean().item() < 2e-3
