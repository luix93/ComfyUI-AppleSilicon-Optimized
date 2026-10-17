"""Protect layout, GQA, mask and backend-override contracts."""
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from _patches import torch_attention as patch
from _patches import attn_router_mps as router


@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("mask_rank", [None, 2, 3, 4])
@pytest.mark.parametrize("scale", [None, 0.0, 0.125])
def test_layout_mask_gqa_and_zero_scale(gqa, mask_rank, scale):
    torch.manual_seed(12)
    b, h, kv, seq, keys, dim = 2, 4, 2 if gqa else 4, 17, 13, 64
    q = torch.randn(b, seq, h * dim)
    k, v = (torch.randn(b, keys, kv * dim) for _ in range(2))
    mask = torch.rand(seq, keys) > .2 if mask_rank else None
    if mask_rank in (3, 4):
        mask = mask.unsqueeze(0).expand(b, -1, -1)
    if mask_rank == 4:
        mask = mask.unsqueeze(1)
    expanded = mask.unsqueeze(1) if mask_rank == 3 else mask
    expected = F.scaled_dot_product_attention(
        q.reshape(b, seq, h, dim).transpose(1, 2),
        k.reshape(b, keys, kv, dim).transpose(1, 2),
        v.reshape(b, keys, kv, dim).transpose(1, 2),
        attn_mask=expanded, scale=scale, enable_gqa=gqa)
    got = patch.attention(q, k, v, h, mask=mask, scale=scale, enable_gqa=gqa)
    assert torch.equal(got, expected.transpose(1, 2).reshape(b, seq, h * dim))


def test_bypasses_mtl_interception(monkeypatch):
    monkeypatch.setitem(sys.modules, "metal_flash_attn.sdpa",
                        SimpleNamespace(_orig=lambda *a, **kw: "builtin"))
    monkeypatch.setattr(F, "scaled_dot_product_attention", lambda *a, **kw: "wrapper")
    assert patch.sdpa(None, None, None) == "builtin"


def test_explicit_selection_and_version_guard(monkeypatch):
    monkeypatch.setenv("ASFP8_ATTN_BACKEND", "pytorch")
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    assert not patch.enabled()
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    assert patch.enabled()
    monkeypatch.setenv("ASFP8_ATTN_BACKEND", "legacy")
    assert not patch.enabled()


def test_override_precedes_new_backend(monkeypatch):
    q = SimpleNamespace(device=SimpleNamespace(type="mps"))
    monkeypatch.setattr(patch, "supports", lambda *a, **kw: pytest.fail("override bypassed"))
    monkeypatch.setattr(router, "_route_log", lambda *a: None)
    monkeypatch.setattr(router, "_default", lambda *a, **kw: kw["transformer_options"])
    opts = {"optimized_attention_override": object()}
    assert router._router_tensors(q, q, q, 4, transformer_options=opts) is opts


@pytest.mark.parametrize("masked", [False, True])
def test_max_install_routes_long_h3_to_native_before_legacy_paths(monkeypatch, masked):
    from _patches import _common
    monkeypatch.setattr(_common.sys, "platform", "darwin")
    monkeypatch.setattr(torch, "__version__", "2.14.1")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(_common.subprocess, "check_output", lambda *a, **kw: "Apple M5 Max")
    env = {}
    _common.apply_native_master_switch(env)
    _common.apply_default_attention_backend(env)
    monkeypatch.setenv("ASFP8_ATTN_BACKEND", env["ASFP8_ATTN_BACKEND"])
    # Metadata-only tensors: exercise actual selection, with no GPU allocation.
    q = SimpleNamespace(device=SimpleNamespace(type="mps"), dtype=torch.bfloat16,
                        shape=(1, 56, 32612, 128))
    mask = SimpleNamespace(dtype=torch.bool) if masked else None
    monkeypatch.setattr(router, "_route_log", lambda *a: None)
    monkeypatch.setattr(patch, "attention", lambda *a, **kw: "native_pytorch")
    monkeypatch.setattr(router, "_fa41_attention", lambda *a, **kw: pytest.fail("legacy flash selected"))
    monkeypatch.setattr(router, "_sub_quad", lambda *a, **kw: pytest.fail("legacy masked route selected"))
    assert router._router_tensors(q, q, q, 56, mask=mask, skip_reshape=True) == "native_pytorch"
