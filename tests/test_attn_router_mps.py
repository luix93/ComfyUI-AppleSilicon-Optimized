"""Tests for patch #19: per-call MPS attention router.

Unit tests (always run): install is opt-in; routing decisions pick the right
backend per (mask, seq, dtype) without touching the GPU.

Integration (opt-in ASFP8_ATTN_ROUTER=1 + MPS + mtlflashattn): the flash route
matches a sub-quad reference at a memory-bounded shape (peak transient well
under 1 GB).
"""
import os

import pytest
import torch

from _patches import attn_router_mps as patch

_run_integration = os.environ.get("ASFP8_ATTN_ROUTER") == "1"
requires_router = pytest.mark.skipif(
    not (_run_integration and torch.backends.mps.is_available()),
    reason="set ASFP8_ATTN_ROUTER=1 on an MPS device to test the flash route",
)


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_ATTN_ROUTER", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def _sentinels(monkeypatch, flash_ok=True):
    calls = {}

    def mk(name):
        def f(q, k, v, heads, mask=None, **kw):
            calls["route"] = name
            return name
        return f

    monkeypatch.setattr(patch, "_default", mk("default"), raising=False)
    monkeypatch.setattr(patch, "_default_masked", mk("default_masked"), raising=False)
    monkeypatch.setattr(patch, "_sub_quad", mk("sub_quad"), raising=False)
    monkeypatch.setattr(patch, "_pytorch", mk("pytorch"), raising=False)
    monkeypatch.setattr(patch, "_bmm_attention",
                        lambda q, k, v, heads, **kw: (calls.__setitem__("route", "bmm"), "bmm")[1],
                        raising=False)
    monkeypatch.setattr(patch, "_flash_ok", flash_ok, raising=False)
    return calls


class _Container:
    def __init__(self, value):
        self.value = value

    def peek(self):
        if self.value is None:
            raise RuntimeError("consumed")
        return self.value

    def take(self):
        value = self.peek()
        self.value = None
        return value


def test_router_consumes_all_three_ownership_containers_once(monkeypatch):
    seen = []
    monkeypatch.setattr(patch, "_default", object(), raising=False)
    monkeypatch.setattr(
        patch, "_router_tensors",
        lambda q, k, v, heads, **kwargs: seen.append((q, k, v, heads)) or "ok",
    )
    q, k, v = (_Container(name) for name in "qkv")
    assert patch._router(q, k, v, 7) == "ok"
    assert seen == [("q", "k", "v", 7)]
    assert all(item.value is None for item in (q, k, v))


def test_router_rejects_mixed_tensor_and_container_inputs(monkeypatch):
    monkeypatch.setattr(patch, "_default", object(), raising=False)
    q, k = _Container("q"), _Container("k")
    with pytest.raises(TypeError, match="must all be attention tensor containers"):
        patch._router(q, k, "v", 8)
    # Validate before consuming anything; a caller may still take a fallback.
    assert q.peek() == "q" and k.peek() == "k"


def test_workflow_backend_strips_its_outer_override(monkeypatch):
    seen = {}

    def routed(q, k, v, heads, **kwargs):
        seen.update(kwargs)
        return "routed"

    monkeypatch.setattr(patch, "_router", routed)
    options = {
        "optimized_attention_override": object(),
        "sigmas": "preserved",
    }
    assert patch.workflow_attention(
        "q", "k", "v", 8, transformer_options=options,
        _inside_attn_wrapper=True) == "routed"
    assert "optimized_attention_override" not in seen["transformer_options"]
    assert seen["transformer_options"]["sigmas"] == "preserved"
    assert "_inside_attn_wrapper" not in seen
    assert "optimized_attention_override" in options


def test_comfy_default_backend_bypasses_the_apple_router(monkeypatch):
    seen = {}

    def default(q, k, v, heads, **kwargs):
        seen.update(kwargs)
        return "default"

    monkeypatch.setattr(patch, "_default", default)
    monkeypatch.setattr(patch, "_default_masked", default)
    monkeypatch.setattr(
        patch, "_router", lambda *args, **kwargs: pytest.fail("router fired"))
    options = {"optimized_attention_override": object(), "keep": 1}
    result = patch.comfy_default_attention(
        "q", "k", "v", 8, mask=None, transformer_options=options)
    assert result == "default"
    assert seen["transformer_options"] == {"keep": 1}


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS device")
class TestRouting:
    """Pure routing decisions; sentinel backends, tiny tensors."""

    def _qkv(self, L, heads=8, d_head=64, layout_bhld=False, dtype=torch.bfloat16):
        if layout_bhld:
            t = torch.zeros(1, heads, L, d_head, dtype=dtype, device="mps")
        else:
            t = torch.zeros(1, L, heads * d_head, dtype=dtype, device="mps")
        return t, t, t

    def test_unmasked_long_goes_flash(self, monkeypatch):
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(4096)
        assert patch._router(q, k, v, 8) == "pytorch" and calls["route"] == "pytorch"

    def test_unmasked_long_bhld_layout(self, monkeypatch):
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(5000, layout_bhld=True)
        assert patch._router(q, k, v, 8, skip_reshape=True) == "pytorch"

    def test_unmasked_short_stays_default(self, monkeypatch):
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(1024)
        assert patch._router(q, k, v, 8) == "default"

    def test_masked_long_huge_score_forced_sub_quad(self, monkeypatch):
        # Wan-scale masked attention (score matrix over the safety cap).
        calls = _sentinels(monkeypatch)
        monkeypatch.setattr(patch, "_MASKED_SUB_QUAD_MIN_SCORE_BYTES", 1, raising=False)
        q, k, v = self._qkv(4096)
        mask = torch.zeros(1, 4096, 4096, dtype=torch.bool, device="mps")
        assert patch._router(q, k, v, 8, mask=mask) == "sub_quad"

    def test_masked_long_small_score_stays_default(self, monkeypatch):
        # Long but small score matrix: memory-safe on the default path, and
        # chunked sub-quad would be a large slowdown (LTX regression class).
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(4096)
        mask = torch.zeros(1, 4096, 4096, dtype=torch.bool, device="mps")
        assert patch._router(q, k, v, 8, mask=mask) == "default_masked"

    def test_masked_cross_attn_long_q_short_kv_stays_default(self, monkeypatch):
        # LTX text cross-attention shape: 16k queries over 512 kv — score is
        # tiny; must NOT hit chunked sub-quad despite the long query.
        calls = _sentinels(monkeypatch)
        q, _, _ = self._qkv(16384)
        _, k, v = self._qkv(512)
        mask = torch.zeros(1, 16384, 512, dtype=torch.bool, device="mps")
        assert patch._router(q, k, v, 8, mask=mask) == "default_masked"

    def test_masked_short_stays_default(self, monkeypatch):
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(512)
        mask = torch.zeros(1, 512, 512, dtype=torch.bool, device="mps")
        assert patch._router(q, k, v, 8, mask=mask) == "default_masked"

    def test_flash_off_unmasked_long_goes_bmm(self, monkeypatch):
        # Policy: unmasked long-seq at D>flash_max_head_dim (or flash off) routes
        # to the lean bmm attention (sub-quad only above the score-size cap).
        calls = _sentinels(monkeypatch, flash_ok=False)
        q, k, v = self._qkv(8192)
        assert patch._router(q, k, v, 8) == "bmm"

    def test_unmasked_huge_seq_falls_back_sub_quad(self, monkeypatch):
        # Above _BMM_MAX_SCORE_BYTES (Wan-scale seqs) memory safety wins.
        calls = _sentinels(monkeypatch, flash_ok=False)
        q, k, v = self._qkv(8192)
        monkeypatch.setattr(patch, "_BMM_MAX_SCORE_BYTES", 1, raising=False)
        assert patch._router(q, k, v, 8) == "sub_quad"

    def test_wide_head_dim_unmasked_goes_bmm(self, monkeypatch):
        # D=256 (Ideogram 4): flash kernels can't take it, but unmasked with a
        # bmm-viable score matrix routes to the one-shot bmm, not sub-quad.
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(4096, heads=4, d_head=256)
        assert patch._router(q, k, v, 4) == "bmm"

    def test_very_wide_head_dim_stays_default(self, monkeypatch):
        calls = _sentinels(monkeypatch)
        q, k, v = self._qkv(4096, heads=2, d_head=512)  # D=512 > 256
        assert patch._router(q, k, v, 2) == "default"


@requires_router
def test_fa41_na_kernel_parity():
    """Attention v2 (flash_fwd_na, DrawThings NAX recipe): parity vs fp32
    reference at a gated shape (Lk % 128 == 0, D % 32 == 0), incl. GQA."""
    from _patches.mfa4_ext import loader
    mod = loader.module()
    if mod is None or not hasattr(mod, "flash_fwd_na"):
        pytest.skip("mfa4_ext unavailable or too old")
    torch.manual_seed(0)
    for (B, Hq, Hkv, Lq, Lk, D, dt) in [
        (1, 8, 8, 4352, 4352, 128, torch.float16),
        (1, 12, 3, 4300, 4352, 128, torch.bfloat16),   # GQA + ragged Lq
        (1, 4, 4, 5000, 5000, 128, torch.float16),     # ragged Lk (tail path)
        (1, 6, 6, 1024, 1013, 256, torch.bfloat16),    # D=256 kBlocks + tail
    ]:
        scale = D ** -0.5
        q = torch.randn(B, Hq, Lq, D, dtype=dt, device="mps")
        k = torch.randn(B, Hkv, Lk, D, dtype=dt, device="mps")
        v = torch.randn(B, Hkv, Lk, D, dtype=dt, device="mps")
        kr = k.repeat_interleave(Hq // Hkv, dim=1) if Hq != Hkv else k
        vr = v.repeat_interleave(Hq // Hkv, dim=1) if Hq != Hkv else v
        s = torch.einsum("bhid,bhjd->bhij", q.float() * scale, kr.float())
        ref = torch.einsum("bhij,bhjd->bhid", s.softmax(-1), vr.float())
        out = mod.flash_fwd_na(q, k, v, scale)
        torch.mps.synchronize()
        err = (out.float() - ref).abs().max().item()
        assert err < 5e-3, f"maxerr={err} at {(B, Hq, Hkv, Lq, Lk, D, dt)}"


@requires_router
def test_fa41_na_ragged_dirty_pool():
    """Regression: ragged-Lk tail reads once went past the buffer end; on a
    dirty allocator pool the garbage intermittently corrupted whole images
    (krea, 2026-07-21). Parity on FRESH allocations cannot catch it — this
    poisons the pool with NaN pages first, then checks the exact krea shape."""
    from _patches.mfa4_ext import loader
    mod = loader.module()
    if mod is None or not hasattr(mod, "flash_fwd_na"):
        pytest.skip("mfa4_ext unavailable or too old")
    torch.manual_seed(0)
    junk = [torch.full((64, 1 << 20), float("nan"), dtype=torch.bfloat16,
                       device="mps") for _ in range(4)]
    torch.mps.synchronize()
    del junk
    B, Hq, Hkv, Lq, Lk, D = 1, 48, 12, 4300, 4300, 128
    scale = D ** -0.5
    q = torch.randn(B, Hq, Lq, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, Hkv, Lk, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, Hkv, Lk, D, dtype=torch.bfloat16, device="mps")
    kr = k.repeat_interleave(Hq // Hkv, dim=1)
    vr = v.repeat_interleave(Hq // Hkv, dim=1)
    s = torch.einsum("bhid,bhjd->bhij", q.float() * scale, kr.float())
    ref = torch.einsum("bhij,bhjd->bhid", s.softmax(-1), vr.float())
    out = mod.flash_fwd_na(q, k, v, scale)
    torch.mps.synchronize()
    assert int(torch.isnan(out).sum()) == 0, "NaNs from OOB tail reads"
    err = (out.float() - ref).abs().max().item()
    assert err < 5e-3, f"maxerr={err} (dirty-pool ragged tail)"


@requires_router
def test_flash_route_matches_sub_quad():
    """Parity: flash route vs sub-quad at L=4352 (memory-bounded, <1 GB peak)."""
    import sys
    comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    A = pytest.importorskip("comfy.ldm.modules.attention",
                            reason="requires a ComfyUI checkout two levels up")
    import metal_flash_attn.sdpa as mfa
    mfa.install()
    try:
        torch.manual_seed(0)
        B, H, L, D = 1, 8, 4352, 64
        q = (torch.randn(B, L, H * D) * 0.1).to(torch.bfloat16).to("mps")
        k = (torch.randn(B, L, H * D) * 0.1).to(torch.bfloat16).to("mps")
        v = (torch.randn(B, L, H * D) * 0.1).to(torch.bfloat16).to("mps")
        ref = A.attention_sub_quad(q, k, v, heads=H)
        out = A.attention_pytorch(q, k, v, heads=H)  # SDPA -> mfa flash kernel
        torch.mps.synchronize()
        rel = ((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)).item()
        assert rel < 2e-2, f"rel={rel}"
    finally:
        mfa.uninstall()


# --- dense fallback for attention overrides ---------------------------------
#
# A node that installs `optimized_attention_override` must be able to decline a
# call and still get this router's fast paths, not stock attention.

def test_dense_without_override_strips_only_the_override_key(monkeypatch):
    seen = {}

    def fake_router(q, k, v, heads, **kwargs):
        seen.update(kwargs)
        seen["heads"] = heads
        return "routed"

    monkeypatch.setattr(patch, "_router", fake_router, raising=False)
    monkeypatch.setattr(patch, "_installed", True, raising=False)

    options = {
        "optimized_attention_override": lambda *a, **k: "override",
        "sigmas": "keep-me",
    }
    result = patch.dense_without_override(
        None, "q", "k", "v", 8, mask=None, scale=0.5, skip_reshape=True,
        skip_output_reshape=True, transformer_options=options)

    assert result == "routed"
    assert seen["heads"] == 8
    assert seen["scale"] == 0.5
    assert seen["skip_reshape"] is True
    assert seen["skip_output_reshape"] is True
    # the override is gone for this call, everything else survives
    assert "optimized_attention_override" not in seen["transformer_options"]
    assert seen["transformer_options"]["sigmas"] == "keep-me"
    # the caller's dict is untouched
    assert "optimized_attention_override" in options


def test_dense_without_override_calls_the_router_exactly_once(monkeypatch):
    calls = []
    monkeypatch.setattr(patch, "_router",
                        lambda *a, **k: calls.append(1) or "routed", raising=False)
    monkeypatch.setattr(patch, "_installed", True, raising=False)
    patch.dense_without_override(None, "q", "k", "v", 8, transformer_options={
        "optimized_attention_override": lambda *a, **k: "override"})
    assert len(calls) == 1


def test_dense_without_override_cannot_recurse_into_the_override(monkeypatch):
    """Re-entering must not fire the override again — that would loop forever."""
    fired = []

    def override(func, *args, **kwargs):
        fired.append(1)
        return patch.dense_without_override(func, *args, **kwargs)

    monkeypatch.setattr(patch, "_router",
                        lambda *a, **k: "dense-result", raising=False)
    monkeypatch.setattr(patch, "_installed", True, raising=False)
    options = {"optimized_attention_override": override}
    result = override(None, "q", "k", "v", 8, transformer_options=options)
    assert result == "dense-result"
    assert len(fired) == 1


def test_dense_without_override_uses_comfy_function_when_not_installed(monkeypatch):
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    result = patch.dense_without_override(
        lambda q, k, v, heads, **kw: "comfy", "q", "k", "v", 8,
        transformer_options={"optimized_attention_override": object()})
    assert result == "comfy"


def test_dense_without_override_drops_the_wrap_attn_reentry_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(patch, "_router",
                        lambda q, k, v, h, **kw: seen.update(kw) or "routed",
                        raising=False)
    monkeypatch.setattr(patch, "_installed", True, raising=False)
    patch.dense_without_override(None, "q", "k", "v", 8,
                                 _inside_attn_wrapper=True,
                                 transformer_options={})
    assert "_inside_attn_wrapper" not in seen
