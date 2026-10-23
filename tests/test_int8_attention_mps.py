"""Opt-in dense INT8 attention: accuracy bounds, layouts and fallback contracts."""
import types

import pytest
import torch

from _patches import attention_backend as backend
from _patches import attn_router_mps as router
from _patches import int8_attention_mps as i8
from _patches import minimax_h3_sparse as sparse

requires_m5 = pytest.mark.skipif(not i8.supported(), reason="requires an M5 MPS device")


def _tensors(*shape, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype).to("mps")


def _rel(out, q, k, v, scale=None):
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)
    return ((out.float() - ref).norm() / ref.norm()).item()


@requires_m5
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lq,lk", [(2048, 2048), (3000, 5000), (1, 1025), (2049, 2049)])
def test_dense_int8_tracks_fp32_reference(dtype, lq, lk):
    q = _tensors(1, 4, lq, 128, dtype=dtype, seed=1)
    k, v = _tensors(1, 4, lk, 128, dtype=dtype, seed=2), _tensors(1, 4, lk, 128, dtype=dtype, seed=3)
    out = i8.attention(q, k, v)
    assert out.dtype == dtype and torch.isfinite(out).all()
    assert _rel(out, q, k, v) < 0.035


@requires_m5
def test_batch_scale_and_determinism():
    q, k, v = (_tensors(2, 3, 4096, 128, seed=s) for s in (4, 5, 6))
    first = i8.attention(q, k, v, scale=0.05)
    assert _rel(first, q, k, v, 0.05) < 0.035
    assert torch.equal(first, i8.attention(q, k, v, scale=0.05))


@requires_m5
def test_strided_inputs_and_output_avoid_layout_copies():
    b, n, h = 1, 4096, 6
    flat = [_tensors(b, n, h * 128, seed=s) for s in (7, 8, 9)]
    q, k, v = (t.view(b, n, h, 128).transpose(1, 2) for t in flat)
    buffer = torch.empty(b, n, h, 128, dtype=torch.bfloat16, device="mps")
    i8.attention(q, k, v, out=buffer.transpose(1, 2))
    contiguous = i8.attention(*(t.contiguous() for t in (q, k, v)))
    assert torch.equal(buffer.transpose(1, 2), contiguous)


def test_ineligible_shapes_are_rejected_without_dispatch(monkeypatch):
    monkeypatch.setattr(i8, "supported", lambda: True)
    cpu = torch.zeros(1, 2, 2048, 128, dtype=torch.bfloat16)
    assert not i8.eligible(cpu, cpu, cpu)
    with pytest.raises(ValueError):
        i8.attention(cpu, cpu, cpu)


@pytest.mark.parametrize("kwargs", [dict(mask=torch.zeros(1)), dict(enable_gqa=True)])
def test_comfy_attention_delegates_masked_and_gqa_calls(monkeypatch, kwargs):
    calls = []
    x = torch.zeros(1, 16384, 256)
    def fallback(q, k, v, heads, **kw):
        calls.append(kw)
        return "dense"
    monkeypatch.setattr(i8, "attention", lambda *a, **k: pytest.fail("quantized an ineligible call"))
    assert i8.comfy_attention(x, x, x, 2, fallback=fallback, **kwargs) == "dense"
    assert calls and calls[0].get("mask") is kwargs.get("mask")


def test_comfy_attention_short_sequences_use_fallback(monkeypatch):
    monkeypatch.setattr(i8, "eligible", lambda *a: True)
    monkeypatch.setattr(i8, "attention", lambda *a, **k: pytest.fail("short call quantized"))
    x = torch.zeros(1, 4096, 256)
    assert i8.comfy_attention(x, x, x, 2, fallback=lambda *a, **k: "dense") == "dense"


def test_min_tokens_env_is_bounded(monkeypatch):
    monkeypatch.setenv("ASFP8_INT8_ATTN_MIN_TOKENS", "16")
    assert i8.min_tokens() == i8.MIN_KEYS
    monkeypatch.setenv("ASFP8_INT8_ATTN_MIN_TOKENS", "bad")
    assert i8.min_tokens() == i8.DEFAULT_MIN_TOKENS


@pytest.mark.parametrize("skip_output_reshape", [False, True])
def test_comfy_attention_layouts(monkeypatch, skip_output_reshape):
    monkeypatch.setattr(i8, "eligible", lambda *a: True)
    seen = {}
    def fake(q, k, v, *, scale=None, out=None, numerics="running"):
        seen.update(q=q.shape, out=out.shape, stride=out.stride(), scale=scale)
        out.fill_(1)
        return out
    monkeypatch.setattr(i8, "attention", fake)
    x = torch.zeros(1, 9000, 2 * 128, dtype=torch.bfloat16)
    result = i8.comfy_attention(x, x, x, 2, skip_output_reshape=skip_output_reshape, scale=0.1)
    assert seen["q"] == (1, 2, 9000, 128) and seen["scale"] == 0.1
    assert result.shape == ((1, 2, 9000, 128) if skip_output_reshape else (1, 9000, 256))
    assert result.is_contiguous() and torch.all(result == 1)


@requires_m5
def test_h3_attention_recomputes_protected_ranges_exactly():
    q, k, v = (_tensors(1, 4, 4096, 128, seed=s) for s in (10, 11, 12))
    out, counts, total = i8.h3_attention(q, k, v, video_start=100, protected_ranges=((0, 128), (2048, 2112)))
    exact = torch.nn.functional.scaled_dot_product_attention(q[:, :, :128], k, v)
    assert torch.equal(out[:, :, :128], exact)
    assert counts.shape == (4, 64) and total == 4 * 64 * 64
    assert _rel(out, q, k, v) < 0.035


@requires_m5
@pytest.mark.parametrize("numerics", ["running", "tiled"])
def test_h3_attention_skips_exact_prefix_rows_in_int8(numerics):
    q, k, v = (_tensors(1, 4, 4096, 128, seed=s) for s in (19, 20, 21))
    out, _, _ = i8.h3_attention(q, k, v, video_start=100, numerics=numerics)
    full = i8.attention(q, k, v, numerics=numerics)
    assert torch.equal(out[:, :, :128], torch.nn.functional.scaled_dot_product_attention(q[:, :, :128], k, v))
    assert torch.equal(out[:, :, 128:], full[:, :, 128:])
    everything, _, _ = i8.h3_attention(q, k, v, video_start=4096, numerics=numerics)
    assert torch.equal(everything, torch.nn.functional.scaled_dot_product_attention(q, k, v))


def test_h3_holder_accepts_v2_backend_and_dispatches_it(monkeypatch):
    holder = sparse.H3SparseHolder(backend="dense_int8_v2", topk_ratio=0.15)
    assert holder.topk_ratio == 0.0
    seen = {}
    def fake(q, k, v, **kw):
        seen.update(kw)
        return q + 1, None, 0
    monkeypatch.setattr(i8, "h3_attention", fake)
    monkeypatch.setattr(sparse, "_video_row_start", lambda h, n: 64)
    x = torch.zeros(1, 2, 128, 128)
    out, _, _ = sparse._run_dense_int8(holder, x, x, x, None)
    assert torch.equal(out, x[0] + 1) and seen["video_start"] == 64


def test_backend_node_offers_int8_only_as_opt_in():
    required = backend.AppleSiliconAttentionBackend.INPUT_TYPES()["required"]
    assert backend.BACKEND_INT8 in required["backend"][0]
    assert required["backend"][1]["default"] == backend.BACKEND_CURRENT


def test_int8_backend_falls_back_to_router(monkeypatch):
    monkeypatch.setattr(backend, "_mps_available", lambda: True)
    monkeypatch.setattr(i8, "supported", lambda: True)
    monkeypatch.setattr(router, "workflow_attention", lambda *a, **k: "router")
    function = backend._backend_function(backend.BACKEND_INT8)
    x = torch.zeros(1, 512, 256)
    assert function(x, x, x, 2) == "router"
    assert callable(function.container_function)


def test_int8_backend_requires_m5(monkeypatch):
    monkeypatch.setattr(i8, "supported", lambda: False)
    with pytest.raises(ValueError, match="M5"):
        backend._backend_function(backend.BACKEND_INT8)


@requires_m5
@pytest.mark.parametrize("lq,lk", [(2048, 2048), (3000, 5000), (1, 1025), (2049, 2049)])
def test_tiled_numerics_track_reference(lq, lk):
    q = _tensors(1, 4, lq, 128, seed=13)
    k, v = _tensors(1, 4, lk, 128, seed=14), _tensors(1, 4, lk, 128, seed=15)
    for levels in (255, 127):
        out = i8.attention(q, k, v, numerics="tiled", levels=levels)
        assert torch.isfinite(out).all() and _rel(out, q, k, v) < 0.025


@requires_m5
def test_tiled_rotation_is_optional_and_deterministic():
    q, k, v = (_tensors(1, 3, 4096, 128, seed=s) for s in (16, 17, 18))
    plain = i8.attention(q, k, v, numerics="tiled", rotate=False)
    rotated = i8.attention(q, k, v, numerics="tiled")
    assert not torch.equal(plain, rotated)
    assert torch.equal(rotated, i8.attention(q, k, v, numerics="tiled"))
    with pytest.raises(ValueError):
        i8.attention(q, k, v, rotate=True)


def test_accurate_backend_selects_tiled_numerics(monkeypatch):
    monkeypatch.setattr(backend, "_mps_available", lambda: True)
    monkeypatch.setattr(i8, "supported", lambda: True)
    seen = {}
    monkeypatch.setattr(i8, "comfy_attention", lambda *a, **k: seen.update(k) or "ok")
    assert backend._backend_function(backend.BACKEND_INT8_ACCURATE)(1, 2, 3, 4) == "ok"
    assert seen["numerics"] == "tiled"


def test_h3_v3_backend_dispatches_tiled(monkeypatch):
    holder = sparse.H3SparseHolder(backend="dense_int8_v3")
    seen = {}
    def fake(q, k, v, **kw):
        seen.update(kw)
        return q, None, 0
    monkeypatch.setattr(i8, "h3_attention", fake)
    monkeypatch.setattr(sparse, "_video_row_start", lambda h, n: 0)
    x = torch.zeros(1, 2, 128, 128)
    sparse._run_dense_int8(holder, x, x, x, None)
    assert seen["numerics"] == "tiled"


def test_dense_metal_name_and_former_name_select_the_same_backend(monkeypatch):
    assert backend.BACKEND_INT8_ACCURATE == "dense_metal" and "dense_metal" in backend.BACKENDS
    assert backend.BACKEND_INT8 == "dense_metal_v2" and "dense_metal_v2" in backend.BACKENDS
    assert "int8_dense_accurate" not in backend.BACKENDS and "int8_dense_approximate" not in backend.BACKENDS
    node = backend.AppleSiliconAttentionBackend
    assert node.VALIDATE_INPUTS(backend="int8_dense_accurate") is True
    assert node.VALIDATE_INPUTS(backend="int8_dense_approximate") is True
    assert node.VALIDATE_INPUTS(backend="dense_metal") is True
    assert node.VALIDATE_INPUTS(backend="nope") is not True
    seen = []
    monkeypatch.setattr(backend, "_backend_function", lambda name: seen.append(name) or (lambda *a, **k: None))

    class Model:
        model_options = {"transformer_options": {}}
        def clone(self):
            return Model()
    backend.apply_attention_backend(Model(), "int8_dense_accurate")
    backend.apply_attention_backend(Model(), "int8_dense_approximate")
    assert seen == ["dense_metal", "dense_metal_v2"]
