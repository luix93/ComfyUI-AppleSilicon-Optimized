"""Protect BF16 rounding points, PyTorch reduction order and fallbacks."""
import os
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from _patches import krea2_glue as glue


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


@pytest.fixture(scope="module")
def kernel():
    if not torch.backends.mps.is_available() or os.environ.get("ASFP8_DIT_EXT") != "1":
        pytest.skip("needs MPS and ASFP8_DIT_EXT=1")
    from _patches.krea2_ext.loader import module
    result = module()
    if result is None:
        pytest.skip("native kernel unavailable")
    return result


@pytest.mark.parametrize("width", [128, 2560, 6144])
@pytest.mark.parametrize("seed", [0, 19, 271828])
def test_norm_matches_fp32_reduction_and_bf16_modulation(kernel, width, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        x = torch.randn(1, 37, width, dtype=torch.bfloat16, device="mps")
        weight = torch.randn(width, device="mps") * 0.1
        scale = torch.randn(1, 1, width, dtype=torch.bfloat16, device="mps")
        shift = torch.randn_like(scale)
        norm = F.rms_norm(x.float(), (width,), 1 + weight, 1e-5).to(x.dtype)
        bits_equal(norm, kernel.norm_mod(x, weight, scale, shift, 1e-5, False))
        bits_equal((1 + scale) * norm + shift, kernel.norm_mod(x, weight, scale, shift, 1e-5, True))


def test_all_finite_bf16_sigmoid_and_silu_values(kernel):
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)].to("mps").reshape(1, -1, 1)
    ones = torch.ones_like(values)
    with torch.inference_mode():
        bits_equal(F.silu(values), kernel.point(values, ones, ones, 3))
        bits_equal(torch.sigmoid(values), kernel.point(ones.reshape(1, 1, -1, 1), values, ones, 1))


def test_attention_layout_with_ragged_sequence(kernel):
    with torch.inference_mode():
        out = torch.randn(2, 48, 113, 128, dtype=torch.bfloat16, device="mps")
        gate = torch.randn(2, 113, 6144, dtype=torch.bfloat16, device="mps")
        bits_equal(out.transpose(1, 2).reshape(2, 113, 6144) * gate.sigmoid(), kernel.point(out, gate, gate, 1))


def test_residual_preserves_product_rounding(kernel):
    with torch.inference_mode():
        value = torch.randn(1, 73, 6144, dtype=torch.bfloat16, device="mps")
        gate = torch.randn(1, 1, 6144, dtype=torch.bfloat16, device="mps")
        residual = torch.randn_like(value)
        bits_equal(residual + gate * value, kernel.point(value, gate, residual, 0))


def test_cpu_block_falls_back_with_every_argument(monkeypatch):
    calls = []
    monkeypatch.setattr(glue, "_kernel", object())
    monkeypatch.setattr(glue, "_original_block", lambda *args, **kwargs: calls.append((args, kwargs)) or "original")
    obj = object()
    x = torch.ones(1, 2, 6144, dtype=torch.bfloat16)
    vec, freqs, mask, options = object(), object(), object(), {"test": 1}
    assert glue._block(obj, x, vec, freqs, mask, 42, options) == "original"
    assert calls == [((obj, x, vec, freqs, mask, 42), {"transformer_options": options})]


def test_norm_hooks_or_changed_forward_disable_fusion(monkeypatch):
    class Norm:
        def forward(self, x):
            return x
    norm = Norm()
    norm.scale = torch.ones(6144)
    norm._forward_hooks = {}
    norm._forward_pre_hooks = {}
    monkeypatch.setattr(glue, "_norm_type", Norm)
    monkeypatch.setattr(glue, "_norm_forward", Norm.forward)
    assert glue._norm_ok(norm, 6144)
    norm._forward_hooks[1] = object()
    assert not glue._norm_ok(norm, 6144)
    norm._forward_hooks.clear()
    norm.forward = lambda x: x
    assert not glue._norm_ok(norm, 6144)


def test_disabled_tail_never_calls_kernel(monkeypatch):
    monkeypatch.setattr(glue, "_kernel", SimpleNamespace(point=lambda *args: pytest.fail("unexpected kernel")))
    monkeypatch.setenv("ASFP8_KREA2_GLUE", "0")
    assert glue.attention_gate(None, None) is None


def test_native_shape_errors_fail_before_dispatch(kernel):
    with torch.inference_mode():
        x = torch.ones(1, 7, 128, dtype=torch.bfloat16, device="mps")
        with pytest.raises(RuntimeError, match="Residual shape mismatch"):
            kernel.point(x, x, x, 0)
