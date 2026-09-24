"""INT8 VAE eligibility, fallback ownership, and FP16 epilogue rounding."""
import os
from types import SimpleNamespace

import pytest
import torch

from _patches import minimax_vae_int8 as patch

gpu = pytest.mark.skipif(
    os.environ.get("ASFP8_INT8_EXT") != "1" or not torch.backends.mps.is_available(),
    reason="requires ASFP8_INT8_EXT=1 and MPS")


def same(a, b):
    assert torch.equal(a.cpu().contiguous().view(torch.uint8),
                       b.cpu().contiguous().view(torch.uint8))


def test_disabled_and_unavailable_do_not_access_the_layer(monkeypatch):
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_INT8", "0")
    assert not patch.eligible(object())
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_INT8", "1")
    monkeypatch.setattr(patch.native, "_kernel", None)
    assert not patch.eligible(object())
    monkeypatch.setattr(patch.native, "_kernel", object())
    assert not patch.eligible(object())


def test_cpu_input_is_left_to_original(monkeypatch):
    monkeypatch.setattr(patch, "eligible", lambda layer: pytest.fail("unexpected layer access"))
    x = torch.randn(1, 3, 2048, dtype=torch.float16)
    with torch.inference_mode():
        assert patch.linear(object(), x, None) is None


@pytest.fixture(scope="module")
def kernel():
    from _patches.int8_ext.loader import module
    result = module()
    assert result is not None
    return result


@gpu
@pytest.mark.parametrize("m,k,n,bias", [
    (1, 129, 17, False), (31, 256, 129, True), (128, 257, 64, True),
    (129, 513, 131, False), (261, 2048, 2048, True),
])
def test_fp16_epilogue_preserves_integer_sum_and_both_half_rounds(kernel, m, k, n, bias):
    generator = torch.Generator().manual_seed(9183)
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=generator)
    b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=generator)
    a[:, 0] = -128; b[:, 0] = 127
    exact = (a.long() @ b.long().T).to(torch.float32).to("mps")
    # Nonzero storage offsets must be honored in every Metal buffer binding.
    rows = torch.linspace(0, .007, m + 3, device="mps")[3:]
    cols = torch.linspace(.00001, .003, n + 5, device="mps")[5:]
    offsets = torch.randn(n + 7, device="mps", dtype=torch.float16)[7:] if bias else None
    reference = (exact * (cols[None, :] * rows[:, None])).half()
    if offsets is not None:
        reference = reference + offsets
    actual = kernel.i8_matmul2d_nt_fused_cs_half(a.to("mps"), b.to("mps"), rows, cols, offsets)
    same(reference, actual)


@gpu
def test_invalid_half_epilogue_arguments_fail_before_dispatch(kernel):
    a = torch.zeros(3, 256, device="mps", dtype=torch.int8)
    b = torch.zeros(8, 256, device="mps", dtype=torch.int8)
    r = torch.ones(3, device="mps"); c = torch.ones(8, device="mps")
    with pytest.raises(RuntimeError): kernel.i8_matmul2d_nt_fused_cs_half(a, b, r.cpu(), c)
    with pytest.raises(RuntimeError): kernel.i8_matmul2d_nt_fused_cs_half(a, b, r, c[:1])
    with pytest.raises(RuntimeError): kernel.i8_matmul2d_nt_fused_cs_half(a[:, ::2], b, r, c)
    with pytest.raises(RuntimeError): kernel.i8_matmul2d_nt_fused_cs_half(a[:0], b, r[:0], c)


@gpu
def test_layer_guards_and_failure_latch_preserve_inputs(monkeypatch, kernel):
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
    monkeypatch.setattr(patch.native, "_kernel", kernel)
    monkeypatch.setattr(patch, "_dead", False)
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_INT8", "1")
    qdata = torch.zeros(2048, 2048, device="mps", dtype=torch.int8)
    params = TensorWiseINT8Layout.Params(
        scale=torch.ones(2048, device="mps"), orig_dtype=torch.float16,
        orig_shape=tuple(qdata.shape), convrot=True, convrot_groupsize=256)
    layer = type("NativeLinear", (), {"_asfp8_int8_patched": True})()
    layer.weight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    layer.bias = None
    assert patch.eligible(layer)
    subclass = type("CustomLinear", (type(layer),), {"forward": lambda self, x: x})()
    subclass.weight, subclass.bias = layer.weight, None
    assert not patch.eligible(subclass)
    for name, value in (("_full_precision_mm", True), ("comfy_force_cast_weights", True),
                        ("weight_function", [object()]), ("bias_function", [object()]),
                        ("_forward_hooks", {1: object()}), ("_v", object()),
                        (patch.native._RUNTIME_LORA_ATTR, object()), ("forward", lambda x: x)):
        setattr(layer, name, value)
        assert not patch.eligible(layer), name
        delattr(layer, name)
    x = torch.randn(1, 7, 2048, device="mps", dtype=torch.float16)
    before = x.clone(); calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError("intentional compile failure")
    monkeypatch.setattr(patch.native, "_quant_cached", fail)
    with torch.inference_mode():
        assert patch.linear(layer, x, None) is None
        assert patch._dead and not patch.eligible(layer)
        assert patch.linear(layer, x, None) is None
    assert calls == [1]
    same(x, before)
