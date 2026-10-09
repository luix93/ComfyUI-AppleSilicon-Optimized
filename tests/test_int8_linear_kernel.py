"""Tests for patch #17: INT8 W8A8 via the bit-exact Metal kernel (int8_linear_kernel_mps).

Unit tests (always run): install() is opt-in/guarded; the wrapper falls back to the
original int8_linear when the kernel is unavailable or off-MPS.

Integration test (opt-in, ASFP8_INT8_EXT=1 on an MPS box): the kernel-backed
int8_linear is bit-identical to comfy_kitchen's original across convrot/bias/3D/M=1.
"""
import os
import sys
from types import ModuleType

import pytest
import torch

from _patches import int8_linear_kernel_mps as patch

_run_integration = os.environ.get("ASFP8_INT8_EXT") == "1"
requires_int8_ext = pytest.mark.skipif(
    not (_run_integration and torch.backends.mps.is_available()),
    reason="set ASFP8_INT8_EXT=1 on an MPS device to build + test the int8 kernel",
)


def test_install_noop_without_flag(monkeypatch):
    """install() must be a no-op unless ASFP8_INT8_EXT=1 (opt-in build)."""
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    # Reset module state so a prior install() in-session doesn't mask this.
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_wrapper_falls_back_off_mps(monkeypatch):
    """With no kernel (or a CPU tensor) the wrapper delegates to the original."""
    sentinel = object()
    called = {}

    def fake_orig(x, w, ws, bias, out_dtype, convrot, gs):
        called["hit"] = True
        return sentinel

    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_orig_int8_linear", fake_orig, raising=False)

    x = torch.zeros(4, 8)  # CPU tensor
    w = torch.zeros(8, 8, dtype=torch.int8)
    ws = torch.ones(1)
    out = patch._int8_linear_kernel(x, w, ws)
    assert out is sentinel and called.get("hit") is True


@pytest.fixture
def linear_input_act_wrapper(monkeypatch):
    comfy = ModuleType("comfy")
    ops = ModuleType("comfy.ops")
    comfy.ops = ops
    def factory(quant_config=None, compute_dtype=torch.bfloat16,
                full_precision_mm=False, disabled=()):
        return type('Operations', (), {
            '_quant_config': quant_config, '_compute_dtype': compute_dtype,
            '_full_precision_mm': full_precision_mm, '_disabled': disabled})
    ops.mixed_precision_ops = factory
    calls = []
    fallback = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return fallback

    ops.linear_input_act = original
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    monkeypatch.setattr(patch, "_installed", False)
    monkeypatch.setattr(patch, "_kernel", None)
    monkeypatch.setattr(patch, "_orig_int8_linear", None)
    monkeypatch.setattr(patch.sys, "platform", "darwin")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_load_kernel", lambda: object())
    monkeypatch.setenv("ASFP8_INT8_EXT", "1")
    monkeypatch.delenv("ASFP8_INT8_SWIGLU_VERIFY", raising=False)
    patch.install()
    assert patch._installed
    return ops, calls, fallback


@pytest.mark.parametrize("activation, positional", [
    ("rms_norm", True), ("rms_norm", False),
    ("swiglu", False), (None, True),
])
def test_linear_input_act_preserves_extended_contract(
        monkeypatch, linear_input_act_wrapper, activation, positional):
    ops, calls, fallback = linear_input_act_wrapper
    linear, x, weight, residual, scale = (object() for _ in range(5))
    native_calls = []
    monkeypatch.setattr(
        patch, "_try_int8_swiglu_forward",
        lambda *args: native_calls.append(args))
    args = (linear, x, activation)
    kwargs = {}
    if positional:
        args += ((weight, 1e-5) if activation == "rms_norm"
                 else (weight, 1e-5, residual, scale))
    else:
        kwargs = {"residual": residual, "residual_scale": scale}
        if activation == "rms_norm":
            kwargs.update(act_weight=weight, act_eps=1e-5)
    assert ops.linear_input_act(*args, **kwargs) is fallback
    assert calls == [(args, kwargs)]
    assert native_calls == []


def test_linear_input_act_keeps_native_three_argument_swiglu(
        monkeypatch, linear_input_act_wrapper):
    ops, calls, _ = linear_input_act_wrapper
    result = object()
    monkeypatch.setenv("ASFP8_INT8_SWIGLU", "1")
    monkeypatch.setattr(patch, "_try_int8_swiglu_forward", lambda *args: result)
    assert ops.linear_input_act(object(), object(), "swiglu") is result
    assert calls == []


@pytest.mark.parametrize('positional', [False, True])
@pytest.mark.parametrize('full_precision', [False, True])
def test_factory_enables_only_native_int8_and_preserves_precision_requests(
        linear_input_act_wrapper, positional, full_precision):
    ops, _, _ = linear_input_act_wrapper
    disabled = {'int8_tensorwise', 'nvfp4', 'convrot_w4a4'}
    quant_config = {'layer': {'full_precision_matrix_mult': True}}
    if positional:
        result = ops.mixed_precision_ops(
            quant_config, torch.float16, full_precision, disabled)
    else:
        result = ops.mixed_precision_ops(
            quant_config=quant_config, compute_dtype=torch.float16,
            full_precision_mm=full_precision, disabled=disabled)
    assert set(result._disabled) == {'nvfp4', 'convrot_w4a4'}
    assert disabled == {'int8_tensorwise', 'nvfp4', 'convrot_w4a4'}
    assert result._full_precision_mm is full_precision
    assert result._compute_dtype is torch.float16
    assert result._quant_config is quant_config


def test_factory_keeps_default_arguments(linear_input_act_wrapper):
    ops, _, _ = linear_input_act_wrapper
    result = ops.mixed_precision_ops()
    assert result._disabled == ()
    assert result._full_precision_mm is False


@requires_int8_ext
def test_kernel_matches_original_bit_exact():
    """Kernel-backed int8_linear == comfy_kitchen original (bit-identical bf16)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # original _int_mm -> CPU
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import (
        int8_linear as orig_int8_linear,
    )

    mod = loader.module()
    assert mod is not None, "int8 kernel failed to build"
    mod.warmup()
    assert hasattr(mod, "i8_matmul2d_nt_fused"), "fused entry point missing"

    # Install so the wrapper picks up the freshly built kernel + original.
    patch._kernel = mod
    patch._orig_int8_linear = orig_int8_linear

    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    dev = "mps"
    g = torch.Generator().manual_seed(7)

    def unfused(x, w, ws, b, convrot):
        """Chunked epilogue (int32 store + Python rescale) for cross-checking."""
        if convrot:
            h = _build_hadamard(256, device=x.device, dtype=x.dtype)
            x = _rotate_activation(x, h, 256)
        shp = x.shape
        x8, xs = quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
        C = mod.i8_matmul2d_nt(x8.contiguous(), w.contiguous()).float()
        out = (C * (ws.view(-1) * xs)).to(torch.bfloat16)
        if b is not None:
            out = out + b.to(out.dtype)
        return out.reshape(*shp[:-1], w.shape[0])

    def run(M, K, N, convrot, bias, three_d):
        shape = (2, M, K) if three_d else (M, K)
        x = (torch.randn(shape, generator=g, dtype=torch.bfloat16) * 0.5).to(dev)
        w = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
        ws = (torch.rand(1, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)
        b = torch.randn(N, generator=g, dtype=torch.bfloat16).to(dev) if bias else None
        ref = orig_int8_linear(x, w, ws, b, torch.bfloat16, convrot, 256)
        # _int8_linear_kernel auto-selects the fused bf16 epilogue.
        out = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256)
        assert torch.equal(ref, out), f"fused mismatch M={M} K={K} N={N} convrot={convrot}"
        # The fused epilogue must also be bit-identical to the chunked one.
        unf = unfused(x, w, ws, b, convrot)
        assert torch.equal(out, unf), f"fused!=unfused M={M} K={K} N={N} convrot={convrot}"

    run(256, 2560, 1024, convrot=False, bias=False, three_d=False)
    run(256, 2560, 1024, convrot=False, bias=True, three_d=False)
    run(512, 6144, 6144, convrot=True, bias=False, three_d=False)
    run(512, 6144, 6144, convrot=True, bias=True, three_d=False)
    run(188, 4096, 2560, convrot=True, bias=True, three_d=True)
    run(1, 6144, 6144, convrot=True, bias=False, three_d=False)


@requires_int8_ext
def test_quant_rowwise_matches_eager_bit_exact():
    """Metal i8_quant_rowwise == comfy_kitchen quantize_int8_rowwise (finite inputs).

    Known boundary: fp16 inputs containing inf produce NaN garbage in BOTH
    implementations (inf/inf), just different garbage — finite inputs only.
    """
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_quant_rowwise")

    g = torch.Generator().manual_seed(3)
    for dtype in (torch.bfloat16, torch.float16):
        for M, K in ((33, 6144), (7, 100), (1, 256), (128, 300), (64, 16384)):
            x = (torch.randn((M, K), generator=g) * 3).to("mps").to(dtype)
            q_ref, s_ref = quantize_int8_rowwise(x)
            q_k, s_k = mod.i8_quant_rowwise(x)
            assert torch.equal(q_ref, q_k), f"q mismatch {dtype} {M}x{K}"
            assert torch.equal(s_ref.float(), s_k), f"scale mismatch {dtype} {M}x{K}"
        for edge in (torch.zeros(4, 512), torch.full((4, 512), 1e-30)):
            x = edge.to("mps").to(dtype)
            q_ref, s_ref = quantize_int8_rowwise(x)
            q_k, s_k = mod.i8_quant_rowwise(x)
            assert torch.equal(q_ref, q_k) and torch.equal(s_ref.float(), s_k)


@requires_int8_ext
def test_activation_cache_reuse_and_invalidation():
    """Same live tensor -> cached quant reused (bit-identical); new tensor -> miss."""
    from _patches.int8_ext import loader

    patch._kernel = loader.module()
    assert patch._kernel is not None

    x = torch.randn(64, 512, device="mps", dtype=torch.bfloat16)
    patch._act_cache = None
    q1, s1 = patch._quant_cached(x, True, 256)
    q2, s2 = patch._quant_cached(x, True, 256)
    assert q1 is q2 and s1 is s2  # hit: identical objects, zero recompute

    q3, s3 = patch._quant_cached(x.clone(), True, 256)  # different tensor -> miss
    assert q3 is not q1
    assert torch.equal(q1, q3) and torch.equal(s1, s3)

    q4, _ = patch._quant_cached(x, False, 256)  # different settings -> miss
    assert q4 is not q1


@requires_int8_ext
def test_channelwise_fused_matches_chunked_bit_exact():
    """i8_matmul2d_nt_fused_cs == the chunked epilogue for [N,1] weight scales
    (the layout Z-Image / Krea2 convrot checkpoints actually ship)."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused_cs")
    patch._kernel = mod

    class _NoCS:  # hide the cs entry to force the chunked reference path
        def __getattr__(self, name):
            if name == "i8_matmul2d_nt_fused_cs":
                raise AttributeError(name)
            return getattr(mod, name)

    g = torch.Generator().manual_seed(11)
    for M, K, N, convrot, bias in ((256, 2560, 1024, False, True),
                                   (512, 3840, 1152, True, False),
                                   (65, 768, 129, False, True),
                                   (96, 704, 385, False, False),
                                   (128, 512, 256, True, True),
                                   (33, 512, 256, True, True),
                                   (1, 2560, 512, False, False)):
        x = (torch.randn((M, K), generator=g, dtype=torch.bfloat16) * 0.5).to("mps")
        w = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to("mps")
        ws = (torch.rand((N, 1), generator=g) * 0.01 + 0.001).to("mps")
        b = torch.randn(N, generator=g, dtype=torch.bfloat16).to("mps") if bias else None
        patch._kernel = _NoCS(); patch._act_cache = None
        ref = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256)
        patch._kernel = mod; patch._act_cache = None
        out = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256)
        assert torch.equal(ref, out), f"cs mismatch M={M} N={N} convrot={convrot} bias={bias}"


@requires_int8_ext
def test_channelwise_split_tail_is_bit_exact():
    """The optional small suffix dispatch must not change any output byte."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused_cs")
    g = torch.Generator().manual_seed(19)
    M, K, N = 130, 512, 256  # 128-row prefix + two-row suffix (H3 pattern)
    a = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8).to("mps")
    b = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to("mps")
    rs = (torch.rand(M, generator=g) * 0.01).to("mps")
    cs = (torch.rand(N, generator=g) * 0.01).to("mps")
    bias = torch.randn(N, generator=g, dtype=torch.bfloat16).to("mps")
    ref = mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, bias, False)
    out = mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, bias, True)
    torch.mps.synchronize()
    assert torch.equal(ref, out)


@requires_int8_ext
def test_channelwise_long_schedule_is_bit_exact():
    """The long-M compiler schedule must preserve every output byte."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused_cs")
    g = torch.Generator().manual_seed(23)
    M, K, N = 8209, 96, 129
    a = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8).to("mps")
    b = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to("mps")
    rs = (torch.rand(M, generator=g) * 0.01).to("mps")
    cs = (torch.rand(N, generator=g) * 0.01).to("mps")
    ref = mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None, False, False)
    out = mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None, False, True)
    torch.mps.synchronize()
    assert torch.equal(ref, out)


@requires_int8_ext
@pytest.mark.parametrize("m", [65, 96, 128, 130])
def test_channelwise_qkv_scatter_is_bit_exact(m):
    """The H3 epilogue changes only storage layout, not GEMM arithmetic."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused_cs_qkv")
    g = torch.Generator().manual_seed(31)
    kin, heads, dim = 512, 4, 32
    n = 3 * heads * dim
    a = torch.randint(-128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
    b = torch.randint(-128, 128, (n, kin), generator=g, dtype=torch.int8).to("mps")
    rs = (torch.rand(m, generator=g) * 0.01).to("mps")
    cs = (torch.rand(n, generator=g) * 0.01).to("mps")
    bias = torch.randn(n, generator=g, dtype=torch.bfloat16).to("mps")

    dense = mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, bias, False, False)
    q, k, v = mod.i8_matmul2d_nt_fused_cs_qkv(
        a, b, rs, cs, bias, heads, dim, False)
    expected = tuple(
        part.reshape(m, heads, dim).permute(1, 0, 2).unsqueeze(0).contiguous()
        for part in dense.split(heads * dim, dim=-1)
    )
    torch.mps.synchronize()
    assert all(torch.equal(actual, reference)
               for actual, reference in zip((q, k, v), expected))


@requires_int8_ext
@pytest.mark.parametrize("m", [65, 96, 128, 259])
@pytest.mark.parametrize("rotary_half", [1, 16, 48, 64])
def test_qkv_rms_rope_epilogue_matches_two_kernel_chain_bit_exact(m, rotary_half):
    """The fused H3 epilogue preserves projection, norm, and RoPE bytes."""
    from _patches.dit_ext import loader as dit_loader
    from _patches.int8_ext import loader as int8_loader

    int8 = int8_loader.module()
    dit = dit_loader.module()
    if (int8 is None or dit is None
            or not hasattr(int8, "i8_matmul2d_nt_fused_cs_qkv_rms_rope")
            or not hasattr(dit, "rms_rope_attention_partial")):
        pytest.skip("fused QKV and dit reference kernels unavailable")

    g = torch.Generator().manual_seed(37)
    kin, heads, dim = 512, 4, 128
    n = 3 * heads * dim
    a = torch.randint(
        -128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
    b = torch.randint(
        -128, 128, (n, kin), generator=g, dtype=torch.int8).to("mps")
    row_scale = (torch.rand(m, generator=g) * 0.01 + 1.0e-5).to("mps")
    col_scale = (torch.rand(n, generator=g) * 0.01 + 1.0e-5).to("mps")
    bias = torch.randn(n, generator=g, dtype=torch.bfloat16).to("mps")
    angles = torch.randn(m, rotary_half, generator=g).to("mps")
    cosine, sine = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, m, 1, rotary_half, 2, 2).to(torch.bfloat16).contiguous()
    q_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()
    k_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()

    expected = int8.i8_matmul2d_nt_fused_cs_qkv(
        a, b, row_scale, col_scale, bias, heads, dim, False)
    dit.rms_rope_attention_partial(
        expected[0], expected[1], freqs, q_weight, k_weight, 1.0e-6)
    actual = int8.i8_matmul2d_nt_fused_cs_qkv_rms_rope(
        a, b, row_scale, col_scale, bias, freqs, q_weight, k_weight,
        heads, dim, 1.0e-6, False)
    torch.mps.synchronize()

    for label, observed, reference in zip("qkv", actual, expected):
        assert torch.equal(observed, reference), (
            label,
            (observed.float() - reference.float()).abs().max().item(),
        )


@requires_int8_ext
@pytest.mark.parametrize("strength", [0.375, -0.6, 1.0])
@pytest.mark.parametrize("m", [65, 96, 128, 259])
def test_qkv_lora_epilogue_matches_runtime_adapter_chain_bit_exact(strength, m):
    """LoRA is added with PyTorch's bf16 semantics before Q/K normalization."""
    from _patches.dit_ext import loader as dit_loader
    from _patches.int8_ext import loader as int8_loader

    int8 = int8_loader.module()
    dit = dit_loader.module()
    if (int8 is None or dit is None
            or not hasattr(int8, "i8_matmul2d_nt_fused_cs_qkv_lora_part")
            or not hasattr(dit, "rms_rope_attention_partial")):
        pytest.skip("LoRA QKV epilogue and dit reference kernels unavailable")

    g = torch.Generator().manual_seed(71)
    kin, heads, dim, rotary_half = 512, 4, 128, 48
    part_width = heads * dim
    width = 3 * part_width
    a = torch.randint(
        -128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
    b = torch.randint(
        -128, 128, (width, kin), generator=g, dtype=torch.int8).to("mps")
    row_scale = (torch.rand(m, generator=g) * 0.01 + 1.0e-5).to("mps")
    col_scale = (torch.rand(width, generator=g) * 0.01 + 1.0e-5).to("mps")
    bias = torch.randn(width, generator=g, dtype=torch.bfloat16).to("mps")
    delta = torch.randn(
        m, width, generator=g, dtype=torch.bfloat16).to("mps")
    angles = torch.randn(m, rotary_half, generator=g).to("mps")
    cosine, sine = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, m, 1, rotary_half, 2, 2).to(torch.bfloat16).contiguous()
    q_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()
    k_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()

    expected = int8.i8_matmul2d_nt_fused_cs_qkv(
        a, b, row_scale, col_scale, bias, heads, dim, False)
    for base, part in zip(expected, delta.split(part_width, dim=-1)):
        shaped = part.reshape(m, heads, dim).permute(1, 0, 2).unsqueeze(0)
        base.add_(shaped, alpha=strength)
    dit.rms_rope_attention_partial(
        expected[0], expected[1], freqs, q_weight, k_weight, 1.0e-6)

    actual = tuple(
        int8.i8_matmul2d_nt_fused_cs_qkv_lora_part(
            a, b[kind * part_width:(kind + 1) * part_width].contiguous(),
            row_scale,
            col_scale[kind * part_width:(kind + 1) * part_width].contiguous(),
            bias[kind * part_width:(kind + 1) * part_width].contiguous(),
            delta[:, kind * part_width:(kind + 1) * part_width].contiguous(),
            freqs, q_weight if kind != 1 else k_weight,
            heads, dim, kind, strength, 1.0e-6, False)
        for kind in range(3)
    )
    torch.mps.synchronize()

    for label, observed, reference in zip("qkv", actual, expected):
        assert torch.equal(observed, reference), (
            label,
            strength,
            torch.count_nonzero(observed != reference).item(),
            (observed.float() - reference.float()).abs().max().item(),
        )


def test_qkv_norm_rope_epilogue_rejects_unsupported_runtime_lora(monkeypatch):
    """Unknown adapter protocols fall back instead of being applied out of order."""
    class Kernel:
        @staticmethod
        def i8_matmul2d_nt_fused_cs_qkv_rms_rope(*args):
            raise AssertionError("native fused epilogue must not be called")

    layer = type("Layer", (), {})()
    setattr(layer, patch._RUNTIME_LORA_ATTR, type("LoRA", (), {"fast": True})())
    monkeypatch.setattr(patch, "_kernel", Kernel())
    monkeypatch.setenv("ASFP8_INT8_QKV_NORM_ROPE", "1")

    assert patch.int8_qkv_rms_rope_prequant(
        layer, object(), object(), object(), object(), 56, 128, 1.0e-6
    ) is None


@requires_int8_ext
def test_qkv_norm_rope_python_route_matches_layout_route(monkeypatch):
    """The guarded QuantizedTensor route selects the native fused epilogue."""
    from comfy.quant_ops import TensorWiseINT8Layout
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.dit_ext import loader as dit_loader
    from _patches.int8_ext import loader as int8_loader

    int8 = int8_loader.module()
    dit = dit_loader.module()
    if int8 is None or dit is None:
        pytest.skip("native reference kernels unavailable")
    monkeypatch.setattr(patch, "_kernel", int8)
    monkeypatch.setenv("ASFP8_INT8_QKV_SCATTER", "1")
    monkeypatch.setenv("ASFP8_INT8_QKV_NORM_ROPE", "1")

    g = torch.Generator().manual_seed(43)
    m, hidden, heads, dim, rotary_half = 259, 512, 4, 128, 48
    width = 3 * heads * dim
    x = (torch.randn(m, hidden, generator=g) * 0.25).to(
        device="mps", dtype=torch.bfloat16).contiguous()
    qdata = torch.randint(
        -128, 128, (width, hidden), generator=g,
        dtype=torch.int8).to("mps")
    scale = (torch.rand(width, generator=g) * 0.01 + 1.0e-4).to("mps")
    params = TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape), convrot=False,
        convrot_groupsize=256)
    weight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    layer = type("Linear", (), {
        "weight": weight,
        "bias": None,
        "weight_function": [],
        "bias_function": [],
        "_full_precision_mm": False,
        "comfy_force_cast_weights": False,
    })()
    angles = torch.randn(m, rotary_half, generator=g).to("mps")
    cosine, sine = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, m, 1, rotary_half, 2, 2).to(torch.bfloat16).contiguous()
    q_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()
    k_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()

    patch._act_cache = None
    expected = patch.int8_qkv_prequant(layer, x, heads, dim)
    assert expected is not None
    dit.rms_rope_attention_partial(
        expected[0], expected[1], freqs, q_weight, k_weight, 1.0e-6)
    patch._act_cache = None
    actual = patch.int8_qkv_rms_rope_prequant(
        layer, x, freqs, q_weight, k_weight, heads, dim, 1.0e-6)
    torch.mps.synchronize()

    assert actual is not None
    for label, observed, reference in zip("qkv", actual, expected):
        assert torch.equal(observed, reference), (
            label,
            torch.count_nonzero(observed != reference).item(),
            (observed.float() - reference.float()).abs().max().item(),
        )
    monkeypatch.setenv("ASFP8_INT8_QKV_NORM_ROPE", "0")
    assert patch.int8_qkv_rms_rope_prequant(
        layer, x, freqs, q_weight, k_weight, heads, dim, 1.0e-6) is None


@requires_int8_ext
@pytest.mark.parametrize("stacked", [False, True])
def test_qkv_norm_rope_python_route_applies_standard_runtime_lora_exactly(
    monkeypatch, stacked,
):
    from comfy.quant_ops import TensorWiseINT8Layout
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.dit_ext import loader as dit_loader
    from _patches.int8_ext import loader as int8_loader
    from _patches.minimax_h3_turbo import H3RuntimeLoRA

    int8 = int8_loader.module()
    dit = dit_loader.module()
    if int8 is None or dit is None:
        pytest.skip("native reference kernels unavailable")
    monkeypatch.setattr(patch, "_kernel", int8)
    monkeypatch.setenv("ASFP8_INT8_QKV_SCATTER", "1")
    monkeypatch.setenv("ASFP8_INT8_QKV_NORM_ROPE", "1")

    g = torch.Generator().manual_seed(73)
    m, hidden, heads, dim, rank, rotary_half = 259, 512, 4, 128, 8, 48
    width = 3 * heads * dim
    x = torch.randn(m, hidden, generator=g).to(
        device="mps", dtype=torch.bfloat16).contiguous()
    qdata = torch.randint(
        -128, 128, (width, hidden), generator=g,
        dtype=torch.int8).to("mps")
    scale = (torch.rand(width, generator=g) * 0.01 + 1.0e-4).to("mps")
    params = TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape), convrot=False,
        convrot_groupsize=256)
    weight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    layer = type("Linear", (), {
        "weight": weight,
        "bias": None,
        "weight_function": [],
        "bias_function": [],
        "_full_precision_mm": False,
        "comfy_force_cast_weights": False,
    })()
    adapter = H3RuntimeLoRA((
        torch.randn(width, rank, generator=g),
        torch.randn(rank, hidden, generator=g),
        5.0, None, None, None,
    ), fast=True)
    adapter.multiplier = -0.6
    if stacked:
        from _patches.minimax_h3_turbo import H3RuntimeLoRAStack
        second = H3RuntimeLoRA((
            torch.randn(width, 3, generator=g), torch.randn(3, hidden, generator=g),
            None, None, None, None), fast=True)
        second.multiplier = 0.4
        adapter = H3RuntimeLoRAStack([adapter, second])
    setattr(layer, patch._RUNTIME_LORA_ATTR, adapter)
    angles = torch.randn(m, rotary_half, generator=g).to("mps")
    cosine, sine = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, m, 1, rotary_half, 2, 2).to(torch.bfloat16).contiguous()
    q_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()
    k_weight = torch.randn(
        dim, generator=g, dtype=torch.bfloat16).float().to("mps").contiguous()

    patch._act_cache = None
    if stacked:
        delattr(layer, patch._RUNTIME_LORA_ATTR)
    expected = patch.int8_qkv_prequant(layer, x, heads, dim)
    assert expected is not None
    if stacked:
        for entry in adapter.adapters:
            entry.add_qkv_to(*expected, x, heads, dim)
        setattr(layer, patch._RUNTIME_LORA_ATTR, adapter)
    dit.rms_rope_attention_partial(
        expected[0], expected[1], freqs, q_weight, k_weight, 1.0e-6)
    patch._act_cache = None
    actual = patch.int8_qkv_rms_rope_prequant(
        layer, x, freqs, q_weight, k_weight, heads, dim, 1.0e-6)
    if stacked:
        assert actual is None  # Never send only the first delta to the fused epilogue.
        actual = patch.int8_qkv_prequant(layer, x, heads, dim)
        assert actual is not None
        dit.rms_rope_attention_partial(actual[0], actual[1], freqs, q_weight, k_weight, 1.0e-6)
    torch.mps.synchronize()

    assert actual is not None
    for label, observed, reference in zip("qkv", actual, expected):
        assert torch.equal(observed, reference), (
            label,
            torch.count_nonzero(observed != reference).item(),
            (observed.float() - reference.float()).abs().max().item(),
        )


@requires_int8_ext
def test_mlp_bridge_prequant_matches_materialized_pipeline_bit_exact():
    """The bridge removes FC1's wide output without changing FC2 input bytes."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_mlp_bridge_prequant")
    g = torch.Generator().manual_seed(41)
    m, kin, ffn = 130, 512, 512
    activation = torch.randint(
        -128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
    weight = torch.randint(
        -128, 128, (2 * ffn, kin), generator=g, dtype=torch.int8).to("mps")
    row_scale = (torch.rand(m, generator=g) * 0.01 + 1.0e-4).to("mps")
    col_scale = (
        torch.rand(2 * ffn, generator=g) * 0.01 + 1.0e-4).to("mps")
    bias = torch.randn(
        2 * ffn, generator=g, dtype=torch.bfloat16).to("mps")

    wide = mod.i8_matmul2d_nt_fused_cs(
        activation, weight, row_scale, col_scale, bias, False, False)
    rotated = mod.i8_swiglu_rot(wide)
    expected_q, expected_scale = mod.i8_quant_rowwise(rotated)
    actual_q, actual_scale = mod.i8_mlp_bridge_prequant(
        activation, weight, row_scale, col_scale, bias, False)
    torch.mps.synchronize()

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scale, expected_scale)


@requires_int8_ext
@pytest.mark.parametrize('m,kin,ffn', [(8209, 96, 256), (8193, 5376, 14336)])
def test_mlp_bridge_long_schedule_is_bit_exact(m, kin, ffn):
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_mlp_bridge_prequant")
    g = torch.Generator().manual_seed(43)
    activation = torch.randint(
        -128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
    weight = torch.randint(
        -128, 128, (2 * ffn, kin), generator=g, dtype=torch.int8).to("mps")
    row_scale = (torch.rand(m, generator=g) * 0.01 + 1.0e-4).to("mps")
    col_scale = (
        torch.rand(2 * ffn, generator=g) * 0.01 + 1.0e-4).to("mps")

    reference = mod.i8_mlp_bridge_prequant(
        activation, weight, row_scale, col_scale, None, False)
    actual = mod.i8_mlp_bridge_prequant(
        activation, weight, row_scale, col_scale, None, True)
    torch.mps.synchronize()
    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])


@requires_int8_ext
def test_mlp_bridge_tiled_scratch_reuse_is_stream_safe():
    """Queued bridge calls may reuse their private scratch without corruption."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_mlp_bridge_prequant")
    g = torch.Generator().manual_seed(45)
    m, kin, ffn = 385, 96, 256
    weight = torch.randint(
        -128, 128, (2 * ffn, kin), generator=g, dtype=torch.int8).to("mps")
    col_scale = (
        torch.rand(2 * ffn, generator=g) * 0.01 + 1.0e-4).to("mps")

    def inputs():
        activation = torch.randint(
            -128, 128, (m, kin), generator=g, dtype=torch.int8).to("mps")
        row_scale = (
            torch.rand(m, generator=g) * 0.01 + 1.0e-4).to("mps")
        return activation, row_scale

    first_a, first_s = inputs()
    second_a, second_s = inputs()
    first_wide = mod.i8_matmul2d_nt_fused_cs(
        first_a, weight, first_s, col_scale, None, False, False)
    first_rot = mod.i8_swiglu_rot(first_wide)
    expected_first = mod.i8_quant_rowwise(first_rot)
    second_wide = mod.i8_matmul2d_nt_fused_cs(
        second_a, weight, second_s, col_scale, None, False, False)
    second_rot = mod.i8_swiglu_rot(second_wide)
    expected_second = mod.i8_quant_rowwise(second_rot)

    actual_first = mod.i8_mlp_bridge_prequant(
        first_a, weight, first_s, col_scale, None, False, 128)
    actual_second = mod.i8_mlp_bridge_prequant(
        second_a, weight, second_s, col_scale, None, False, 128)
    torch.mps.synchronize()

    for actual, expected in zip(actual_first, expected_first):
        assert torch.equal(actual, expected)
    for actual, expected in zip(actual_second, expected_second):
        assert torch.equal(actual, expected)


@requires_int8_ext
def test_h3_mlp_bridge_matches_existing_python_pipeline_bit_exact(monkeypatch):
    from comfy.quant_ops import TensorWiseINT8Layout
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None
    monkeypatch.setattr(patch, "_kernel", mod)
    monkeypatch.setenv("ASFP8_INT8_MLP_BRIDGE", "1")
    monkeypatch.setenv("ASFP8_INT8_SWIGLU", "1")
    monkeypatch.setenv("ASFP8_INT8_SWIGLU_ROT", "1")

    g = torch.Generator().manual_seed(47)
    m, hidden, ffn, output = 130, 512, 512, 256
    x = (torch.randn((m, hidden), generator=g) * 0.25).to(
        device="mps", dtype=torch.bfloat16).contiguous()
    w1 = torch.randint(
        -128, 128, (2 * ffn, hidden), generator=g,
        dtype=torch.int8).to("mps")
    w2 = torch.randint(
        -128, 128, (output, ffn), generator=g,
        dtype=torch.int8).to("mps")
    s1 = (torch.rand(2 * ffn, generator=g) * 0.01 + 1.0e-4).to("mps")
    s2 = (torch.rand(output, generator=g) * 0.01 + 1.0e-4).to("mps")

    def layer(weight, scale):
        params = TensorWiseINT8Layout.Params(
            scale=scale, orig_dtype=torch.bfloat16,
            orig_shape=tuple(weight.shape), convrot=True,
            convrot_groupsize=256)
        quantized = QuantizedTensor(weight, "TensorWiseINT8Layout", params)
        return type("Linear", (), {
            "weight": quantized,
            "bias": None,
            "weight_function": [],
            "bias_function": [],
            "_full_precision_mm": False,
            "comfy_force_cast_weights": False,
        })()

    fc1, fc2 = layer(w1, s1), layer(w2, s2)
    patch._act_cache = None
    wide = patch._int8_linear_kernel(
        x, w1, s1, None, torch.bfloat16, True, 256)
    reference = patch._try_int8_swiglu_forward(fc2, wide)
    assert reference is not None

    patch._act_cache = None
    actual = patch.int8_mlp_bridge(fc1, fc2, x)
    torch.mps.synchronize()
    assert actual is not None
    assert torch.equal(actual, reference)

    monkeypatch.setenv("ASFP8_INT8_MLP_BRIDGE", "0")
    assert patch.int8_mlp_bridge(fc1, fc2, x) is None
    monkeypatch.setenv("ASFP8_INT8_MLP_BRIDGE", "1")
    setattr(fc1, patch._RUNTIME_LORA_ATTR, object())
    assert patch.int8_mlp_bridge(fc1, fc2, x) is None


@requires_int8_ext
@pytest.mark.parametrize(
    ("adapt_fc1", "adapt_fc2"),
    [(True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("adapter_count", [1, 2, 3])
def test_h3_mlp_bridge_applies_partial_or_full_runtime_lora_exactly(
    monkeypatch, adapt_fc1, adapt_fc2, adapter_count,
):
    from comfy.quant_ops import TensorWiseINT8Layout
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.int8_ext import loader
    from _patches.minimax_h3_turbo import H3RuntimeLoRA

    mod = loader.module()
    assert mod is not None
    monkeypatch.setattr(patch, "_kernel", mod)
    monkeypatch.setenv("ASFP8_INT8_MLP_BRIDGE", "1")
    monkeypatch.setenv("ASFP8_INT8_SWIGLU", "1")
    monkeypatch.setenv("ASFP8_INT8_SWIGLU_ROT", "1")
    monkeypatch.setenv("ASFP8_INT8_MLP_LORA_TILE_ROWS", "128")

    g = torch.Generator().manual_seed(79)
    rows, hidden, ffn = 259, 512, 512
    x = torch.randn(rows, hidden, generator=g).to(
        device="mps", dtype=torch.bfloat16).contiguous()
    w1 = torch.randint(
        -128, 128, (2 * ffn, hidden), generator=g,
        dtype=torch.int8).to("mps")
    w2 = torch.randint(
        -128, 128, (hidden, ffn), generator=g,
        dtype=torch.int8).to("mps")
    s1 = (torch.rand(2 * ffn, generator=g) * 0.01 + 1.0e-4).to("mps")
    s2 = (torch.rand(hidden, generator=g) * 0.01 + 1.0e-4).to("mps")

    def layer(weight, scale):
        params = TensorWiseINT8Layout.Params(
            scale=scale, orig_dtype=torch.bfloat16,
            orig_shape=tuple(weight.shape), convrot=True,
            convrot_groupsize=256)
        quantized = QuantizedTensor(weight, "TensorWiseINT8Layout", params)
        return type("Linear", (), {
            "weight": quantized,
            "bias": None,
            "weight_function": [],
            "bias_function": [],
            "_full_precision_mm": False,
            "comfy_force_cast_weights": False,
        })()

    fc1, fc2 = layer(w1, s1), layer(w2, s2)
    lora1 = H3RuntimeLoRA((
        torch.randn(2 * ffn, 8, generator=g),
        torch.randn(8, hidden, generator=g),
        5.0, None, None, None,
    ), fast=True)
    lora1.multiplier = -0.6
    lora2 = H3RuntimeLoRA((
        torch.randn(hidden, 6 if adapter_count == 1 else 8, generator=g),
        torch.randn(6 if adapter_count == 1 else 8, ffn, generator=g),
        3.0, None, None, None,
    ), fast=True)
    lora2.multiplier = 0.7
    if adapter_count > 1:
        from _patches.minimax_h3_turbo import H3RuntimeLoRAStack
        def stack(first, nout, nin):
            adapters = [first]
            for index in range(1, adapter_count):
                rank = 8 * (index + 1)
                adapter = H3RuntimeLoRA((
                    torch.randn(nout, rank, generator=g),
                    torch.randn(rank, nin, generator=g),
                    float(rank + 1), None, None, None,
                ), fast=True)
                adapter.multiplier = 0.35 if index % 2 else -0.4
                adapters.append(adapter)
            return H3RuntimeLoRAStack(adapters)
        lora1 = stack(lora1, 2 * ffn, hidden)
        lora2 = stack(lora2, hidden, ffn)
    if adapt_fc1:
        setattr(fc1, patch._RUNTIME_LORA_ATTR, lora1)
    if adapt_fc2:
        setattr(fc2, patch._RUNTIME_LORA_ATTR, lora2)

    patch._act_cache = None
    wide = patch._int8_linear_kernel(
        x, w1, s1, None, torch.bfloat16, True, 256)
    if adapt_fc1:
        lora1.add_to(wide, x)
    act = patch._swiglu_activation(wide)
    reference = patch._int8_linear_kernel(
        act, w2, s2, None, torch.bfloat16, True, 256)
    if adapt_fc2:
        lora2.add_to(reference, act)

    patch._act_cache = None
    actual = patch.int8_mlp_bridge(fc1, fc2, x)
    torch.mps.synchronize()
    if adapter_count > 1:
        # Changing BF16 GEMM row shapes changed rounding for stacked adapters
        # during screening. Decline the tiled bridge; ordinary INT8 forwards
        # and the fused FC2 activation path still apply the stack.
        assert actual is None
        return
    assert actual is not None
    assert torch.equal(actual, reference), (
        torch.count_nonzero(actual != reference).item(),
        (actual.float() - reference.float()).abs().max().item(),
    )


@requires_int8_ext
def test_merge_rot_quant_matches_eager_pipeline():
    """i8_merge_rot_quant (fused head-merge + convrot + rowwise quant) vs the
    three-pass reference (transpose copy -> bf16 Hadamard matmul -> rowwise
    quant). NOT bit-identical by construction: the rotation differs only in
    fp32 summation order, so rotated bf16 values flip their last ulp on rare
    ties; a flipped row-absmax then shifts that row's scale by ~2^-8. Bounds
    below encode "same math, rounding-order noise only"."""
    from _patches.int8_ext import loader
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_merge_rot_quant")

    g = torch.Generator().manual_seed(5)
    for B, H, S, D in ((1, 18, 777, 256), (2, 18, 64, 256),
                       (1, 1, 256, 256), (1, 4, 5037, 256),
                       (1, 56, 257, 128)):
        x = (torch.randn((B, H, S, D), generator=g) * 2).to("mps").to(torch.bfloat16)
        # poison the pool between allocations so stale garbage would surface
        junk = torch.full((32, 1 << 20), float("nan"), dtype=torch.bfloat16, device="mps")
        del junk

        merged = x.transpose(1, 2).reshape(B * S, H * D).contiguous()
        # Convrot groups are 256 wide. For H3 D=128, each group spans two
        # adjacent attention heads in the merged layout.
        h = _build_hadamard(256, device="mps", dtype=torch.bfloat16)
        rot_ref = _rotate_activation(merged, h, 256)
        q_ref, s_ref = mod.i8_quant_rowwise(rot_ref.contiguous())

        q_k, s_k = mod.i8_merge_rot_quant(x)
        torch.mps.synchronize()

        assert q_k.shape == (B * S, H * D) and s_k.shape == (B * S, 1)
        assert int(torch.isnan(s_k).sum()) == 0
        # scales: equal up to a last-ulp absmax flip (2^-8 relative)
        srel = ((s_k - s_ref).abs() / s_ref).max().item()
        assert srel < 6e-3, f"scale rel diff {srel} at B={B} H={H} S={S}"
        # int8: off-by-one on rounding boundaries only, and rarely
        dq = (q_k.int() - q_ref.int()).abs()
        assert dq.max().item() <= 1, f"int8 diff >1 at B={B} H={H} S={S}"
        frac = (dq > 0).float().mean().item()
        assert frac < 5e-3, f"int8 mismatch fraction {frac} at B={B} H={H} S={S}"


@requires_int8_ext
def test_o_proj_prequant_matches_layer_path():
    """int8_o_proj_prequant output ~= merge + _int8_linear_kernel on the same
    QuantizedTensor-shaped inputs (channelwise scales, convrot, no bias —
    the Ideogram 4 o-projection contract)."""
    from _patches.int8_ext import loader

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_merge_rot_quant")
    patch._kernel = mod

    g = torch.Generator().manual_seed(9)
    B, H, S, D, N = 1, 18, 333, 256, 4608
    x = (torch.randn((B, H, S, D), generator=g) * 0.7).to("mps").to(torch.bfloat16)
    w = torch.randint(-128, 128, (N, H * D), generator=g, dtype=torch.int8).to("mps")
    ws = (torch.rand((N, 1), generator=g) * 0.01 + 0.001).to("mps")

    patch._act_cache = None
    merged = x.transpose(1, 2).reshape(B, S, H * D)
    ref = patch._int8_linear_kernel(merged, w, ws, None, torch.bfloat16, True, 256)

    q_k, s_k = mod.i8_merge_rot_quant(x)
    out = mod.i8_matmul2d_nt_fused_cs(
        q_k, w.contiguous(), s_k.reshape(-1).float().contiguous(),
        ws.view(-1).float().contiguous(), None).reshape(B, S, N)
    torch.mps.synchronize()

    # ±1 int8 flips on <0.5% of entries bound the output difference to ~one
    # quant step through the GEMM; relative to signal it must stay tiny.
    denom = ref.float().abs().max().item() + 1e-9
    rel = (out.float() - ref.float()).abs().max().item() / denom
    assert rel < 2e-2, f"o-proj prequant rel err {rel}"


@requires_int8_ext
@pytest.mark.parametrize("stacked", [False, True])
def test_h3_o_proj_prequant_applies_runtime_lora_exactly(monkeypatch, stacked):
    from comfy.quant_ops import TensorWiseINT8Layout
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.int8_ext import loader
    from _patches.minimax_h3_turbo import H3RuntimeLoRA

    mod = loader.module()
    assert mod is not None
    monkeypatch.setattr(patch, "_kernel", mod)
    monkeypatch.setenv("ASFP8_INT8_OPROJ_FUSE", "1")

    g = torch.Generator().manual_seed(83)
    batch, heads, seq, dim, output, rank = 1, 4, 259, 128, 512, 8
    merged = heads * dim
    attention = torch.randn(
        batch, heads, seq, dim, generator=g,
        dtype=torch.bfloat16).to("mps")
    qdata = torch.randint(
        -128, 128, (output, merged), generator=g,
        dtype=torch.int8).to("mps")
    scale = (torch.rand(output, generator=g) * 0.01 + 1.0e-4).to("mps")
    params = TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape), convrot=True,
        convrot_groupsize=256)
    layer = type("Linear", (), {
        "weight": QuantizedTensor(qdata, "TensorWiseINT8Layout", params),
        "bias": None,
        "weight_function": [],
        "bias_function": [],
        "_full_precision_mm": False,
        "comfy_force_cast_weights": False,
    })()
    adapter = H3RuntimeLoRA((
        torch.randn(output, rank, generator=g),
        torch.randn(rank, merged, generator=g),
        5.0, None, None, None,
    ), fast=True)
    adapter.multiplier = -0.6

    if stacked:
        from _patches.minimax_h3_turbo import H3RuntimeLoRAStack
        second = H3RuntimeLoRA((
            torch.randn(output, 3, generator=g), torch.randn(3, merged, generator=g),
            None, None, None, None), fast=True)
        second.multiplier = 0.4
        adapter = H3RuntimeLoRAStack([adapter, second])

    expected = patch.int8_o_proj_prequant(layer, attention)
    assert expected is not None
    adapter.add_bhsd_to(expected, attention)
    setattr(layer, patch._RUNTIME_LORA_ATTR, adapter)
    actual = patch.int8_o_proj_prequant(layer, attention)
    torch.mps.synchronize()

    assert actual is not None
    assert torch.equal(actual, expected), (
        torch.count_nonzero(actual != expected).item(),
        (actual.float() - expected.float()).abs().max().item(),
    )


@requires_int8_ext
def test_merge_quant_norotate_bit_exact():
    """rotate=False mode (the default o-proj fuse path): rotation done by the
    eager bf16 matmul in [B,H,S,D] layout, kernel does gather+quant only.
    Must be BIT-IDENTICAL to the merged-layout eager pipeline — this is the
    guarantee the bit-exact mode ships on. Also asserts the 4D-layout matmul
    equals the merged-layout matmul (the load-bearing assumption)."""
    from _patches.int8_ext import loader
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_merge_rot_quant")

    g = torch.Generator().manual_seed(13)
    for B, H, S in ((1, 18, 777), (2, 18, 64), (1, 1, 256), (1, 4, 5037)):
        D = 256
        x = (torch.randn((B, H, S, D), generator=g) * 2).to("mps").to(torch.bfloat16)
        # inject structured data (exact zeros, duplicates) — the tie-prone
        # patterns real activations carry
        x[:, :, : S // 4] = 0
        x[:, :, S // 4: S // 2] = x[:, :, : S // 4].flip(2) + 1

        had = _build_hadamard(D, device="mps", dtype=torch.bfloat16)
        merged = x.transpose(1, 2).reshape(B * S, H, D).contiguous()
        rot_ref = torch.matmul(merged, had).reshape(B * S, H * D)
        q_ref, s_ref = mod.i8_quant_rowwise(rot_ref.contiguous())

        rot4 = torch.matmul(x, had)
        assert torch.equal(
            rot4.transpose(1, 2).reshape(B * S, H * D).contiguous(), rot_ref
        ), f"4D-layout matmul != merged-layout matmul at B={B} H={H} S={S}"
        q_k, s_k = mod.i8_merge_rot_quant(rot4.contiguous(), False)
        torch.mps.synchronize()
        assert torch.equal(q_ref, q_k), f"q not bit-identical at B={B} H={H} S={S}"
        assert torch.equal(s_ref, s_k), f"s not bit-identical at B={B} H={H} S={S}"
