"""Tests for patch #17: INT8 W8A8 via the bit-exact Metal kernel (int8_linear_kernel_mps).

Unit tests (always run): install() is opt-in/guarded; the wrapper falls back to the
original int8_linear when the kernel is unavailable or off-MPS.

Integration test (opt-in, ASFP8_INT8_EXT=1 on an MPS box): the kernel-backed
int8_linear is bit-identical to comfy_kitchen's original across convrot/bias/3D/M=1.
"""
import os

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
