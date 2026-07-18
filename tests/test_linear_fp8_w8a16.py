"""Tests for patch #18: standard-fp8 Linear via the half×fp8 W8A16 Metal kernel.

Unit tests (always run): install() is opt-in/guarded; _eligible gates correctly;
the patched forward delegates to the original on any ineligibility.

Integration tests (opt-in, ASFP8_FP8_EXT=1 on an MPS box): kernel-backed forward
matches the decode+F.linear path (f32-accumulate tolerance), across bias dtypes,
3D inputs and ragged shapes.
"""
import os

import pytest
import torch

from _patches import linear_fp8_w8a16 as patch
from _patches._common import decode_fp8

_run_integration = os.environ.get("ASFP8_FP8_EXT") == "1"
requires_fp8_ext = pytest.mark.skipif(
    not (_run_integration and torch.backends.mps.is_available()),
    reason="set ASFP8_FP8_EXT=1 on an MPS device to build + test the fp8 kernel",
)
_run_sgemm = os.environ.get("ASFP8_FP8_SGEMM") == "1"
requires_sgemm = pytest.mark.skipif(
    not (_run_sgemm and torch.backends.mps.is_available()),
    reason="set ASFP8_FP8_SGEMM=1 on an MPS device to build + test the staged kernel",
)


class _FakeLinear:
    """Duck-typed stand-in for comfy.ops disable_weight_init.Linear."""

    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias
        self.weight_function = []
        self.bias_function = []


def _mk(N, K, device="cpu", bias_dtype=torch.bfloat16):
    torch.manual_seed(0)
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).to(device)
    b = (torch.randn(N) * 0.1).to(bias_dtype).to(device) if bias_dtype else None
    return _FakeLinear(torch.nn.Parameter(w, requires_grad=False), b)


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_SGEMM", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_sgemm", None, raising=False)
    patch.install()
    assert patch._installed is False and patch._sgemm is None


def test_install_respects_kill_switch(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    monkeypatch.setenv("ASFP8_FP8_W8A16", "0")
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_eligible_rejects_cpu_and_small_and_lora(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "8192")
    x = torch.zeros(4, 64, dtype=torch.bfloat16)

    m = _mk(32, 64)                       # CPU + tiny -> ineligible
    assert patch._eligible(m, x) is False

    m = _FakeLinear(torch.nn.Parameter(torch.zeros(32, 64), requires_grad=False))
    assert patch._eligible(m, x) is False  # non-fp8 weight

    m = _mk(32, 64)
    m.weight_function = [lambda w: w]      # LoRA -> ineligible
    assert patch._eligible(m, x) is False


def test_patched_forward_delegates_when_ineligible(monkeypatch):
    sentinel = object()
    called = {}

    def fake_orig(module, input):
        called["hit"] = True
        return sentinel

    monkeypatch.setattr(patch, "_orig_forward", fake_orig, raising=False)
    monkeypatch.setattr(patch, "_kernel", None, raising=False)

    m = _mk(32, 64)
    out = patch._patched_forward(m, torch.zeros(4, 64, dtype=torch.bfloat16))
    assert out is sentinel and called.get("hit") is True


@requires_fp8_ext
class TestKernelParity:
    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        from _patches.fp8_ext.loader import module as load
        kernel = load()
        if kernel is None:
            pytest.skip("fp8 extension failed to build")
        monkeypatch.setattr(patch, "_kernel", kernel, raising=False)
        monkeypatch.setattr(patch, "_sgemm", None, raising=False)
        monkeypatch.setattr(patch, "_self_checked", False, raising=False)
        monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "256")
        monkeypatch.setenv("ASFP8_FP8_W8A16_M_MAX", "4096")

        def decode_forward(module, input):
            wb = decode_fp8(module.weight, torch.float32)
            bias = module.bias
            if bias is not None and bias.dtype not in (torch.float32,):
                bias = (decode_fp8(bias, torch.float32)
                        if bias.dtype in patch.FP8_DTYPES else bias.to(torch.float32))
            out = input.float() @ wb.t()
            if bias is not None:
                out = out + bias
            return out.to(input.dtype)

        monkeypatch.setattr(patch, "_orig_forward", decode_forward, raising=False)

    def _check(self, m, x, rel_tol=5e-3):
        ref = patch._orig_forward(m, x).float()
        out = patch._patched_forward(m, x)
        assert patch._kernel, "kernel was disabled during forward"
        out = out.float()
        rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        assert rel < rel_tol, f"rel={rel}"
        assert out.shape == ref.shape

    @pytest.mark.parametrize("M,K,N", [(64, 512, 384), (257, 384, 512), (1, 512, 512)])
    def test_parity_2d(self, M, K, N):
        m = _mk(N, K, device="mps")
        x = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
        self._check(m, x)

    def test_parity_3d_and_fp8_bias(self):
        m = _mk(384, 512, device="mps", bias_dtype=None)
        m.bias = (torch.randn(384) * 0.1).to(torch.float8_e4m3fn).to("mps")
        x = (torch.randn(2, 33, 512) * 0.3).to(torch.bfloat16).to("mps")
        self._check(m, x)

    def test_parity_fp16_input(self):
        m = _mk(512, 384, device="mps")
        x = (torch.randn(96, 384) * 0.3).to(torch.float16).to("mps")
        self._check(m, x)

    def test_m_max_gate_falls_back(self, monkeypatch):
        monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "512")
        monkeypatch.setenv("ASFP8_FP8_W8A16_M_MAX", "64")
        m = _mk(512, 256, device="mps")  # N >= min_dim, K < min_dim
        x = (torch.randn(128, 256) * 0.3).to(torch.bfloat16).to("mps")
        assert patch._eligible(m, x) is False  # M=128 > m_max and K small
        x_small = x[:32].contiguous()
        assert patch._eligible(m, x_small) is True


@requires_sgemm
class TestStagedKernelParity:
    """Same parity contract, staged backend active (fused bias epilogue path)."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        from _patches.fp8s_ext.loader import module as load_sg
        sg = load_sg()
        if sg is None:
            pytest.skip("staged fp8 extension failed to build")
        monkeypatch.setattr(patch, "_sgemm", sg, raising=False)
        monkeypatch.setattr(patch, "_kernel", False, raising=False)
        monkeypatch.setattr(patch, "_self_checked", False, raising=False)
        monkeypatch.setenv("ASFP8_FP8_SGEMM_MIN_DIM", "256")
        monkeypatch.setenv("ASFP8_FP8_W8A16_M_MAX", "4096")

        def decode_forward(module, input):
            wb = decode_fp8(module.weight, torch.float32)
            bias = module.bias
            if bias is not None:
                bias = (decode_fp8(bias, torch.float32)
                        if bias.dtype in patch.FP8_DTYPES else bias.to(torch.float32))
            out = input.float() @ wb.t()
            if bias is not None:
                out = out + bias
            return out.to(input.dtype)

        monkeypatch.setattr(patch, "_orig_forward", decode_forward, raising=False)

    @pytest.mark.parametrize("M,K,N", [(64, 512, 384), (257, 384, 512), (1, 512, 512),
                                       (3000, 512, 384)])
    def test_parity_2d_staged(self, M, K, N):
        m = _mk(N, K, device="mps")
        x = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
        ref = patch._orig_forward(m, x).float()
        out = patch._patched_forward(m, x)
        assert patch._sgemm, "staged kernel got disabled during forward"
        rel = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        assert rel < 5e-3, f"rel={rel}"

    def test_staged_3d_and_fp16_input(self):
        m = _mk(384, 512, device="mps")
        x3 = (torch.randn(2, 33, 512) * 0.3).to(torch.bfloat16).to("mps")
        ref = patch._orig_forward(m, x3).float()
        out = patch._patched_forward(m, x3)
        rel = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        assert rel < 5e-3 and out.shape == ref.shape

        x16 = (torch.randn(96, 512) * 0.3).to(torch.float16).to("mps")
        ref = patch._orig_forward(m, x16).float()
        out = patch._patched_forward(m, x16)  # non-bf16 input -> f32 kernel + python epilogue
        rel = ((out.float() - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        assert rel < 5e-3 and out.dtype == torch.float16

    def test_staged_k_alignment_gate(self, monkeypatch):
        monkeypatch.setenv("ASFP8_FP8_SGEMM_MIN_DIM", "256")
        m = _mk(512, 264, device="mps")  # K=264 not a multiple of 16
        x = (torch.randn(32, 264) * 0.3).to(torch.bfloat16).to("mps")
        assert patch._eligible(m, x) is False
