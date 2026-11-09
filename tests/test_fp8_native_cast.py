"""Native MPS FP8 decode must reproduce the LUT path's bits and layout."""
import pytest
import torch

from _patches import _common
from conftest import requires_mps

E4M3 = torch.float8_e4m3fn


def _bits(x):
    x = x.cpu().contiguous()
    return x.view(torch.int16 if x.element_size() == 2 else torch.int32)


def _lut(t, out_dtype, monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_NATIVE_CAST", "0")
    try:
        return _common.decode_fp8(t, out_dtype)
    finally:
        monkeypatch.delenv("ASFP8_FP8_NATIVE_CAST")


def test_cpu_and_fp16_targets_never_use_native_cast():
    assert not _common.native_fp8_cast_ok(E4M3, torch.device("cpu"), torch.bfloat16)
    assert not _common.native_fp8_cast_ok(E4M3, torch.device("mps"), torch.float16)


def test_opt_out(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_NATIVE_CAST", "0")
    assert not _common.native_fp8_cast_ok(E4M3, torch.device("mps"), torch.bfloat16)


@requires_mps
@pytest.mark.parametrize("fp8", [E4M3, torch.float8_e5m2])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_all_codes_and_layouts_match_lut(monkeypatch, fp8, out_dtype):
    # e5m2 has no MPS cast on 2.15: the probe must fail closed onto the LUT.
    codes = torch.arange(256, dtype=torch.uint8).view(fp8).to("mps")
    assert torch.equal(_bits(_common.decode_fp8(codes, out_dtype)), _bits(_lut(codes, out_dtype, monkeypatch)))
    raw = torch.randint(0, 256, (96, 160), dtype=torch.uint8, generator=torch.Generator().manual_seed(7))
    w = raw.view(fp8).to("mps").t()  # non-contiguous, as torch._scaled_mm receives weights
    got, want = _common.decode_fp8(w, out_dtype), _lut(w, out_dtype, monkeypatch)
    assert got.is_contiguous() and want.is_contiguous() and got.shape == want.shape
    assert torch.equal(_bits(got), _bits(want))


@requires_mps
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_decode_as_matches_two_step(monkeypatch, out_dtype):
    raw = torch.randint(0, 256, (64, 48), dtype=torch.uint8, generator=torch.Generator().manual_seed(11))
    t = raw.view(E4M3).to("mps")
    got = _common.decode_fp8_as(t, out_dtype)
    want = _lut(t, torch.float32, monkeypatch).to(out_dtype)
    finite = torch.isfinite(want.cpu())
    assert got.dtype == out_dtype and got.is_contiguous()
    assert torch.equal(_bits(got)[finite], _bits(want)[finite])
    assert torch.equal(torch.isnan(got.cpu()), torch.isnan(want.cpu()))
