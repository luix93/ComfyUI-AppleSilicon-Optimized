"""Tests for patch #26: Conv3d as sliced conv2d on MPS."""
import os

import pytest
import torch
import torch.nn.functional as F

from _patches import conv3d_mps as patch

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs MPS device")


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_CONV3D", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


@requires_mps
@pytest.mark.parametrize("case", [
    # (Ci, Co, T, H, W, kT, kH, kW, stride, padding, dilation, groups)
    (8, 16, 7, 12, 12, 3, 3, 3, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1),   # causal-style
    (8, 8, 9, 10, 10, 3, 3, 3, (1, 1, 1), (1, 1, 1), (1, 1, 1), 1),   # symmetric pT
    (8, 8, 9, 16, 16, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1), 1),   # stride 2
    (8, 8, 11, 10, 10, 3, 3, 3, (1, 1, 1), (0, 1, 1), (2, 1, 1), 1),  # time dilation
    (8, 8, 7, 10, 10, 1, 3, 3, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1),   # kT=1
    (8, 8, 7, 10, 10, 3, 3, 3, (1, 1, 1), (0, 1, 1), (1, 1, 1), 4),   # groups
])
def test_decomposition_matches_conv3d(case):
    Ci, Co, T, H, W, kT, kH, kW, stride, padding, dilation, groups = case
    torch.manual_seed(0)
    x = torch.randn(2, Ci, T, H, W, device="mps", dtype=torch.float32)
    w = torch.randn(Co, Ci // groups, kT, kH, kW, device="mps",
                    dtype=torch.float32) * 0.1
    b = torch.randn(Co, device="mps", dtype=torch.float32)

    ref = F.conv3d(x, w, b, stride, padding, dilation, groups)
    out = patch._conv3d_decomposed(x, w, b, stride, padding, dilation, groups)
    assert out.shape == ref.shape
    d = (out - ref).abs().max().item()
    assert d < 1e-4, f"maxdiff {d}"


@requires_mps
def test_chunking_path():
    """Force multi-chunk execution via a tiny _MAX_ELEMS."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 9, 8, 8, device="mps")
    w = torch.randn(4, 4, 3, 3, 3, device="mps") * 0.1
    ref = F.conv3d(x, w, None, (1, 1, 1), (0, 1, 1), (1, 1, 1), 1)
    old = patch._MAX_ELEMS
    try:
        patch._MAX_ELEMS = 4 * 8 * 8 * 2   # ~2 frames per chunk
        out = patch._conv3d_decomposed(x, w, None, (1, 1, 1), (0, 1, 1),
                                       (1, 1, 1), 1)
    finally:
        patch._MAX_ELEMS = old
    assert (out - ref).abs().max().item() < 1e-4
