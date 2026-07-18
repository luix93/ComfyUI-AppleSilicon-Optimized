import pytest
import torch

from conftest import requires_mps

from _patches import fp8_fill_mps as ff

FP8 = torch.float8_e4m3fn


# --- wrapper logic (no MPS needed): guard, CPU-bounce, don't-mask-real-errors ---


def test_factory_passthrough_non_fp8():
    calls = []
    def orig(*a, **k):
        calls.append((a, k))
        return "OUT"
    assert ff._wrap_factory(orig)(2, 2, dtype=torch.float32, device="mps") == "OUT"
    assert len(calls) == 1  # called once, directly, no bounce


def test_factory_cpu_bounce_on_float8_error():
    seen = []
    def orig(*a, **k):
        seen.append(k.get("device"))
        if k.get("device") is not None:
            raise RuntimeError("Undefined type Float8_e4m3fn")
        class _R:
            def to(self, dev):
                seen.append(f"to:{dev}")
                return self
        return _R()
    ff._wrap_factory(orig)(2, 2, dtype=FP8, device="mps")
    assert seen == ["mps", None, "to:mps"]  # tried mps, rebuilt device-less, moved back


def test_factory_real_error_not_masked():
    def orig(*a, **k):
        raise RuntimeError("some unrelated MPS failure")
    with pytest.raises(RuntimeError, match="unrelated"):
        ff._wrap_factory(orig)(2, 2, dtype=FP8, device="mps")


# --- real behaviour on MPS: the exact crash sites (torch 2.13 zeros regression) ---


@requires_mps
def test_zeros_ones_full_fp8_mps():
    ff.install()
    for fn, ref in ((torch.zeros, 0.0), (torch.ones, 1.0)):
        t = fn(4, 4, dtype=FP8, device="mps")
        assert t.device.type == "mps" and t.dtype == FP8
        assert torch.equal(t.cpu().view(torch.uint8),
                           fn(4, 4, dtype=FP8).view(torch.uint8))  # bit-exact vs cpu
    t = torch.full((4, 4), 0.5, dtype=FP8, device="mps")
    assert torch.equal(t.cpu().view(torch.uint8),
                       torch.full((4, 4), 0.5, dtype=FP8).view(torch.uint8))


@requires_mps
def test_like_variants_fp8_mps():
    ff.install()
    base = torch.zeros(3, 3, dtype=FP8).to("mps")
    for fn in (torch.zeros_like, torch.ones_like):
        t = fn(base)
        assert t.device.type == "mps" and t.dtype == FP8
        assert torch.equal(t.cpu().view(torch.uint8), fn(base.cpu()).view(torch.uint8))
    t = torch.full_like(base, 2.0)
    assert torch.equal(t.cpu().view(torch.uint8),
                       torch.full_like(base.cpu(), 2.0).view(torch.uint8))


@requires_mps
def test_inplace_fill_fp8_mps():
    ff.install()
    t = torch.empty(4, 4, dtype=FP8, device="mps")
    t.zero_()
    assert torch.equal(t.cpu().view(torch.uint8),
                       torch.zeros(4, 4, dtype=FP8).view(torch.uint8))
    t.fill_(1.0)
    assert torch.equal(t.cpu().view(torch.uint8),
                       torch.ones(4, 4, dtype=FP8).view(torch.uint8))


@requires_mps
def test_e5m2_and_eye_fp8_mps():
    ff.install()
    t = torch.zeros(2, 2, dtype=torch.float8_e5m2, device="mps")
    assert t.dtype == torch.float8_e5m2 and t.device.type == "mps"
    e = torch.eye(3, dtype=FP8, device="mps")
    assert torch.equal(e.cpu().view(torch.uint8),
                       torch.eye(3, dtype=FP8).view(torch.uint8))


@requires_mps
def test_non_fp8_factories_unaffected():
    ff.install()
    t = torch.zeros(4, 4, device="mps")
    assert t.dtype == torch.float32 and t.abs().sum().item() == 0.0
    assert torch.ones(2, device="mps", dtype=torch.float16).sum().item() == 2.0
