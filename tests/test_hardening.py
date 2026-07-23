"""Correctness-hardening series regression tests (Sol audit, 2026-07-23).

Covers: Tensor.to positional-bool overload parsing, exact native _int_mm
routing, self-check RNG hygiene, and bmm-route scale semantics (incl. the
explicit scale=0 case). E2E gate for the same series: same-seed krea +
ideogram renders byte-identical before/after (verified at commit time).
"""

import os

import pytest
import torch

_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not _MPS, reason="requires MPS")


def test_tensor_to_positional_bool_overloads():
    from _patches import tensor_to_fp8
    tensor_to_fp8.install()
    t = torch.randn(4, 8).to(torch.float8_e4m3fn).to("mps")

    # positional non_blocking after dtype: bool must not clobber the device
    # (bool is a subclass of int, int is a valid device index).
    o1 = t.to(torch.float16, False)
    assert o1.dtype == torch.float16 and o1.device.type == "mps"
    assert torch.equal(o1.cpu(), t.cpu().to(torch.float16))

    # device + dtype + positional non_blocking: the device must survive.
    o2 = t.to("cpu", torch.float32, False)
    assert o2.device.type == "cpu" and o2.dtype == torch.float32

    # float -> fp8 with a positional bool.
    f = torch.randn(4, 8, device="mps")
    o3 = f.to(torch.float8_e4m3fn, False)
    assert o3.dtype == torch.float8_e4m3fn and o3.device.type == "mps"

    # kwarg forms unchanged.
    o4 = t.to(dtype=torch.bfloat16, device="mps", non_blocking=True)
    assert o4.dtype == torch.bfloat16 and o4.device.type == "mps"


def test_int_mm_native_routing_exact():
    os.environ.setdefault("ASFP8_INT8_EXT", "1")
    from _patches import int_mm_mps
    int_mm_mps.install()
    if int_mm_mps._native_mod() is None:
        pytest.skip("int8 Metal extension unavailable on this host")
    g = torch.Generator().manual_seed(0)
    for M, K, N in ((7, 333, 11), (64, 4096, 128)):
        a = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8)
        b = torch.randint(-128, 128, (K, N), generator=g, dtype=torch.int8)
        a[:, :2] = -128
        b[:2, :] = 127
        ref = (a.long() @ b.long()).to(torch.int32)
        out = torch._int_mm(a.to("mps"), b.to("mps"))
        torch.mps.synchronize()
        assert out.dtype == torch.int32
        assert torch.equal(out.cpu(), ref), f"mismatch at {(M, K, N)}"
    # the common caller pattern: torch._int_mm(x, weight.T) on [N,K] storage
    w = torch.randint(-128, 128, (64, 512), generator=g, dtype=torch.int8).to("mps")
    x = torch.randint(-128, 128, (16, 512), generator=g, dtype=torch.int8).to("mps")
    out = torch._int_mm(x, w.t())
    ref = (x.cpu().long() @ w.cpu().long().t()).to(torch.int32)
    assert torch.equal(out.cpu(), ref)


def test_self_checks_preserve_global_rng():
    from _patches import na_gemm, sg_gemm
    before = torch.random.get_rng_state()
    na_gemm.self_check_ok()
    sg_gemm.self_check_ok()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after), "self-check mutated the global RNG stream"


def test_bmm_attention_scale_semantics():
    from _patches import attn_router_mps as R
    torch_g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, 64, 32, generator=torch_g).to("mps").to(torch.bfloat16)
    k = torch.randn(1, 4, 64, 32, generator=torch_g).to("mps").to(torch.bfloat16)
    v = torch.randn(1, 4, 64, 32, generator=torch_g).to("mps").to(torch.bfloat16)

    # explicit default scale must be bit-identical to the implicit default
    o_def = R._bmm_attention(q, k, v, 4, skip_reshape=True)
    o_exp = R._bmm_attention(q, k, v, 4, scale=32 ** -0.5, skip_reshape=True)
    assert torch.equal(o_def, o_exp)

    # scale=0 is explicit (uniform attention = per-head mean of V), not
    # "absent" — the old truthiness test silently substituted d**-0.5.
    o_zero = R._bmm_attention(q, k, v, 4, scale=0.0, skip_reshape=True)
    o4 = o_zero[0].reshape(64, 4, 32).transpose(0, 1)  # [B,S,H*D] -> [H,S,D]
    ref0 = v[0].float().mean(dim=1, keepdim=True).expand(4, 64, 32)
    assert (o4.float() - ref0).abs().max().item() < 1e-2


def test_vae_slice_attention_floor_and_parity(monkeypatch):
    """Patch #31: INT_MAX floor forces extra slices; result parity holds."""
    import sys as _sys
    import types

    names = ["comfy", "comfy.ldm", "comfy.ldm.modules",
             "comfy.ldm.modules.diffusionmodules",
             "comfy.ldm.modules.diffusionmodules.model",
             "comfy.model_management"]
    for n in names:
        monkeypatch.setitem(_sys.modules, n, types.ModuleType(n))
    mm = _sys.modules["comfy.model_management"]
    mm.get_free_memory = lambda dev=None: 1 << 44  # "infinite" free memory

    def _raise(e):
        raise e
    mm.raise_non_oom = _raise
    mm.soft_empty_cache = lambda *a, **k: None
    M = _sys.modules["comfy.ldm.modules.diffusionmodules.model"]

    from _patches import vae_slice_attn_mps as P
    monkeypatch.setattr(P, "_installed", False)
    monkeypatch.setenv("ASFP8_VAE_SLICE_FIX", "1")
    P.install()
    assert hasattr(M, "slice_attention"), "patch did not install"

    g = torch.Generator().manual_seed(0)
    # VAE layout: q [B, hw, C], k/v [B, C, hw]
    q = torch.randn(3, 256, 32, generator=g).to("mps").to(torch.float16)
    k = torch.randn(3, 32, 256, generator=g).to("mps").to(torch.float16)
    v = torch.randn(3, 32, 256, generator=g).to("mps").to(torch.float16)

    ref = M.slice_attention(q, k, v)  # floor inert: 3*256*256 << budget

    # Shrink the budget so the floor binds (3*256*256 = 196608 > 65536 ->
    # min_steps 3 -> steps 4) and verify the sliced result matches.
    # (the installed closure reads the module global at call time)
    monkeypatch.setattr(P, "_MAX_SCORE_ELEMS", 65536)
    forced = M.slice_attention(q, k, v)
    assert (forced.float() - ref.float()).abs().max().item() < 2e-3

    # eager reference: softmax over full key length per query row
    attn = torch.softmax(
        torch.bmm(q.float(), k.float()) * (32 ** -0.5), dim=2)
    eager = torch.bmm(v.float(), attn.permute(0, 2, 1))
    assert (forced.float() - eager).abs().max().item() < 2e-2
