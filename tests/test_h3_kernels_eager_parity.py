"""H3 fused norm/RoPE kernels must equal the eager calls production makes, byte for byte.

References are the live ops on the installed PyTorch: native F.rms_norm with a
same-dtype weight (fused MLX-derived kernel on MPS since 2.14) and comfy_kitchen's
unwrapped eager rms_rope_split_half_. A PyTorch rounding change fails here
instead of silently shifting H3 output.
"""
import os

import pytest
import torch

requires_dit_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and os.environ.get("ASFP8_DIT_EXT") == "1"),
    reason="needs MPS + ASFP8_DIT_EXT=1",
)
F = torch.nn.functional


@pytest.fixture(scope="module")
def kernel():
    from _patches.dit_ext import loader
    k = loader.module()
    if k is None:
        pytest.skip("dit_ext kernel unavailable")
    return k


def _bf16(*shape, scale=0.5, generator=None):
    return (torch.randn(*shape, generator=generator) * scale).to(
        "mps", torch.bfloat16).contiguous()


def _assert_bytes(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    mismatches = int((actual != expected).sum())
    assert mismatches == 0, f"{mismatches} of {expected.numel()} elements differ"


@requires_dit_mps
@pytest.mark.parametrize("dm", [5376, 3072, 1024, 128])
def test_segmented_norm_modulation_matches_native_rms_norm(kernel, dm):
    from _patches import minimax_fast as patch
    g = torch.Generator().manual_seed(dm)
    s, mods = 333, 3
    segments = [(0, 70, 1), (70, 260, 2), (260, s, 0)]
    rows = patch._seg_rows(segments, s, torch.device("mps"))
    x, attention = _bf16(s, dm, scale=2, generator=g), _bf16(s, dm, generator=g)
    gate = _bf16(mods, dm, generator=g)
    weight = (1 + _bf16(dm, generator=g) * 0.1).contiguous()
    scale, shift = _bf16(mods, dm, generator=g), _bf16(mods, dm, generator=g)

    def eager(h):
        h = F.rms_norm(h, (dm,), weight, eps=1e-6)
        for a, b, row in segments:
            h[a:b].mul_(1.0 + scale[row]).add_(shift[row])
        return h

    _assert_bytes(kernel.rms_mod_seg(x, weight, scale, shift, rows, 1e-6), eager(x.clone()))
    _assert_bytes(kernel.rms_mod_seg_f32(x, weight, scale, shift, rows, 1e-6),
                  eager(x.clone()).float())
    updated = x.clone()
    for a, b, row in segments:
        updated[a:b].addcmul_(attention[a:b], gate[row])
    actual_x, actual_h = kernel.gate_norm_mod_seg(
        x.clone(), attention, gate, weight, scale, shift, rows, 1e-6)
    _assert_bytes(actual_x, updated)
    _assert_bytes(actual_h, eager(updated.clone()))


@requires_dit_mps
def test_qk_norm_partial_rope_kernels_match_kitchen_eager(kernel):
    from comfy_kitchen.backends.eager import rope as kitchen
    g = torch.Generator().manual_seed(11)
    s, heads, dim, pairs = 257, 6, 128, 48
    qkv = _bf16(s, 3 * heads * dim, scale=2, generator=g)
    freqs = _bf16(1, s, 1, pairs, 2, 2, scale=1, generator=g)
    q_weight = (1 + _bf16(dim, generator=g) * 0.1).contiguous()
    k_weight = (1 + _bf16(dim, generator=g) * 0.1).contiguous()

    def views(t):
        q, k, v = t.split(heads * dim, dim=-1)
        return q.view(1, s, heads, dim), k.view(1, s, heads, dim), v.view(s, heads, dim)

    def attention_layout(t):
        return t[0].transpose(0, 1).unsqueeze(0)

    reference = qkv.clone()
    ref_q, ref_k, ref_v = views(reference)
    kitchen.rms_rope_split_half_(ref_q, ref_k, freqs, q_weight, k_weight,
                                 epsilon=1e-6, rot_dim=2 * pairs)

    in_place = qkv.clone()
    got_q, got_k, _ = views(in_place)
    kernel.rms_rope_sh_partial(got_q, got_k, freqs, q_weight.float(), k_weight.float(), 1e-6)
    _assert_bytes(got_q, ref_q)
    _assert_bytes(got_k, ref_k)

    out_q, out_k, out_v = kernel.qkv_rms_rope_sh_partial(
        qkv, freqs, q_weight.float(), k_weight.float(), heads, dim, 1e-6)
    _assert_bytes(out_q, attention_layout(ref_q))
    _assert_bytes(out_k, attention_layout(ref_k))
    _assert_bytes(out_v, ref_v.transpose(0, 1).unsqueeze(0))

    pre_q, pre_k, _ = views(qkv)
    att_q = attention_layout(pre_q).contiguous()
    att_k = attention_layout(pre_k).contiguous()
    kernel.rms_rope_attention_partial(att_q, att_k, freqs, q_weight.float(),
                                      k_weight.float(), 1e-6)
    _assert_bytes(att_q, attention_layout(ref_q))
    _assert_bytes(att_k, attention_layout(ref_k))
