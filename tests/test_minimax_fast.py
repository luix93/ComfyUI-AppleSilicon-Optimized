"""Tests for the MiniMax H3 model-specific MPS fusions."""

import os
import sys
import types
import weakref

import pytest
import torch

from _patches import minimax_fast as patch


def test_owned_attention_inputs_fall_back_on_old_comfyui(monkeypatch):
    real_import = __import__

    def missing_attention(name, *args, **kwargs):
        if name == "comfy.ldm.modules.attention":
            raise ImportError("old ComfyUI")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_attention)
    q, k, v = object(), object(), object()
    assert patch._owned_attention_inputs(q, k, v) == (q, k, v)


def test_owned_attention_inputs_use_new_comfyui_container(monkeypatch):
    class Container:
        def __init__(self, tensor):
            self.tensor = tensor

        def take(self):
            tensor = self.tensor
            self.tensor = None
            return tensor

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules")
    modules.__path__ = []
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.AttentionTensorContainer = Container
    for name, module in (
        ("comfy", comfy),
        ("comfy.ldm", ldm),
        ("comfy.ldm.modules", modules),
        ("comfy.ldm.modules.attention", attention),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    q, k, v = object(), object(), object()
    owned = patch._owned_attention_inputs(q, k, v)
    assert all(isinstance(value, Container) for value in owned)
    assert tuple(value.take() for value in owned) == (q, k, v)


def test_fp32_weight_cache_accepts_inference_tensors():
    patch._wcache.clear()
    with torch.inference_mode():
        weight = torch.randn(128, dtype=torch.bfloat16)
        first = patch._fp32_w(weight)
        second = patch._fp32_w(weight)
    assert first is second
    assert first.dtype == torch.float32
    assert torch.equal(first, weight.float())


def test_seg_rows_requires_a_contiguous_tiling():
    rows = patch._seg_rows([(0, 2, 1), (2, 5, 0)], 5, torch.device("cpu"))
    assert torch.equal(rows, torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32))
    assert patch._seg_rows([(0, 3, 0), (2, 4, 1)], 5, torch.device("cpu")) is None


def test_cross_block_links_are_weak_and_idempotent():
    class Block:
        pass

    class Model:
        def __init__(self, count=4):
            self.blocks = [Block() for _ in range(count)]

    module = types.SimpleNamespace(MiniMaxH3Model=Model)
    patch._install_cross_block_links(module)
    wrapped = module.MiniMaxH3Model.__init__
    patch._install_cross_block_links(module)
    assert module.MiniMaxH3Model.__init__ is wrapped

    model = module.MiniMaxH3Model()
    for current, following in zip(model.blocks, model.blocks[1:]):
        assert isinstance(current._asfp8_next_block, weakref.ReferenceType)
        assert current._asfp8_next_block() is following
    assert not hasattr(model.blocks[-1], "_asfp8_next_block")


def test_mlp_bridge_wrapper_routes_and_falls_back(monkeypatch):
    from _patches import int8_linear_kernel_mps as int8_patch

    calls = []

    class MLP:
        def __init__(self):
            self.fc1 = object()
            self.fc2 = object()

        def forward(self, x):
            calls.append("original")
            return x + 1

    module = types.SimpleNamespace(MLP=MLP)
    patch._install_mlp_bridge(module)
    wrapped = MLP.forward
    patch._install_mlp_bridge(module)
    assert MLP.forward is wrapped

    monkeypatch.setattr(
        int8_patch, "int8_mlp_bridge", lambda fc1, fc2, x: x + 2)
    value = torch.tensor(3)
    assert MLP().forward(value).item() == 5
    assert not calls

    monkeypatch.setattr(
        int8_patch, "int8_mlp_bridge", lambda fc1, fc2, x: None)
    assert MLP().forward(value).item() == 4
    assert calls == ["original"]


requires_dit_mps = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and os.environ.get("ASFP8_DIT_EXT") == "1"),
    reason="needs MPS + ASFP8_DIT_EXT=1",
)


@requires_dit_mps
def test_qkv_layout_fusion_matches_existing_rope_path_bit_exact():
    from _patches.dit_ext import loader

    kernel = loader.module()
    if kernel is None:
        pytest.skip("dit_ext kernel unavailable")

    torch.manual_seed(23)
    s, heads, dim, rotary_half = 37, 4, 128, 48
    qkv = torch.randn(
        s, 3 * heads * dim, device="mps", dtype=torch.bfloat16
    ).contiguous()
    angles = torch.randn(s, rotary_half, device="mps", dtype=torch.float32)
    c, sn = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((c, -sn, sn, c), dim=-1).reshape(
        1, s, 1, rotary_half, 2, 2
    ).to(torch.bfloat16).contiguous()
    qw = torch.randn(dim, device="mps", dtype=torch.bfloat16)
    kw = torch.randn(dim, device="mps", dtype=torch.bfloat16)

    old = qkv.clone()
    q0, k0, v0 = old.split(heads * dim, dim=-1)
    q0 = q0.view(1, s, heads, dim)
    k0 = k0.view(1, s, heads, dim)
    kernel.rms_rope_sh_partial(q0, k0, freqs, qw.float(), kw.float(), 1.0e-6)
    expected_q = q0.permute(0, 2, 1, 3).contiguous()
    expected_k = k0.permute(0, 2, 1, 3).contiguous()
    expected_v = v0.view(1, s, heads, dim).permute(0, 2, 1, 3).contiguous()

    actual_q, actual_k, actual_v = kernel.qkv_rms_rope_sh_partial(
        qkv, freqs, qw.float(), kw.float(), heads, dim, 1.0e-6
    )
    torch.mps.synchronize()
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)


@requires_dit_mps
def test_attention_layout_rope_matches_existing_qkv_kernel_bit_exact():
    from _patches.dit_ext import loader

    kernel = loader.module()
    if kernel is None or not hasattr(kernel, "rms_rope_attention_partial"):
        pytest.skip("attention-layout H3 kernel unavailable")

    torch.manual_seed(29)
    s, heads, dim, rotary_half = 43, 4, 128, 48
    qkv = torch.randn(
        s, 3 * heads * dim, device="mps", dtype=torch.bfloat16
    ).contiguous()
    angles = torch.randn(s, rotary_half, device="mps", dtype=torch.float32)
    c, sn = torch.cos(angles), torch.sin(angles)
    freqs = torch.stack((c, -sn, sn, c), dim=-1).reshape(
        1, s, 1, rotary_half, 2, 2
    ).to(torch.bfloat16).contiguous()
    qw = torch.randn(dim, device="mps", dtype=torch.bfloat16).float().contiguous()
    kw = torch.randn(dim, device="mps", dtype=torch.bfloat16).float().contiguous()

    expected_q, expected_k, expected_v = kernel.qkv_rms_rope_sh_partial(
        qkv.clone(), freqs, qw, kw, heads, dim, 1.0e-6)
    q_raw, k_raw, v_raw = qkv.split(heads * dim, dim=-1)
    actual_q = q_raw.view(1, s, heads, dim).permute(0, 2, 1, 3).contiguous()
    actual_k = k_raw.view(1, s, heads, dim).permute(0, 2, 1, 3).contiguous()
    actual_v = v_raw.view(1, s, heads, dim).permute(0, 2, 1, 3).contiguous()
    kernel.rms_rope_attention_partial(
        actual_q, actual_k, freqs, qw, kw, 1.0e-6)
    torch.mps.synchronize()

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert torch.equal(actual_v, expected_v)


@requires_dit_mps
def test_h3_fused_attention_releases_large_inputs_before_output_projection(
    monkeypatch,
):
    """Python references must not extend QKV/Q/K/V into o-projection."""
    refs = {}
    s, heads, dim = 17, 4, 128

    class Kernel:
        @staticmethod
        def qkv_rms_rope_sh_partial(qkv, *args):
            width = heads * dim
            outputs = tuple(
                qkv[:, index * width:(index + 1) * width]
                .reshape(1, s, heads, dim).transpose(1, 2).contiguous()
                for index in range(3)
            )
            refs["q"], refs["k"], refs["v"] = map(weakref.ref, outputs)
            return outputs

    class Projection:
        def __call__(self, x):
            result = torch.zeros(
                s, 3 * heads * dim, device=x.device, dtype=torch.bfloat16)
            refs["qkv"] = weakref.ref(result)
            return result

    class OutputProjection:
        def __call__(self, x):
            assert refs["qkv"]() is None
            assert refs["q"]() is None
            assert refs["k"]() is None
            assert refs["v"]() is None
            return x

    class Attention:
        def forward(self, *args, **kwargs):
            raise AssertionError("fallback should not run")

    def optimized_attention(q, k, v, head_count, **kwargs):
        # This test runs against the pre-container ComfyUI checkout; keep it
        # valid after an update as well.
        values = []
        for item in (q, k, v):
            values.append(item.take() if hasattr(item, "peek") else item)
        return values[2].clone()

    model_module = types.SimpleNamespace(
        Attention=Attention,
        optimized_attention=optimized_attention,
    )
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.cast_to = lambda value, **kwargs: value.to(
        device=kwargs["device"], dtype=kwargs["dtype"])
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setattr(patch, "_kernel", Kernel())
    monkeypatch.setattr(patch, "_attn_dead", False)
    patch._install_attention_fusion(model_module)

    instance = Attention()
    instance.heads = heads
    instance.head_dim = dim
    instance.qkv_proj = Projection()
    instance.q_norm = types.SimpleNamespace(
        weight=torch.ones(dim, device="mps", dtype=torch.bfloat16), eps=1e-6)
    instance.k_norm = types.SimpleNamespace(
        weight=torch.ones(dim, device="mps", dtype=torch.bfloat16), eps=1e-6)
    instance.out_proj = OutputProjection()
    x = torch.zeros(s, dim, device="mps", dtype=torch.bfloat16)
    rope = torch.zeros(
        1, s, 1, 48, 2, 2, device="mps", dtype=torch.bfloat16).contiguous()

    result = instance.forward(x, rope_freqs=rope, transformer_options={})
    assert result.shape == (s, heads * dim)


@requires_dit_mps
def test_int8_qkv_epilogue_skips_separate_norm_rope(monkeypatch):
    """A successful native epilogue must not run the old Q/K post-pass."""
    from _patches import int8_linear_kernel_mps as int8_patch

    calls = []
    s, heads, dim = 13, 4, 128

    class Kernel:
        @staticmethod
        def qkv_rms_rope_sh_partial(*args):
            raise AssertionError("dense QKV route must not run")

        @staticmethod
        def rms_rope_attention_partial(*args):
            raise AssertionError("separate norm/RoPE pass must not run")

    class Projection:
        def __call__(self, x):
            raise AssertionError("ordinary projection must not run")

    class OutputProjection:
        def __call__(self, x):
            return x

    class Attention:
        def forward(self, *args, **kwargs):
            raise AssertionError("fallback should not run")

    def fused_epilogue(layer, x, *args):
        calls.append("fused")
        return tuple(
            torch.full(
                (1, heads, s, dim), value, device="mps",
                dtype=torch.bfloat16)
            for value in (1.0, 2.0, 3.0)
        )

    def optimized_attention(q, k, v, head_count, **kwargs):
        values = [item.take() if hasattr(item, "peek") else item
                  for item in (q, k, v)]
        return values[2].clone()

    model_module = types.SimpleNamespace(
        Attention=Attention,
        optimized_attention=optimized_attention,
    )
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.cast_to = lambda value, **kwargs: value.to(
        device=kwargs["device"], dtype=kwargs["dtype"])
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setattr(patch, "_kernel", Kernel())
    monkeypatch.setattr(patch, "_attn_dead", False)
    monkeypatch.setattr(
        int8_patch, "int8_qkv_rms_rope_prequant", fused_epilogue)
    monkeypatch.setattr(
        int8_patch, "int8_qkv_prequant",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("layout-only QKV route must not run")))
    monkeypatch.setattr(int8_patch, "int8_o_proj_prequant", lambda *args: None)
    patch._install_attention_fusion(model_module)

    instance = Attention()
    instance.heads = heads
    instance.head_dim = dim
    instance.qkv_proj = Projection()
    instance.q_norm = types.SimpleNamespace(
        weight=torch.ones(dim, device="mps", dtype=torch.bfloat16), eps=1e-6)
    instance.k_norm = types.SimpleNamespace(
        weight=torch.ones(dim, device="mps", dtype=torch.bfloat16), eps=1e-6)
    instance.out_proj = OutputProjection()
    x = torch.zeros(s, dim, device="mps", dtype=torch.bfloat16)
    rope = torch.zeros(
        1, s, 1, 48, 2, 2, device="mps", dtype=torch.bfloat16).contiguous()

    result = instance.forward(x, rope_freqs=rope, transformer_options={})
    assert calls == ["fused"]
    assert result.shape == (s, heads * dim)
    assert torch.all(result == 3.0)


@requires_dit_mps
def test_gate_norm_mod_and_final_gate_match_eager_bf16():
    from _patches.dit_ext import loader

    kernel = loader.module()
    if kernel is None:
        pytest.skip("dit_ext kernel unavailable")

    torch.manual_seed(31)
    device = torch.device("mps")
    s, hidden, mods = 17, 5376, 3
    segments = [(0, 5, 1), (5, 12, 2), (12, 17, 0)]
    rows = patch._seg_rows(segments, s, device)

    def rand(*shape):
        return (torch.randn(*shape, dtype=torch.float32) * 0.25).to(
            device=device, dtype=torch.bfloat16
        ).contiguous()

    x = rand(s, hidden)
    attention = rand(s, hidden)
    gate1 = rand(mods, hidden)
    weight = (1.0 + rand(hidden) * 0.05).contiguous()
    scale = rand(mods, hidden)
    shift = rand(mods, hidden)

    expected_x = x.clone()
    for a, b, row in segments:
        expected_x[a:b].addcmul_(attention[a:b], gate1[row])
    expected_h = torch.nn.functional.rms_norm(
        expected_x, (hidden,), weight, eps=1.0e-6
    )
    for a, b, row in segments:
        expected_h[a:b].mul_(1.0 + scale[row]).add_(shift[row])

    actual_x, actual_h = kernel.gate_norm_mod_seg(
        x.clone(), attention, gate1, weight, scale, shift, rows, 1.0e-6
    )
    torch.mps.synchronize()
    assert torch.equal(actual_x.cpu(), expected_x.cpu())
    assert torch.equal(actual_h.cpu(), expected_h.cpu())

    mlp = rand(s, hidden)
    gate2 = rand(mods, hidden)
    expected_final = expected_x.clone()
    for a, b, row in segments:
        expected_final[a:b].addcmul_(mlp[a:b], gate2[row])
    actual_final = kernel.gate_residual_seg(actual_x.clone(), mlp, gate2, rows)
    torch.mps.synchronize()
    assert torch.equal(actual_final.cpu(), expected_final.cpu())


@requires_dit_mps
def test_final_layer_f32_graph_kernel_matches_eager_rounding():
    from _patches.dit_ext import loader

    kernel = loader.module()
    if kernel is None or not hasattr(kernel, "rms_mod_seg_f32"):
        pytest.skip("final-layer graph kernel unavailable")

    torch.manual_seed(37)
    device = torch.device("mps")
    rows_count, hidden, mods = 11, 128, 3

    def rand(*shape):
        return (torch.randn(*shape) * 0.25).to(
            device=device, dtype=torch.bfloat16).contiguous()

    x = rand(rows_count, hidden)
    weight = (1.0 + rand(hidden) * 0.05).contiguous()
    scale, shift = rand(mods, hidden), rand(mods, hidden)
    segments = [(0, 4, 2), (4, rows_count, 0)]
    rows = patch._seg_rows(segments, rows_count, device)

    expected = torch.nn.functional.rms_norm(
        x, (hidden,), weight, eps=1.0e-6)
    for start, stop, row in segments:
        expected[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    expected = expected.float()
    actual = kernel.rms_mod_seg_f32(
        x, weight, scale, shift, rows, 1.0e-6)
    torch.mps.synchronize()
    assert actual.dtype == torch.float32
    assert torch.equal(actual.cpu(), expected.cpu())
