"""Unit coverage for the H3 Turbo LoRA loader and runtime adapter protocol."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from _patches import minimax_h3_turbo as turbo


def _adapter_state():
    state = {}
    for module in turbo.EXPECTED_MODULES:
        state[f"{module}.lora_A.weight"] = torch.zeros(1, 1)
        state[f"{module}.lora_B.weight"] = torch.zeros(1, 1)
    return state


def test_expected_released_h3_turbo_layout_is_complete():
    state = _adapter_state()
    assert len(turbo.EXPECTED_MODULES) == 208
    assert turbo._module_names(state) == set(turbo.EXPECTED_MODULES)


def test_standard_diffusion_model_prefix_is_normalized():
    state = {
        f"diffusion_model.{key}": value
        for key, value in _adapter_state().items()
    }
    assert turbo._module_names(state) == set(turbo.EXPECTED_MODULES)


def test_complete_full_adaln_layout_is_allowed_for_native_base_validation():
    state = _adapter_state()
    for module in turbo.ADALN_MODULES:
        state[f"{module}.lora_A.weight"] = torch.zeros(1, 1)
        state[f"{module}.lora_B.weight"] = torch.zeros(1, 1)
    assert turbo._module_names(state) == set(turbo.FULL_MODULES)


def test_partial_or_unknown_turbo_layout_is_rejected():
    state = _adapter_state()
    state.pop("blocks.0.attn.qkv_proj.lora_B.weight")
    with pytest.raises(ValueError, match="incomplete A/B pairs"):
        turbo._module_names(state)

    state = _adapter_state()
    state["not_h3.lora_A.weight"] = torch.zeros(1, 1)
    state["not_h3.lora_B.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="unsupported targets"):
        turbo._module_names(state)

    state = _adapter_state()
    state["not_h3.alpha"] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="without matching A/B pairs"):
        turbo._module_names(state)

    state = _adapter_state()
    module = next(iter(turbo.ADALN_MODULES))
    state[f"{module}.lora_A.weight"] = torch.zeros(1, 1)
    state[f"{module}.lora_B.weight"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="incomplete AdaLN target set"):
        turbo._module_names(state)


def test_generic_h3_layout_accepts_any_nonempty_supported_target_subset():
    modules = {
        "blocks.0.attn.qkv_proj",
        "blocks.17.mlp.fc1",
        "token_refiner.blocks.1.mlp.fc2",
        "final_layer.adaln_proj.linear",
    }
    state = {}
    for index, module in enumerate(sorted(modules), 1):
        state[f"diffusion_model.{module}.lora_A.weight"] = torch.zeros(index, 7)
        state[f"diffusion_model.{module}.lora_B.weight"] = torch.zeros(9, index)
        state[f"diffusion_model.{module}.alpha"] = torch.tensor(float(index))

    assert turbo._module_names(state, official=False) == modules

    with pytest.raises(ValueError, match="does not contain any"):
        turbo._module_names({}, official=False)


def test_generic_loader_accepts_arbitrary_rank_and_validates_model_shape(monkeypatch):
    class LoRAAdapter:
        def __init__(self, weights):
            self.weights = weights

    def load_lora(state, key_map, log_missing=False):
        loaded = {}
        for source, target_key in key_map.items():
            a_key = f"{source}.lora_A.weight"
            b_key = f"{source}.lora_B.weight"
            if a_key in state and b_key in state:
                alpha = state.get(f"{source}.alpha")
                if isinstance(alpha, torch.Tensor):
                    alpha = alpha.item()
                loaded[target_key] = LoRAAdapter((
                    state[b_key], state[a_key], alpha, None, None, None))
        return loaded

    def get_attr(obj, path):
        for name in path.split("."):
            obj = getattr(obj, name)
        return obj

    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.lora = ModuleType("comfy.lora")
    comfy.lora.load_lora = load_lora
    comfy.utils = ModuleType("comfy.utils")
    comfy.utils.get_attr = get_attr
    comfy.weight_adapter = ModuleType("comfy.weight_adapter")
    comfy.weight_adapter.LoRAAdapter = LoRAAdapter
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.lora", comfy.lora)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy.utils)
    monkeypatch.setitem(sys.modules, "comfy.weight_adapter", comfy.weight_adapter)

    target = "blocks.0.attn.qkv_proj"
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.qkv_proj = nn.Linear(7, 15, bias=False)
    dm = nn.Module()
    dm.blocks = nn.ModuleList([block])
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=dm))
    state = {
        f"{target}.lora_A.weight": torch.randn(3, 7),
        f"{target}.lora_B.weight": torch.randn(15, 3),
        f"{target}.alpha": torch.tensor(2.0),
    }

    loaded, key_map = turbo._load_adapters(model, state, {target})
    key = f"diffusion_model.{target}.weight"
    assert set(loaded) == {key}
    assert key_map == {target: key}
    assert loaded[key].weights[0].shape == (15, 3)
    assert loaded[key].weights[1].shape == (3, 7)

    bad = dict(state)
    bad[f"{target}.lora_B.weight"] = torch.randn(14, 3)
    with pytest.raises(ValueError, match="model expects"):
        turbo._load_adapters(model, bad, {target})


def test_generic_exact_runtime_node_is_registered():
    assert turbo.NODE_CLASS_MAPPINGS["AppleSiliconH3LoRA"] is turbo.AppleSiliconH3LoRA
    assert "Exact Runtime" in turbo.NODE_DISPLAY_NAME_MAPPINGS["AppleSiliconH3LoRA"]


def test_runtime_adapter_matches_linear_lora_and_reuses_base_output():
    torch.manual_seed(4)
    x = torch.randn(2, 3, 7)
    up = torch.randn(5, 2)
    down = torch.randn(2, 7)
    adapter = turbo.H3RuntimeLoRA(
        (up, down, 1.0, None, None, None), fast=True)
    adapter.multiplier = 0.75

    base = torch.randn(2, 3, 5)
    original_storage = base.data_ptr()
    reference = base.clone() + F.linear(F.linear(x, down), up) * (0.75 / 2.0)
    result = adapter.add_to(base, x)

    assert result.data_ptr() == original_storage
    torch.testing.assert_close(result, reference)


def test_runtime_adapter_bhsd_uses_standard_head_merge_order():
    torch.manual_seed(9)
    x = torch.randn(1, 3, 4, 2)
    up = torch.randn(6, 2)
    down = torch.randn(2, 6)
    adapter = turbo.H3RuntimeLoRA(
        (up, down, None, None, None, None), fast=True)
    base = torch.randn(1, 4, 6)
    expected = base.clone() + F.linear(
        F.linear(x.transpose(1, 2).reshape(1, 4, 6), down), up)

    result = adapter.add_bhsd_to(base, x)
    torch.testing.assert_close(result, expected)


def test_runtime_qkv_adapter_avoids_full_output_and_matches_fused_linear():
    torch.manual_seed(12)
    seq, heads, dim, rank, hidden = 5, 3, 2, 2, 7
    x = torch.randn(seq, hidden)
    up = torch.randn(3 * heads * dim, rank)
    down = torch.randn(rank, hidden)
    adapter = turbo.H3RuntimeLoRA(
        (up, down, 1.0, None, None, None), fast=True)
    q = torch.randn(1, heads, seq, dim)
    k = torch.randn(1, heads, seq, dim)
    v = torch.randn(1, heads, seq, dim)
    bases = tuple(item.clone() for item in (q, k, v))

    delta = F.linear(F.linear(x, down), up) * (1.0 / rank)
    expected = tuple(
        base + part.view(seq, heads, dim).permute(1, 0, 2).unsqueeze(0)
        for base, part in zip(bases, delta.split(heads * dim, dim=-1))
    )
    result = adapter.add_qkv_to(q, k, v, x, heads, dim)

    assert all(actual.data_ptr() == original.data_ptr()
               for actual, original in zip(result, (q, k, v)))
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference)
