"""Small real-ComfyUI injection tests; no model checkpoint is loaded."""
import os
import pytest
import torch
from torch import nn
from _patches import minimax_h3_turbo as h3

pytestmark = pytest.mark.skipif(os.environ.get("ASFP8_INT8_EXT") != "1", reason="requires ComfyUI runtime")


def model_and_loras():
    from comfy.model_patcher import ModelPatcher
    model = nn.Module()
    dm = model.diffusion_model = nn.Module()
    dm.blocks = nn.ModuleList([nn.Module() for _ in range(50)])
    dm.token_refiner = nn.Module()
    dm.token_refiner.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    dm.blocks[0].attn = nn.Module()
    dm.blocks[0].attn.qkv_proj = nn.Linear(7, 12, bias=False)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    target = "blocks.0.attn.qkv_proj"
    torch.manual_seed(61)
    states = [{f"{target}.lora_A.weight": torch.randn(rank, 7),
               f"{target}.lora_B.weight": torch.randn(12, rank),
               f"{target}.alpha": torch.tensor(float(rank + 1))} for rank in (2, 3)]
    return patcher, dm.blocks[0].attn.qkv_proj, states


@pytest.mark.parametrize("mode", ["fast_runtime", "reference_runtime"])
@pytest.mark.parametrize("count", [1, 2])
def test_real_patcher_stack_inject_eject_and_reinject(monkeypatch, mode, count):
    import comfy.model_management
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    model, layer, states = model_and_loras()
    entries = [(state, strength, False) for state, strength in zip(states[:count], (0.7, -0.2))]
    original_weight = layer.weight.detach().clone()
    original_forward = layer.forward
    x = torch.randn(5, 7)
    base = layer(x)
    expected = base.clone()
    for state, strength, _ in entries:
        down, up, alpha = state.values()
        expected.add_(torch.nn.functional.linear(torch.nn.functional.linear(x, down), up),
                      alpha=float(alpha) / down.shape[0] * strength)
    patched = h3.apply_h3_lora_stack(model, entries, mode)
    assert model.injections == {}
    assert not hasattr(layer, h3.RUNTIME_ATTR)
    for _ in range(2):
        patched.inject_model()
        assert torch.equal(layer(x), expected)
        patched.eject_model()
        assert layer.forward == original_forward
        assert not hasattr(layer, h3.RUNTIME_ATTR)
        assert torch.equal(layer(x), base)
        assert torch.equal(layer.weight, original_weight)
    with pytest.raises(ValueError, match="same H3 LoRA node"):
        h3.apply_h3_lora_stack(patched, entries, mode)


def test_merged_stack_keeps_ordered_model_patcher_updates():
    model, layer, states = model_and_loras()
    patched = h3.apply_h3_lora_stack(model, [(states[0], 0.7, False), (states[1], -0.2, False)], "merged")
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    assert model.patches == {}
    assert not patched.injections
    assert [patch[0] for patch in patched.patches[key]] == [0.7, -0.2]


def test_later_invalid_turbo_does_not_partially_attach_valid_adapter():
    model, layer, states = model_and_loras()
    with pytest.raises(ValueError, match="missing"):
        h3.apply_h3_lora_stack(model, [(states[0], 1, False), (states[1], 1, True)], "fast_runtime")
    assert not model.patches and not model.injections
    assert not hasattr(layer, h3.RUNTIME_ATTR)


def test_node_disabled_files_are_never_loaded_and_enabled_files_are_validated(monkeypatch):
    import json
    import folder_paths
    import comfy.utils
    model, layer, states = model_and_loras()
    loaded = []
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda kind: ["a.safetensors"])
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: name)
    def load(path, safe_load):
        assert safe_load
        loaded.append(path)
        return states[0]
    monkeypatch.setattr(comfy.utils, "load_torch_file", load)
    selected = [{"enabled": False, "lora_name": "missing"},
                {"lora_name": "missing", "strength": 0},
                {"lora_name": "a.safetensors", "strength": -0.3}]
    h3.AppleSiliconH3LoRA().apply(model, json.dumps(selected))
    assert loaded == ["a.safetensors"]
    with pytest.raises(ValueError, match="not in the LoRA folder"):
        h3.AppleSiliconH3LoRA().apply(model, '[{"lora_name":"../outside.safetensors"}]')
    assert loaded == ["a.safetensors"]


def test_failed_second_hook_rolls_back_the_first(monkeypatch):
    import comfy.model_management
    from comfy.weight_adapter.bypass import BypassForwardHook
    monkeypatch.setattr(comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    model, layer, states = model_and_loras()
    dm = model.model.diffusion_model
    dm.blocks[0].mlp = nn.Module()
    dm.blocks[0].mlp.fc1 = nn.Linear(7, 12, bias=False)
    second = {key.replace("attn.qkv_proj", "mlp.fc1"): value for key, value in states[0].items()}
    state = {**states[0], **second}
    patched = h3.apply_h3_lora_stack(model, [(state, 0.5, False), (state, 0.7, False)], "fast_runtime")
    inject = BypassForwardHook.inject
    def fail_second(hook):
        if hook.module is dm.blocks[0].mlp.fc1:
            raise RuntimeError("synthetic injection failure")
        return inject(hook)
    monkeypatch.setattr(BypassForwardHook, "inject", fail_second)
    original_forward = layer.forward
    with pytest.raises(RuntimeError, match="synthetic"):
        patched.inject_model()
    assert layer.forward == original_forward
    assert not hasattr(layer, h3.RUNTIME_ATTR)
    assert not hasattr(dm.blocks[0].mlp.fc1, h3.RUNTIME_ATTR)
