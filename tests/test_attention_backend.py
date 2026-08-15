"""Workflow attention backend and ComfyUI ownership compatibility tests."""

import sys
import types

import pytest

from _patches import attention_backend as backend
from _patches import attn_router_mps as router
from _patches import minimax_h3_sparse as sparse


class _FakeModel:
    def __init__(self, options=None):
        self.model_options = {"transformer_options": dict(options or {})}
        self.clones = 0

    def clone(self):
        clone = _FakeModel(self.model_options["transformer_options"])
        clone.clones = self.clones + 1
        return clone


class _FakeDiT:
    def __init__(self):
        self.blocks = list(range(50))


_FakeDiT.__module__ = "comfy.ldm.minimax.model"


class _FakeH3Model(_FakeModel):
    def __init__(self, options=None):
        super().__init__(options)
        self.model = type("_Inner", (), {"diffusion_model": _FakeDiT()})()
        self.wrappers = []

    def clone(self):
        clone = _FakeH3Model(self.model_options["transformer_options"])
        clone.clones = self.clones + 1
        return clone

    def add_wrapper_with_key(self, *args):
        self.wrappers.append(args)


class _Container:
    def __init__(self, value):
        self.value = value

    def peek(self):
        if self.value is None:
            raise RuntimeError("consumed")
        return self.value

    def take(self):
        value = self.peek()
        self.value = None
        return value


def _dense(*args, **kwargs):
    return args, kwargs


def test_current_mode_clones_without_changing_options():
    marker = object()
    model = _FakeModel({"keep": marker})
    result = backend.apply_attention_backend(model, backend.BACKEND_CURRENT)
    assert result is not model
    assert result.clones == 1
    assert result.model_options["transformer_options"] == {"keep": marker}


def test_selected_backend_records_dense_fallback_and_override(monkeypatch):
    monkeypatch.setattr(backend, "_backend_function", lambda name: _dense)
    model = _FakeModel()
    result = backend.apply_attention_backend(model, backend.BACKEND_PYTORCH)
    options = result.model_options["transformer_options"]

    assert options[router.DENSE_BACKEND_KEY] is _dense
    assert options[router.DENSE_BACKEND_NAME_KEY] == backend.BACKEND_PYTORCH
    override = options["optimized_attention_override"]
    assert backend.is_our_dense_override(override)
    args, kwargs = override(None, "q", "k", "v", 8, scale=0.5)
    assert args == ("q", "k", "v", 8)
    assert kwargs["scale"] == 0.5


def test_container_function_is_propagated_to_model_override(monkeypatch):
    consumed = []

    def dense(*args, **kwargs):
        raise AssertionError("tensor entry should not be used")

    def containers(q, k, v, heads, **kwargs):
        values = (q.take(), k.take(), v.take())
        consumed.extend(values)
        return values, heads

    dense.container_function = containers
    monkeypatch.setattr(backend, "_backend_function", lambda name: dense)
    result = backend.apply_attention_backend(
        _FakeModel(), backend.BACKEND_APPLE)
    override = result.model_options["transformer_options"][
        "optimized_attention_override"]
    q, k, v = (_Container(name) for name in "qkv")

    assert override.container_function(q, k, v, 4) == (("q", "k", "v"), 4)
    assert consumed == ["q", "k", "v"]
    assert all(item.value is None for item in (q, k, v))


def test_unknown_override_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(backend, "_backend_function", lambda name: _dense)
    model = _FakeModel({"optimized_attention_override": object()})
    with pytest.raises(ValueError, match="another attention override"):
        backend.apply_attention_backend(model, backend.BACKEND_PYTORCH)
    assert model.clones == 0


def test_backend_after_sparse_keeps_sparse_outermost(monkeypatch):
    monkeypatch.setattr(backend, "_backend_function", lambda name: _dense)
    model = _FakeModel({
        sparse.STATE_KEY: object(),
        "optimized_attention_override": sparse.sparse_attention_override,
    })
    result = backend.apply_attention_backend(model, backend.BACKEND_PYTORCH)
    options = result.model_options["transformer_options"]
    assert options["optimized_attention_override"] is sparse.sparse_attention_override
    assert options[router.DENSE_BACKEND_KEY] is _dense


def test_sparse_state_does_not_legalize_an_unrelated_override(monkeypatch):
    monkeypatch.setattr(backend, "_backend_function", lambda name: _dense)
    model = _FakeModel({
        sparse.STATE_KEY: object(),
        "optimized_attention_override": object(),
    })
    with pytest.raises(ValueError, match="another attention override"):
        backend.apply_attention_backend(model, backend.BACKEND_PYTORCH)


def test_backend_before_sparse_is_composed_instead_of_rejected(monkeypatch):
    monkeypatch.setattr(backend, "_backend_function", lambda name: _dense)
    monkeypatch.setattr(sparse, "_kernel", lambda: object())
    monkeypatch.setattr(sparse.torch.backends.mps, "is_available", lambda: True)

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    patcher_extension = types.ModuleType("comfy.patcher_extension")
    patcher_extension.WrappersMP = type("WrappersMP", (), {
        "OUTER_SAMPLE": "outer_sample",
        "DIFFUSION_MODEL": "diffusion_model",
    })
    comfy.patcher_extension = patcher_extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(
        sys.modules, "comfy.patcher_extension", patcher_extension)

    selected = backend.apply_attention_backend(
        _FakeH3Model(), backend.BACKEND_PYTORCH)
    result = sparse.apply_h3_sparse(
        selected, mode="corrected", tau=1.0, start_percent=0.5,
        dense_leading_blocks=2, min_sequence_tokens=8192, verbose=False)
    options = result.model_options["transformer_options"]
    assert sparse.STATE_KEY in options
    assert options[router.DENSE_BACKEND_KEY] is _dense
    assert backend.is_our_dense_override(
        options["optimized_attention_override"])
    assert len(result.wrappers) == 2


def test_sparse_dense_fallback_uses_workflow_selected_backend(monkeypatch):
    calls = []

    def selected(q, k, v, heads, **kwargs):
        calls.append((q, k, v, heads, kwargs))
        return "selected-dense"

    monkeypatch.setattr(router, "_installed", True)
    monkeypatch.setattr(
        router, "_router",
        lambda *args, **kwargs: pytest.fail("global router should not win"),
    )
    options = {
        "optimized_attention_override": sparse.sparse_attention_override,
        router.DENSE_BACKEND_KEY: selected,
    }
    result = router.dense_without_override(
        None, "q", "k", "v", 8,
        transformer_options=options, skip_reshape=True)

    assert result == "selected-dense"
    assert len(calls) == 1
    forwarded = calls[0][4]["transformer_options"]
    assert "optimized_attention_override" not in forwarded
    assert options["optimized_attention_override"] is sparse.sparse_attention_override


def test_sparse_container_entry_consumes_once_and_declines_to_selected(monkeypatch):
    calls = []

    def selected(q, k, v, heads, **kwargs):
        calls.append((q, k, v, heads))
        return "dense"

    options = {router.DENSE_BACKEND_KEY: selected}
    q, k, v = (_Container(name) for name in "qkv")
    result = sparse.sparse_attention_override.container_function(
        q, k, v, 8, transformer_options=options)

    assert result == "dense"
    assert calls == [("q", "k", "v", 8)]
    assert all(item.value is None for item in (q, k, v))


def test_node_schema_exposes_all_exact_backend_choices():
    required = backend.AppleSiliconAttentionBackend.INPUT_TYPES()["required"]
    assert tuple(required["backend"][0]) == backend.BACKENDS
    assert backend.BACKEND_CURRENT == required["backend"][1]["default"]
    assert "comfy_kitchen" not in " ".join(backend.BACKENDS)


def test_new_comfyui_registry_receives_apple_backend(monkeypatch):
    registered = {}
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    ldm = types.ModuleType("comfy.ldm")
    ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules")
    modules.__path__ = []
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.register_attention_function = registered.__setitem__
    for name, module in (
        ("comfy", comfy),
        ("comfy.ldm", ldm),
        ("comfy.ldm.modules", modules),
        ("comfy.ldm.modules.attention", attention),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(backend, "_registered", False)
    monkeypatch.setattr(backend, "_mps_available", lambda: True)

    backend.install()
    assert registered["apple_silicon_mps"] is router.workflow_attention
