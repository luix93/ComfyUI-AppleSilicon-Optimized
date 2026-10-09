"""SeedVR2 tile scheduling must not affect other forwards or leak scope."""
from types import SimpleNamespace

import pytest

from _patches import seedvr2_vae_mps as patch


def tensor(device, frames=1):
    return SimpleNamespace(device=SimpleNamespace(type=device), ndim=5,
                           shape=(1, 3, frames, 32, 32))


def test_tiled_mps_preserves_arguments_results_and_tile_count(monkeypatch):
    calls = []
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: calls.append("wait"))
    model = object()
    x = tensor("mps")
    result = object()
    def original(self, value, *, option):
        assert self is model and value is x and option == 17
        calls.append("tile")
        return result
    tile = patch._wrap_tile_call(original)
    def tiled(value, selected, *, repeats):
        return [tile(selected, value, option=17) for _ in range(repeats)]
    assert patch._wrap_tiling(tiled)(x, model, repeats=3) == [result] * 3
    assert calls == ["tile", "wait"] * 3
    assert patch._tiled_model.get() is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_other_devices_never_wait(monkeypatch, device):
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: pytest.fail("unexpected wait"))
    x = tensor(device)
    model = object()
    tile = patch._wrap_tile_call(lambda self, x: x)
    tiled = patch._wrap_tiling(lambda x, model: tile(model, x))
    assert tiled(x, model) is x


def test_non_tiled_and_unrelated_models_do_not_wait(monkeypatch):
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: pytest.fail("unexpected wait"))
    x = tensor("mps")
    model, unrelated = object(), object()
    tile = patch._wrap_tile_call(lambda self, x: x)
    assert tile(model, x) is x
    tiled = patch._wrap_tiling(lambda x, model: tile(unrelated, x))
    assert tiled(x, model) is x


def test_exception_restores_context_and_preserves_error(monkeypatch):
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: pytest.fail("unexpected wait"))
    error = RuntimeError("original failure")
    def fail(*args):
        raise error
    tiled = patch._wrap_tiling(patch._wrap_tile_call(fail))
    with pytest.raises(RuntimeError) as captured:
        tiled(tensor("mps"), object())
    assert captured.value is error
    assert patch._tiled_model.get() is None


def test_nested_tiling_restores_outer_model(monkeypatch):
    waits = []
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: waits.append(patch._tiled_model.get()))
    outer, inner = object(), object()
    x = tensor("mps")
    tile = patch._wrap_tile_call(lambda self, x: x)
    def nested(x, model):
        patch._wrap_tiling(lambda x, model: tile(model, x))(x, inner)
        return tile(model, x)
    assert patch._wrap_tiling(nested)(x, outer) is x
    assert waits == [inner, outer]
    assert patch._tiled_model.get() is None


def test_disabled_and_repeated_install_return_early(monkeypatch):
    monkeypatch.setattr(patch.sys, "platform", "darwin")
    monkeypatch.setattr(patch, "_installed", False)
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_SYNC", "0")
    monkeypatch.setattr(patch.torch.backends.mps, "is_available", lambda: pytest.fail("unexpected probe"))
    patch.install()
    assert not patch._installed
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_SYNC", "1")
    monkeypatch.setattr(patch, "_installed", True)
    patch.install()


@pytest.mark.parametrize("device,frames,enabled,expected", [
    ("mps", 121, "1", 2), ("mps", 1, "1", 0),
    ("mps", 121, None, 2),
    ("mps", 121, "0", 0), ("cpu", 121, "1", 0),
    ("cuda", 121, "1", 0),
])
def test_video_blocks_wait_only_in_enabled_mps_video_scope(
        monkeypatch, device, frames, enabled, expected):
    waits = []
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: waits.append(1))
    if enabled is None:
        monkeypatch.delenv("ASFP8_SEEDVR2_VAE_VIDEO_SYNC", raising=False)
    else:
        monkeypatch.setenv("ASFP8_SEEDVR2_VAE_VIDEO_SYNC", enabled)
    x = tensor(device, frames)
    # Later blocks can have one latent frame; the outer video scope owns policy.
    latent = tensor(device, 1)
    result = object()
    def original(self, value, *, cache):
        assert value is latent and cache == {"causal": 7}
        return result
    block = patch._wrap_video_block(original)
    def tiled(value, model):
        assert block(model, latent, cache={"causal": 7}) is result
        return block(model, input_tensor=latent, cache={"causal": 7})
    assert patch._wrap_tiling(tiled)(x, object()) is result
    assert len(waits) == expected
    assert not patch._video_tiled.get()
    assert block(object(), latent, cache={"causal": 7}) is result
    assert len(waits) == expected


def test_video_scope_restores_after_nested_image_and_exception(monkeypatch):
    waits = []
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: waits.append(1))
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_VIDEO_SYNC", "1")
    block = patch._wrap_video_block(lambda self, x: x)
    def inner(x, model):
        assert not patch._video_tiled.get()
        return block(model, x)
    def outer(x, model):
        patch._wrap_tiling(inner)(tensor("mps"), model)
        assert patch._video_tiled.get()
        block(model, x)
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        patch._wrap_tiling(outer)(tensor("mps", 121), object())
    assert waits == [1]
    assert not patch._video_tiled.get()
    assert patch._tiled_model.get() is None
