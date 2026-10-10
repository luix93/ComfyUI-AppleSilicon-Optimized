"""SeedVR2 tile scheduling must not affect other forwards or leak scope."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from _patches import seedvr2_vae_mps as patch


def tensor(device, frames=1):
    return SimpleNamespace(device=SimpleNamespace(type=device), ndim=5,
                           shape=(1, 3, frames, 32, 32), dtype=patch.torch.float16)


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


@pytest.mark.parametrize("modern", [False, True])
def test_install_supports_both_vae_tile_interfaces(monkeypatch, modern):
    waits = []
    monkeypatch.setattr(patch.sys, "platform", "darwin")
    monkeypatch.setattr(patch, "_installed", False)
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_SYNC", "1")
    monkeypatch.setattr(patch.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch.torch.mps, "synchronize", lambda: waits.append("wait"))
    class Wrapper:
        def encode(self, x, device=None):
            return x
        def slicing_decode(self, x, output_buffer=None):
            return output_buffer if output_buffer is not None else x
    class LegacyDecoder:
        def decode_(self, x):
            return x
    class Block:
        def forward(self, input_tensor, *args):
            return input_tensor
    vae = ModuleType("comfy.ldm.seedvr.vae")
    vae.VideoAutoencoderKLWrapper = Wrapper
    vae.ResnetBlock3D = Block
    if not modern:
        vae.VideoAutoencoderKL = LegacyDecoder
    vae.tiled_vae = lambda x, model: model.encode(x) if modern else model.decode_(x)
    names = ("comfy", "comfy.ldm", "comfy.ldm.seedvr", "comfy.ldm.seedvr.vae")
    modules = [ModuleType(name) for name in names[:-1]] + [vae]
    for name, module in zip(names, modules):
        monkeypatch.setitem(sys.modules, name, module)
    for parent, child, name in zip(modules, modules[1:], names[1:]):
        setattr(parent, name.rsplit(".", 1)[-1], child)
    monkeypatch.setattr(patch.torch, "is_grad_enabled", lambda: True)
    patch.install()
    assert patch._installed
    model = Wrapper() if modern else LegacyDecoder()
    x = tensor("mps")
    assert vae.tiled_vae(x, model) is x
    assert waits == ["wait"]
    assert patch._tiled_model.get() is None
    if modern:
        destination = object()
        wrapped = patch._wrap_tiling(lambda x, model: model.slicing_decode(x, output_buffer=destination))
        assert wrapped(x, model) is destination
        assert waits == ["wait", "wait"]


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

def reviewed_block_policy(monkeypatch, driver):
    monkeypatch.setattr(patch.seedvr2_fast, "_current_vae_fusions", True)
    monkeypatch.setenv("ASFP8_SEEDVR2_FAST", "1")
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_BLOCK_BATCH", "4")
    monkeypatch.setattr(patch.torch.mps, "driver_allocated_memory", driver)


def test_bounded_video_blocks_flush_partial_tail_and_restore_scope(monkeypatch):
    waits=[]
    reviewed_block_policy(monkeypatch,lambda:10*2**30)
    monkeypatch.setattr(patch.torch.mps,"synchronize",lambda:waits.append(1))
    block=patch._wrap_video_block(lambda self,x:x)
    def tiled(x,model):
        for _ in range(5):block(model,x)
        return x
    x=tensor("mps",9)
    assert patch._wrap_tiling(tiled)(x,object()) is x
    assert waits==[1,1]
    assert patch._pending_blocks.get() is None


@pytest.mark.parametrize("initial,after",[(10,12.1),(41,41)])
def test_bounded_video_driver_budget_fences_before_batch_is_full(monkeypatch,initial,after):
    values=iter([initial,after,after])
    waits=[]
    reviewed_block_policy(monkeypatch,lambda:int(next(values)*2**30))
    monkeypatch.setattr(patch.torch.mps,"synchronize",lambda:waits.append(1))
    block=patch._wrap_video_block(lambda self,x:x)
    x=tensor("mps",9)
    assert patch._wrap_tiling(lambda x,model:block(model,x))(x,object()) is x
    assert waits==[1]
    assert patch._pending_blocks.get() is None


def test_bounded_video_tile_fence_releases_pending_work(monkeypatch):
    waits=[]
    reviewed_block_policy(monkeypatch,lambda:10*2**30)
    monkeypatch.setattr(patch.torch.mps,"synchronize",lambda:waits.append(1))
    block=patch._wrap_video_block(lambda self,x:x)
    def forward(model,x):
        for _ in range(3):block(model,x)
        return x
    tile=patch._wrap_tile_call(forward)
    x=tensor("mps",9)
    assert patch._wrap_tiling(lambda x,model:tile(model,x))(x,object()) is x
    assert waits==[1]


def test_bounded_video_exception_flushes_and_restores_scope(monkeypatch):
    waits=[]
    reviewed_block_policy(monkeypatch,lambda:10*2**30)
    monkeypatch.setattr(patch.torch.mps,"synchronize",lambda:waits.append(1))
    block=patch._wrap_video_block(lambda self,x:x)
    error=RuntimeError("original failure")
    def tiled(x,model):
        block(model,x)
        raise error
    with pytest.raises(RuntimeError) as captured:
        patch._wrap_tiling(tiled)(tensor("mps",9),object())
    assert captured.value is error
    assert waits==[1]
    assert patch._pending_blocks.get() is None
    assert patch._tiled_model.get() is None
    assert not patch._video_tiled.get()


@pytest.mark.parametrize("value,expected",[("1",1),("2",2),("4",4),("99",4),("bad",1)])
def test_bounded_video_batch_configuration_is_limited(monkeypatch,value,expected):
    reviewed_block_policy(monkeypatch,lambda:10*2**30)
    monkeypatch.setenv("ASFP8_SEEDVR2_VAE_BLOCK_BATCH",value)
    assert patch._block_batch_size()==expected
    monkeypatch.setattr(patch.seedvr2_fast,"_current_vae_fusions",False)
    assert patch._block_batch_size()==1


def test_bounded_policy_keeps_fp32_video_on_per_block_schedule(monkeypatch):
    waits=[]
    reviewed_block_policy(monkeypatch,lambda:pytest.fail("unexpected memory query"))
    monkeypatch.setattr(patch.torch.mps,"synchronize",lambda:waits.append(1))
    block=patch._wrap_video_block(lambda self,x:x)
    x=tensor("mps",9);x.dtype=patch.torch.float32
    assert patch._wrap_tiling(lambda x,model:block(model,x))(x,object()) is x
    assert waits==[1]
