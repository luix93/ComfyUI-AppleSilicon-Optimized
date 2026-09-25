"""Raw checkpoint compatibility must never replace weights or conditioning."""
import hashlib

import pytest
import torch

from _patches import seedvr2_loader as patch


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_raw_model_gets_exact_upstream_buffers_and_keeps_weight_objects(prefix):
    weight = torch.randn(3, 5, dtype=torch.float16)
    state = {prefix + "txt_in.weight": weight}
    calls = []
    def original(config, sd, key_prefix):
        calls.append((config, sd, key_prefix))
        assert sd[prefix + "txt_in.weight"] is weight
        for key in patch._SHAPES:
            assert sd[prefix + key].dtype == torch.bfloat16
        return "matched"
    config = dict(patch._CONFIG)
    assert patch._wrap_match(original)(config, state, prefix) == "matched"
    assert calls == [(config, state, prefix)]
    hashes = {"positive_conditioning": "92ff754e036702f20f1782b0f5327c3d120aac6370ddc2ef0cdfdb06988f70ab",
              "negative_conditioning": "f7279b7df52f3eb2e290019528c9794a5d02999c804f02f44a2e2a0d029c5410"}
    for key, expected in hashes.items():
        assert hashlib.sha256(state[prefix + key].view(torch.uint8).numpy().tobytes()).hexdigest() == expected


@pytest.mark.parametrize("keys", [list(patch._SHAPES), ["positive_conditioning"], ["negative_conditioning"]])
def test_existing_or_partial_conditioning_untouched(monkeypatch, keys):
    monkeypatch.setattr(patch, "_conditioning", lambda: pytest.fail("unexpected asset access"))
    state = {key: object() for key in keys}
    before = dict(state)
    result = patch._wrap_match(lambda config, sd, prefix: sd)(patch._CONFIG, state)
    assert result is state and state == before


@pytest.mark.parametrize("config", [{}, {**patch._CONFIG, "image_model": "flux"},
                                      {**patch._CONFIG, "vid_dim": 2560},
                                      {**patch._CONFIG, "mlp_type": "swiglu"}])
def test_other_architectures_untouched(monkeypatch, config):
    monkeypatch.setattr(patch, "_conditioning", lambda: pytest.fail("unexpected asset access"))
    state = {"weight": object()}
    assert patch._wrap_match(lambda c, sd, p: sd)(config, state) is state
    assert list(state) == ["weight"]


def test_config_only_probe_and_upstream_exception_are_preserved(monkeypatch):
    monkeypatch.setattr(patch, "_conditioning", lambda: pytest.fail("unexpected asset access"))
    error = ValueError("upstream")
    def fail(config, state, prefix):
        assert state is None
        raise error
    with pytest.raises(ValueError) as captured:
        patch._wrap_match(fail)(patch._CONFIG)
    assert captured.value is error


def test_damaged_asset_fails_before_mutating_state(monkeypatch, tmp_path):
    bad = tmp_path / "conditioning.safetensors"
    bad.write_bytes(b"damaged")
    monkeypatch.setattr(patch, "_ASSET", bad)
    state = {"weight": object()}
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        patch._wrap_match(lambda *args: pytest.fail("unexpected loader call"))(patch._CONFIG, state)
    assert list(state) == ["weight"]


def test_disabled_install(monkeypatch):
    monkeypatch.setattr(patch, "_installed", False)
    monkeypatch.setenv("ASFP8_SEEDVR2_LOADER", "0")
    patch.install()
    assert not patch._installed
