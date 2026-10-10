"""Automatic native defaults retain platform gates and explicit user choices."""
import pytest

from _patches import _common
from _patches import torch_mpp_qcache as qc


@pytest.mark.parametrize("platform,mps", [("linux", False), ("win32", False), ("darwin", False)])
def test_non_mps_default_is_unchanged(monkeypatch, platform, mps):
    monkeypatch.setattr(_common.sys, "platform", platform)
    monkeypatch.setattr(_common.torch.backends.mps, "is_available", lambda: mps)
    env = {}
    assert _common.apply_native_master_switch(env) == []
    assert env == {}


def test_macos_mps_enables_native_without_launch_flags(monkeypatch):
    monkeypatch.setattr(_common.sys, "platform", "darwin")
    monkeypatch.setattr(_common.torch.backends.mps, "is_available", lambda: True)
    env = {"ASFP8_DIT_EXT": "0", "ASFP8_INT8_EXT": "1"}
    implied = _common.apply_native_master_switch(env)
    assert env["ASFP8_NATIVE"] == "1"
    assert env["ASFP8_DIT_EXT"] == "0"
    assert "ASFP8_DIT_EXT" not in implied
    assert "ASFP8_INT8_EXT" not in implied
    assert set(implied) == set(_common.NATIVE_IMPLIES) - {"ASFP8_DIT_EXT", "ASFP8_INT8_EXT"}
    assert _common.apply_native_master_switch(env) == []


def test_native_zero_is_not_on():
    env = {"ASFP8_NATIVE": "0"}
    assert _common.apply_native_master_switch(env) == []
    assert "ASFP8_INT8_EXT" not in env


def test_native_implies_all_optins():
    env = {"ASFP8_NATIVE": "1"}
    implied = _common.apply_native_master_switch(env)
    assert set(implied) == set(_common.NATIVE_IMPLIES)
    for var in _common.NATIVE_IMPLIES:
        assert env[var] == "1"


def test_explicit_zero_wins_over_master():
    env = {"ASFP8_NATIVE": "1", "ASFP8_DIT_EXT": "0"}
    implied = _common.apply_native_master_switch(env)
    assert env["ASFP8_DIT_EXT"] == "0"          # kill switch preserved
    assert "ASFP8_DIT_EXT" not in implied
    assert env["ASFP8_INT8_EXT"] == "1"          # the rest still implied


def test_explicit_one_not_reported_as_implied():
    env = {"ASFP8_NATIVE": "1", "ASFP8_INT8_EXT": "1"}
    implied = _common.apply_native_master_switch(env)
    assert "ASFP8_INT8_EXT" not in implied       # was already set by the user
    assert env["ASFP8_INT8_EXT"] == "1"


def _reviewed_stack(monkeypatch, supported=True, hardware=True):
    monkeypatch.setattr(_common.sys, "platform", "darwin")
    monkeypatch.setattr(_common.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(qc, "_torch_build_supported", lambda: supported)
    monkeypatch.setattr(qc, "_hardware", lambda: hardware)


def test_reviewed_attention_defaults_and_idempotence(monkeypatch):
    _reviewed_stack(monkeypatch)
    env = {}
    _common.apply_native_master_switch(env)
    assert _common.apply_default_attention_backend(env) == ["ASFP8_ATTN_BACKEND", "MTLFLASHATTN_SDPA"]
    assert env["ASFP8_ATTN_BACKEND"] == "pytorch"
    assert env["MTLFLASHATTN_SDPA"] == "off"
    assert _common.apply_default_attention_backend(env) == []
    assert "ASFP8_MINIMAX_RELAXED" not in env


@pytest.mark.parametrize("supported,hardware", [(False, True), (True, False)])
def test_unreviewed_stack_keeps_existing_attention(monkeypatch, supported, hardware):
    _reviewed_stack(monkeypatch, supported, hardware)
    env = {"ASFP8_NATIVE": "1"}
    assert _common.apply_default_attention_backend(env) == []
    assert env == {"ASFP8_NATIVE": "1"}


@pytest.mark.parametrize("overrides", [
    {"ASFP8_NATIVE": "0"}, {"ASFP8_ATTN_BACKEND": ""}, {"ASFP8_ATTN_BACKEND": "pytorch"},
    {"ASFP8_ATTN_BACKEND": "comfy_default"},
])
def test_explicit_master_or_backend_keeps_attention_settings(monkeypatch, overrides):
    _reviewed_stack(monkeypatch)
    env = {"ASFP8_NATIVE": "1", **overrides}
    original = dict(env)
    assert _common.apply_default_attention_backend(env) == []
    assert env == original


@pytest.mark.parametrize("flag", ["MTLFLASHATTN_SDPA", "APPLESILICON_FP8_SDPA"])
def test_explicit_sdpa_setting_survives_automatic_backend(monkeypatch, flag):
    _reviewed_stack(monkeypatch)
    env = {"ASFP8_NATIVE": "1", flag: "on"}
    assert _common.apply_default_attention_backend(env) == ["ASFP8_ATTN_BACKEND"]
    assert env[flag] == "on"
    if flag == "APPLESILICON_FP8_SDPA":
        assert "MTLFLASHATTN_SDPA" not in env


@pytest.mark.parametrize("platform,mps", [("linux", True), ("darwin", False)])
def test_attention_default_is_macos_mps_only(monkeypatch, platform, mps):
    _reviewed_stack(monkeypatch)
    monkeypatch.setattr(_common.sys, "platform", platform)
    monkeypatch.setattr(_common.torch.backends.mps, "is_available", lambda: mps)
    env = {"ASFP8_NATIVE": "1"}
    assert _common.apply_default_attention_backend(env) == []
