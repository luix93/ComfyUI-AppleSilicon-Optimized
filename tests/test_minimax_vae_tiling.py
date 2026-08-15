"""Tests for the opt-in approximate MiniMax H3 VAE tiling controls."""

import pytest

from _patches import minimax_vae_tiling as patch


def test_no_environment_request_is_a_noop(monkeypatch):
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_TILE_SIZE", raising=False)
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_OVERLAP", raising=False)
    assert patch._requested_tiling() is None


def test_overlap_only_preserves_reference_tile_size(monkeypatch):
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_TILE_SIZE", raising=False)
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_OVERLAP", "32")
    assert patch._requested_tiling() == (256, 32)


@pytest.mark.parametrize("value", ["0", "17", "not-an-int", "1024"])
def test_invalid_overlap_is_rejected(monkeypatch, value):
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_TILE_SIZE", raising=False)
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_OVERLAP", value)
    with pytest.raises(ValueError):
        patch._requested_tiling()


def test_overlap_must_be_smaller_than_tile(monkeypatch):
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_TILE_SIZE", "128")
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_OVERLAP", "128")
    with pytest.raises(ValueError, match="smaller"):
        patch._requested_tiling()


def test_tile_sizes_above_validated_h3_limit_are_rejected(monkeypatch):
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_TILE_SIZE", "336")
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_OVERLAP", "64")
    with pytest.raises(ValueError, match=r"\[64, 320\]"):
        patch._requested_tiling()
