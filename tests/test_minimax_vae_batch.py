"""Batched MiniMax H3 VAE tile decode must be exactly the reference decode.

The decoder itself is batch-independent; what these tests pin is the restructure
— that tiles are gathered in the same order, grouped correctly, and fed to the
same blending pass — using a stand-in decoder whose output depends on its input,
so any mis-ordering or mis-grouping changes the canvas.
"""

import math

import pytest
import torch

from _patches import minimax_vae_batch as batch_patch


class _FakeVAE:
    """Reproduces the reference tiling/blending contract, minus the weights."""

    vae_ratio = 8

    def __init__(self, tile_size=256, overlap=64):
        self.tile_size = tile_size
        self.tile_overlap_min = overlap
        self.calls = []

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []
        n = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (n - 1)
            remaining = tile_size * n - sum(overlaps) - input_len
            if remaining < 0:
                n += 1
            else:
                break
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (n - 1)] += self.vae_ratio
        start = [0]
        for i in range(n - 1):
            start.append(start[-1] + tile_size - overlaps[i])
        return start, [tile_size] * n, overlaps

    def _decode_pixels(self, z):
        """Content-dependent and batch-independent, like the real decoder."""
        self.calls.append(tuple(z.shape))
        return z.repeat_interleave(self.vae_ratio, -2).repeat_interleave(
            self.vae_ratio, -1) * 2.0 + 1.0

    @staticmethod
    def blend(a, b, overlap, dim):
        weight = torch.linspace(0, 1, overlap, device=b.device, dtype=b.dtype)
        shape = [1] * b.dim()
        shape[dim] = overlap
        weight = weight.view(shape)
        merged = b.clone()
        if dim == -2:
            merged[..., :overlap, :] = a * (1 - weight) + b[..., :overlap, :] * weight
        else:
            merged[..., :, :overlap] = a * (1 - weight) + b[..., :, :overlap] * weight
        return merged

    # Legacy core tiling contract; modern core uses already blended neighbours.
    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        canvas = None
        row_tails = []
        out_y = 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            new_tails = []
            left_tail = None
            out_x = 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi:zi + zl, zj:zj + zw])
                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i]:, :].clone())
                next_left_tail = (tile[..., :, -x_overlap[j]:].clone()
                                  if j < len(x_idx) - 1 else None)
                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail
                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width,
                                         dtype=tile.dtype, device=tile.device)
                canvas[..., out_y:out_y + tile.shape[-2],
                       out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas


def _latent(height_px, width_px, frames=2, channels=3):
    torch.manual_seed(0)
    return torch.randn(1, channels, frames,
                       height_px // _FakeVAE.vae_ratio,
                       width_px // _FakeVAE.vae_ratio)


@pytest.mark.parametrize("overlap,batch", [
    (64, 2), (64, 4), (64, 16), (32, 2), (32, 8),
])
def test_batched_decode_is_identical_to_the_reference(overlap, batch):
    z = _latent(480, 864)
    reference = _FakeVAE(overlap=overlap).tiled_decode(z)
    got = batch_patch.batched_tiled_decode(_FakeVAE(overlap=overlap), z, batch)
    assert torch.equal(got, reference)


def test_batch_of_one_matches_and_makes_the_same_number_of_calls():
    z = _latent(480, 864)
    ref_vae, bat_vae = _FakeVAE(), _FakeVAE()
    reference = ref_vae.tiled_decode(z)
    got = batch_patch.batched_tiled_decode(bat_vae, z, 1)
    assert torch.equal(got, reference)
    assert len(bat_vae.calls) == len(ref_vae.calls) == 15   # 3x5 grid at 864x480


def test_batching_reduces_the_number_of_decoder_launches():
    z = _latent(480, 864)
    vae = _FakeVAE()
    batch_patch.batched_tiled_decode(vae, z, 4)
    assert len(vae.calls) == 4                    # 15 tiles -> ceil(15/4)
    assert [c[0] for c in vae.calls] == [4, 4, 4, 3]


def test_a_single_tile_grid_is_handled():
    """tile_size >= input_len returns one tile of the real length, not tile_size."""
    z = _latent(128, 128)
    vae = _FakeVAE()
    reference = _FakeVAE().tiled_decode(z)
    got = batch_patch.batched_tiled_decode(vae, z, 8)
    assert torch.equal(got, reference)
    assert len(vae.calls) == 1


def test_non_square_and_ragged_grids_match():
    for h, w in ((480, 864), (512, 512), (720, 1280), (256, 1024)):
        z = _latent(h, w)
        reference = _FakeVAE().tiled_decode(z)
        got = batch_patch.batched_tiled_decode(_FakeVAE(), z, 6)
        assert torch.equal(got, reference), (h, w)


def test_env_parsing_rejects_nonsense(monkeypatch):
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_DECODE_BATCH", "nope")
    with pytest.raises(ValueError, match="must be an integer"):
        batch_patch._requested_batch()
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_DECODE_BATCH", "0")
    with pytest.raises(ValueError, match=r"must be in \[1, 64\]"):
        batch_patch._requested_batch()
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_DECODE_BATCH")
    assert batch_patch._requested_batch() == 1


def test_install_is_a_noop_without_the_variable(monkeypatch):
    monkeypatch.delenv("ASFP8_MINIMAX_VAE_DECODE_BATCH", raising=False)
    batch_patch._installed = False
    batch_patch.install()
    assert batch_patch._installed is False


@pytest.mark.parametrize("samples,batch", [(1, 2), (2, 4), (2, 8)])
def test_current_row_keeps_sample_groups_and_partial_last_batch(samples, batch):
    vae = _FakeVAE()
    z = _latent(256, 864).repeat(samples, 1, 1, 1, 1)
    z[-1] += 0.3
    idx, lengths, _ = vae.split_tiles(864)
    expected = [vae._decode_pixels(z[..., p // vae.vae_ratio:(p + n) // vae.vae_ratio])
                for p, n in zip(idx, lengths)]
    actual = list(batch_patch.batched_tile_row(vae, z, idx, lengths, batch))
    assert len(actual) == len(expected)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_current_core_install_preserves_native_blending(monkeypatch):
    import sys
    import types
    class Current:
        tiled_decode = object()
        _decode_tile_row = lambda *_: None
    tiled = Current.tiled_decode
    vae_module = types.SimpleNamespace(MiniMaxH3VideoVAE=Current)
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax", types.SimpleNamespace(vae=vae_module))
    monkeypatch.setenv("ASFP8_MINIMAX_VAE_DECODE_BATCH", "8")
    monkeypatch.setattr(batch_patch, "_installed", False)
    batch_patch.install()
    assert Current.tiled_decode is tiled
    assert Current._decode_tile_row._asfp8_batched
