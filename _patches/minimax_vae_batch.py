"""Optional MiniMax H3 VAE tile batching (ASFP8_MINIMAX_VAE_DECODE_BATCH).

Current ComfyUI already batches tile rows. We only change the row batch size,
retaining core's current overlap blending and streamed output. Older cores
without the row API use the original whole-grid adapter below.

Tile coordinates and blend arithmetic are preserved. Batch shapes may change
backend arithmetic, so byte equality is validated on measured configurations,
not assumed for every device and dtype. This flag remains off by default;
larger batches increase decoder activation memory and need not be faster.
"""

from __future__ import annotations

import os

import torch


TAG = "[AppleSilicon-Optimized/minimax_vae_batch]"

_installed = False


def _requested_batch() -> int:
    raw = os.environ.get("ASFP8_MINIMAX_VAE_DECODE_BATCH")
    if raw is None:
        return 1
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"ASFP8_MINIMAX_VAE_DECODE_BATCH must be an integer, got {raw!r}"
        ) from exc
    if value < 1 or value > 64:
        raise ValueError(
            f"ASFP8_MINIMAX_VAE_DECODE_BATCH must be in [1, 64], got {value}")
    return value


def batched_tiled_decode(self, z, batch: int):
    """Same tiles, same blending, fewer decoder launches."""
    height = z.shape[-2] * self.vae_ratio
    width = z.shape[-1] * self.vae_ratio
    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    # every tile in the grid is the same size, so the whole grid stacks
    parts = []
    for i_pos, i_len in zip(y_idx, y_len):
        zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
        for j_pos, j_len in zip(x_idx, x_len):
            zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
            parts.append(z[..., zi:zi + zl, zj:zj + zw])

    decoded = []
    for start in range(0, len(parts), batch):
        group = parts[start:start + batch]
        if len(group) == 1:
            decoded.append(self._decode_pixels(group[0]))
            continue
        out = self._decode_pixels(torch.cat(group, dim=0))
        decoded.extend(out[k:k + 1] for k in range(out.shape[0]))

    # blending is the reference implementation, reading precomputed tiles
    canvas = None
    row_tails = []
    out_y = 0
    cursor = 0
    for i in range(len(y_idx)):
        new_tails = []
        left_tail = None
        out_x = 0
        for j in range(len(x_idx)):
            tile = decoded[cursor]
            cursor += 1
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


def batched_tile_row(self, z_row, x_idx, x_len, batch: int):
    """Feed current core's blend loop without replacing its corner semantics."""
    for start in range(0, len(x_idx), batch):
        parts = [
            z_row[..., pos // self.vae_ratio:(pos + length) // self.vae_ratio]
            for pos, length in zip(x_idx[start:start + batch], x_len[start:start + batch])
        ]
        yield from self._decode_pixels(torch.cat(parts, dim=0)).chunk(len(parts), dim=0)


def install():
    global _installed
    if _installed:
        return
    try:
        batch = _requested_batch()
    except ValueError as exc:
        print(f"{TAG} invalid decode-batch request ({exc}); disabled.")
        return
    if batch <= 1:
        return

    try:
        from comfy.ldm.minimax import vae as minimax_vae
    except Exception as exc:
        print(f"{TAG} MiniMax H3 VAE unavailable ({exc!r}); not installed.")
        return

    cls = minimax_vae.MiniMaxH3VideoVAE
    if hasattr(cls, "_decode_tile_row"):
        original_row = cls._decode_tile_row
        if not getattr(original_row, "_asfp8_batched", False):
            def row(self, z_row, x_idx, x_len):
                yield from batched_tile_row(self, z_row, x_idx, x_len, batch)
            row._asfp8_batched = True
            row._asfp8_original = original_row
            cls._decode_tile_row = row
        _installed = True
        print(f"{TAG} current core tile-row batching set to {batch}; "
              "native overlap blending retained (experimental scheduling).")
        return

    original = cls.tiled_decode
    if getattr(original, "_asfp8_batched", False):
        _installed = True
        return

    def wrapped(self, z):
        try:
            return batched_tiled_decode(self, z, batch)
        except Exception as exc:      # never lose a decode to this
            print(f"{TAG} batched decode failed ({str(exc)[:200]}); "
                  "falling back to the reference tile loop.")
            return original(self, z)

    wrapped._asfp8_batched = True
    wrapped._asfp8_original = original
    cls.tiled_decode = wrapped
    _installed = True
    print(f"{TAG} MiniMax H3 VAE tiles decoded {batch} at a time "
          f"(exact; scheduling only, peak decoder memory scales with the batch).")
