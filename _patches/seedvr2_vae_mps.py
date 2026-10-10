"""Bound pending MPS work within SeedVR2 VAE video tiles and between tiles.

The tiled encoder can accumulate MPS/MPSGraph temporary allocations even when
live tensor memory stays flat. Waiting after each tile allows those buffers to
be reused. A video tile contains multiple temporal chunks, so it also needs
waits between residual blocks. Arithmetic, tile geometry, causal caches, and
non-tiled forwards are unchanged. The reviewed current-core M5 Ultra path can queue up to four
residual blocks, with early fences at 2 GiB driver growth or 40 GiB total.
ASFP8_SEEDVR2_VAE_BLOCK_BATCH=1 restores per-block waits.
ASFP8_SEEDVR2_VAE_SYNC=0 restores the original scheduling.
"""
from contextvars import ContextVar
from contextlib import nullcontext
from functools import wraps
import os
import sys

import torch

from . import seedvr2_fast

TAG = "[AppleSilicon-Optimized/seedvr2_vae]"
_installed = False
_tiled_model = ContextVar("asfp8_seedvr2_tiled_model", default=None)
_video_tiled = ContextVar("asfp8_seedvr2_video_tiled", default=False)
_pending_blocks = ContextVar("asfp8_seedvr2_pending_blocks", default=None)


def _block_batch_size():
    if not seedvr2_fast._current_vae_fusions or not seedvr2_fast._enabled("FAST"):
        return 1
    try:
        return min(4, max(1, int(os.environ.get("ASFP8_SEEDVR2_VAE_BLOCK_BATCH", "4"))))
    except ValueError:
        return 1


def _flush_blocks(state):
    torch.mps.synchronize()
    state["count"] = 0
    state["driver"] = torch.mps.driver_allocated_memory()



def _wrap_tiling(original):
    @wraps(original)
    def tiled(x, vae_model, *args, **kwargs):
        token = _tiled_model.set(vae_model if x.device.type == "mps" else None)
        video_token = _video_tiled.set(
            x.device.type == "mps" and x.ndim == 5 and x.shape[2] > 1
            and os.environ.get("ASFP8_SEEDVR2_VAE_VIDEO_SYNC", "1") == "1")
        limit = _block_batch_size() if (_video_tiled.get() and x.dtype == torch.float16) else 1
        state = ({"limit": limit, "count": 0, "driver": torch.mps.driver_allocated_memory()}
                 if limit > 1 else None)
        block_token = _pending_blocks.set(state)
        try:
            from .conv3d_mps import packing_scope
            pack = (x.device.type == 'mps' and not torch.is_grad_enabled()
                    and os.environ.get('ASFP8_SEEDVR2_FAST', '1') == '1'
                    and os.environ.get('ASFP8_SEEDVR2_VAE_PACK', '1') == '1')
            with packing_scope() if pack else nullcontext():
                return original(x, vae_model, *args, **kwargs)
        finally:
            try:
                if state is not None and state["count"]:
                    _flush_blocks(state)
            finally:
                _pending_blocks.reset(block_token)
                _video_tiled.reset(video_token)
                _tiled_model.reset(token)
    return tiled


def _wrap_tile_call(original):
    @wraps(original)
    def tile(self, x, *args, **kwargs):
        out = original(self, x, *args, **kwargs)
        if _tiled_model.get() is self and x.device.type == "mps":
            state = _pending_blocks.get()
            if state is None:
                torch.mps.synchronize()
            else:
                _flush_blocks(state)
        return out
    return tile


def _wrap_video_block(original):
    @wraps(original)
    def block(self, input_tensor, *args, **kwargs):
        out = original(self, input_tensor, *args, **kwargs)
        if _video_tiled.get() and input_tensor.device.type == "mps":
            state = _pending_blocks.get()
            if state is None:
                torch.mps.synchronize()
            else:
                state["count"] += 1
                driver = torch.mps.driver_allocated_memory()
                if (state["count"] >= state["limit"] or driver >= 40 * 2**30
                        or driver - state["driver"] >= 2 * 2**30):
                    _flush_blocks(state)
        return out
    return block


def install():
    global _installed
    if (_installed or sys.platform != "darwin"
            or os.environ.get("ASFP8_SEEDVR2_VAE_SYNC", "1") != "1"
            or not torch.backends.mps.is_available()):
        return
    try:
        import comfy.ldm.seedvr.vae as vae
    except ImportError as exc:
        print(f"{TAG} SeedVR2 VAE unavailable ({exc}); not installed.")
        return

    tiled = vae.tiled_vae
    encode = vae.VideoAutoencoderKLWrapper.encode
    if hasattr(vae, "VideoAutoencoderKL"):
        decoder, decode_name = vae.VideoAutoencoderKL, "decode_"
    else:
        # Current ComfyUI owns temporal slicing directly on the wrapper.
        decoder, decode_name = vae.VideoAutoencoderKLWrapper, "slicing_decode"
    decode = getattr(decoder, decode_name)
    vae.tiled_vae = _wrap_tiling(tiled)
    vae.VideoAutoencoderKLWrapper.encode = _wrap_tile_call(encode)
    setattr(decoder, decode_name, _wrap_tile_call(decode))
    vae.ResnetBlock3D.forward = _wrap_video_block(vae.ResnetBlock3D.forward)
    _installed = True
    video = (" and video residual blocks"
             if os.environ.get("ASFP8_SEEDVR2_VAE_VIDEO_SYNC", "1") == "1" else "")
    print(f"{TAG} MPS tiled encode/decode waits between tiles{video} "
          "to bound temporary memory; "
          "ASFP8_SEEDVR2_VAE_SYNC=0 restores original scheduling.")
