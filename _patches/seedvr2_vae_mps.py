"""Bound pending MPS work between SeedVR2 VAE spatial tiles.

The tiled encoder can accumulate MPS/MPSGraph temporary allocations even when
live tensor memory stays flat. Waiting after each tile allows those buffers to
be reused. Arithmetic, tile geometry, and non-tiled forwards are unchanged.
ASFP8_SEEDVR2_VAE_SYNC=0 restores the original scheduling.
"""
from contextvars import ContextVar
from contextlib import nullcontext
from functools import wraps
import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/seedvr2_vae]"
_installed = False
_tiled_model = ContextVar("asfp8_seedvr2_tiled_model", default=None)


def _wrap_tiling(original):
    @wraps(original)
    def tiled(x, vae_model, *args, **kwargs):
        token = _tiled_model.set(vae_model if x.device.type == "mps" else None)
        try:
            from .conv3d_mps import packing_scope
            pack = (x.device.type == 'mps' and not torch.is_grad_enabled()
                    and os.environ.get('ASFP8_SEEDVR2_FAST', '1') == '1'
                    and os.environ.get('ASFP8_SEEDVR2_VAE_PACK', '1') == '1')
            with packing_scope() if pack else nullcontext():
                return original(x, vae_model, *args, **kwargs)
        finally:
            _tiled_model.reset(token)
    return tiled


def _wrap_tile_call(original):
    @wraps(original)
    def tile(self, x, *args, **kwargs):
        out = original(self, x, *args, **kwargs)
        if _tiled_model.get() is self and x.device.type == "mps":
            torch.mps.synchronize()
        return out
    return tile


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
    decode = vae.VideoAutoencoderKL.decode_
    vae.tiled_vae = _wrap_tiling(tiled)
    vae.VideoAutoencoderKLWrapper.encode = _wrap_tile_call(encode)
    vae.VideoAutoencoderKL.decode_ = _wrap_tile_call(decode)
    _installed = True
    print(f"{TAG} MPS tiled encode/decode waits between tiles to bound temporary memory; "
          "ASFP8_SEEDVR2_VAE_SYNC=0 restores original scheduling.")
