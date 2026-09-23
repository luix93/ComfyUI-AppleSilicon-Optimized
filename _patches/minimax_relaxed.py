"""Explicit, H3-scoped numerical relaxations; never implied by native mode.

ASFP8_MINIMAX_RELAXED=1 selects the measured relaxed preset. Individual
settings override it, including explicit zero/exact. Startup settings only;
cache, sparse attention, token reduction and VAE tiling remain independent.
"""
import functools
import os
import sys

TAG = "[AppleSilicon-Optimized/minimax_relaxed]"
_installed = False


def vae_dense_enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("ASFP8_MINIMAX_VAE_DENSE_ATTENTION",
                   env.get("ASFP8_MINIMAX_RELAXED", "0")) == "1"


def vae_qkv_enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("ASFP8_MINIMAX_VAE_QKV",
                   env.get("ASFP8_MINIMAX_RELAXED", "0")) == "1"


def vae_tuning_enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("ASFP8_MINIMAX_VAE_ATTN_TUNING",
                   env.get("ASFP8_MINIMAX_RELAXED", "0")) == "1"


def attention_mode(environ=None):
    env = os.environ if environ is None else environ
    mode = env.get("ASFP8_MINIMAX_ATTN_MODE",
                   "tuned" if env.get("ASFP8_MINIMAX_RELAXED") == "1" else "exact")
    if mode not in ("exact", "tail", "tuned"):
        raise ValueError("ASFP8_MINIMAX_ATTN_MODE must be exact, tail or tuned")
    return mode


def _eligible(q, k, v, heads, args, kwargs):
    import torch
    from .attn_router_mps import _is_attention_container
    # Preserve explicit workflow backends and sparse/third-party overrides.
    options = kwargs.get("transformer_options") or {}
    supported = {"mask", "attn_precision", "skip_reshape", "skip_output_reshape",
                 "transformer_options", "scale"}
    if (args or set(kwargs) - supported or heads != 56 or kwargs.get("mask") is not None
            or kwargs.get("attn_precision") is not None
            or not kwargs.get("skip_reshape", False)
            or "optimized_attention_override" in options
            or "asfp8_dense_attention_backend" in options
            or torch.is_grad_enabled()):
        return False
    values = tuple(t.peek() if _is_attention_container(t) else t for t in (q, k, v))
    if not all(isinstance(t, torch.Tensor) for t in values):
        return False
    first = values[0]
    return (first.ndim == 4 and first.shape[0] > 0 and first.shape[1] == 56
            and first.shape[2] >= 8192 and first.shape[3] == 128
            and all(t.device.type == "mps" and t.device == first.device
                    and t.dtype == torch.bfloat16 and t.shape == first.shape
                    for t in values))


def _make_attention(original, kernel, mode="tail"):
    from .attn_router_mps import _take_attention_inputs
    dead = False

    @functools.wraps(original)
    def relaxed_attention(q, k, v, heads, *args, **kwargs):
        nonlocal dead
        if dead or not _eligible(q, k, v, heads, args, kwargs):
            return original(q, k, v, heads, *args, **kwargs)
        q, k, v, _ = _take_attention_inputs(q, k, v)
        scale = kwargs.get("scale")
        scale = 128 ** -0.5 if scale is None else float(scale)
        try:
            options = dict(nsg=8, bc=96, blk=1) if mode == "tuned" else {}
            out = kernel.flash_fwd_na(q, k, v, scale, **options)
        except Exception as exc:
            dead = True
            print(f"{TAG} {mode} kernel disabled ({str(exc)[:180]}); restoring exact attention.")
            # Containers were consumed; fallback receives their intact tensors.
            return original(q, k, v, heads, *args, **kwargs)
        if not kwargs.get("skip_output_reshape", False):
            out = out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * 128)
        return out

    relaxed_attention._asfp8_minimax_relaxed = True
    relaxed_attention._asfp8_original = original
    return relaxed_attention


def install():
    global _installed
    if (_installed or sys.platform != "darwin" or attention_mode() == "exact"
            or os.environ.get("ASFP8_DIT_EXT") != "1"
            or os.environ.get("ASFP8_MINIMAX_FAST", "1") != "1"):
        return
    from . import attn_router_mps, minimax_fast
    if (not minimax_fast._installed or not attn_router_mps._FA41_NA
            or attn_router_mps._na_dead or attn_router_mps._fa41 is None):
        return
    from comfy.ldm.minimax import model
    original = model.optimized_attention
    if getattr(original, "_asfp8_minimax_relaxed", False):
        _installed = True
        return
    from .mfa4_ext import relaxed_loader
    mode = attention_mode()
    kernel = relaxed_loader.module(mode)
    if kernel is None:
        return
    model.optimized_attention = _make_attention(original, kernel, mode)
    _installed = True
    print(f"{TAG} NON-BIT-EXACT long H3 {mode} attention enabled; "
          "ASFP8_MINIMAX_ATTN_MODE=exact and restart restores the exact route.")
