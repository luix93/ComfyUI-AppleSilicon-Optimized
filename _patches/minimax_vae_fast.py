"""MiniMax video VAE glue; existing FP16 GEMMs and attention stay the default.

The decoder's [head, qkv, dim] layout and FP16 rounding differ from the BF16
denoiser. These kernels are deliberately separate from minimax_fast. Native
mode enables the patch; ASFP8_MINIMAX_VAE_FAST=0 restores the core forwards.
ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1 independently opts into non-bit-exact
dense FlashAttention. The explicit ASFP8_MINIMAX_RELAXED preset also enables
it unless overridden with zero; native mode alone never enables it.
Loading the INT8 ConvRot VAE selects a separate native W8A8 projection path;
ASFP8_MINIMAX_VAE_INT8=0 restores the previous quantized decoder forwards.
"""
from __future__ import annotations

import functools
import inspect
import os
import sys

import torch

from . import minimax_vae_int8

TAG = "[AppleSilicon-Optimized/minimax_vae_fast]"
_installed = False
_kernel = None
_attention_dead = False
_ff_dead = False
_merge = True
_dense = False
_dense_dead = False
_dense_kernel = None
_dense_tuning = False
_qkv_kernel = None
_qkv_dead = False
_vm = None
_ops = None
_original_attention = None
_original_ff = None


def _half_mps(t):
    return (isinstance(t, torch.Tensor) and t.device.type == "mps"
            and t.dtype == torch.float16 and t.is_contiguous())

def _input_ok(x):
    return (not torch.is_grad_enabled() and _half_mps(x) and x.ndim == 3
            and x.shape[0] > 0 and x.shape[1] > 0 and x.shape[2] == 2048)

def _plain_linear(layer):
    weight = getattr(layer, "weight", None)
    return (isinstance(weight, torch.Tensor) and weight.dtype == torch.float16
            and not isinstance(weight, _ops.QuantizedTensor))

def _supported_linear(layer):
    return _plain_linear(layer) or minimax_vae_int8.eligible(layer)

def _linear_input_act(layer, x, input_act, *args, **kwargs):
    if not _plain_linear(layer):
        result = minimax_vae_int8.linear(layer, x, input_act, *args, **kwargs)
        if result is not None:
            return result
    return _ops.linear_input_act(layer, x, input_act, *args, **kwargs)

def _attention_ok(self, x, rope):
    return (not _attention_dead and _kernel is not None and _input_ok(x)
            and self.heads == 32 and self.dim_head == 64
            and _supported_linear(self.to_qkv) and _supported_linear(self.to_out)
            and _half_mps(self.qk_norm_scale) and self.qk_norm_scale.shape == (64,)
            and _half_mps(rope) and rope.ndim == 6
            and rope.shape[:3] == (*x.shape[:2], 1)
            and 0 < rope.shape[3] <= 32 and rope.shape[4:] == (2, 2))

def _run_attention(q, k, v, heads):
    """Dense FP16 FlashAttention is separately opt-in and changes output bytes."""
    global _dense_dead
    if _dense and not _dense_dead and _dense_kernel is not None:
        try:
            options = {}
            if _dense_tuning:
                options = dict(nsg=4, bc=32, blk=2) if q.shape[0] == 1 else dict(nsg=4, bc=64, blk=1)
            out = _dense_kernel.flash_fwd_na(q, k, v, 64 ** -0.5, **options)
            if not _merge:
                out = out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * 64)
            return out
        except Exception as exc:
            _dense_dead = True
            print(f"{TAG} optional dense attention disabled ({str(exc)[:180]}); "
                  "restoring the exact attention path.")
    return _vm.optimized_attention(q, k, v, heads, skip_reshape=True,
                                   skip_output_reshape=_merge)


def _qkv_layer_ok(layer):
    # Preserve lazy materialization, LoRA functions, hooks and custom forwards.
    supported = (torch.nn.Linear, _ops.disable_weight_init.Linear, _ops.manual_cast.Linear)
    return (type(layer) in supported and "forward" not in vars(layer)
            and not hasattr(layer, "_v")
            and not any(getattr(layer, name, ()) for name in (
                "weight_function", "bias_function", "_forward_hooks", "_forward_pre_hooks")))


def _project_qkv(self, x, rope, pre_norm):
    """Return None to retain the exact projection and QKV glue."""
    global _qkv_dead
    if (_qkv_kernel is None or _qkv_dead or not _qkv_layer_ok(self.to_qkv)
            or not _half_mps(self.to_qkv.weight)
            or self.to_qkv.weight.shape != (6144, 2048)
            or not _half_mps(self.to_qkv.bias)
            or self.to_qkv.bias.shape != (6144,)):
        return None
    try:
        h = _ops._eager_input_act(x, "rms_norm", pre_norm.weight, pre_norm.eps)
        return _qkv_kernel(h, self.to_qkv.weight, self.to_qkv.bias, rope,
                           self.qk_norm_scale, self.norm_q.eps)
    except Exception as exc:
        _qkv_dead = True
        print(f"{TAG} optional QKV projection disabled ({str(exc)[:180]}); "
              "restoring the exact projection.")
        return None


def _attention(self, x, rotary_pos_emb, pre_norm, residual, residual_scale):
    global _attention_dead
    if not _attention_ok(self, x, rotary_pos_emb):
        return _original_attention(self, x, rotary_pos_emb, pre_norm, residual, residual_scale)
    projected = _project_qkv(self, x, rotary_pos_emb, pre_norm)
    if projected is None:
        h = _linear_input_act(self.to_qkv, x, "rms_norm", pre_norm.weight, pre_norm.eps)
        try:
            projected = _kernel.qkv(h, rotary_pos_emb, self.qk_norm_scale, self.heads, self.norm_q.eps)
        except Exception as exc:
            _attention_dead = True
            print(f"{TAG} QKV fusion disabled ({str(exc)[:180]}).")
            return _original_attention(self, x, rotary_pos_emb, pre_norm, residual, residual_scale)
    q, k, v = projected
    out = _run_attention(q, k, v, self.heads)
    if _merge:
        if _half_mps(out) and out.shape == q.shape:
            try:
                out = _kernel.merge_sanitize(out)
            except Exception as exc:
                # The attention result is still valid; no attention/GEMM is repeated.
                _attention_dead = True
                print(f"{TAG} output fusion disabled ({str(exc)[:180]}).")
                out = torch.nan_to_num(out.transpose(1, 2).reshape(*x.shape))
        else:
            out = torch.nan_to_num(out.transpose(1, 2).reshape(*x.shape))
    else:
        out = torch.nan_to_num(out)
    return _linear_input_act(self.to_out, out, None,
                                residual=residual, residual_scale=residual_scale)

def _feed_forward(self, x, pre_norm, residual, residual_scale):
    global _ff_dead
    if (_ff_dead or _kernel is None or not _input_ok(x)
            or not _supported_linear(self.w1) or not _supported_linear(self.w2)):
        return _original_ff(self, x, pre_norm, residual, residual_scale)
    h = _linear_input_act(self.w1, x, "rms_norm", pre_norm.weight, pre_norm.eps)
    if not _half_mps(h) or h.ndim != 3 or h.shape[-1] != 16384:
        return _linear_input_act(self.w2, h, "swiglu",
                                    residual=residual, residual_scale=residual_scale)
    try:
        activated = _kernel.swiglu(h)
    except Exception as exc:
        _ff_dead = True
        print(f"{TAG} SwiGLU fusion disabled ({str(exc)[:180]}).")
        return _linear_input_act(self.w2, h, "swiglu",
                                    residual=residual, residual_scale=residual_scale)
    return _linear_input_act(self.w2, activated, None,
                                residual=residual, residual_scale=residual_scale)

def _compatible(vm, rope_impl):
    """Reject older forwards and the older two-rounded-products rotary formula."""
    try:
        if tuple(inspect.signature(vm.Attention.forward).parameters) != (
                "self", "x", "rotary_pos_emb", "pre_norm", "residual", "residual_scale"):
            return False
        if tuple(inspect.signature(vm.FeedForward.forward).parameters) != (
                "self", "x", "pre_norm", "residual", "residual_scale"):
            return False
        attn_names = set(vm.Attention.forward.__code__.co_names)
        rope_names = set(rope_impl.apply_rope_split_half1.__code__.co_names)
        return ({"linear_input_act", "qk_norm_scale", "nan_to_num", "optimized_attention"}
                <= attn_names and {"diagonal", "addcmul_"} <= rope_names)
    except (AttributeError, TypeError, ValueError):
        return False

def install():
    global _installed, _kernel, _vm, _ops, _original_attention, _original_ff, _merge
    global _dense, _dense_kernel, _qkv_kernel, _dense_tuning
    if (_installed or sys.platform != "darwin" or os.environ.get("ASFP8_DIT_EXT") != "1"
            or os.environ.get("ASFP8_MINIMAX_VAE_FAST", "1") != "1"
            or not torch.backends.mps.is_available()):
        return
    from comfy.ldm.minimax import vae as vm
    import comfy.ops as ops
    from comfy_kitchen.backends.eager import rope
    if getattr(vm.Attention.forward, "_asfp8_vae_fp16", False):
        _installed = True
        return
    if not _compatible(vm, rope):
        print(f"{TAG} unrecognized decoder/rotary API; keeping core forwards.")
        return
    from .minimax_vae_ext import loader
    kernel = loader.module()
    if kernel is None:
        return
    _vm, _ops, _kernel = vm, ops, kernel
    _merge = os.environ.get("ASFP8_MINIMAX_VAE_MERGE", "1") == "1"
    # Native mode enables exact kernels only. Numerical relaxation requires
    # an explicit per-feature or relaxed-preset switch.
    from .minimax_relaxed import vae_dense_enabled, vae_qkv_enabled, vae_tuning_enabled
    _dense = vae_dense_enabled()
    _dense_tuning = vae_tuning_enabled()
    if _dense:
        from . import attn_router_mps
        candidate = attn_router_mps._fa41
        if (attn_router_mps._FA41_NA and not attn_router_mps._na_dead
                and hasattr(candidate, "flash_fwd_na")):
            _dense_kernel = candidate
        else:
            _dense = False
            print(f"{TAG} optional dense attention unavailable; keeping exact attention.")
    if vae_qkv_enabled() and hasattr(ops, "_eager_input_act"):
        from .minimax_vae_qkv import Qkv
        _qkv_kernel = Qkv((64, 256, 4))
        print(f"{TAG} optional FP32-accumulating FP16 QKV projection enabled; "
              "ASFP8_MINIMAX_VAE_QKV=0 and restart restores the exact projection.")
    _original_attention, _original_ff = vm.Attention.forward, vm.FeedForward.forward
    attention = functools.wraps(_original_attention)(_attention)
    attention._asfp8_vae_fp16 = True
    attention._asfp8_original = _original_attention
    vm.Attention.forward = attention
    if os.environ.get("ASFP8_MINIMAX_VAE_SWIGLU", "1") == "1":
        ff = functools.wraps(_original_ff)(_feed_forward)
        ff._asfp8_original = _original_ff
        vm.FeedForward.forward = ff
    _installed = True
    if _dense:
        print(f"{TAG} optional NON-BIT-EXACT dense FP16 VAE attention enabled; "
              "set ASFP8_MINIMAX_VAE_DENSE_ATTENTION=0 and restart for exact attention.")
    else:
        print(f"{TAG} FP16 decoder QK norm/RoPE/layout and elementwise fusions enabled; "
              "attention backend and tiling preserved.")
