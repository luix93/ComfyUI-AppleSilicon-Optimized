"""Exact Krea2 BF16 memory-pass fusions, validated on M5 Ultra / torch 2.13/2.14.

Retains the original projections, attention arithmetic and BF16 rounding.
RMSNorm uses PyTorch's FP32 reduction order while reading BF16 directly.
"""
import os

import torch

TAG = "[AppleSilicon-Optimized/krea2_glue]"
_kernel = None
_installed = False
_original_block = _original_mlp = _original_last = None
_norm_type = _norm_forward = None


def _enabled():
    return _kernel is not None and os.environ.get("ASFP8_KREA2_GLUE", "1") == "1"


def _bf16(x):
    return (type(x) is torch.Tensor and x.device.type == "mps"
            and x.dtype == torch.bfloat16 and x.is_contiguous()
            and 0 < x.numel() < 2**32 and not torch.is_grad_enabled())


def _norm_ok(norm, width):
    return (type(norm) is _norm_type
            and getattr(norm.forward, "__func__", None) is _norm_forward
            and not norm._forward_hooks and not norm._forward_pre_hooks
            and norm.scale.numel() == width)


def _norm_mod(x, norm, scale, shift):
    weight = norm.scale.to(device=x.device, dtype=torch.float32).contiguous()
    return _kernel.norm_mod(x, weight, scale.contiguous(), shift.contiguous(), norm.eps, True)


def _gate_residual(value, gate, residual):
    if _bf16(value) and value.shape == residual.shape:
        return _kernel.point(value, gate.contiguous(), residual, 0)
    return residual + gate * value


def _block(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
    if (not _enabled() or not _bf16(x) or x.dim() != 3 or x.shape[0] != 1
            or x.shape[-1] != 6144 or timestep_zero_index is not None
            or vec.shape != (1, 1, 6 * 6144) or not _bf16(vec)
            or not _norm_ok(self.prenorm, 6144) or not _norm_ok(self.postnorm, 6144)):
        return _original_block(self, x, vec, freqs, mask, timestep_zero_index,
                               transformer_options=transformer_options)
    sc, sh, gate, sc2, sh2, gate2 = self.mod(vec)
    # Custom modulation hooks can change the geometry/dtype; preserve those
    # results while using the original arithmetic if they do.
    if any(t.numel() != 6144 or t.dtype != x.dtype or t.device != x.device
           for t in (sc, sh, gate, sc2, sh2, gate2)):
        x = x + gate * self.attn((1 + sc) * self.prenorm(x) + sh, freqs, mask,
                                transformer_options=transformer_options)
        return x + gate2 * self.mlp((1 + sc2) * self.postnorm(x) + sh2)
    attn = self.attn(_norm_mod(x, self.prenorm, sc, sh), freqs, mask,
                     transformer_options=transformer_options)
    x = _gate_residual(attn, gate, x)
    return _gate_residual(self.mlp(_norm_mod(x, self.postnorm, sc2, sh2)), gate2, x)


def _mlp(self, x):
    if not _enabled() or not _bf16(x):
        return _original_mlp(self, x)
    gate, up = self.gate(x), self.up(x)
    if _bf16(gate) and _bf16(up) and gate.shape == up.shape:
        act = _kernel.point(gate, up, up, 3)
    else:
        act = torch.nn.functional.silu(gate).mul_(up)
    return self.down(act)


def _last(self, x, tvec):
    if (not _enabled() or not _bf16(x) or x.dim() != 3 or x.shape[0] != 1
            or x.shape[-1] != 6144 or not _norm_ok(self.norm, 6144)
            or tvec.shape != (1, 1, 6144) or not _bf16(tvec)):
        return _original_last(self, x, tvec)
    sc, sh = self.modulation(tvec)
    if any(t.numel() != 6144 or t.dtype != x.dtype or t.device != x.device for t in (sc, sh)):
        return self.linear((1 + sc) * self.norm(x) + sh)
    return self.linear(_norm_mod(x, self.norm, sc, sh))


def attention_gate(out, gate):
    """Return fused BSHD-layout gated attention, or None for the eager tail."""
    if (not _enabled() or not _bf16(out) or not _bf16(gate)
            or out.dim() != 4 or gate.dim() != 3):
        return None
    b, h, s, d = out.shape
    if gate.shape != (b, s, h * d):
        return None
    return _kernel.point(out, gate, gate, 1)


def install():
    global _kernel, _installed, _original_block, _original_mlp, _original_last
    global _norm_type, _norm_forward
    if _installed or os.environ.get("ASFP8_KREA2_GLUE", "1") != "1":
        return
    # This reduction matches the tested PyTorch release and GPU. Other
    # versions/devices retain the existing Krea2 implementation.
    if not torch.__version__.split("+")[0].startswith(("2.13.", "2.14.")):
        return
    from .krea2_ext.loader import module
    kernel = module()
    if kernel is None or "M5 Ultra" not in kernel.device_name():
        return
    import comfy.ldm.krea2.model as krea
    _original_block, _original_mlp, _original_last = (
        krea.SingleStreamBlock.forward, krea.SwiGLU.forward, krea.LastLayer.forward)
    _norm_type, _norm_forward = krea.RMSNorm, krea.RMSNorm.forward
    _kernel = kernel
    krea.SingleStreamBlock.forward = _block
    krea.SwiGLU.forward = _mlp
    krea.LastLayer.forward = _last
    _installed = True
    print(f"{TAG} exact normalization/modulation, residual, SwiGLU and attention-layout fusions enabled.")
