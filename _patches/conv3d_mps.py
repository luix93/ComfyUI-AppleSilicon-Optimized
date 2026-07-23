"""Patch #26 — Conv3d as sliced conv2d on MPS (opt-in ASFP8_CONV3D=1).

MPS has no tuned 3D convolution: torch's conv3d lowers to an MPSGraph path
that runs ~9-14x slower than the mathematically identical sum of per-time-
slice 2D convolutions (measured M5 Max at LTX video-VAE decoder shapes,
every dtype: e.g. 64c/121f/512px: 1951 ms conv3d vs 229 ms decomposed).
Video VAEs (LTX, Wan, Hunyuan, Mochi) and latent upsamplers are almost pure
conv3d, so this dominates their decode time.

    out[:, :, t] = sum_kt conv2d(x[:, :, t*sT + kt*dT - pT], W[:, :, kt])

The decomposition is exact up to accumulation order (partials add in the
conv's dtype; measured relerr ~1e-3 of activation std in fp16, tighter in
fp32). Frames fold into the conv2d batch dim, chunked to respect MPSGraph's
INT_MAX element limit. Bias folds into the last slice's conv2d call.

Patched at comfy.ops.disable_weight_init.Conv3d._conv_forward, so every
comfy conv3d user (manual_cast included) gets it; non-MPS inputs and exotic
cases (transposed, output_padding) fall through to the original.
Kill switch: ASFP8_CONV3D=0 (ASFP8_NATIVE=1 implies on).
"""

import os
import sys

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-Optimized/conv3d]"

_installed = False
_orig_conv_forward = None

# MPSGraph rejects tensors over INT_MAX elements; keep conv2d batches under
# a comfortable ceiling (elements per activation tensor).
_MAX_ELEMS = 1 << 30


def _conv3d_decomposed(input, weight, bias, stride, padding, dilation, groups):
    B, Ci, T, H, W = input.shape
    Co, _, kT, kH, kW = weight.shape
    sT, sH, sW = stride
    pT, pH, pW = padding
    dT, dH, dW = dilation

    if pT:
        input = F.pad(input, (0, 0, 0, 0, pT, pT))
    T_in = input.shape[2]
    T_out = (T_in - (kT - 1) * dT - 1) // sT + 1
    if T_out <= 0:
        return None  # degenerate; let the original path raise/handle

    Ho = (H + 2 * pH - (kH - 1) * dH - 1) // sH + 1
    Wo = (W + 2 * pW - (kW - 1) * dW - 1) // sW + 1

    # Folded input is kT*Ci wide; frames fold into conv2d's batch dim TIMES
    # the real batch B, so B belongs in the per-frame footprint (B>1 would
    # otherwise overshoot the element budget by exactly B).
    per_frame = B * max(kT * Ci * H * W, Co * Ho * Wo)
    chunk = max(1, min(T_out, _MAX_ELEMS // max(per_frame, 1)))

    # Fold the temporal taps into conv2d's CHANNEL dim: conv3d(kT) over frame
    # window [t, t+kT) == one conv2d with kT*Ci input channels. The single
    # conv2d accumulates ALL taps in its internal fp32 accumulator and rounds
    # once — exactly conv3d's numerics. (Summing separately-rounded per-tap
    # conv2d outputs looks harmless in one decode (~8e-4) but Wan-style
    # recursive causal VAEs feed each chunk's output into the next chunk's
    # cache, and the per-tap rounding compounds into horizontal banding that
    # grows over the clip.) Weight blocks are ordered [W[:,:,0] | W[:,:,1] |
    # ...] to match the channel cat; .contiguous() is load-bearing — MPS
    # conv2d silently miscomputes with non-contiguous weights.
    # groups > 1 would interleave wrongly under the fold; use fp32-accumulated
    # per-tap convs there (rare in VAEs).
    fold = groups == 1 or kT == 1
    if fold:
        wf = weight.transpose(1, 2).reshape(Co, kT * (Ci // groups), kH, kW).contiguous()
    else:
        w2 = [weight[:, :, kt].float().contiguous() for kt in range(kT)]

    outs = []
    for f0 in range(0, T_out, chunk):
        f1 = min(f0 + chunk, T_out)
        n = f1 - f0
        if fold:
            taps = []
            for kt in range(kT):
                t0 = f0 * sT + kt * dT
                xt = input[:, :, t0:t0 + (n - 1) * sT + 1:sT]    # [B,Ci,n,H,W]
                taps.append(xt.transpose(1, 2).reshape(B * n, Ci, H, W))
            x2 = taps[0] if kT == 1 else torch.cat(taps, 1)      # [B*n, kT*Ci, H, W]
            acc = F.conv2d(x2, wf, bias, (sH, sW), (pH, pW), (dH, dW), groups)
        else:
            acc = None
            for kt in range(kT):
                t0 = f0 * sT + kt * dT
                xt = input[:, :, t0:t0 + (n - 1) * sT + 1:sT]
                x2 = xt.transpose(1, 2).reshape(B * n, Ci, H, W).float()
                y = F.conv2d(x2, w2[kt], None, (sH, sW), (pH, pW), (dH, dW), groups)
                acc = y if acc is None else acc.add_(y)
            if bias is not None:
                acc = acc.add_(bias.float().view(1, -1, 1, 1))
            acc = acc.to(input.dtype)
        outs.append(acc.reshape(B, n, Co, Ho, Wo))
    return torch.cat(outs, 1).transpose(1, 2)


def _patched_conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
    if (
        input.device.type == "mps"
        and input.dim() == 5
        and not self.transposed
        and getattr(self, "padding_mode", "zeros") == "zeros"
        and isinstance(self.padding, (tuple, list)) and len(self.padding) == 3
        and len(self.stride) == 3 and len(self.dilation) == 3
    ):
        if autopad == "causal_zero":
            weight = weight[:, :, -input.shape[2]:, :, :]
            autopad = None
        try:
            out = _conv3d_decomposed(input, weight, bias, self.stride,
                                     self.padding, self.dilation, self.groups)
            if out is not None:
                return out
        except Exception as e:
            sys.stderr.write(f"{TAG} decomposition fell back ({e!r})\n")
    return _orig_conv_forward(self, input, weight, bias, autopad=autopad,
                              *args, **kwargs)


def install():
    global _installed, _orig_conv_forward
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_CONV3D") != "1":
        return  # opt-in (ASFP8_NATIVE=1 implies)
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ops as O
    except Exception as e:
        print(f"{TAG} comfy.ops unavailable ({e!r}); not installed.")
        return

    _orig_conv_forward = O.disable_weight_init.Conv3d._conv_forward
    O.disable_weight_init.Conv3d._conv_forward = _patched_conv_forward
    _installed = True
    print(f"{TAG} Conv3d -> sliced conv2d on MPS (video VAEs / upsamplers; "
          f"~10x at LTX decoder shapes; exact up to accumulation order).")
