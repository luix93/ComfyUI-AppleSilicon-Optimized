"""Patch #29 — Wan 2.1 VAE fast path on MPS (opt-in ASFP8_CONV3D=1).

Attribution of a 21-latent-frame 512^2 decode (M5 Max, ~10s eager):
conv2d ~3.2s (already at MPS's ~60 TF/s ceiling), channelwise RMS_norm
~3.1s (F.normalize(dim=1) over NCTHW = a 42MB-strided reduction, ~80 GB/s
effective), tensor cats ~2.4s (causal zero-pad cat + per-tap transpose
copies + channel-fold cat: ~14 tensor volumes of traffic per causal conv),
SiLU ~0.8s. Three fixes:

  * chunk batching: decode/encode outer loops and Decoder3d.run_up stream
    ASFP8_WAN_VAE_CHUNK (default 16, forced even) frames per causal pass
    instead of 2. The CausalConv3d cache protocol (keep last CACHE_T=2
    input frames, zero-pad the remainder) is chunk-size invariant for any
    even chunk >= 2; the first pass stays T=1 (the 'Rep' special case only
    applies to frame 0). Exact per frame; mostly a memory-shape enabler.
  * chan_rms (wan_ext): channelwise L2 RMS_norm + optional fused SiLU at
    ~400 GB/s (TG-tiled 16KB contiguous runs; fp32 accumulation). Patched
    at RMS_norm.forward and fused with SiLU inside ResidualBlock.forward.
    Handles both standard-contiguous and the [B,T,C,H,W]-transpose layout
    the conv3d decomposition emits.
  * fold_taps (wan_ext): CausalConv3d.forward builds the folded conv2d
    input (kT*Ci channels) with ONE gather straight from
    [zeros | cache | x] — the pad-cat, tap copies, and channel cat never
    materialize. Same folded conv2d as patch #26, so numerics unchanged.

Peak transient memory scales with the chunk (~5 GB at 512^2 x chunk 16,
inside comfy's own memory_used_decode estimate). Wan 2.2 VAE (vae2_2.py)
has a different decoder and is NOT touched. Kill switch:
ASFP8_WAN_VAE_FAST=0 (under ASFP8_CONV3D=1; kernels need ASFP8_DIT_EXT=1).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/wan_vae_fast]"

_installed = False
_kernel = None
_orig_decode = None
_orig_encode = None
_orig_run_up = None
_orig_rms = None
_orig_resblock = None
_orig_causal = None


_auto_chunk = None


def _chunk(env, default=None):
    """Frames per causal pass. Explicit env wins; otherwise auto-size from
    total RAM — decode transients run ~5 GB at 512^2 x chunk 16 (and ~4x
    that at 1024^2), which is fine at 128 GB but squeezes <64 GB machines
    into swap. Tiers: >=96 GB -> 16, >=64 -> 8, below -> 4."""
    global _auto_chunk
    if default is None:
        if _auto_chunk is None:
            try:
                import psutil
                total = psutil.virtual_memory().total
            except Exception:
                total = 128 << 30
            _auto_chunk = 16 if total >= (96 << 30) else (8 if total >= (64 << 30) else 4)
        default = _auto_chunk
    try:
        c = int(os.environ.get(env, str(default)))
    except ValueError:
        c = default
    return max(2, min(c - (c % 2), 32))


def _fast_decode(self, z):
    # z: [b,c,t,h,w]
    if z.shape[2] <= 2 or z.device.type != "mps":
        return _orig_decode(self, z)
    from comfy.ldm.wan.vae import count_cache_layers
    feat_map = [None] * count_cache_layers(self.decoder)
    x = self.conv2(z)
    T = x.shape[2]
    C = _chunk("ASFP8_WAN_VAE_CHUNK")
    out = self.decoder(x[:, :, :1], feat_cache=feat_map, feat_idx=[0])
    for s in range(1, T, C):
        out += self.decoder(x[:, :, s:min(s + C, T)],
                            feat_cache=feat_map, feat_idx=[0])
    return torch.cat(out, 2)


def _fast_encode(self, x):
    t = x.shape[2]
    t = 1 + ((t - 1) // 4) * 4  # truncate to 4N+1 like eager
    if t <= 2 or x.device.type != "mps":
        return _orig_encode(self, x)
    from comfy.ldm.wan.vae import count_cache_layers
    feat_map = [None] * count_cache_layers(self.encoder)
    C = _chunk("ASFP8_WAN_VAE_CHUNK")
    outs = [self.encoder(x[:, :, :1], feat_cache=feat_map, feat_idx=[0])]
    for s in range(1, t, C):
        e = min(s + C, t)
        o = self.encoder(x[:, :, s:e], feat_cache=feat_map, feat_idx=[0],
                         final=(e == t))
        if o is not None:  # downsample3d may defer a lone trailing frame
            outs.append(o)
    mu, log_var = self.conv1(torch.cat(outs, 2)).chunk(2, dim=1)
    return mu


def _fast_run_up(self, layer_idx, x_ref, feat_cache, feat_idx, out_chunks):
    from comfy.ldm.wan.vae import CACHE_T, CausalConv3d, Resample

    x = x_ref[0]
    x_ref[0] = None
    if layer_idx >= len(self.upsamples):
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                cache_x = x[:, :, -CACHE_T:, :, :]
                x = layer(x, feat_cache[feat_idx[0]])
                feat_cache[feat_idx[0]] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        out_chunks.append(x)
        return

    layer = self.upsamples[layer_idx]
    if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
    else:
        x = layer(x)

    split = _chunk("ASFP8_WAN_VAE_CHUNK")
    if isinstance(layer, Resample) and layer.mode == 'upsample3d' and x.shape[2] > split:
        for frame_idx in range(0, x.shape[2], split):
            self.run_up(
                layer_idx + 1,
                [x[:, :, frame_idx:frame_idx + split, :, :]],
                feat_cache,
                feat_idx.copy(),
                out_chunks,
            )
        del x
        return

    next_x_ref = [x]
    del x
    self.run_up(layer_idx + 1, next_x_ref, feat_cache, feat_idx, out_chunks)


def _chan_rms(mod, x, silu):
    """RMS_norm module via the wan_ext kernel; None if unsupported."""
    K = _kernel
    if (
        K is None
        or not mod.channel_first
        or mod.bias is not None
        or x.device.type != "mps"
        or x.dtype not in (torch.float16, torch.bfloat16)
        or x.dim() < 3
    ):
        return None
    B, C = x.shape[0], x.shape[1]
    g = mod.gamma
    if g.numel() != C:
        return None
    ga = g.reshape(C).to(x.dtype)
    sc = float(mod.scale)
    if x.is_contiguous():
        return K.chan_rms(x.view(B, C, -1), ga, sc, 1e-12, silu).view(x.shape)
    # conv3d-decomposition outputs arrive as [B,C,T,H,W] transpose views of a
    # contiguous [B,T,C,H,W]; that layout IS [B*T, C, H*W] contiguous, which
    # the kernel reduces over C just as well. Result keeps the same strides.
    if x.dim() == 5:
        xt = x.transpose(1, 2)
        if xt.is_contiguous():
            T, HW = x.shape[2], x.shape[3] * x.shape[4]
            out = K.chan_rms(xt.reshape(B * T, C, HW), ga, sc, 1e-12, silu)
            return out.view(B, T, C, x.shape[3], x.shape[4]).transpose(1, 2)
    return None


def _fast_rms_forward(self, x):
    out = _chan_rms(self, x, False)
    return out if out is not None else _orig_rms(self, x)


def _fast_resblock_forward(self, x, feat_cache=None, feat_idx=[0], final=False):
    """ResidualBlock with RMS_norm+SiLU fused into one kernel pass.

    residual = [RMS, SiLU, CausalConv3d, RMS, SiLU, Dropout, CausalConv3d];
    cache handling mirrors eager (cache_x snapshotted from each conv INPUT).
    """
    from comfy.ldm.wan.vae import CACHE_T, CausalConv3d, RMS_norm

    r = self.residual
    ok = (
        _kernel is not None
        and len(r) == 7
        and isinstance(r[0], RMS_norm) and isinstance(r[3], RMS_norm)
        and isinstance(r[1], torch.nn.SiLU) and isinstance(r[4], torch.nn.SiLU)
        and isinstance(r[2], CausalConv3d) and isinstance(r[6], CausalConv3d)
        and isinstance(r[5], torch.nn.Dropout)
        and (not self.training or r[5].p == 0.0)
    )
    if not ok:
        return _orig_resblock(self, x, feat_cache=feat_cache, feat_idx=feat_idx,
                              final=final)

    old_x = x
    for norm, conv in ((r[0], r[2]), (r[3], r[6])):
        h = _chan_rms(norm, x, True)
        if h is None:
            # per-op eager fallback — never restart the block, the causal
            # cache slots for earlier convs are already consumed
            h = torch.nn.functional.silu(_orig_rms(norm, x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = h[:, :, -CACHE_T:, :, :]
            x = conv(h, cache_list=feat_cache, cache_idx=idx)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = conv(h)
    return x + self.shortcut(old_x)


def _x_layout(x):
    """(storage_tensor, tlayout) for the kernel, or (None, _) if unsupported.

    Standard contiguous [1,Ci,T,H,W] -> tlayout 0. The transpose view of a
    contiguous [1,T,Ci,H,W] (what conv3d decomposition / chan_rms emit) ->
    tlayout 1, no copy.
    """
    if x.is_contiguous():
        return x, False
    xt = x.transpose(1, 2)
    if xt.is_contiguous():
        return xt, True
    return None, False


def _fast_causal_forward(self, x, cache_x=None, cache_list=None, cache_idx=None):
    if cache_list is not None:
        cache_x = cache_list[cache_idx]
        cache_list[cache_idx] = None

    K = _kernel
    kT = self.weight.shape[2]
    xs = tl = None
    if (
        K is not None
        and x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dim() == 5 and x.shape[0] == 1
        and self.groups == 1
        and self.stride[0] == 1 and self.dilation[0] == 1
        and getattr(self, "padding_mode", "zeros") == "zeros"
        and not getattr(self, "comfy_cast_weights", False)
        and self.weight.dtype == x.dtype
        and not (cache_x is None and x.shape[2] == 1 and self._padding > 0)
    ):
        xs, tl = _x_layout(x)
    if xs is None:
        return _orig_causal(self, x, cache_x=cache_x)

    _, Ci, T, H, W = x.shape
    cache = None
    Tc = 0
    Z = 0
    if self._padding > 0:
        if cache_x is not None:
            cache = cache_x.to(dtype=x.dtype, device=x.device).contiguous()
            Tc = cache.shape[2]
            if Tc > self._padding:
                return _orig_causal(self, x, cache_x=cache_x)
        Z = self._padding - Tc
    T_out = Z + Tc + T - kT + 1
    if T_out <= 0:
        return _orig_causal(self, x, cache_x=cache_x)

    import torch.nn.functional as F
    # Cache key = weight identity: storage pointer + in-place version counter
    # + dtype/device. A reload, LoRA-style in-place mutation, or offload/
    # restore cycle changes the key and forces a refold — a dtype-only check
    # would silently reuse stale folded weights (plain attributes are not
    # moved by Module._apply, so they survive .to()/offload unchanged).
    w = self.weight
    try:
        ver = w._version
    except RuntimeError:
        # Inference tensors don't track a version counter — and can't be
        # mutated by tracked in-place ops either; storage identity suffices.
        ver = None
    key = (w.data_ptr(), ver, w.dtype, w.device)
    wf = getattr(self, "_asfp8_wf", None)
    if wf is None or getattr(self, "_asfp8_wf_key", None) != key:
        wf = w.transpose(1, 2).reshape(w.shape[0], kT * Ci,
                                       w.shape[3], w.shape[4]).contiguous()
        self._asfp8_wf = wf
        self._asfp8_wf_key = key

    Co = wf.shape[0]
    sH, sW = self.stride[1], self.stride[2]
    pH, pW = self.padding[1], self.padding[2]
    dH, dW = self.dilation[1], self.dilation[2]
    Ho = (H + 2 * pH - (wf.shape[2] - 1) * dH - 1) // sH + 1
    Wo = (W + 2 * pW - (wf.shape[3] - 1) * dW - 1) // sW + 1

    per_frame = max(kT * Ci * H * W, Co * Ho * Wo)
    chunk = max(1, min(T_out, (1 << 30) // max(per_frame, 1)))
    outs = []
    for f0 in range(0, T_out, chunk):
        n = min(chunk, T_out - f0)
        if kT == 1 and Z == 0 and Tc == 0 and tl:
            x2 = xs.reshape(T, Ci, H, W)[f0:f0 + n]   # pure view, no gather
        else:
            x2 = K.fold_taps(xs, cache if cache is not None else torch.Tensor(),
                             Ci, T, H, W, Z, kT, f0, n, tl)
        outs.append(F.conv2d(x2, wf, self.bias, (sH, sW), (pH, pW), (dH, dW), 1))
    y = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
    return y.reshape(1, T_out, Co, Ho, Wo).transpose(1, 2)


def install():
    global _installed, _kernel, _orig_decode, _orig_encode, _orig_run_up
    global _orig_rms, _orig_resblock, _orig_causal
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_CONV3D") != "1":
        return  # opt-in, same switch as the conv3d decomposition
    if os.environ.get("ASFP8_WAN_VAE_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.wan.vae as WV
    except Exception as e:
        print(f"{TAG} comfy wan vae unavailable ({e!r}); not installed.")
        return

    try:
        from .wan_ext import loader
        _kernel = loader.module()
        if _kernel is not None and not (hasattr(_kernel, "chan_rms")
                                        and hasattr(_kernel, "fold_taps")):
            _kernel = None
    except Exception:
        _kernel = None

    _orig_decode = WV.WanVAE.decode
    WV.WanVAE.decode = _fast_decode
    _orig_encode = WV.WanVAE.encode
    WV.WanVAE.encode = _fast_encode
    _orig_run_up = WV.Decoder3d.run_up
    WV.Decoder3d.run_up = _fast_run_up
    _orig_rms = WV.RMS_norm.forward
    _orig_resblock = WV.ResidualBlock.forward
    _orig_causal = WV.CausalConv3d.forward
    if _kernel is not None:
        WV.RMS_norm.forward = _fast_rms_forward
        WV.ResidualBlock.forward = _fast_resblock_forward
        WV.CausalConv3d.forward = _fast_causal_forward
    _installed = True
    print(f"{TAG} Wan 2.1 VAE fast path on MPS: chunk batching "
          f"({_chunk('ASFP8_WAN_VAE_CHUNK')} frames/pass)"
          + (", fused channelwise RMS_norm+SiLU kernel" if _kernel is not None
             else " (chan_rms kernel unavailable — norms stay eager)")
          + ".")
