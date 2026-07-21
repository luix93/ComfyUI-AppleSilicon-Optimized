"""Patch #32 — Ideogram 4 fused block glue (opt-in ASFP8_DIT_EXT=1).

Ideogram 4 is a NextDiT/Lumina2-family single-stream DiT (34 layers, 18
heads x D=256, packed [text, image] sequence). Profiling a 12-step 1024^2
run showed ~43% of GPU time in unwrapped eager elementwise — the same
modulation/norm sandwich the Z-Image sprint already built kernels for:

  block:  mod = adaln(c); s_msa, g_msa, s_mlp, g_mlp = chunk4
          h  = RMSNorm_w(x) * (1 + s_msa)          -> dit_ext norm_modulate
          x  = x + tanh(g_msa) * RMSNorm_w(attn)   -> dit_ext norm_gate_residual
          h  = RMSNorm_w(x) * (1 + s_mlp)          -> dit_ext norm_modulate
          x  = x + tanh(g_mlp) * RMSNorm_w(ffn)    -> dit_ext norm_gate_residual

The FFN's silu(w1)*w3 gating is already fused: Ideogram imports Lumina's
FeedForward class, which lumina_attn_fast patches at class level.

qkv/attention/o stay untouched here (attention routing is the router's
job; the fused qkv+QKNorm+MRoPE kernel needs a D=256 variant — future
stage). Per-call guards fall back to the original forward; kill switch
ASFP8_IDEOGRAM_FAST=0.
"""

import os
import sys

import torch

TAG = "[AppleSilicon-Optimized/ideogram_fast]"

_installed = False
_orig_forward = None
_orig_attn = None
_kernel = None
_announced = False


def _announce():
    global _announced
    if not _announced:
        _announced = True
        sys.stderr.write(f"{TAG} fused block glue engaged.\n")


def _norm_w(norm, dtype, device):
    return norm.weight.to(dtype=dtype, device=device)


# ── freqs cache: the MRoPE cos/sin build is identical for all 24 forwards ──
# (it depends only on head_dim/theta/sections and the position grid, which is
# fixed for a given text length + latent size). Keyed by shape + a 2-element
# content probe (last image token's h/w coords) so different aspect ratios
# with equal L can't collide. Bit-exact: we return the same tensors the
# original call produced.
_freqs_cache = {}
_orig_precompute = None


def _cached_precompute_freqs_cis(head_dim, positions, theta, rope_scale=None,
                                 rope_dims=None, device=None,
                                 interleaved_mrope=False):
    try:
        probe = (int(positions[-1, 0]), int(positions[-1, -1]))
        key = (int(head_dim), str(theta), tuple(positions.shape),
               tuple(rope_dims) if rope_dims is not None else None,
               bool(interleaved_mrope), str(positions.device), probe)
    except Exception:
        return _orig_precompute(head_dim, positions, theta, rope_scale=rope_scale,
                                rope_dims=rope_dims, device=device,
                                interleaved_mrope=interleaved_mrope)
    hit = _freqs_cache.get(key)
    if hit is not None:
        return hit
    out = _orig_precompute(head_dim, positions, theta, rope_scale=rope_scale,
                           rope_dims=rope_dims, device=device,
                           interleaved_mrope=interleaved_mrope)
    if len(_freqs_cache) > 4:
        _freqs_cache.clear()
    _freqs_cache[key] = out
    return out


def _fast_attn_forward(self, x, attn_mask, freqs_cis, transformer_options={}):
    """Fused qkv-split + per-head RMSNorm + split-half MRoPE via dit_ext:
    replaces ~7s/run of eager rope + reshape copies (measured) and hands
    attention contiguous [B,H,S,D] tensors so the bmm route stops copying."""
    if (_kernel is None or x.device.type != "mps" or x.dtype != torch.bfloat16
            or not isinstance(freqs_cis, (tuple, list)) or len(freqs_cis) != 3):
        return _orig_attn(self, x, attn_mask, freqs_cis,
                          transformer_options=transformer_options)
    try:
        from comfy.ldm.modules.attention import optimized_attention_masked
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        cos, sin, nsin = freqs_cis
        if (D % 64 != 0 or D > 512 or cos.numel() != S * D
                or sin.numel() != S * (D // 2)):
            return _orig_attn(self, x, attn_mask, freqs_cis,
                              transformer_options=transformer_options)
        qw = getattr(self, "_asfp8_qw", None)
        if qw is None or qw.numel() != D:
            qw = self.norm_q.weight.to(torch.float32).contiguous()
            kw = self.norm_k.weight.to(torch.float32).contiguous()
            self._asfp8_qw, self._asfp8_kw = qw, kw
        else:
            kw = self._asfp8_kw
        qkv = self.qkv(x)
        q, k, v = _kernel.qkv_norm_rope_sh(
            qkv.contiguous(), cos.reshape(S, D).contiguous(),
            sin.reshape(S, D // 2).contiguous(),
            nsin.reshape(S, D // 2).contiguous(),
            qw, kw, H, D, float(self.norm_q.eps))
        out = optimized_attention_masked(q, k, v, H, attn_mask,
                                         skip_reshape=True,
                                         transformer_options=transformer_options)
        return self.o(out)
    except Exception:
        return _orig_attn(self, x, attn_mask, freqs_cis,
                          transformer_options=transformer_options)


# ── dev attribution (ASFP8_IDEO_TRACE=1): sync-timer per attention segment ──
_trace = {"n": 0}


def _t(key, t0):
    import time
    torch.mps.synchronize()
    t1 = time.perf_counter()
    _trace[key] = _trace.get(key, 0.0) + (t1 - t0)
    return t1


def _traced_attn_forward(self, x, attn_mask, freqs_cis, transformer_options={}):
    import time
    from comfy.ldm.modules.attention import optimized_attention_masked
    from comfy.text_encoders.llama import apply_rope
    torch.mps.synchronize()
    t0 = time.perf_counter()
    batch_size, seq_len, _ = x.shape
    qkv = self.qkv(x)
    t0 = _t("qkv_linear", t0)
    qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = self.norm_q(q)
    k = self.norm_k(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    t0 = _t("split_norm_transpose", t0)
    q, k = apply_rope(q, k, freqs_cis)
    t0 = _t("rope", t0)
    out = optimized_attention_masked(q, k, v, self.num_heads, attn_mask,
                                     skip_reshape=True,
                                     transformer_options=transformer_options)
    t0 = _t("attention", t0)
    out = self.o(out)
    _t("o_linear", t0)
    _trace["n"] += 1
    if _trace["n"] % 136 == 0:   # every 4 forwards (34 layers)
        total = sum(v for k_, v in _trace.items() if k_ != "n")
        parts = " ".join(f"{k_}={v:.2f}s" for k_, v in sorted(_trace.items())
                         if k_ != "n")
        sys.stderr.write(f"{TAG} trace n={_trace['n']} total={total:.2f}s {parts}\n")
    return out


def _fast_block_forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):
    if (_kernel is None or x.device.type != "mps"
            or x.dtype != torch.bfloat16 or x.dim() != 3):
        return _orig_forward(self, x, attn_mask, freqs_cis, adaln_input,
                             transformer_options=transformer_options)
    try:
        B, S, C = x.shape
        eps = float(self.attention_norm1.eps)
        mod = self.adaln_modulation(adaln_input)          # [B, 1, 4C]
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        tg_msa = torch.tanh(gate_msa).reshape(B, C)
        tg_mlp = torch.tanh(gate_mlp).reshape(B, C)
        s_msa = scale_msa.reshape(B, C)                    # RAW: kernel adds the 1+
        s_mlp = scale_mlp.reshape(B, C)

        dt, dev = x.dtype, x.device
        h = _kernel.norm_modulate(x.contiguous(),
                                  _norm_w(self.attention_norm1, dt, dev),
                                  s_msa.contiguous(), eps)
        attn_out = self.attention(h, attn_mask, freqs_cis,
                                  transformer_options=transformer_options)
        x = _kernel.norm_gate_residual(x.contiguous(), attn_out.contiguous(),
                                       _norm_w(self.attention_norm2, dt, dev),
                                       tg_msa.contiguous(), eps)
        h = _kernel.norm_modulate(x, _norm_w(self.ffn_norm1, dt, dev),
                                  s_mlp.contiguous(), eps)
        ff = self.feed_forward(h)
        x = _kernel.norm_gate_residual(x, ff.contiguous(),
                                       _norm_w(self.ffn_norm2, dt, dev),
                                       tg_mlp.contiguous(), eps)
        _announce()
        return x
    except Exception:
        return _orig_forward(self, x, attn_mask, freqs_cis, adaln_input,
                             transformer_options=transformer_options)


def install():
    global _installed, _orig_forward, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_IDEOGRAM_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.ideogram4.model as IM
    except Exception:
        return  # comfy too old for Ideogram 4 — nothing to patch

    from .dit_ext import loader
    _kernel = loader.module()
    if (_kernel is None or not hasattr(_kernel, "norm_modulate")
            or not hasattr(_kernel, "norm_gate_residual")):
        print(f"{TAG} dit_ext kernels unavailable; ideogram fast path stays off.")
        return

    global _orig_attn, _orig_precompute
    _orig_forward = IM.Ideogram4TransformerBlock.forward
    _orig_attn = IM.Ideogram4Attention.forward
    # Freqs cache: rebind the name the ideogram module resolves at call time
    # (comfy.text_encoders.llama itself stays untouched — the TE is unaffected).
    _orig_precompute = IM.precompute_freqs_cis
    IM.precompute_freqs_cis = _cached_precompute_freqs_cis

    # Cold-run helper: compile the hot PSOs on the MPS stream while the two
    # ~9GB checkpoints stream from disk (pure overlap; no output effect).
    def _prewarm():
        try:
            t = torch.zeros(256, 4608, dtype=torch.bfloat16, device="mps")
            w = torch.zeros(256, dtype=torch.float32, device="mps")
            c = torch.zeros(256, 256, dtype=torch.float32, device="mps")
            sn = torch.zeros(256, 128, dtype=torch.float32, device="mps")
            _kernel.qkv_norm_rope_sh(
                torch.zeros(1, 256, 3 * 18 * 256, dtype=torch.bfloat16, device="mps"),
                c, sn, sn, w, w, 18, 256, 1e-5)
            _kernel.norm_modulate(t.view(1, 256, 4608),
                                  torch.zeros(4608, dtype=torch.bfloat16, device="mps"),
                                  torch.zeros(1, 4608, dtype=torch.bfloat16, device="mps"),
                                  1e-5)
            torch.mps.synchronize()
        except Exception:
            pass
    import threading
    threading.Thread(target=_prewarm, daemon=True).start()
    IM.Ideogram4TransformerBlock.forward = _fast_block_forward
    if os.environ.get("ASFP8_IDEO_TRACE") == "1":
        IM.Ideogram4Attention.forward = _traced_attn_forward
        print(f"{TAG} ATTRIBUTION TRACE ON (sync timers — wall time inflated).")
    elif hasattr(_kernel, "qkv_norm_rope_sh"):
        IM.Ideogram4Attention.forward = _fast_attn_forward
    _installed = True
    print(f"{TAG} Ideogram 4 fused block glue on MPS: RMSNorm+modulate and "
          f"gate+residual via Metal (FFN silu-gating fused via the shared "
          f"Lumina FeedForward; eager fallback per call).")
