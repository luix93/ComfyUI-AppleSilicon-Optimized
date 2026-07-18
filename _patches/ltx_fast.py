"""Patch #27 — LTX fused rope + norm-modulate on MPS (opt-in ASFP8_DIT_EXT=1).

LTX 2.3's sampler is elementwise-bound on MPS (~56% of serialized GPU time in
unfused eager ops; attention is ~2% at 512^2). The two dominant chains:

  * split-RoPE (apply_split_rotary_emb): per call, TWO transposed copies of
    the [B,T,H*D] activation (reshape/swapaxes in and out) plus ~7 elementwise
    passes — for q and k in every attention of every block. The dit_ext
    rope_split kernel is one pass with zero copies (per-head cos/sin read via
    strides straight from the swapaxes view).
  * `rms_norm(x) * (1 + scale) + shift` — 6 sites per BasicAVTransformerBlock
    (video/audio msa, a2v/v2a cross-modal, video/audio mlp), 3+ passes each,
    plus a hidden full-size copy per site when timesteps are per-frame
    CompressedTimestep (eager expands frame modulation to per-token via
    expand().reshape()). The rms_mod kernel fuses norm+modulate in one pass
    and indexes the per-frame rows directly — the expansion never happens.

Numerics: kernels mirror the eager rounding chains op-by-op (fp32 math,
bf16_rte at each eager store point) — exact-or-1ulp. Gates, attention,
linears, and the cross-attention adaln path stay eager.

Guarded per call: non-MPS, non-bf16, interleaved rope, or unexpected shapes
fall back to the originals. Kill switch: ASFP8_LTX_FAST=0 (under
ASFP8_DIT_EXT=1).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/ltx_fast]"

_installed = False
_kernel = None
_orig_rope = None
_orig_block_forward = None
_announced = set()


def _announce(which):
    if which not in _announced:
        _announced.add(which)
        sys.stderr.write(f"{TAG} fused {which} engaged.\n")


def _fast_apply_rotary_emb(input_tensor, freqs_cis):
    K = _kernel
    if (
        K is not None
        and len(freqs_cis) > 2 and freqs_cis[2]          # split mode
        and input_tensor.device.type == "mps"
        and input_tensor.dtype == torch.bfloat16
        and input_tensor.dim() == 3
        and freqs_cis[0].dim() == 4
        and freqs_cis[0].dtype == torch.bfloat16
        and freqs_cis[1].dtype == torch.bfloat16
        and freqs_cis[0].shape == freqs_cis[1].shape
        and freqs_cis[0].stride() == freqs_cis[1].stride()
        and input_tensor.shape[2] == freqs_cis[0].shape[1] * 2 * freqs_cis[0].shape[3]
        and input_tensor.shape[1] == freqs_cis[0].shape[2]
    ):
        try:
            out = K.rope_split(input_tensor.contiguous(), freqs_cis[0],
                               freqs_cis[1], freqs_cis[0].shape[1])
            _announce("split-rope")
            return out
        except Exception:
            pass
    return _orig_rope(input_tensor, freqs_cis)


def _ada_rows(table, bsz, ts, idx, CT):
    """get_ada_values WITHOUT the spatial expansion: returns (vals, ppf) where
    each val is [B, Sm, Cd] and rows repeat over ppf consecutive tokens."""
    num = table.shape[0]
    if isinstance(ts, CT):
        data = ts.data
        reshaped = data.reshape(bsz, data.shape[1], num, -1)[:, :, idx, :]
        tv = table[idx].unsqueeze(0).unsqueeze(0).to(device=data.device, dtype=data.dtype)
        return (tv + reshaped).unbind(dim=2), ts.patches_per_frame
    vals = (
        table[idx].unsqueeze(0).unsqueeze(0).to(device=ts.device, dtype=ts.dtype)
        + ts.reshape(bsz, ts.shape[1], num, -1)[:, :, idx, :]
    ).unbind(dim=2)
    return vals, 1


def _rm(x, scale, shift, ppf):
    return _kernel.rms_mod(x.contiguous(), scale, shift, 1e-6, ppf)


def _fast_av_block_forward(self, x, v_context=None, a_context=None, attention_mask=None,
                           v_timestep=None, a_timestep=None, v_pe=None, a_pe=None,
                           v_cross_pe=None, a_cross_pe=None,
                           v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
                           v_cross_gate_timestep=None, a_cross_gate_timestep=None,
                           transformer_options=None, self_attention_mask=None,
                           v_prompt_timestep=None, a_prompt_timestep=None):
    from comfy.ldm.lightricks.av_model import CompressedTimestep as CT

    vx, ax = x
    K = _kernel
    ok = (
        K is not None
        and vx.device.type == "mps"
        and vx.dtype == torch.bfloat16
        and (ax is None or ax.numel() == 0 or ax.dtype == torch.bfloat16)
    )
    if not ok:
        return _orig_block_forward(self, x, v_context=v_context, a_context=a_context,
                                   attention_mask=attention_mask, v_timestep=v_timestep,
                                   a_timestep=a_timestep, v_pe=v_pe, a_pe=a_pe,
                                   v_cross_pe=v_cross_pe, a_cross_pe=a_cross_pe,
                                   v_cross_scale_shift_timestep=v_cross_scale_shift_timestep,
                                   a_cross_scale_shift_timestep=a_cross_scale_shift_timestep,
                                   v_cross_gate_timestep=v_cross_gate_timestep,
                                   a_cross_gate_timestep=a_cross_gate_timestep,
                                   transformer_options=transformer_options,
                                   self_attention_mask=self_attention_mask,
                                   v_prompt_timestep=v_prompt_timestep,
                                   a_prompt_timestep=a_prompt_timestep)

    run_vx = transformer_options.get("run_vx", True)
    run_ax = transformer_options.get("run_ax", True)
    run_ax = run_ax and ax.numel() > 0
    run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
    run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)

    # video self-attention
    if run_vx:
        (vshift_msa, vscale_msa), ppf = _ada_rows(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2), CT)
        norm_vx = _rm(vx, vscale_msa, vshift_msa, ppf)
        _announce("av-block")
        del vshift_msa, vscale_msa
        attn1_out = self.attn1(norm_vx, pe=v_pe, mask=self_attention_mask,
                               transformer_options=transformer_options)
        del norm_vx
        vgate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0],
                                        v_timestep, slice(2, 3))[0]
        vx.addcmul_(attn1_out, vgate_msa)
        del vgate_msa, attn1_out
        vx.add_(self._apply_text_cross_attention(
            vx, v_context, self.attn2, self.scale_shift_table,
            getattr(self, 'prompt_scale_shift_table', None),
            v_timestep, v_prompt_timestep, attention_mask, transformer_options,))

    # audio self-attention
    if run_ax:
        (ashift_msa, ascale_msa), appf = _ada_rows(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2), CT)
        norm_ax = _rm(ax, ascale_msa, ashift_msa, appf)
        del ashift_msa, ascale_msa
        attn1_out = self.audio_attn1(norm_ax, pe=a_pe,
                                     transformer_options=transformer_options)
        del norm_ax
        agate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0],
                                        a_timestep, slice(2, 3))[0]
        ax.addcmul_(attn1_out, agate_msa)
        del agate_msa, attn1_out
        ax.add_(self._apply_text_cross_attention(
            ax, a_context, self.audio_attn2, self.audio_scale_shift_table,
            getattr(self, 'audio_prompt_scale_shift_table', None),
            a_timestep, a_prompt_timestep, attention_mask, transformer_options,))

    # video <-> audio cross attention
    if run_a2v or run_v2a:
        if run_a2v:
            (sa_a2v, sha_a2v), appf2 = _ada_rows(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0],
                a_cross_scale_shift_timestep, slice(0, 2), CT)
            (sv_a2v, shv_a2v), vppf2 = _ada_rows(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0],
                v_cross_scale_shift_timestep, slice(0, 2), CT)
            vx_scaled = _rm(vx, sv_a2v, shv_a2v, vppf2)
            ax_scaled = _rm(ax, sa_a2v, sha_a2v, appf2)
            del sa_a2v, sha_a2v, sv_a2v, shv_a2v
            a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled,
                                               pe=v_cross_pe, k_pe=a_cross_pe,
                                               transformer_options=transformer_options)
            del vx_scaled, ax_scaled
            gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :],
                                               vx.shape[0], v_cross_gate_timestep)[0]
            vx.addcmul_(a2v_out, gate_out_a2v)
            del gate_out_a2v, a2v_out

        if run_v2a:
            (sa_v2a, sha_v2a), appf3 = _ada_rows(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0],
                a_cross_scale_shift_timestep, slice(2, 4), CT)
            (sv_v2a, shv_v2a), vppf3 = _ada_rows(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0],
                v_cross_scale_shift_timestep, slice(2, 4), CT)
            ax_scaled = _rm(ax, sa_v2a, sha_v2a, appf3)
            vx_scaled = _rm(vx, sv_v2a, shv_v2a, vppf3)
            del sa_v2a, sha_v2a, sv_v2a, shv_v2a
            v2a_out = self.video_to_audio_attn(ax_scaled, context=vx_scaled,
                                               pe=a_cross_pe, k_pe=v_cross_pe,
                                               transformer_options=transformer_options)
            del ax_scaled, vx_scaled
            gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :],
                                               ax.shape[0], a_cross_gate_timestep)[0]
            ax.addcmul_(v2a_out, gate_out_v2a)
            del gate_out_v2a, v2a_out

    # video feedforward
    if run_vx:
        (vshift_mlp, vscale_mlp), ppf = _ada_rows(
            self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5), CT)
        vx_scaled = _rm(vx, vscale_mlp, vshift_mlp, ppf)
        del vshift_mlp, vscale_mlp
        ff_out = self.ff(vx_scaled)
        del vx_scaled
        vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0],
                                        v_timestep, slice(5, 6))[0]
        vx.addcmul_(ff_out, vgate_mlp)
        del vgate_mlp, ff_out

    # audio feedforward
    if run_ax:
        (ashift_mlp, ascale_mlp), appf = _ada_rows(
            self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5), CT)
        ax_scaled = _rm(ax, ascale_mlp, ashift_mlp, appf)
        del ashift_mlp, ascale_mlp
        ff_out = self.audio_ff(ax_scaled)
        del ax_scaled
        agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0],
                                        a_timestep, slice(5, 6))[0]
        ax.addcmul_(ff_out, agate_mlp)
        del agate_mlp, ff_out

    return vx, ax


def install():
    global _installed, _kernel, _orig_rope, _orig_block_forward
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_LTX_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.lightricks.model as LM
        import comfy.ldm.lightricks.av_model as LAV
    except Exception as e:
        print(f"{TAG} comfy lightricks modules unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "rope_split") \
            or not hasattr(_kernel, "rms_mod"):
        print(f"{TAG} dit_ext LTX kernels unavailable; LTX fast path stays off.")
        return

    _orig_rope = LM.apply_rotary_emb
    LM.apply_rotary_emb = _fast_apply_rotary_emb
    rebound = 0
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "apply_rotary_emb", None) is _orig_rope and mod is not LM:
                mod.apply_rotary_emb = _fast_apply_rotary_emb
                rebound += 1
        except Exception:
            continue

    _orig_block_forward = LAV.BasicAVTransformerBlock.forward
    LAV.BasicAVTransformerBlock.forward = _fast_av_block_forward

    _installed = True
    print(f"{TAG} LTX fast path on MPS: one-pass split-RoPE (zero transposed "
          f"copies) + fused rms_norm+modulate with per-frame timestep rows "
          f"({rebound} rope bindings rerouted; eager fallback per call).")
