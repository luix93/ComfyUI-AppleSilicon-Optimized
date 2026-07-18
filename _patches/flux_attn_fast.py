"""Patch #24 — fused Flux2 double/single block attention prep (opt-in ASFP8_DIT_EXT=1).

Flux-family DoubleStreamBlock spends ~8 ms/layer of eager time between the qkv
Linears and the attention call, and SingleStreamBlock ~7 ms, on:

    view -> permute -> q_norm -> k_norm (F.rms_norm over [B,H,S,128]
    non-contiguous views — pathological on MPS) -> 3x torch.cat along seq
    (double block, ~140 MB traffic each) -> eager/kernel RoPE

The dit_ext kernels do the whole chain in one pass and write q/k/v directly in
concatenated [B, H, S, D] attention layout:

  * double block: qkv2_norm_rope — two sources (txt+img) with separate QK norm
    weights, RoPE over the concatenated position table, seq-cat for free;
  * single block: qkv_norm_rope with a row stride — linear1's fused qkv+mlp
    output feeds the kernel directly (no split copy).

RoPE is folded in, so attention runs via optimized_attention directly
(comfy.ldm.flux.math.attention's apply_rope would double-rotate). The rest of
the block (LayerNorms, modulation, MLPs, projections) stays eager torch.

Numerics: same class as patch #21 — norm is fp32 normalize -> bf16 round ->
bf16 weight multiply (verified bit-identical to F.rms_norm on MPS), RoPE is
fp32 pair rotation with RTE store; only the row-sum reduction order differs,
so outputs are exact-or-1ulp vs eager.

Guarded per call: non-MPS, non-bf16, masks, modulation_dims, attn1 patches,
missing/mismatched pe, or D > 128 all fall back to the original forward.
Kill switch: ASFP8_DIT_EXT unset/0 (opt-in; ASFP8_NATIVE=1 implies it).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/flux_fast]"

_installed = False
_orig_double = None
_orig_single = None
_kernel = None
_announced = set()   # {"double", "single"} once the fused path first engages


def _announce(which):
    # stderr: ComfyUI's log interception swallows stdout during execution.
    if which not in _announced:
        _announced.add(which)
        sys.stderr.write(f"{TAG} fused {which}-block path engaged.\n")


_reasons = set()


def _fallback(which, reason):
    """Debug aid: with ASFP8_FLUX_FAST_DEBUG=1, log each distinct fallback
    reason once so silent per-call fallbacks are diagnosable."""
    if os.environ.get("ASFP8_FLUX_FAST_DEBUG") == "1" and (which, reason) not in _reasons:
        _reasons.add((which, reason))
        sys.stderr.write(f"{TAG} {which}-block fallback: {reason}\n")
    return True


def _eps(norm, dtype):
    e = getattr(norm, "eps", None)
    return float(e) if e is not None else float(torch.finfo(dtype).eps)


def _norm_w(norm, device):
    return norm.weight.to(dtype=torch.bfloat16, device=device)


def _pe_ok(pe, B, S, D):
    return (
        pe is not None
        and pe.dtype == torch.float32
        and pe.shape[-3:] == (D // 2, 2, 2)
        and pe.numel() == B * S * (D // 2) * 4   # broadcast over heads only
    )


def _fast_double_forward(self, img, txt, vec, pe, attn_mask=None,
                         modulation_dims_img=None, modulation_dims_txt=None,
                         transformer_options={}):
    K = _kernel
    patches = transformer_options.get("patches", {})
    B, S_img, _ = img.shape
    S_txt = txt.shape[1]
    H = self.num_heads
    D = self.hidden_size // H
    if (
        (K is None and _fallback("double", "kernel unavailable"))
        or (img.device.type != "mps" and _fallback("double", "non-MPS"))
        or ((img.dtype != torch.bfloat16 or txt.dtype != torch.bfloat16)
            and _fallback("double", f"dtype {img.dtype}/{txt.dtype}"))
        or (attn_mask is not None and _fallback("double", "attn_mask"))
        or ((modulation_dims_img is not None or modulation_dims_txt is not None)
            and _fallback("double", "modulation_dims"))
        or (("attn1_patch" in patches or "attn1_output_patch" in patches)
            and _fallback("double", "attn1 patches"))
        or ((D > 128 or D % 32 != 0) and _fallback("double", f"D={D}"))
        or (not _pe_ok(pe, B, S_txt + S_img, D)
            and _fallback("double", f"pe {None if pe is None else (pe.dtype, tuple(pe.shape))}"))
    ):
        return _orig_double(self, img, txt, vec, pe, attn_mask=attn_mask,
                            modulation_dims_img=modulation_dims_img,
                            modulation_dims_txt=modulation_dims_txt,
                            transformer_options=transformer_options)

    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.modules.attention import optimized_attention

    if self.modulation:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
    else:
        # global modulation (Flux2): vec is the precomputed ModulationOut pairs
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

    img_qkv = self.img_attn.qkv(
        apply_mod(self.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift, None))
    txt_qkv = self.txt_attn.qkv(
        apply_mod(self.txt_norm1(txt), (1 + txt_mod1.scale), txt_mod1.shift, None))
    if img_qkv.dtype != torch.bfloat16 or txt_qkv.dtype != torch.bfloat16:
        return _orig_double(self, img, txt, vec, pe, attn_mask=attn_mask,
                            modulation_dims_img=modulation_dims_img,
                            modulation_dims_txt=modulation_dims_txt,
                            transformer_options=transformer_options)

    dev = img.device
    inorm, tnorm = self.img_attn.norm, self.txt_attn.norm
    q, k, v = K.qkv2_norm_rope(
        txt_qkv.contiguous(), img_qkv.contiguous(),
        pe.reshape(B * (S_txt + S_img), -1).contiguous(),
        _norm_w(tnorm.query_norm, dev), _norm_w(tnorm.key_norm, dev),
        _norm_w(inorm.query_norm, dev), _norm_w(inorm.key_norm, dev),
        H, D,
        _eps(inorm.query_norm, img.dtype), _eps(inorm.key_norm, img.dtype),
    )
    del img_qkv, txt_qkv
    _announce("double")

    # RoPE already applied — go straight to the router, not flux math.attention.
    attn = optimized_attention(q, k, v, H, skip_reshape=True,
                               transformer_options=transformer_options)
    del q, k, v
    txt_attn, img_attn = attn[:, :S_txt], attn[:, S_txt:]

    # gate residual as one addcmul pass (eager: mul then +=, two passes).
    img = torch.addcmul(img, self.img_attn.proj(img_attn), img_mod1.gate)
    del img_attn
    img = torch.addcmul(
        img,
        self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale),
                               img_mod2.shift, None)),
        img_mod2.gate)

    txt = torch.addcmul(txt, self.txt_attn.proj(txt_attn), txt_mod1.gate)
    del txt_attn
    txt = torch.addcmul(
        txt,
        self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale),
                               txt_mod2.shift, None)),
        txt_mod2.gate)

    return img, txt


def _fast_single_forward(self, x, vec, pe, attn_mask=None, modulation_dims=None,
                         transformer_options={}):
    K = _kernel
    patches = transformer_options.get("patches", {})
    B, S, _ = x.shape
    H = self.num_heads
    D = self.hidden_size // H
    if (
        (K is None and _fallback("single", "kernel unavailable"))
        or (x.device.type != "mps" and _fallback("single", "non-MPS"))
        or (x.dtype != torch.bfloat16 and _fallback("single", f"dtype {x.dtype}"))
        or (attn_mask is not None and _fallback("single", "attn_mask"))
        or (modulation_dims is not None and _fallback("single", "modulation_dims"))
        or (("attn1_patch" in patches or "attn1_output_patch" in patches)
            and _fallback("single", "attn1 patches"))
        or ((D > 128 or D % 32 != 0) and _fallback("single", f"D={D}"))
        or (not _pe_ok(pe, B, S, D)
            and _fallback("single", f"pe {None if pe is None else (pe.dtype, tuple(pe.shape))}"))
    ):
        return _orig_single(self, x, vec, pe, attn_mask=attn_mask,
                            modulation_dims=modulation_dims,
                            transformer_options=transformer_options)

    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.modules.attention import optimized_attention

    if self.modulation:
        mod, _ = self.modulation(vec)
    else:
        mod = vec   # global modulation (Flux2): precomputed ModulationOut
    lin = self.linear1(
        apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, None))
    if lin.dtype != torch.bfloat16:
        return _orig_single(self, x, vec, pe, attn_mask=attn_mask,
                            modulation_dims=modulation_dims,
                            transformer_options=transformer_options)

    # linear1 output is [B, S, 3*H*D + mlp_first]; the kernel reads the qkv
    # columns via the row stride — no split copy.
    q, k, v = K.qkv_norm_rope(
        lin, pe.reshape(B * S, -1).contiguous(),
        _norm_w(self.norm.query_norm, x.device), _norm_w(self.norm.key_norm, x.device),
        H, H, D,
        _eps(self.norm.query_norm, x.dtype), _eps(self.norm.key_norm, x.dtype),
    )

    _announce("single")
    attn = optimized_attention(q, k, v, H, skip_reshape=True,
                               transformer_options=transformer_options)
    del q, k, v

    mlp = lin[..., 3 * self.hidden_size:]
    if self.yak_mlp:
        mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) \
            * mlp[..., :self.mlp_hidden_dim_first // 2]
    else:
        mlp = self.mlp_act(mlp)
    output = self.linear2(torch.cat((attn, mlp), 2))
    x = torch.addcmul(x, output, mod.gate)
    return x


def install():
    global _installed, _orig_double, _orig_single, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_FLUX_FAST", "1") != "1":
        return  # per-patch kill switch (A/B isolation from rope_fast/lumina)
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.flux.layers as FL
    except Exception as e:
        print(f"{TAG} comfy flux layers unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "qkv2_norm_rope"):
        print(f"{TAG} dit_ext qkv2 kernel unavailable; flux fast path stays off.")
        return

    _orig_double = FL.DoubleStreamBlock.forward
    FL.DoubleStreamBlock.forward = _fast_double_forward
    _orig_single = FL.SingleStreamBlock.forward
    FL.SingleStreamBlock.forward = _fast_single_forward
    _installed = True
    print(f"{TAG} Flux double/single block fast path on MPS: fused "
          f"qkv+QKNorm+RoPE+seq-cat via Metal (eager fallback per call).")
