"""Patch #25 — GQA-native Krea2 attention (opt-in ASFP8_DIT_EXT=1).

Krea2 is GQA (48 query heads, 12 KV heads, D=128) but ComfyUI's
optimized_attention API takes one head count, so the eager path
repeat_interleaves K and V to 48 heads before every attention call — two
~107 MB materializations per call (224 calls per 8-step run) and 4x the KV
read traffic inside the attention kernel. DrawThings runs GQA natively;
this patch closes that structural gap:

  * dit_ext qkv3_norm_rope fuses the post-linear chain (rearrange -> per-head
    QK RMSNorm -> RoPE -> attention layout) in one pass, keeping k/v at their
    NATIVE 12 heads. Krea2 norm semantics differ from flux: fp32 (1+scale)
    weight with a single final bf16 rounding, eps 1e-5 — the kernel matches
    them exactly.
  * ASFP8_ATTN_BACKEND=pytorch selects native PyTorch 2.14 GQA while
    keeping the same fused preparation and output gate. Otherwise attention
    goes straight to mfa4_ext flash41, whose kernel has supported
    grouped KV (Hq % Hkv == 0) all along — each query head indexes its KV
    group in-kernel; nothing is repeated.

The output projection stays eager. On the validated M5 Ultra stack,
krea2_glue fuses the sigmoid gate with attention-layout conversion.
TextFusion adapter blocks use
the same Attention class but pass masks/no freqs, so they fall back cleanly.

Guarded per call: non-MPS, non-bf16, mask, no/mismatched freqs, D > 128,
Hq % Hkv != 0, or a missing flash41 build all fall back to the original
forward. Kill switch: ASFP8_KREA2_FAST=0 (under ASFP8_DIT_EXT=1;
ASFP8_KREA2_FAST_DEBUG=1 logs each distinct fallback reason once).
"""

import os
import sys

import torch

from . import torch_attention

TAG = "[AppleSilicon-Optimized/krea2_fast]"

_installed = False
_orig_forward = None
_kernel = None
_fa41 = None
_glue = None
_announced = False
_reasons = set()
_na_dead = False   # latched when flash_fwd_na fails once (e.g. OS API break)
_tile32_library = None
_tile32_dead = False
_tile32_calls = 0

# Same rationale as the router default: bf16 score tiles match the stock bmm
# path's rounding; ASFP8_FA41_F32S=1 reverts to fp32 tiles everywhere.
_BF16S = os.environ.get("ASFP8_FA41_F32S") != "1"


def _fallback(reason):
    if os.environ.get("ASFP8_KREA2_FAST_DEBUG") == "1" and reason not in _reasons:
        _reasons.add(reason)
        sys.stderr.write(f"{TAG} fallback: {reason}\n")
    return True


def _announce():
    global _announced
    if not _announced:
        _announced = True
        sys.stderr.write(f"{TAG} GQA-native fused attention engaged.\n")


def _w1(norm, device):
    # Krea2 RMSNorm stores the weight zero-centered; effective weight is 1+scale.
    return (norm.scale.to(dtype=torch.float32, device=device) + 1.0).contiguous()


def _query_tile32(self, q, k, v):
    """Exact query scheduling for the measured 1.5–3.5 MP floating-weight path."""
    global _tile32_library, _tile32_dead, _tile32_calls
    if (_tile32_dead or os.environ.get("ASFP8_KREA2_TILE32", "1") != "1"
            or torch.__version__.split("+")[0] != "2.14.1"
            or os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0"
            or q.ndim != 4 or q.shape[1] != 48 or k.shape[1] != 12
            or not 6144 <= q.shape[2] <= 14336
            or any(type(getattr(layer, "weight", None)) not in (torch.Tensor, torch.nn.Parameter)
                   for layer in (self.wq, self.wk, self.wv, self.wo, self.gate))):
        return None
    from . import torch_mpp_qcache as mpp
    kwargs = dict(scale=128 ** -0.5, enable_gqa=True)
    if not mpp._supports(q, k, v, kwargs):
        return None
    try:
        if _tile32_library is None:
            old = 'bfloat, 64, 32, 128, 4, 1, false, false, true, bfloat'
            new = 'bfloat, 32, 32, 128, 2, 1, false, false, true, bfloat'
            source = mpp._source()
            if source.count(old) != 1:
                raise RuntimeError('unreviewed query-cache entry')
            _tile32_library = torch.mps.compile_shader(source.replace(old, new))
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        params = mpp._parameters(tuple(q.shape), tuple(k.shape), q.stride(), k.stride(), v.stride(), 128 ** -0.5, False)
        _tile32_library.custom_attn(q, k, v, out, params, params, params,
                                   threads=(((q.shape[2] + 31) // 32) * 64, 48, 1),
                                   group_size=(64, 1, 1))
        _tile32_calls += 1
        return out
    except Exception as exc:
        _tile32_dead = True
        sys.stderr.write(f"{TAG} query tile32 unavailable; retaining existing attention ({exc}).\n")
        return None


def _fast_attn_forward(self, x, freqs=None, mask=None, transformer_options={}):
    Hq, Hkv, D = self.heads, self.kvheads, self.headdim
    B, S, _ = x.shape
    if (
        (_kernel is None and _fallback("kernel unavailable"))
        or (_fa41 is None and not torch_attention.enabled()
            and _fallback("attention backend unavailable"))
        or (x.device.type != "mps" and _fallback("non-MPS"))
        or (x.dtype != torch.bfloat16 and _fallback(f"dtype {x.dtype}"))
        or (mask is not None and _fallback("mask"))
        # comfy's wrap_attn contract: an active third-party attention hook
        # must not be silently bypassed — this path calls flash41 directly,
        # so hand the call back to the original (hook-aware) forward.
        or (("optimized_attention_override" in (transformer_options or {}))
            and _fallback("optimized_attention_override"))
        or (("block_index" in (transformer_options or {}) and any(
            (transformer_options.get("patches") or {}).get(name)
            for name in ("attn1_patch", "attn1_output_patch")))
            and _fallback("attention patches"))
        or ((D > 128 or D % 32 != 0) and _fallback(f"D={D}"))
        or ((Hkv <= 0 or Hq % Hkv != 0) and _fallback(f"heads {Hq}/{Hkv}"))
        or ((freqs is None or freqs.dtype != torch.float32
             or freqs.shape[-3:] != (D // 2, 2, 2)
             or freqs.numel() != B * S * (D // 2) * 4)
            and _fallback("freqs layout"))
    ):
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)

    q_lin, k_lin, v_lin = self.wq(x), self.wk(x), self.wv(x)
    gate = self.gate(x)
    if q_lin.dtype != torch.bfloat16 or k_lin.dtype != torch.bfloat16 \
            or v_lin.dtype != torch.bfloat16:
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)

    dev = x.device
    q, k, v = _kernel.qkv3_norm_rope(
        q_lin.contiguous(), k_lin.contiguous(), v_lin.contiguous(),
        freqs.reshape(B * S, -1).contiguous(),
        _w1(self.qknorm.qnorm, dev), _w1(self.qknorm.knorm, dev),
        Hq, Hkv, D, float(self.qknorm.qnorm.eps),
    )
    del q_lin, k_lin, v_lin
    _announce()

    global _na_dead
    try:
        # Native GQA: k/v stay at Hkv heads; flash41 groups them in-kernel.
        # Prefer the v2 per-simdgroup kernel where its Lk % 128 gate allows
        # (kill switch shared with the router: ASFP8_ATTN_NA=0).
        if torch_attention.supports(q, k, v, Hq, skip_reshape=True):
            out = _query_tile32(self, q, k, v)
            if out is None:
                out = torch_attention.sdpa(q, k, v, scale=D ** -0.5,
                                           enable_gqa=Hq != Hkv)
        elif (os.environ.get("ASFP8_ATTN_NA", "1") == "1" and not _na_dead
                and hasattr(_fa41, "flash_fwd_na")):
            try:
                out = _fa41.flash_fwd_na(
                    q, k, v, D ** -0.5,
                    krea_schedule=os.environ.get("ASFP8_KREA2_ATTN_SCHEDULE", "1") == "1")
            except Exception:
                _na_dead = True   # latch: don't re-attempt the compile per call
                out = _fa41.flash_fwd_bf16(q, k, v, D ** -0.5, False, 0, 0, _BF16S)
        else:
            out = _fa41.flash_fwd_bf16(q, k, v, D ** -0.5, False, 0, 0, _BF16S)
    except Exception:
        return _orig_forward(self, x, freqs=freqs, mask=mask,
                             transformer_options=transformer_options)
    del q, k, v
    if _glue is not None:
        gated = _glue.attention_gate(out, gate)
        if gated is not None:
            return self.wo(gated)
    out = out.transpose(1, 2).reshape(B, S, Hq * D)
    return self.wo(out * torch.sigmoid(gate))


def install():
    global _installed, _orig_forward, _kernel, _fa41, _glue
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_KREA2_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.ldm.krea2.model as K2
    except Exception as e:
        print(f"{TAG} comfy krea2 model unavailable ({e!r}); not installed.")
        return

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "qkv3_norm_rope"):
        print(f"{TAG} dit_ext qkv3 kernel unavailable; krea2 fast path stays off.")
        return

    # flash41 (mfa4_ext) for the native-GQA attention; without it the fused
    # prep would still force a repeat for comfy attention — not worth it.
    try:
        from .mfa4_ext import loader as _fa41_loader
        _fa41 = _fa41_loader.module()
        if _fa41 is not None and not hasattr(_fa41, "flash_fwd_bf16"):
            _fa41 = None
    except Exception:
        _fa41 = None
    if _fa41 is None and not torch_attention.enabled():
        print(f"{TAG} flash41 unavailable; krea2 GQA fast path stays off.")
        return

    _orig_forward = K2.Attention.forward
    K2.Attention.forward = _fast_attn_forward
    _installed = True
    from . import krea2_glue
    krea2_glue.install()
    _glue = krea2_glue
    print(f"{TAG} Krea2 GQA-native attention on MPS: fused q/k/v norm+RoPE at "
          f"native head counts + {'PyTorch SDPA' if torch_attention.enabled() else 'flash41'} grouped-KV (no repeat_interleave; "
          f"eager fallback per call).")
