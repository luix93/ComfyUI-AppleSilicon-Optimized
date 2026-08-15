"""Patch #33 — MiniMax H3 fused RMSNorm + partial split-half RoPE (ASFP8_DIT_EXT=1).

H3's Attention.forward hands q and k to comfy_kitchen's eager
``rms_rope_split_half_``, which for every one of the 50 DiT blocks walks ~10
elementwise passes over two 77 MB tensors: full-width RMSNorm, a slice for the
rotary part, a reshape/movedim pair, two broadcast multiplies, an add, a cat,
and two in-place copies. Measured M5 Max at the 864x480 1s shape (S=5378,
H=56, D=128): **13.6 ms per call, 683 ms per step** — the single largest
eager-overhead item in the model.

`dit_ext.rms_rope_sh_partial` does the whole thing in one pass, in place, on
the strided q/k views of the fused qkv buffer: one threadgroup per (token,
slot), fp32 norm accumulation, the rotary pair (i, i+R) resolved through
threadgroup memory so no cross-lane traffic is needed, and the non-rotary tail
written straight through. **1.46 ms per call, 73 ms per step — 9.4x.**

Numerics: the kernel reproduces the eager rounding sequence exactly, which is
NOT one fused fp32 expression — F.rms_norm rounds the normalized value to bf16
and then multiplies by the bf16 weight (a second round), and the 2x2 rotation
rounds each product before rounding their sum (the rotation table is stored at
model dtype, so torch rounds after every op; verified against torch's own
`a*b + c*d`). With that sequence the fused output is bit-identical on the norm
and matches the full path to within fp32->bf16 tie-breaking: measured
**67 of 38.5M elements differ by exactly 1 bf16 ULP** (1.7e-6 of the tensor) at
the production shape — three orders of magnitude below the W8A8 error this
model's linears already carry.

Kill switch: ASFP8_MINIMAX_FAST=0. Non-MPS, non-bf16, unexpected layouts and
any other model's rope all fall through to the original implementation.
"""

import os
import sys
import weakref

import torch

TAG = "[AppleSilicon-Optimized/minimax_fast]"

_installed = False
_kernel = None
_orig = None
_announced = False
_dead = False           # latched if the kernel ever raises
_wcache = {}            # id(w) -> (weakref, version, data_ptr, fp32 copy)


def _fp32_w(w):
    """Cached fp32 copy of a [D] norm weight (bf16->fp32 is exact).

    Tensor identity + version are both checked.  A storage-pointer-only cache
    can return stale data after allocator reuse, while H3's 100 live Q/K norm
    weights exceed the old 64-entry clear-all cache and made it churn every
    denoising step.
    """
    if w.dtype == torch.float32 and w.is_contiguous():
        return w
    key = id(w)
    entry = _wcache.get(key)
    # Inference-mode tensors deliberately have no version counter; accessing
    # the property raises rather than returning a sentinel. Model norm weights
    # are immutable during inference, so identity + storage pointer remains a
    # safe cache key for that case.
    try:
        version = w._version
    except RuntimeError:
        version = None
    ptr = w.data_ptr()
    c = None
    if entry is not None:
        wr, old_version, old_ptr, cached = entry
        if wr() is w and old_version == version and old_ptr == ptr \
                and cached.shape == w.shape:
            c = cached
    if c is None:
        c = w.float().contiguous()
        try:
            wr = weakref.ref(w, lambda _r, k=key: _wcache.pop(k, None))
            _wcache[key] = (wr, version, ptr, c)
        except TypeError:
            pass
        if len(_wcache) > 256:
            for dead_key, dead_entry in list(_wcache.items()):
                if dead_entry[0]() is None:
                    _wcache.pop(dead_key, None)
    return c


_orig_block = None
_rows_cache = {}        # (segments, S) -> int32 [S] token->mod-row map
_block_announced = False
_block_dead = False
_orig_attn = None
_attn_announced = False
_attn_dead = False
_CROSS_BLOCK_KEY = "_asfp8_h3_cross_block"


def _owned_attention_inputs(q, k, v):
    """Use ComfyUI's single-owner Q/K/V handoff when the API exists."""
    try:
        import comfy.ldm.modules.attention as attention
        container = getattr(attention, "AttentionTensorContainer", None)
    except ImportError:
        container = None
    if container is None:
        return q, k, v
    return container(q), container(k), container(v)


def _seg_rows(segments, S, device):
    """int32 [S] mapping each packed token to its adaln modulation row."""
    key = (tuple((int(a), int(b), int(r)) for a, b, r in segments), int(S),
           device.type, getattr(device, "index", None))
    t = _rows_cache.get(key)
    if t is None:
        t = torch.zeros(S, dtype=torch.int32)
        cursor = 0
        for a, b, r in segments:
            a, b, r = int(a), int(b), int(r)
            if a != cursor or b < a or b > S or r < 0:
                return None
            t[a:b] = int(r)
            cursor = b
        if cursor != S:
            return None          # segments must tile the sequence exactly
        t = t.to(device)
        if len(_rows_cache) > 16:
            _rows_cache.clear()
        _rows_cache[key] = t
    return t


def _norm_mod(x, norm, shift, scale, segments):
    """RMSNorm(x)*w*(1+scale[row]) + shift[row], fused. None if ineligible."""
    if _kernel is None or _block_dead:
        return None
    if x.dim() != 2 or not x.is_contiguous() or x.dtype != torch.bfloat16:
        return None
    w = getattr(norm, "weight", None)
    if w is None:
        return None
    S, Dm = x.shape
    if Dm % 32 or scale.dim() != 2 or scale.shape[1] != Dm:
        return None
    rows = _seg_rows(segments, S, x.device)
    if rows is None:
        return None
    import comfy.model_management as mm
    wb = mm.cast_to(w, dtype=torch.bfloat16, device=x.device).contiguous()
    # eager does scale[row].to(h.dtype) per segment — same single bf16 round
    sc = scale.to(torch.bfloat16).contiguous()
    sh = shift.to(torch.bfloat16).contiguous()
    return _kernel.rms_mod_seg(x, wb, sc, sh, rows, float(norm.eps))


def _gate_norm_mod(x, other, gate, norm, shift, scale, segments):
    """Gated residual update plus the next RMSNorm/modulation, or None."""
    if (_kernel is None or _block_dead
            or not hasattr(_kernel, "gate_norm_mod_seg")):
        return None
    if (x.dim() != 2 or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or x.device.type != "mps"
            or not isinstance(other, torch.Tensor) or other.shape != x.shape
            or other.device != x.device or other.dtype != torch.bfloat16
            or not other.is_contiguous()
            or not all(isinstance(t, torch.Tensor)
                       for t in (gate, scale, shift))):
        return None
    w = getattr(norm, "weight", None)
    if w is None:
        return None
    S, Dm = x.shape
    if (Dm % 32 or Dm * 2 > 30 * 1024 or gate.dim() != 2
            or gate.shape[1] != Dm or scale.shape != gate.shape
            or shift.shape != gate.shape):
        return None
    rows = _seg_rows(segments, S, x.device)
    if rows is None:
        return None
    import comfy.model_management as mm
    wb = mm.cast_to(w, dtype=torch.bfloat16, device=x.device).contiguous()
    ga = gate.to(device=x.device, dtype=torch.bfloat16).contiguous()
    sc = scale.to(device=x.device, dtype=torch.bfloat16).contiguous()
    sh = shift.to(device=x.device, dtype=torch.bfloat16).contiguous()
    return _kernel.gate_norm_mod_seg(
        x, other, ga, wb, sc, sh, rows, float(norm.eps))


def _gate_residual(x, other, gate, segments):
    """One mapped pass for all segmented gated residual updates, or None."""
    if (_kernel is None or _block_dead
            or not hasattr(_kernel, "gate_residual_seg")):
        return None
    if (x.dim() != 2 or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or x.device.type != "mps"
            or not isinstance(other, torch.Tensor) or other.shape != x.shape
            or other.device != x.device or other.dtype != torch.bfloat16
            or not other.is_contiguous() or not isinstance(gate, torch.Tensor)
            or gate.dim() != 2 or gate.shape[1] != x.shape[1]):
        return None
    rows = _seg_rows(segments, x.shape[0], x.device)
    if rows is None:
        return None
    return _kernel.gate_residual_seg(
        x, other, gate.to(device=x.device, dtype=torch.bfloat16).contiguous(), rows)


def _install_block_fusion(M):
    """Fuse H3's segmented residual, norm, and modulation block glue."""
    global _orig_block
    _orig_block = M.DiTBlock.forward

    def fast_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        global _block_announced, _block_dead
        # This prefix does not mutate x, so it remains safe to use the original
        # block as a fallback if model/layout eligibility changes.
        try:
            carry = transformer_options.pop(_CROSS_BLOCK_KEY, None)
            if (isinstance(carry, tuple) and len(carry) == 4
                    and carry[0]() is self and carry[1] is x):
                mods, h = carry[2], carry[3]
                (shift_msa, scale_msa, gate_msa,
                 shift_mlp, scale_mlp, gate_mlp) = mods
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                    self.adaln_proj(t_emb)
                h = _norm_mod(
                    x, self.norm1, shift_msa, scale_msa, mod_segments)
            if h is None:
                return _orig_block(self, x, t_emb, mod_segments, rope_freqs,
                                   transformer_options=transformer_options)
        except Exception as e:
            _block_dead = True
            sys.stderr.write(f"{TAG} block fusion disabled ({str(e)[:200]})\n")
            return _orig_block(self, x, t_emb, mod_segments, rope_freqs,
                               transformer_options=transformer_options)

        attention = self.attn(h, rope_freqs=rope_freqs,
                              transformer_options=transformer_options)
        try:
            fused = _gate_norm_mod(
                x, attention, gate_msa, self.norm2,
                shift_mlp, scale_mlp, mod_segments)
        except Exception as e:
            # The kernel updates x in place. Never rerun the original block on
            # a possibly mutated stream; surface the real error for this run.
            _block_dead = True
            sys.stderr.write(f"{TAG} fused gate+norm disabled "
                             f"({str(e)[:200]})\n")
            raise
        if fused is None:
            x = M._mod_gate(x, gate_msa, attention, mod_segments)
            del attention
            h = _norm_mod(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
            if h is None:
                h = M._mod_scale_shift(
                    self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        else:
            x, h = fused
            # The kernel has consumed attention. Drop its Python owner before
            # fc1 creates H3's much wider MLP activation; otherwise both large
            # buffers overlap even though no later operation can use attention.
            del attention

        mlp = self.mlp(h)
        del h
        out = None
        # No block-replacement may intervene between the two blocks. When the
        # next AdaLN weight is already resident, let the current MLP gate also
        # produce the next block's attention-normalized input and carry it
        # across the Python loop. This is the exact h3.c cross-block fusion;
        # caches/token reduction/Spectrum automatically disable it by owning a
        # block boundary.
        replacements = transformer_options.get(
            "patches_replace", {}).get("dit", {})
        next_ref = getattr(self, "_asfp8_next_block", None)
        next_block = next_ref() if next_ref is not None else None
        next_weight = getattr(
            getattr(getattr(next_block, "adaln_proj", None), "linear", None),
            "weight", None)
        if (not replacements and next_block is not None
                and os.environ.get("ASFP8_MINIMAX_CROSS_BLOCK", "1") == "1"
                and getattr(getattr(next_weight, "device", None), "type", None)
                    == "mps"):
            try:
                next_mods = next_block.adaln_proj(t_emb)
                cross = _gate_norm_mod(
                    x, mlp, gate_mlp, next_block.norm1,
                    next_mods[0], next_mods[1], mod_segments)
                if cross is not None:
                    out, next_h = cross
                    transformer_options[_CROSS_BLOCK_KEY] = (
                        weakref.ref(next_block), out, next_mods, next_h)
            except Exception as e:
                _block_dead = True
                sys.stderr.write(f"{TAG} cross-block gate+AdaLN disabled "
                                 f"({str(e)[:200]})\n")
                raise
        try:
            if out is None:
                out = _gate_residual(x, mlp, gate_mlp, mod_segments)
        except Exception as e:
            _block_dead = True
            sys.stderr.write(f"{TAG} fused final gate disabled "
                             f"({str(e)[:200]})\n")
            raise
        if out is None:
            out = M._mod_gate(x, gate_mlp, mlp, mod_segments)
        del mlp
        if not _block_announced:
            _block_announced = True
            sys.stderr.write(f"{TAG} fused block glue engaged "
                             "(segmented residual + RMSNorm + modulation).\n")
        return out

    M.DiTBlock.forward = fast_forward


def _install_cross_block_links(M):
    """Give each newly constructed H3 block a non-registering weak next link."""
    original = M.MiniMaxH3Model.__init__
    if getattr(original, "_asfp8_cross_links", False):
        return

    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for current, following in zip(self.blocks, self.blocks[1:]):
            current._asfp8_next_block = weakref.ref(following)

    wrapped._asfp8_cross_links = True
    wrapped._asfp8_original = original
    M.MiniMaxH3Model.__init__ = wrapped


def _install_attention_fusion(M):
    """Write H3 q/k/v directly in flash-attention layout.

    The original path applies the in-place q/k kernel and then asks the router
    to materialize three transposed contiguous tensors.  The new kernel folds
    that layout conversion into Q/K RMSNorm + partial RoPE and copies V in the
    same pass.  It is bit-identical to the existing optimized chain.
    """
    global _orig_attn
    _orig_attn = M.Attention.forward

    def fast_attention(self, x, rope_freqs=None, transformer_options={}):
        global _attn_announced, _attn_dead
        if (_attn_dead or rope_freqs is None
                or not hasattr(_kernel, "qkv_rms_rope_sh_partial")
                or not isinstance(x, torch.Tensor) or x.device.type != "mps"
                or x.dtype != torch.bfloat16 or x.dim() != 2
                or not x.is_contiguous()):
            return _orig_attn(self, x, rope_freqs=rope_freqs,
                              transformer_options=transformer_options)

        s = x.shape[0]
        h, d = int(self.heads), int(self.head_dim)
        if (d > 256 or d % 32 or rope_freqs.dtype != torch.bfloat16
                or not rope_freqs.is_contiguous() or rope_freqs.dim() < 3
                or rope_freqs.numel() != s * rope_freqs.shape[-3] * 4
                or 2 * rope_freqs.shape[-3] > d):
            return _orig_attn(self, x, rope_freqs=rope_freqs,
                              transformer_options=transformer_options)

        import comfy.model_management as mm
        qw = mm.cast_to(self.q_norm.weight, dtype=torch.bfloat16,
                        device=x.device).contiguous()
        kw = mm.cast_to(self.k_norm.weight, dtype=torch.bfloat16,
                        device=x.device).contiguous()
        qw_fp32 = _fp32_w(qw)
        kw_fp32 = _fp32_w(kw)
        direct_qkv = None
        fused_int8_epilogue = False
        try:
            from . import int8_linear_kernel_mps as i8k
            direct_qkv = i8k.int8_qkv_rms_rope_prequant(
                self.qkv_proj, x, rope_freqs, qw_fp32, kw_fp32,
                h, d, float(self.q_norm.eps))
            fused_int8_epilogue = direct_qkv is not None
            if hasattr(_kernel, "rms_rope_attention_partial"):
                if direct_qkv is None:
                    direct_qkv = i8k.int8_qkv_prequant(
                        self.qkv_proj, x, h, d)
        except Exception:
            direct_qkv = None
            fused_int8_epilogue = False

        used_direct_qkv = direct_qkv is not None
        if used_direct_qkv:
            q, k, v = direct_qkv
            del direct_qkv
            if not fused_int8_epilogue:
                try:
                    _kernel.rms_rope_attention_partial(
                        q, k, rope_freqs, qw_fp32, kw_fp32,
                        float(self.q_norm.eps))
                except Exception as e:
                    _attn_dead = True
                    sys.stderr.write(f"{TAG} direct QKV norm/RoPE disabled "
                                     f"({str(e)[:200]})\n")
                    del q, k, v
                    return _orig_attn(self, x, rope_freqs=rope_freqs,
                                      transformer_options=transformer_options)
        else:
            # Projection failures are real model/runtime failures, not a reason
            # to recompute the same projection through the original path.
            qkv = self.qkv_proj(x)
            if (not isinstance(qkv, torch.Tensor) or qkv.dtype != torch.bfloat16
                    or qkv.dim() != 2 or qkv.shape != (s, 3 * h * d)
                    or not qkv.is_contiguous()):
                del qkv
                return _orig_attn(self, x, rope_freqs=rope_freqs,
                                  transformer_options=transformer_options)
        try:
            if not used_direct_qkv:
                q, k, v = _kernel.qkv_rms_rope_sh_partial(
                    qkv, rope_freqs, qw_fp32, kw_fp32, h, d,
                    float(self.q_norm.eps))
        except Exception as e:
            _attn_dead = True
            sys.stderr.write(f"{TAG} fused attention layout disabled "
                             f"({str(e)[:200]})\n")
            if not used_direct_qkv:
                del qkv
            return _orig_attn(self, x, rope_freqs=rope_freqs,
                              transformer_options=transformer_options)
        # q/k/v are independent output allocations. The projection buffer is
        # no longer needed and should not overlap attention/output-projection
        # allocations merely because this Python frame still references it.
        if not used_direct_qkv:
            del qkv

        if not _attn_announced:
            _attn_announced = True
            sys.stderr.write(f"{TAG} fused qkv + partial-RoPE + attention "
                             "layout engaged.\n")
        q_arg, k_arg, v_arg = _owned_attention_inputs(q, k, v)
        del q, k, v
        out4 = M.optimized_attention(
            q_arg, k_arg, v_arg, h, mask=None, skip_reshape=True,
            skip_output_reshape=True,
            transformer_options=transformer_options)
        del q_arg, k_arg, v_arg
        # Keep the native [B,H,S,D] result through the INT8 output projection:
        # the prequant kernel merges heads, rotates 256-wide convrot groups,
        # and rowwise-quantizes in one pass.  H3's D=128 means each group spans
        # two adjacent heads; the generalized kernel handles that directly.
        try:
            from . import int8_linear_kernel_mps as i8k
            projected = i8k.int8_o_proj_prequant(self.out_proj, out4)
        except Exception:
            projected = None
        if projected is not None:
            del out4
            return projected.squeeze(0)
        out = out4.transpose(1, 2).reshape(1, s, h * d)
        del out4
        return self.out_proj(out.squeeze(0))

    M.Attention.forward = fast_attention


def _install_mlp_bridge(M):
    """Give H3's paired INT8 FC1/FC2 layers one allocation-aware fast path."""
    original = M.MLP.forward
    if getattr(original, "_asfp8_mlp_bridge", False):
        return

    def fast_mlp(self, x):
        from . import int8_linear_kernel_mps as i8k
        result = i8k.int8_mlp_bridge(self.fc1, self.fc2, x)
        if result is not None:
            return result
        return original(self, x)

    fast_mlp._asfp8_mlp_bridge = True
    fast_mlp._asfp8_original = original
    M.MLP.forward = fast_mlp


def _eligible(q, k, freqs, qw, kw, rot_dim):
    if _kernel is None or _dead:
        return False
    if q.device.type != "mps":
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if freqs.dtype != torch.bfloat16 or not freqs.is_contiguous():
        return False
    if q.dim() != 4 or q.shape != k.shape or q.shape[0] != 1:
        return False
    S, H, D = q.shape[1], q.shape[2], q.shape[3]
    if D > 256 or D % 32 or D < 32:
        return False
    # inner [H,D] contiguous, rows strided in lockstep (the qkv split views)
    if q.stride(3) != 1 or q.stride(2) != D or k.stride(3) != 1 or k.stride(2) != D:
        return False
    if q.stride(1) != k.stride(1):
        return False
    if freqs.dim() < 3:
        return False
    R = freqs.shape[-3]
    if freqs.numel() != S * R * 4 or 2 * R > D:
        return False
    # rot_dim=0 means "rotate everything"; anything else must match the table
    if rot_dim and rot_dim != 2 * R:
        return False
    if not rot_dim and 2 * R != D:
        return False
    if qw is None or kw is None:
        return False
    if qw.numel() != D or kw.numel() != D:
        return False
    return True


def install():
    global _installed, _kernel, _orig
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return  # opt-in, same switch as the other DiT kernels
    if os.environ.get("ASFP8_MINIMAX_FAST", "1") != "1":
        return  # per-patch kill switch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        import comfy.quant_ops as qops
    except Exception as e:
        print(f"{TAG} comfy.quant_ops unavailable ({e!r}); not installed.")
        return
    ck = getattr(qops, "ck", None)
    if ck is None or not hasattr(ck, "rms_rope_split_half_"):
        return  # older comfy without the kitchen rope entry point

    from .dit_ext import loader
    _kernel = loader.module()
    if _kernel is None or not hasattr(_kernel, "rms_rope_sh_partial"):
        print(f"{TAG} dit_ext rms_rope_sh_partial unavailable; H3 rope stays eager.")
        return

    _orig = ck.rms_rope_split_half_
    if getattr(_orig, "_asfp8_wrapped", False):
        _installed = True
        return

    def rms_rope_split_half_wrapped(q, k, freqs_cis, q_scale, k_scale=None,
                                    epsilon=1e-6, rot_dim=0):
        global _dead, _announced
        ks = q_scale if k_scale is None else k_scale
        if _eligible(q, k, freqs_cis, q_scale, ks, rot_dim):
            try:
                _kernel.rms_rope_sh_partial(
                    q, k, freqs_cis, _fp32_w(q_scale), _fp32_w(ks), float(epsilon))
                if not _announced:
                    _announced = True
                    sys.stderr.write(f"{TAG} fused RMSNorm+partial-RoPE engaged "
                                     f"(in place on the qkv views).\n")
                return q, k
            except Exception as e:
                # Latch: a Metal compile break would otherwise re-attempt the
                # expensive build on every one of the 50 blocks, every step.
                _dead = True
                sys.stderr.write(f"{TAG} kernel unavailable, using eager rope "
                                 f"({str(e)[:200]})\n")
        return _orig(q, k, freqs_cis, q_scale, k_scale, epsilon, rot_dim)

    rms_rope_split_half_wrapped._asfp8_wrapped = True
    ck.rms_rope_split_half_ = rms_rope_split_half_wrapped

    try:
        import comfy.ldm.minimax.model as MM
        _install_cross_block_links(MM)
        if (os.environ.get("ASFP8_MINIMAX_ATTN", "1") == "1"
                and hasattr(_kernel, "qkv_rms_rope_sh_partial")
                and not getattr(MM.Attention.forward, "_asfp8_wrapped", False)):
            _install_attention_fusion(MM)
            MM.Attention.forward._asfp8_wrapped = True
        if (os.environ.get("ASFP8_INT8_EXT") == "1"
                and os.environ.get("ASFP8_INT8_MLP_BRIDGE", "1") == "1"):
            _install_mlp_bridge(MM)

        # Block glue: fuse the two RMSNorm + per-segment modulate chains per
        # block (eager runs 6 elementwise passes over a 57 MB tensor for each).
        if (os.environ.get("ASFP8_MINIMAX_BLOCK", "1") == "1"
                and hasattr(_kernel, "rms_mod_seg")):
            if not getattr(MM.DiTBlock.forward, "_asfp8_wrapped", False):
                _install_block_fusion(MM)
                MM.DiTBlock.forward._asfp8_wrapped = True
        if os.environ.get("ASFP8_MINIMAX_GRAPH", "1") == "1":
            from .minimax_h3_graph import rewrite_h3_graph
            for result in rewrite_h3_graph(MM, _kernel, _seg_rows):
                if not result.applied:
                    sys.stderr.write(
                        f"{TAG} graph pattern {result.pattern} skipped "
                        f"({result.reason}).\n")
    except Exception as e:
        print(f"{TAG} model-specific fusion not installed ({e!r}).")

    _installed = True
    print(f"{TAG} MiniMax H3 rope fused: RMSNorm + partial split-half RoPE in "
          f"one in-place Metal pass (9.4x over the eager chain).")
