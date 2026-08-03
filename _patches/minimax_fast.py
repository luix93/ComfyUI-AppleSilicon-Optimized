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

import torch

TAG = "[AppleSilicon-Optimized/minimax_fast]"

_installed = False
_kernel = None
_orig = None
_announced = False
_dead = False           # latched if the kernel ever raises
_wcache = {}            # id-keyed bf16 -> fp32 norm-weight cache


def _fp32_w(w):
    """Cached fp32 copy of a [D] norm weight (bf16->fp32 is exact).

    Keyed by storage pointer: these are module buffers that live for the whole
    run, and a reload/offload allocates a new one. Cheap insurance against
    doing 100 tiny casts per step.
    """
    if w.dtype == torch.float32 and w.is_contiguous():
        return w
    key = (w.data_ptr(), w.dtype, w.device.index)
    c = _wcache.get(key)
    if c is None or c.shape != w.shape:
        c = w.float().contiguous()
        if len(_wcache) > 64:
            _wcache.clear()
        _wcache[key] = c
    return c


_orig_block = None
_rows_cache = {}        # (segments, S) -> int32 [S] token->mod-row map
_block_announced = False
_block_dead = False


def _seg_rows(segments, S, device):
    """int32 [S] mapping each packed token to its adaln modulation row."""
    key = (tuple((int(a), int(b), int(r)) for a, b, r in segments), int(S),
           device.type, getattr(device, "index", None))
    t = _rows_cache.get(key)
    if t is None:
        t = torch.zeros(S, dtype=torch.int32)
        covered = 0
        for a, b, r in segments:
            t[a:b] = int(r)
            covered += int(b) - int(a)
        if covered != S:
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


def _install_block_fusion(M):
    """Fuse each DiT block's two norm+modulate chains into one Metal pass."""
    global _orig_block
    _orig_block = M.DiTBlock.forward

    def fast_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        global _block_announced, _block_dead
        try:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaln_proj(t_emb)
            h = _norm_mod(x, self.norm1, shift_msa, scale_msa, mod_segments)
            if h is None:
                return _orig_block(self, x, t_emb, mod_segments, rope_freqs,
                                   transformer_options=transformer_options)
            if not _block_announced:
                _block_announced = True
                sys.stderr.write(f"{TAG} fused block glue engaged "
                                 f"(segmented norm+modulate).\n")
            x = M._mod_gate(x, gate_msa,
                            self.attn(h, rope_freqs=rope_freqs,
                                      transformer_options=transformer_options),
                            mod_segments)
            h = _norm_mod(x, self.norm2, shift_mlp, scale_mlp, mod_segments)
            if h is None:
                h = M._mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
            return M._mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
        except Exception as e:
            _block_dead = True
            sys.stderr.write(f"{TAG} block fusion disabled ({str(e)[:200]})\n")
            return _orig_block(self, x, t_emb, mod_segments, rope_freqs,
                               transformer_options=transformer_options)

    M.DiTBlock.forward = fast_forward


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

    # Block glue: fuse the two RMSNorm + per-segment modulate chains per block
    # (eager runs 6 elementwise passes over a 57 MB tensor for each).
    if (os.environ.get("ASFP8_MINIMAX_BLOCK", "1") == "1"
            and hasattr(_kernel, "rms_mod_seg")):
        try:
            import comfy.ldm.minimax.model as MM
            if not getattr(MM.DiTBlock.forward, "_asfp8_wrapped", False):
                _install_block_fusion(MM)
                MM.DiTBlock.forward._asfp8_wrapped = True
        except Exception as e:
            print(f"{TAG} block fusion not installed ({e!r}).")

    _installed = True
    print(f"{TAG} MiniMax H3 rope fused: RMSNorm + partial split-half RoPE in "
          f"one in-place Metal pass (9.4x over the eager chain).")
