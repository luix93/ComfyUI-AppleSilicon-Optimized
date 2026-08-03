"""Patch: route comfy_kitchen INT8 convrot layers through our bit-exact Metal kernel.

Background
----------
comfy ships int8 (`int8_tensorwise`, e.g. Krea2 convrot int8mixed) but **disables
the real int8 compute path on every non-CUDA platform**: `comfy.quant_ops` sets
``QUANT_ALGOS["int8_tensorwise"]["quantize_input"] = False`` (no CUDA -> no fast
int8 matmul). So on MPS every step instead runs **weight-only W8A16**: it
dequantizes the full int8 weight to bf16 *and un-rotates the whole convrot weight
in fp32* per layer per step (~50% of GPU time), then a bf16 ``F.linear``.

Naively flipping ``quantize_input = True`` does NOT work: comfy's forward then
pre-quantizes the activation **tensorwise** (a single scalar scale over the whole
[M,K] activation) via ``QuantizedTensor.from_float`` *before* convrot can spread
the outliers, then re-quantizes inside ``int8_linear``. Measured: that pre-quant
adds ~16% error (17x worse than W8A16) -> structured garbage. (That is exactly why
comfy disabled it for int8.)

What works (measured ~1.3% err, on par with W8A16's ~0.9%): the **clean W8A8**
path -- feed ``comfy_kitchen.int8_linear`` the *raw bf16* activation so it does the
convrot rotation first and *then* per-row quantization, then an int8xint8 matmul.

So this patch wraps the ``mixed_precision_ops`` factory and replaces the int8
``Linear.forward`` so that, on MPS, eligible int8 layers compute via our
kernel-backed ``int8_linear`` on the raw bf16 input -- skipping both comfy's lossy
pre-quant and the per-step fp32 weight dequant/un-rotation. The matmul runs on our
bit-exact INT8xINT8->INT32 Metal kernel (~102 TF/s, ~1.75x over bf16). For the
tensorwise-scale bf16 case (Krea2's int8mixed), the rescale ``float(C)*row_scale[m]``
and bias add are **fused into the kernel's store epilogue** (Cider's
``w8a8_matmul_fused_dequant``), so the int32 product never round-trips through
global memory -- bit-identical to the chunked path, ~1.2-1.65x faster per call.
Everything else (other quant formats, transposed weights, LoRA weight/bias
functions, non-MPS) falls back to comfy's original forward unchanged.

Opt-in: only active when ``ASFP8_INT8_EXT=1`` (the kernel build flag).
"""

import os
import sys

import os
import weakref

import torch

TAG = "[AppleSilicon-Optimized/int8_kernel]"

_installed = False
_orig_int8_linear = None
_kernel = None
_oproj_announced = False
_verify_dumped = False

# 1-slot cache of the rotated+quantized activation. Krea2-style blocks call the
# kernel with the SAME input tensor for wq/wk/wv/gate (and SwiGLU gate/up), so
# 3-4 of the ~2.3 ms rotate+quant passes per block are pure repeats. Keyed by a
# weakref to the pre-rotation tensor object: a live identical object guarantees
# identical bytes unless someone mutates it in-place between projections, which
# comfy's functional model blocks never do. Disable: ASFP8_INT8_ACT_CACHE=0.
_act_cache = None  # (x_weakref, convrot, groupsize, x_8, x_scale)


def _quant_cached(x, convrot, convrot_groupsize):
    """Rotate (if convrot) + rowwise-quantize x, reusing the previous result when
    the same live tensor object is quantized again with the same settings."""
    global _act_cache
    c = _act_cache
    if (
        c is not None
        and c[0]() is x
        and c[1] == convrot
        and c[2] == convrot_groupsize
    ):
        return c[3], c[4]

    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    xr = x
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        xr = _rotate_activation(x, h, convrot_groupsize)
    x_2d = xr.reshape(-1, xr.shape[-1])
    if (
        _kernel is not None
        and hasattr(_kernel, "i8_quant_rowwise")
        and x_2d.dtype in (torch.bfloat16, torch.float16)
    ):
        # Metal rowwise quant: single fused absmax+scale+quant pass, bit-exact
        # vs the eager path (verified incl. zero/tiny/huge rows; ~6x faster).
        x_8, x_scale = _kernel.i8_quant_rowwise(x_2d.contiguous())
    else:
        from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
        x_8, x_scale = quantize_int8_rowwise(x_2d)
        x_8 = x_8.contiguous()
    if os.environ.get("ASFP8_INT8_ACT_CACHE", "1") != "0":
        try:
            _act_cache = (weakref.ref(x), convrot, convrot_groupsize, x_8, x_scale)
        except TypeError:  # tensor subclass without weakref support
            _act_cache = None
    return x_8, x_scale


def _load_kernel():
    try:
        from .int8_ext import loader
    except Exception as e:  # pragma: no cover - import wiring
        print(f"{TAG} loader import failed: {e!r}")
        return None
    return loader.module()


def _int8_linear_kernel(
    x,
    weight,
    weight_scale,
    bias=None,
    out_dtype=torch.bfloat16,
    convrot=False,
    convrot_groupsize=256,
):
    """Kernel-backed drop-in for comfy_kitchen eager int8_linear.

    Semantics match the original exactly (optional convrot activation rotation,
    per-row int8 activation quant, int8xint8->int32 matmul, chunked rescale by
    weight_scale*row_scale, optional bias). Only the matmul backend differs (our
    NT kernel vs torch._int_mm) and the weight is used in its stored [N,K] layout.
    """
    if (
        _kernel is None
        or x.device.type != "mps"
        or weight.device.type != "mps"
        or weight.dtype != torch.int8
    ):
        return _orig_int8_linear(
            x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
        )

    orig_shape = x.shape
    x_8, x_scale = _quant_cached(x, convrot, convrot_groupsize)
    weight_scale = weight_scale.view(-1)

    # Fused fast path: when the rescale is purely per-row (tensorwise weight
    # scale, the int8_tensorwise case) and the output is bf16, fold
    # float(C)*row_scale[m] (+bias) straight into the kernel's store epilogue so
    # the int32 product never round-trips through global memory. Bit-identical to
    # the chunked path below (verified equal across convrot/bias/3D/M=1).
    if (
        out_dtype == torch.bfloat16
        and weight_scale.numel() == 1
        and hasattr(_kernel, "i8_matmul2d_nt_fused")
    ):
        row_scale = (weight_scale.float() * x_scale.reshape(-1).float()).contiguous()
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        result = _kernel.i8_matmul2d_nt_fused(
            x_8.contiguous(), weight.contiguous(), row_scale, bias_arg
        )
        return result.reshape(*orig_shape[:-1], weight.shape[0])

    # Channelwise fast path: per-output-channel weight scales ([N,1] checkpoints
    # — Z-Image / Krea2 convrot ship these, so the scalar path above never fires
    # for them). The kernel epilogue computes (col_scale[n] * row_scale[m]) first
    # and then rescales C, matching the chunked path's fp32 op order bit-exactly.
    if (
        out_dtype == torch.bfloat16
        and weight_scale.numel() == weight.shape[0]
        and hasattr(_kernel, "i8_matmul2d_nt_fused_cs")
    ):
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        result = _kernel.i8_matmul2d_nt_fused_cs(
            x_8.contiguous(), weight.contiguous(),
            x_scale.reshape(-1).float().contiguous(),
            weight_scale.float().contiguous(), bias_arg,
        )
        return result.reshape(*orig_shape[:-1], weight.shape[0])

    # C[M,N] int32 = x_8[M,K] @ weight[N,K]^T  (NT: weight in stored layout).
    result = _kernel.i8_matmul2d_nt(x_8.contiguous(), weight.contiguous())

    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()
        chunk = chunk * (weight_scale * x_scale[i:end_i])
        scaled_parts.append(chunk.to(out_dtype))
    result = torch.cat(scaled_parts, dim=0)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype)

    return result.reshape(*orig_shape[:-1], weight.shape[0])


def int8_o_proj_prequant(layer, out_bhsd):
    """o-projection straight from the attention layout [B,H,S,D].

    When head_dim == the convrot groupsize (Ideogram 4: D=256), each head is
    exactly one rotation group, so the head-merge transpose copy and the
    rowwise-quant pass collapse into one Metal kernel (i8_merge_rot_quant);
    the GEMM then runs on the pre-quantized (x_8, x_scale).

    ASFP8_INT8_OPROJ_FUSE selects the mode:
      "1" (default) BIT-EXACT: the convrot rotation stays the eager bf16
          torch.matmul, applied directly in [B,H,S,D] layout (measured
          bit-identical to the merged-layout matmul, so no merge copy);
          the kernel does gather + quant only — pixel-identical output,
          verified by same-seed e2e A/B.
      "2" FULL FUSION: the rotation runs as in-kernel FWHT butterflies —
          fastest (one pass total), same quality, but fp32 summation order
          differs from the eager matmul, so outputs drift within the usual
          bf16 rounding class (NOT pixel-identical).
      "0" off: ordinary merge + Linear path.

    Returns bf16 [B, S, N], or None when any eligibility check fails (caller
    falls back to the ordinary merge + layer(input) path).
    """
    if _kernel is None or not hasattr(_kernel, "i8_merge_rot_quant"):
        return None
    mode = os.environ.get("ASFP8_INT8_OPROJ_FUSE", "1")
    if mode == "0":
        return None  # kill switch: fall back to merge + ordinary Linear path
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if (not isinstance(out_bhsd, torch.Tensor) or out_bhsd.dim() != 4
                or out_bhsd.device.type != "mps"
                or out_bhsd.dtype != torch.bfloat16):
            return None
        w = layer.weight
        if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
            return None
        if getattr(layer, "_full_precision_mm", False):
            return None
        if getattr(layer, "comfy_force_cast_weights", False):
            return None
        if len(getattr(layer, "weight_function", [])) or len(getattr(layer, "bias_function", [])):
            return None
        params = w._params
        if getattr(params, "transposed", False):
            return None
        if not getattr(params, "convrot", False):
            return None
        B, H, S, D = out_bhsd.shape
        if D != 256 or getattr(params, "convrot_groupsize", 256) != D:
            return None
        qdata = w._qdata
        if H * D != qdata.shape[1]:
            return None
        weight_scale = params.scale.view(-1)
        bias = layer.bias
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None

        if mode == "2":
            x_8, x_scale = _kernel.i8_merge_rot_quant(out_bhsd.contiguous())
        else:
            from comfy_kitchen.tensor.int8_utils import _build_hadamard
            had = _build_hadamard(D, device=out_bhsd.device, dtype=out_bhsd.dtype)
            rot4 = torch.matmul(out_bhsd.contiguous(), had)
            x_8, x_scale = _kernel.i8_merge_rot_quant(rot4, False)
        if weight_scale.numel() == 1 and hasattr(_kernel, "i8_matmul2d_nt_fused"):
            row_scale = (weight_scale.float() * x_scale.reshape(-1).float()).contiguous()
            r = _kernel.i8_matmul2d_nt_fused(x_8, qdata.contiguous(), row_scale, bias_arg)
        elif (weight_scale.numel() == qdata.shape[0]
              and hasattr(_kernel, "i8_matmul2d_nt_fused_cs")):
            r = _kernel.i8_matmul2d_nt_fused_cs(
                x_8, qdata.contiguous(),
                x_scale.reshape(-1).float().contiguous(),
                weight_scale.float().contiguous(), bias_arg)
        else:
            return None
        global _oproj_announced
        if not _oproj_announced:
            _oproj_announced = True
            what = ("full FWHT fusion (rounding-equivalent)" if mode == "2"
                    else "bit-exact merge+quant fusion")
            sys.stderr.write(f"{TAG} fused o-projection engaged: {what}.\n")
        return r.reshape(B, S, qdata.shape[0])
    except Exception as e:
        print(f"{TAG} o-proj prequant fell back ({e!r})")
        return None


_dit = None   # dit_ext module for the fused silu*mul; False = latched off


def _dit_silu_mul():
    global _dit
    if _dit is None:
        _dit = False
        try:
            from .dit_ext import loader as _dl
            m = _dl.module()
            if m is not None and hasattr(m, "silu_mul"):
                _dit = m
        except Exception:
            pass
    return _dit or None


def _try_int8_swiglu_forward(linear, x):
    """linear(swiglu(x)) via fused silu*mul + the kernel W8A8 path, or None.

    Same eligibility contract as _try_int8_kernel_forward. The silu*mul
    kernel computes silu in fp32 with RTE exactly like the eager MPS op, so
    the activation bytes match comfy's _swiglu_eager."""
    global _kernel
    if _kernel is None:
        return None
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(x, torch.Tensor) or x.device.type != "mps":
            return None
        if isinstance(x, QuantizedTensor):
            return None
        w = linear.weight
        if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
            return None
        if getattr(linear, "_full_precision_mm", False):
            return None
        if getattr(linear, "comfy_force_cast_weights", False):
            return None
        if len(getattr(linear, "weight_function", [])) or len(getattr(linear, "bias_function", [])):
            return None
        params = w._params
        if getattr(params, "transposed", False):
            return None
        wide = x.shape[-1]
        if wide % 2:
            return None
        half = wide // 2
        dk = _dit_silu_mul()
        if (dk is not None and x.dtype == torch.bfloat16
                and os.environ.get("ASFP8_INT8_SWIGLU_EAGERACT") != "1"):
            act = dk.silu_mul(x[..., :half].contiguous(), x[..., half:].contiguous())
        else:
            act = torch.nn.functional.silu(x[..., :half]) * x[..., half:]
        return _int8_linear_kernel(
            act, w._qdata, params.scale, linear.bias, x.dtype,
            getattr(params, "convrot", False),
            getattr(params, "convrot_groupsize", 256))
    except Exception:
        return None


def _try_int8_kernel_forward(self, input):
    """Return the layer output via the kernel W8A8 path, or None to fall back.

    Eligible iff: kernel built; input is a plain MPS Tensor (not a QuantizedTensor);
    weight is a TensorWiseINT8Layout QuantizedTensor; not full-precision-mm; no
    force-cast; no LoRA weight/bias functions; weight not logically transposed.
    """
    global _kernel
    if _kernel is None:
        return None
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(input, torch.Tensor) or input.device.type != "mps":
            return None
        if isinstance(input, QuantizedTensor):
            return None

        w = self.weight
        if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
            return None
        if getattr(self, "_full_precision_mm", False):
            return None
        if getattr(self, "comfy_force_cast_weights", False):
            return None
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return None

        params = w._params
        if getattr(params, "transposed", False):
            return None

        qdata = w._qdata
        scale = params.scale
        convrot = getattr(params, "convrot", False)
        gs = getattr(params, "convrot_groupsize", 256)
        bias = self.bias

        return _int8_linear_kernel(input, qdata, scale, bias, input.dtype, convrot, gs)
    except Exception as e:
        # Never take down a render; fall back to comfy's original forward.
        # LATCH on Metal compile failures (e.g. a macOS beta changing the
        # tensor-ops API): without it every Linear call re-attempts the
        # expensive newLibraryWithSource compile — 33 s/it measured.
        if "compile failed" in str(e):
            _kernel = None
            print(f"{TAG} kernel DISABLED for this session — Metal compile "
                  f"failed (likely an OS update changed the tensor-ops API): "
                  f"{str(e)[:180]}")
        else:
            print(f"{TAG} kernel forward fell back ({e!r})")
        return None


def install():
    global _installed, _orig_int8_linear, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_INT8_EXT") != "1":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    _kernel = _load_kernel()
    if _kernel is None:
        return  # loader already explained why

    # Capture the original eager int8_linear for the fallback inside the wrapper.
    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear as _orig
        _orig_int8_linear = _orig
    except Exception as e:
        print(f"{TAG} could not import eager int8_linear: {e!r}")
        return

    # Wrap the mixed_precision_ops factory so each generated int8 Linear routes
    # through the kernel W8A8 path. pick_operations() calls this by module name at
    # model load, so wrapping the module attribute is sufficient.
    try:
        import comfy.ops as ops

        orig_factory = ops.mixed_precision_ops
        if getattr(orig_factory, "_asfp8_wrapped", False):
            _installed = True
            return

        def wrapped_factory(*a, **k):
            cls = orig_factory(*a, **k)
            linear_cls = getattr(cls, "Linear", None)
            if linear_cls is not None and not getattr(linear_cls, "_asfp8_int8_patched", False):
                orig_forward = linear_cls.forward

                def forward(self, input, *args, **kwargs):
                    res = _try_int8_kernel_forward(self, input)
                    if res is not None:
                        return res
                    return orig_forward(self, input, *args, **kwargs)

                linear_cls.forward = forward
                linear_cls._asfp8_int8_patched = True
            return cls

        wrapped_factory._asfp8_wrapped = True
        ops.mixed_precision_ops = wrapped_factory

        # comfy >= 0.30 routes SwiGLU'd down-projections (e.g. MiniMax H3's
        # MLP fc2) through ops.linear_input_act, which never calls
        # Linear.forward in the quantized case — the factory wrap above can't
        # see those calls. Wrap the helper too: fused silu*mul (dit_ext,
        # bit-matched to the eager swiglu) + the same kernel W8A8 path.
        # NUMERICS: in the live pipeline comfy's cast_bias_weight hands this
        # layer back a DEQUANTIZED weight (W8A16); this route is the same
        # W8A8-for-W8A16 trade patch #17 already applies to every other int8
        # Linear (~1.3% err class) — NOT bit-identical to stock, by design.
        # ASFP8_INT8_SWIGLU=0 restores stock; _VERIFY=1 runs both + dumps.
        # Measured M5 Max, H3 fc2 [5378,28672]->[5378,5376]: 23.5 -> 14.3 ms
        # (H3 864x480 1s 20st: 139.2 -> 122.5 s).
        if hasattr(ops, "linear_input_act") and not getattr(
                ops.linear_input_act, "_asfp8_wrapped", False):
            _orig_lia = ops.linear_input_act

            def linear_input_act_wrapped(linear, x, input_act):
                if (input_act == "swiglu"
                        and os.environ.get("ASFP8_INT8_SWIGLU_VERIFY") == "1"):
                    # diagnostic: run BOTH paths, dump the first divergence
                    res = _try_int8_swiglu_forward(linear, x)
                    ref = _orig_lia(linear, x, input_act)
                    if res is not None and not torch.equal(res, ref):
                        global _verify_dumped
                        if not _verify_dumped:
                            _verify_dumped = True
                            md = (res.float() - ref.float()).abs().max().item()
                            sys.stderr.write(f"{TAG} SWIGLU VERIFY MISMATCH "
                                             f"maxdiff={md:.3e} x{tuple(x.shape)} "
                                             f"w{tuple(linear.weight.shape)}\n")
                            try:
                                import comfy.ops as _ops2
                                cw, cb, _os_ = _ops2.cast_bias_weight(
                                    linear, x, offloadable=True,
                                    compute_dtype=x.dtype, want_requant=True)
                                from comfy.quant_ops import TensorWiseINT8Layout as _TL
                                cqd, csc = _TL.get_plain_tensors(cw)
                                sys.stderr.write(
                                    f"{TAG} VERIFY weights: qdata_eq="
                                    f"{torch.equal(cqd, linear.weight._qdata)} "
                                    f"scale_eq={torch.equal(csc.view(-1), linear.weight._params.scale.view(-1))} "
                                    f"bias={cb is not None}\n")
                            except Exception as _e:
                                sys.stderr.write(f"{TAG} VERIFY weight-compare failed: {_e!r}\n")
                            torch.save({"x": x.cpu(), "qdata": linear.weight._qdata.cpu(),
                                        "scale": linear.weight._params.scale.cpu(),
                                        "convrot": getattr(linear.weight._params, "convrot", False)},
                                       "/tmp/asfp8_swiglu_mismatch.pt")
                    return ref
                if (input_act == "swiglu"
                        and os.environ.get("ASFP8_INT8_SWIGLU", "1") == "1"):
                    res = _try_int8_swiglu_forward(linear, x)
                    if res is not None:
                        return res
                return _orig_lia(linear, x, input_act)

            linear_input_act_wrapped._asfp8_wrapped = True
            ops.linear_input_act = linear_input_act_wrapped
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    print(
        f"{TAG} INT8 convrot Linear routed through bit-exact Metal kernel on MPS "
        f"(clean W8A8: rotate->per-row quant->int8 matmul; weight-only fp32 "
        f"dequant/un-rotation bypassed)."
    )
