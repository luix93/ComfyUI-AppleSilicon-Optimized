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

import inspect
import os
import sys
import weakref

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-Optimized/int8_kernel]"

_installed = False
_orig_int8_linear = None
_kernel = None
_oproj_announced = False
_qkv_scatter_announced = False
_qkv_norm_rope_announced = False
_qkv_lora_norm_rope_announced = False
_swiglu_rot_announced = False
_mlp_bridge_announced = False
_mlp_lora_bridge_announced = False
_verify_dumped = False
_RUNTIME_LORA_ATTR = "_asfp8_runtime_lora"

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


def _int8_linear_prequant(
    x_8, x_scale, weight, weight_scale, bias, out_dtype, input_shape,
    split_tail=False,
):
    """Run the INT8 GEMM/epilogue from an already quantized activation."""
    weight_scale = weight_scale.view(-1)

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
        return result.reshape(*input_shape[:-1], weight.shape[0])

    if (
        out_dtype == torch.bfloat16
        and weight_scale.numel() == weight.shape[0]
        and hasattr(_kernel, "i8_matmul2d_nt_fused_cs")
    ):
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        args = (
            x_8.contiguous(), weight.contiguous(),
            x_scale.reshape(-1).float().contiguous(),
            weight_scale.float().contiguous(), bias_arg,
        )
        # H3's fc2 has M=5378: the 128-row main kernel wastes 3/4 of
        # its final tile. A second 32-row dispatch for that two-row suffix is
        # exact and measures ~1.3% faster for fc2; it is neutral/slower on the
        # other H3 projections, so only the SwiGLU down-projection opts in.
        long_schedule = os.environ.get(
            "ASFP8_INT8_LONG_SCHEDULE", "1") == "1"
        result = _kernel.i8_matmul2d_nt_fused_cs(
            *args, split_tail, long_schedule)
        return result.reshape(*input_shape[:-1], weight.shape[0])

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
    return result.reshape(*input_shape[:-1], weight.shape[0])


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
    # MiniMax H3 long clips leave a small suffix in the 128-row GEMM tile.
    # At M=15,787 the exact 32-row suffix kernel is ~2% faster for QKV, while
    # it is slightly slower at the 1-second shape; keep the dispatch narrowly
    # shape/length gated. The INT32 reduction and epilogue are unchanged.
    split_tail = (
        x_8.shape[0] >= 8192
        and weight.shape == (21504, 5376)
        and os.environ.get("ASFP8_INT8_LONG_QKV_TAIL", "1") == "1"
    )
    return _int8_linear_prequant(
        x_8, x_scale, weight, weight_scale, bias, out_dtype, orig_shape,
        split_tail=split_tail,
    )


def _int8_qkv_prequant(layer, x, heads, head_dim, norm_rope):
    """Shared guarded H3 QKV projection route.

    ``norm_rope`` is either None for the layout-only epilogue or
    (freqs, q_weight_fp32, k_weight_fp32, eps) for the D=128 fused epilogue.
    A fast standard runtime LoRA uses a partitioned native epilogue which adds
    each bf16 delta before Q/K normalization without materializing full QKV.
    """
    fused_norm_rope = norm_rope is not None
    runtime_lora = getattr(layer, _RUNTIME_LORA_ATTR, None)
    lora_norm_rope = fused_norm_rope and runtime_lora is not None
    kernel_name = (
        "i8_matmul2d_nt_fused_cs_qkv_lora_part" if lora_norm_rope else
        "i8_matmul2d_nt_fused_cs_qkv_rms_rope"
        if fused_norm_rope else "i8_matmul2d_nt_fused_cs_qkv"
    )
    if (_kernel is None
            or not hasattr(_kernel, kernel_name)
            or os.environ.get("ASFP8_INT8_QKV_SCATTER", "1") != "1"
            or (fused_norm_rope and os.environ.get(
                "ASFP8_INT8_QKV_NORM_ROPE", "1") != "1")):
        return None
    if (runtime_lora is not None and (
            not getattr(runtime_lora, "fast", False)
            or not hasattr(runtime_lora, "linear_parts"))):
        return None
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if (not isinstance(x, torch.Tensor) or x.dim() != 2
                or x.device.type != "mps" or x.dtype != torch.bfloat16
                or not x.is_contiguous()):
            return None
        w = layer.weight
        if (not isinstance(w, QuantizedTensor)
                or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout"):
            return None
        if (getattr(layer, "_full_precision_mm", False)
                or getattr(layer, "comfy_force_cast_weights", False)
                or len(getattr(layer, "weight_function", []))
                or len(getattr(layer, "bias_function", []))):
            return None
        params = w._params
        if getattr(params, "transposed", False):
            return None
        qdata = w._qdata
        if qdata.shape != (3 * heads * head_dim, x.shape[1]):
            return None
        weight_scale = params.scale.view(-1)
        if weight_scale.numel() != qdata.shape[0]:
            return None
        convrot = getattr(params, "convrot", False)
        groupsize = getattr(params, "convrot_groupsize", 256)
        x_8, x_scale = _quant_cached(x, convrot, groupsize)
        bias = layer.bias
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        long_schedule = os.environ.get(
            "ASFP8_INT8_LONG_SCHEDULE", "1") == "1"
        common = (
            x_8, qdata.contiguous(),
            x_scale.reshape(-1).float().contiguous(),
            weight_scale.float().contiguous(), bias_arg,
        )
        if fused_norm_rope:
            freqs, q_weight, k_weight, eps = norm_rope
            if (head_dim != 128 or not isinstance(freqs, torch.Tensor)
                    or freqs.device.type != "mps"
                    or freqs.dtype != torch.bfloat16
                    or not freqs.is_contiguous() or freqs.dim() < 3
                    or freqs.numel() != x.shape[0] * freqs.shape[-3] * 4
                    or 2 * freqs.shape[-3] > head_dim
                    or not isinstance(q_weight, torch.Tensor)
                    or not isinstance(k_weight, torch.Tensor)
                    or q_weight.device.type != "mps"
                    or k_weight.device.type != "mps"
                    or q_weight.dtype != torch.float32
                    or k_weight.dtype != torch.float32
                    or not q_weight.is_contiguous()
                    or not k_weight.is_contiguous()
                    or q_weight.numel() != head_dim
                    or k_weight.numel() != head_dim):
                return None
            if lora_norm_rope:
                up, down, lora_scale = runtime_lora.linear_parts(x)
                part_width = heads * head_dim
                if (up.shape[0] != 3 * part_width
                        or up.shape[1] != down.shape[0]
                        or down.shape[1] != x.shape[1]):
                    return None
                low_rank = F.linear(x, down)
                outputs = []
                for kind in range(3):
                    start = kind * part_width
                    end = start + part_width
                    delta = F.linear(low_rank, up[start:end]).contiguous()
                    part_bias = (
                        bias_arg[start:end].contiguous()
                        if bias_arg is not None else None)
                    outputs.append(getattr(_kernel, kernel_name)(
                        x_8, qdata[start:end].contiguous(),
                        x_scale.reshape(-1).float().contiguous(),
                        weight_scale[start:end].float().contiguous(),
                        part_bias, delta, freqs,
                        k_weight if kind == 1 else q_weight,
                        heads, head_dim, kind, float(lora_scale),
                        float(eps), long_schedule))
                    del delta
                q, k, v = outputs
            else:
                q, k, v = getattr(_kernel, kernel_name)(
                    *common, freqs, q_weight, k_weight, heads, head_dim,
                    float(eps), long_schedule)
        else:
            q, k, v = getattr(_kernel, kernel_name)(
                *common, heads, head_dim, long_schedule)
        if runtime_lora is not None and not fused_norm_rope:
            q, k, v = runtime_lora.add_qkv_to(
                q, k, v, x, heads, head_dim)
        global _qkv_scatter_announced, _qkv_norm_rope_announced
        global _qkv_lora_norm_rope_announced
        if lora_norm_rope:
            if not _qkv_lora_norm_rope_announced:
                _qkv_lora_norm_rope_announced = True
                sys.stderr.write(
                    f"{TAG} exact runtime-LoRA INT8 QKV + Q/K RMSNorm + "
                    "partial RoPE epilogue engaged.\n")
        elif fused_norm_rope:
            if not _qkv_norm_rope_announced:
                _qkv_norm_rope_announced = True
                sys.stderr.write(
                    f"{TAG} INT8 QKV + Q/K RMSNorm + partial RoPE "
                    "epilogue engaged.\n")
        elif not _qkv_scatter_announced:
            _qkv_scatter_announced = True
            sys.stderr.write(
                f"{TAG} direct INT8 QKV attention-layout epilogue engaged; "
                "fused qkv bf16 buffer eliminated.\n")
        return q, k, v
    except Exception as e:
        route = "qkv norm/RoPE epilogue" if fused_norm_rope else "qkv scatter"
        print(f"{TAG} {route} fell back ({e!r})")
        return None


def int8_qkv_prequant(layer, x, heads, head_dim):
    """H3 QKV projection directly to contiguous [1,H,S,D] Q/K/V.

    A fast H3 Turbo adapter adds its Q/K/V deltas after the base projection;
    normalization therefore remains a later pass for that route.
    """
    return _int8_qkv_prequant(layer, x, heads, head_dim, None)


def int8_qkv_rms_rope_prequant(
    layer, x, freqs, q_weight, k_weight, heads, head_dim, eps,
):
    """Bit-exact H3 INT8 projection + Q/K RMSNorm/partial-RoPE epilogue.

    Eligible standard runtime adapters are added before the nonlinear norm;
    unsupported adapter protocols fall back to the ordinary exact route.
    """
    return _int8_qkv_prequant(
        layer, x, heads, head_dim, (freqs, q_weight, k_weight, eps))


def int8_o_proj_prequant(layer, out_bhsd):
    """o-projection straight from the attention layout [B,H,S,D].

    When the merged head width tiles the convrot groupsize (Ideogram 4: D=256;
    MiniMax H3: D=128, two heads per group), the head-merge transpose copy and
    rowwise-quant pass collapse into one Metal kernel (i8_merge_rot_quant); the
    GEMM then runs on the pre-quantized (x_8, x_scale).

    ASFP8_INT8_OPROJ_FUSE selects the mode:
      "1" (default) BIT-EXACT when D=256: the convrot rotation stays the eager bf16
          torch.matmul, applied directly in [B,H,S,D] layout (measured
          bit-identical to the merged-layout matmul, so no merge copy);
          the kernel does gather + quant only.  For D<256 (H3), a rotation
          group spans heads and mode 1 uses the full fused FWHT instead; its
          rounding class is therefore the same as mode 2.
      "2" FULL FUSION: the rotation runs as in-kernel FWHT butterflies —
          fastest (one pass total), same quality, but fp32 summation order
          differs from the eager matmul, so outputs drift within the usual
          bf16 rounding class (NOT pixel-identical).
      "0" off: ordinary merge + Linear path.

    Returns bf16 [B, S, N], or None when any eligibility check fails (caller
    falls back to the ordinary merge + layer(input) path).
    """
    runtime_lora = getattr(layer, _RUNTIME_LORA_ATTR, None)
    if runtime_lora is not None and not getattr(runtime_lora, "fast", False):
        return None
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
        if (D <= 0 or D > 256 or (H * D) % 256
                or getattr(params, "convrot_groupsize", 256) != 256):
            return None
        qdata = w._qdata
        if H * D != qdata.shape[1]:
            return None
        weight_scale = params.scale.view(-1)
        bias = layer.bias
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None

        if mode == "2" or D != 256:
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
            long_schedule = os.environ.get(
                "ASFP8_INT8_LONG_SCHEDULE", "1") == "1"
            r = _kernel.i8_matmul2d_nt_fused_cs(
                x_8, qdata.contiguous(),
                x_scale.reshape(-1).float().contiguous(),
                weight_scale.float().contiguous(), bias_arg,
                False, long_schedule)
        else:
            return None
        global _oproj_announced
        if not _oproj_announced:
            _oproj_announced = True
            what = ("full FWHT fusion (rounding-equivalent)" if mode == "2" or D != 256
                    else "bit-exact merge+quant fusion")
            sys.stderr.write(f"{TAG} fused o-projection engaged: {what}.\n")
        r = r.reshape(B, S, qdata.shape[0])
        if runtime_lora is not None:
            # The direct o-projection intentionally bypasses Linear.forward.
            # Apply the registered H3 runtime LoRA explicitly so the fused path
            # cannot silently drop the output-projection delta.
            r = runtime_lora.add_bhsd_to(r, out_bhsd)
        return r
    except Exception as e:
        print(f"{TAG} o-proj prequant fell back ({e!r})")
        return None


def int8_mlp_bridge(fc1, fc2, x):
    """Run an H3 INT8 FC1/SwiGLU/FC2 without FC1's [M,2F] output."""
    if (_kernel is None
            or not hasattr(_kernel, "i8_mlp_bridge_prequant")
            or os.environ.get("ASFP8_INT8_MLP_BRIDGE", "1") != "1"
            or os.environ.get("ASFP8_INT8_SWIGLU", "1") != "1"
            or os.environ.get("ASFP8_INT8_SWIGLU_ROT", "1") != "1"):
        return None
    lora1 = getattr(fc1, _RUNTIME_LORA_ATTR, None)
    lora2 = getattr(fc2, _RUNTIME_LORA_ATTR, None)
    runtime_lora = lora1 is not None or lora2 is not None
    if runtime_lora and any(
            adapter is not None and (
                not getattr(adapter, "fast", False)
                or not hasattr(adapter, "linear_parts"))
            for adapter in (lora1, lora2)):
        return None
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if (not isinstance(x, torch.Tensor) or x.dim() != 2
                or x.device.type != "mps" or x.dtype != torch.bfloat16
                or not x.is_contiguous()):
            return None
        weights = (fc1.weight, fc2.weight)
        if any(not isinstance(weight, QuantizedTensor)
               or getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout"
               for weight in weights):
            return None
        if any(getattr(layer, "_full_precision_mm", False)
               or getattr(layer, "comfy_force_cast_weights", False)
               or len(getattr(layer, "weight_function", []))
               or len(getattr(layer, "bias_function", []))
               for layer in (fc1, fc2)):
            return None
        p1, p2 = weights[0]._params, weights[1]._params
        if (getattr(p1, "transposed", False)
                or getattr(p2, "transposed", False)
                or not getattr(p2, "convrot", False)
                or getattr(p2, "convrot_groupsize", 256) != 256):
            return None
        w1, w2 = weights[0]._qdata, weights[1]._qdata
        if (w1.dim() != 2 or w2.dim() != 2 or w1.shape[0] % 512
                or w1.shape[1] != x.shape[1]
                or w2.shape[1] * 2 != w1.shape[0]):
            return None
        s1, s2 = p1.scale.view(-1), p2.scale.view(-1)
        if s1.numel() != w1.shape[0] or s2.numel() != w2.shape[0]:
            return None

        x_8, x_scale = _quant_cached(
            x, getattr(p1, "convrot", False),
            getattr(p1, "convrot_groupsize", 256))
        bias1 = fc1.bias
        bias1_arg = bias1.to(torch.bfloat16) if bias1 is not None else None
        long_schedule = os.environ.get(
            "ASFP8_INT8_LONG_SCHEDULE", "1") == "1"
        if not runtime_lora:
            fc2_8, fc2_scale = _kernel.i8_mlp_bridge_prequant(
                x_8.contiguous(), w1.contiguous(),
                x_scale.reshape(-1).float().contiguous(),
                s1.float().contiguous(), bias1_arg, long_schedule)
        else:
            if (not hasattr(_kernel, "i8_matmul2d_nt_fused_cs")
                    or not hasattr(_kernel, "i8_quant_rowwise")):
                return None
            rows = x.shape[0]
            ffn = w2.shape[1]
            parts1 = lora1.linear_parts(x) if lora1 is not None else None
            parts2 = lora2.linear_parts(x) if lora2 is not None else None
            if parts1 is not None:
                up1, down1, scale1 = parts1
                if (up1.shape != (2 * ffn, down1.shape[0])
                        or down1.shape[1] != x.shape[1]):
                    return None
                low1 = F.linear(x, down1)
            if parts2 is not None:
                up2, down2, scale2 = parts2
                if (up2.shape != (w2.shape[0], down2.shape[0])
                        or down2.shape[1] != ffn):
                    return None
                low2 = torch.empty(
                    (rows, down2.shape[0]), device=x.device, dtype=x.dtype)

            fc2_8 = torch.empty(
                (rows, ffn), device=x.device, dtype=torch.int8)
            fc2_scale = torch.empty(
                (rows, 1), device=x.device, dtype=torch.float32)
            tile_rows = max(128, int(os.environ.get(
                "ASFP8_INT8_MLP_LORA_TILE_ROWS", "2048")))
            tile_rows = (tile_rows // 128) * 128
            row_scales = x_scale.reshape(-1).float().contiguous()
            col_scales = s1.float().contiguous()
            for start in range(0, rows, tile_rows):
                end = min(start + tile_rows, rows)
                wide = _kernel.i8_matmul2d_nt_fused_cs(
                    x_8[start:end].contiguous(), w1.contiguous(),
                    row_scales[start:end].contiguous(), col_scales,
                    bias1_arg, False, long_schedule)
                if parts1 is not None:
                    delta1 = F.linear(low1[start:end], up1)
                    wide.add_(delta1, alpha=scale1)
                    del delta1
                act = _swiglu_activation(wide)
                del wide
                if parts2 is not None:
                    low2[start:end].copy_(F.linear(act, down2))
                quantized, scales = _quant_cached(
                    act, getattr(p2, "convrot", False),
                    getattr(p2, "convrot_groupsize", 256))
                fc2_8[start:end].copy_(quantized)
                fc2_scale[start:end].copy_(scales)
                del act, quantized, scales
        bias2 = fc2.bias
        result = _int8_linear_prequant(
            fc2_8, fc2_scale, w2, s2, bias2, x.dtype, x.shape,
            split_tail=True)
        if runtime_lora and parts2 is not None:
            delta2 = F.linear(low2, up2)
            result.add_(delta2, alpha=scale2)
            del delta2
        global _mlp_bridge_announced, _mlp_lora_bridge_announced
        if runtime_lora and not _mlp_lora_bridge_announced:
            _mlp_lora_bridge_announced = True
            sys.stderr.write(
                f"{TAG} exact runtime-LoRA H3 FC1/SwiGLU/FC2 tiled bridge "
                "engaged.\n")
        elif not runtime_lora and not _mlp_bridge_announced:
            _mlp_bridge_announced = True
            sys.stderr.write(
                f"{TAG} fused H3 FC1/SwiGLU/FC2 bridge engaged; "
                "wide FC1 and separate rotated BF16 allocations eliminated.\n")
        return result
    except Exception as e:
        print(f"{TAG} MLP bridge fell back ({e!r})")
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


def _swiglu_activation(x):
    """Materialize H3's SwiGLU once, preferring the bit-matched Metal helper."""
    wide = x.shape[-1]
    half = wide // 2
    x2 = x.reshape(-1, wide)
    dk = _dit_silu_mul()
    if (dk is not None and x.dtype == torch.bfloat16
            and os.environ.get("ASFP8_INT8_SWIGLU_EAGERACT") != "1"):
        if hasattr(dk, "silu_mul_split") and x2.is_contiguous():
            return dk.silu_mul_split(x2).reshape(*x.shape[:-1], half)
        return dk.silu_mul(
            x[..., :half].contiguous(), x[..., half:].contiguous())
    return torch.nn.functional.silu(x[..., :half]) * x[..., half:]


def _try_int8_swiglu_forward(linear, x):
    """linear(swiglu(x)) via fused silu*mul + the kernel W8A8 path, or None.

    Same eligibility contract as _try_int8_kernel_forward. The silu*mul
    kernel computes silu in fp32 with RTE exactly like the eager MPS op, so
    the activation bytes match comfy's _swiglu_eager."""
    global _kernel, _swiglu_rot_announced
    runtime_lora = getattr(linear, _RUNTIME_LORA_ATTR, None)
    if runtime_lora is not None and not getattr(runtime_lora, "fast", False):
        return None
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
        x2 = x.reshape(-1, wide)
        convrot = getattr(params, "convrot", False)
        groupsize = getattr(params, "convrot_groupsize", 256)
        if runtime_lora is not None:
            # A low-rank delta needs the unrotated SwiGLU activation.  Keep the
            # quantized base GEMM, materialize that activation once, and add the
            # small-rank projection in activation space.  This avoids both the
            # lossy INT8 weight merge and the silent forward-hook bypass.
            act = _swiglu_activation(x)
            base = _int8_linear_kernel(
                act, w._qdata, params.scale, linear.bias, x.dtype,
                convrot, groupsize)
            return runtime_lora.add_to(base, act)
        if (os.environ.get("ASFP8_INT8_SWIGLU_ROT", "1") == "1"
                and convrot and groupsize == 256
                and x.dtype == torch.bfloat16 and x2.is_contiguous()
                and hasattr(_kernel, "i8_swiglu_rot")
                and hasattr(_kernel, "i8_quant_rowwise")):
            rot = _kernel.i8_swiglu_rot(x2)
            x_8, x_scale = _kernel.i8_quant_rowwise(rot)
            if not _swiglu_rot_announced:
                _swiglu_rot_announced = True
                sys.stderr.write(f"{TAG} fused SwiGLU+convrot pre-pass "
                                 "engaged for INT8 fc2.\n")
            return _int8_linear_prequant(
                x_8, x_scale, w._qdata, params.scale, linear.bias,
                x.dtype, x.shape, split_tail=True)

        act = _swiglu_activation(x)
        return _int8_linear_kernel(
            act, w._qdata, params.scale, linear.bias, x.dtype,
            convrot, groupsize)
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

        factory_signature = inspect.signature(orig_factory)

        def wrapped_factory(*a, **k):
            bound = factory_signature.bind(*a, **k)
            disabled = bound.arguments.get("disabled")
            if disabled is not None:
                # ComfyUI's hardware probe does not know about this MPS INT8
                # implementation. Keep other exclusions and explicit
                # full_precision_mm / checkpoint precision requests intact.
                bound.arguments["disabled"] = [
                    fmt for fmt in disabled if fmt != "int8_tensorwise"]
            cls = orig_factory(*bound.args, **bound.kwargs)
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

            def linear_input_act_wrapped(linear, x, input_act, *args, **kwargs):
                # Newer ComfyUI callers pass RMSNorm weights/eps and residual
                # epilogues. Our SwiGLU fast path only implements the original
                # three-argument contract; preserve extended calls unchanged.
                if args or kwargs:
                    return _orig_lia(linear, x, input_act, *args, **kwargs)
                runtime_lora = getattr(linear, _RUNTIME_LORA_ATTR, None)
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
                    if runtime_lora is None:
                        return ref
                    # The stock reference bypasses Linear.forward for INT8 and
                    # therefore omits a runtime adapter. Fall through to the
                    # explicit forward path below when one is registered.
                if (input_act == "swiglu"
                        and os.environ.get("ASFP8_INT8_SWIGLU", "1") == "1"):
                    res = _try_int8_swiglu_forward(linear, x)
                    if res is not None:
                        return res
                if input_act == "swiglu" and runtime_lora is not None:
                    # Eligibility changes (dtype/device/layout/kill switch)
                    # must fall back through Linear.forward, not ComfyUI's
                    # quantized linear_input_act helper, because only the
                    # former invokes the runtime LoRA hook.
                    return linear(_swiglu_activation(x))
                return _orig_lia(linear, x, input_act)

            linear_input_act_wrapped._asfp8_wrapped = True
            ops.linear_input_act = linear_input_act_wrapped
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    print(
        f"{TAG} native INT8 convrot Linear registered on MPS for eligible layers "
        f"(clean W8A8: rotate->per-row quant->int8 matmul; weight-only fp32 "
        f"dequant/un-rotation bypassed)."
    )
