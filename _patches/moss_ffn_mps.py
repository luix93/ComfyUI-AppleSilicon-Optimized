"""Optional exact MOSS FC1/SiTU fusion on the reviewed M5 Ultra stack.

The paired-column GEMM reads original weights and preserves BF16 stores via
explicit rounding before SiTU table lookup. ConvRot, rowwise quantization and
FC2 remain the existing path. No retained weight cache or compact-tensor marker.
The block forward follows ComfyUI's GPL-3.0 MOSS implementation; its source and
sampling scope are validated by moss_fast before this helper is called.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import inspect
import logging
import os
import torch

_library = None
_dead = False
_LINEAR_FORWARD_HASH = '7883f2da541430032a08e2417ead94e86bb0c688c9555407f26cdaccb616dabb'


@lru_cache(maxsize=32)
def _known_forward(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest() == _LINEAR_FORWARD_HASH
    except (OSError, TypeError):
        return False


def _eligible_layer(layer, shape, device):
    from . import int8_linear_kernel_mps as i8
    from comfy_kitchen.tensor import QuantizedTensor
    if (i8._kernel is None or hasattr(layer, '_v')
            or getattr(layer, 'bias', None) is not None
            or any(getattr(layer, key, None) for key in (
                'weight_function', 'bias_function', 'weight_lowvram_function',
                'bias_lowvram_function', '_forward_hooks', '_forward_pre_hooks',
                '_full_precision_mm', 'comfy_force_cast_weights', i8._RUNTIME_LORA_ATTR))):
        return False
    if not _known_forward(getattr(type(layer), 'forward', None)):
        return False
    w = getattr(layer, 'weight', None)
    if (not isinstance(w, QuantizedTensor)
            or getattr(w, '_layout_cls', None) != 'TensorWiseINT8Layout'
            or w.dtype != torch.bfloat16 or tuple(w.shape) != shape):
        return False
    params = w._params
    q, scale = w._qdata, params.scale
    return (q.device == device and q.dtype == torch.int8 and q.is_contiguous()
            and scale.device == device and scale.dtype == torch.float32
            and scale.numel() == shape[0] and scale.is_contiguous()
            and getattr(params, 'convrot', False)
            and getattr(params, 'convrot_groupsize', None) == 256
            and not getattr(params, 'transposed', False))


def _ready():
    global _library, _dead
    if _dead:
        return False
    if _library is None:
        try:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead = True
            logging.warning('[AppleSilicon-Optimized/MOSS] FC1/SiTU fusion unavailable; retaining existing FFN: %s', exc)
    return _library is not None


def project(q, scales, weight, weight_scale, tables):
    """Prevalidated [M,3072] INT8 -> [M,8192] activated BF16 tensor."""
    m, k = q.shape
    n = weight.shape[0]
    out = torch.empty((m, n // 2), device=q.device, dtype=torch.bfloat16)
    tm, tn = (m + 127) // 128, (n + 127) // 128
    _library.fc1_situ(q, weight, out, m, n, k, tm, tn,
                     scales.reshape(-1), weight_scale.reshape(-1), *tables,
                     threads=(tn * 512, tm, 1), group_size=(512, 1, 1))
    return out


def block(original, module, owner, layer, x, mod, pe, transformer_options):
    # All fallbacks occur before the first in-place gated residual update.
    state = owner._scope.get()
    if (not owner._enabled('FFN_EPILOGUE') or not owner._enabled('FFN_DISPATCH')
            or not owner._enabled('SITU')
            or os.environ.get('ASFP8_MOSS_SITU_QUANT', '0') == '1'
            or not owner._direct_ffn_supported
            or not owner._bf16(x) or x.ndim != 3 or x.shape[0] != 1
            or x.shape[-1] != 3072 or not 4096 <= x.shape[1] <= 6656
            or id(layer.proj['ffn_down']) not in state['modules']):
        return original(layer, x, mod, pe, transformer_options)
    up, down = layer.proj['ffn_up_gate'], layer.proj['ffn_down']
    import comfy.ops as ops
    if (not owner._owned_dequantizing_cast(ops.cast_bias_weight)
            or not _eligible_layer(up, (16384, 3072), x.device)
            or not _eligible_layer(down, (3072, 8192), x.device)
            or not owner._ready() or not _ready()):
        return original(layer, x, mod, pe, transformer_options)
    from . import int8_linear_kernel_mps as i8
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
    y = module.modulated_rms_norm(x, layer.norm1, scale_a, shift_a)
    if layer.fused_qkv:
        q, k, v = layer.proj['qkv'](y).chunk(3, dim=-1)
    else:
        q, k, v = (layer.proj[name](y) for name in ('q', 'k', 'v'))
    out = module.attention(q, k, v, layer.q_norm, layer.k_norm, pe,
                           layer.num_heads, layer.comfy_attention, transformer_options)
    x = module.gated_residual(x, layer.proj['o'](out), gate_a)
    y = module.modulated_rms_norm(x, layer.norm2, scale_f, shift_f)
    ops.run_every_op()
    q8, scales = i8._quant_cached(y, True, 256)
    activation = project(q8, scales, up.weight._qdata, up.weight._params.scale,
                         owner._tables).reshape(*y.shape[:-1], 8192)
    result = down(activation)
    state['ffn_epilogue_calls'] = state.get('ffn_epilogue_calls', 0) + 1
    return module.gated_residual(x, result, gate_f)
