"""Exact single-token BF16 YuE2 RoPE on the validated M5 Ultra / Torch build.

Keep the two FP32 products and their addition separately rounded, then convert
once to BF16. The MPS eager addcmul path uses this order, not an FMA. No change
to frequencies, matrix multiplication, attention, token sampling or CPU RNG.
"""
import functools
import hashlib
import logging
import os
from pathlib import Path

import torch

_LLAMA_SHA = '3e8cbb319e0c727d1a75525cbe29863463aa8263485fac22303c58fa0316ab4a'
_LLAMA_SHAS = {_LLAMA_SHA, '630a9d0f66448ae1b8de53041fde7dc2c4df1e7a16f29fab8ed6c9bb44ff7fdd'}
_EAGER_SHA = '630a3d287bf602046935086f9d43eb6db2ec4971913a1cb3fb35bae6ea4ac9ee'
_library = None
_dead = False
_SOURCE = '''
#include <metal_stdlib>
using namespace metal;
kernel void yue2_rope(device const bfloat* Q, device const bfloat* K,
                      device const float* R, device bfloat* OQ,
                      device bfloat* OK, uint i [[thread_position_in_grid]]) {
    bool is_q = i < 1024;
    uint p = is_q ? i : i - 1024;
    uint head = p / 64, col = p % 64;
    device const bfloat* X = is_q ? Q : K;
    device bfloat* Y = is_q ? OQ : OK;
    float a = float(X[head * 128 + col]);
    float b = float(X[head * 128 + 64 + col]);
    // volatile retains the native intermediate rounding; contraction changes
    // several BF16 output bytes even for ordinary finite model activations.
    volatile float pa = a * R[col * 4];
    volatile float pb = b * R[col * 4 + 3];
    volatile float ca = b * R[col * 4 + 1];
    volatile float cb = a * R[col * 4 + 2];
    Y[head * 128 + col] = bfloat(pa + ca);
    Y[head * 128 + 64 + col] = bfloat(pb + cb);
}
'''


@functools.lru_cache(maxsize=1)
def _sources_supported():
    try:
        from comfy.text_encoders import llama
        from comfy_kitchen.backends.eager import rope
        return (hashlib.sha256(Path(llama.__file__).read_bytes()).hexdigest() in _LLAMA_SHAS
                and hashlib.sha256(Path(rope.__file__).read_bytes()).hexdigest() == _EAGER_SHA)
    except (ImportError, OSError, TypeError):
        return False


@functools.lru_cache(maxsize=1)
def _dependencies():
    from comfy.text_encoders import llama
    from comfy_kitchen.backends.eager import rope
    from comfy_kitchen.registry import registry
    return llama, rope, registry


def prepare(config):
    """One small scope per AR phase; a changed runtime retains the core path."""
    from . import torch_mpp_qcache as qc
    if (_dead or getattr(config, 'hidden_size', None) != 2048
            or getattr(config, 'num_attention_heads', None) != 16
            or getattr(config, 'num_key_value_heads', None) != 8
            or not qc._torch_build_supported()
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'
            or not qc._hardware() or not _sources_supported()):
        return None
    return dict(freqs=None, matrix=None, calls=0, matrices=0)


def _input_supported(q, k):
    return (type(q) is torch.Tensor and type(k) is torch.Tensor
            and q.device.type == 'mps' and k.device == q.device
            and q.dtype == k.dtype == torch.bfloat16
            and tuple(q.shape) == (1, 16, 1, 128)
            and tuple(k.shape) == (1, 8, 1, 128)
            and q.is_contiguous() and k.is_contiguous())


def _fused(q, k, matrix):
    global _library
    if _library is None:
        _library = torch.mps.compile_shader(_SOURCE)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    _library.yue2_rope(q, k, matrix, oq, ok, threads=1536, group_size=256)
    return oq, ok


def maybe_rope(q, k, freqs, state):
    """None means use the caller's original implementation, including layout."""
    global _dead
    rope_state = None if state is None else state.get('rope_state')
    if rope_state is None or _dead or not _input_supported(q, k):
        return None
    llama, rope, registry = _dependencies()
    # The checked Llama forward constructs a new frequency tuple each call and
    # shares it unchanged across the 28 layers. Retain its identity so it cannot
    # be reused for a later token even if Python recycles object addresses.
    if rope_state['freqs'] is not freqs:
        matrix = llama.rope_matrix(freqs)
        if matrix.ndim == 5:
            matrix = matrix.unsqueeze(0)
        if (type(matrix) is not torch.Tensor or matrix.dtype != torch.float32
                or matrix.device != q.device or not matrix.is_contiguous()
                or tuple(matrix.shape) != (1, 1, 1, 64, 2, 2)):
            return None
        # Respect a user-installed Kitchen backend or its thread-local override.
        impl = registry.get_implementation('apply_rope_split_half', kwargs={
            'xq': q, 'xk': k, 'freqs_cis': matrix})
        if impl is not rope.apply_rope_split_half:
            return None
        rope_state['freqs'], rope_state['matrix'] = freqs, matrix
        rope_state['matrices'] += 1
    try:
        out = _fused(q, k, rope_state['matrix'])
    except Exception as exc:
        _dead = True
        logging.warning('[AppleSilicon-Optimized/yue2] RoPE fusion disabled: %s', str(exc)[:180])
        return None
    rope_state['calls'] += 1
    return out
