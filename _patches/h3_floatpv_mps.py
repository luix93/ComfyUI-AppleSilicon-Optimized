"""Opt-in H3 dense INT8 QK / BF16 PV attention; numerically approximate.

CCV e499321 kernel, with pack-owned BHSD binding/layout adaptation. Every key
is retained. K centering and Hadamard rotation precede Q/K quantization. P/V
remain BF16 with FP32 accumulation. Does not change the exact dispatcher.
"""
from __future__ import annotations
import functools
import math
from pathlib import Path
import re
import torch
from .h3_dense_mps import supported, workspace_budget, WorkspaceBudgetExceeded


def scratch_bytes(tokens, query_tokens=None):
    """Conservative live scratch including input copies, prefix and result copies."""
    r = tokens if query_tokens is None else query_tokens
    h, d = 56, 128
    chunks = (tokens + 511) // 512
    return (h * d * (6 * tokens + 7 * r + 64 * 2)
            + 4 * h * (2 * ((r + 15) // 16) + 4 * ((tokens + 63) // 64)
                       + 2 * d + 2 * r + 2 * chunks * d)
            + 8 * h * ((tokens + 63) // 64))


def check_workspace(tokens, memory_budget_gib=0.0):
    required = scratch_bytes(tokens)
    budget, headroom = workspace_budget(memory_budget_gib)
    if required > budget:
        raise WorkspaceBudgetExceeded(
            f"H3 floating PV needs {required / 2**30:.2f} GiB temporary memory; "
            f"budget {budget / 2**30:.2f} GiB, safe headroom {headroom / 2**30:.2f} GiB. "
            "Adjust Temporary memory (GiB), free memory or reduce the sequence.")


@functools.lru_cache(maxsize=1)
def _library():
    source = (Path(__file__).parent / 'h3_floatpv_shaders/int8_qk_float_pv.metal').read_text()
    values = {'QUANTIZE_Q_HEADS': 56, 'QUANTIZE_KV_HEADS': 56,
              'QUANTIZE_Q_TILE_SIZE': 16, 'QUANTIZE_KV_TILE_SIZE': 64,
              'V_mean_batch_stride': 56 * 128,
              # Shared upstream declarations; backward kernels are not called.
              'dO_batch_stride': 0, 'dV_batch_stride': 0, 'dK_batch_stride': 0,
              'dQ_batch_stride': 0, 'dO_scale_batch_stride': 0}
    def constant(match):
        name = match[1]
        if name not in values:
            raise RuntimeError(f"Unknown floating-PV shader constant: {name}")
        return f'constant uint {name} = {values[name]};'
    source = re.sub(r'constant uint (\w+) \[\[function_constant\(\d+\)\]\];', constant, source)
    bindings = {}
    for match in reversed(list(re.finditer(r'kernel void (\w+)\(', source))):
        start, end = match.start(), source.index('{', match.start())
        signature = source[start:end]
        indices = [int(m[1]) for m in re.finditer(r'\[\[buffer\((\d+)\)\]\]', signature)]
        bindings[match[1]] = indices
        remap = {old: new for new, old in enumerate(indices)}
        signature = re.sub(r'\[\[buffer\((\d+)\)\]\]',
                           lambda m: f'[[buffer({remap[int(m[1])]})]]', signature)
        source = source[:start] + signature + source[end:]
    return torch.mps.compile_shader(source), bindings


@functools.lru_cache(maxsize=16)
def _dimensions(r, c):
    return torch.tensor([r, c, r*56*128, c*56*128, c*56*128, r*56*128,
                         0, 0, 0, 0, 56*((r+15)//16), 56*((c+63)//64),
                         56*((c+63)//64)], dtype=torch.int32, device='mps')


def _video_attention(q, k, v):
    r, c = q.shape[2], k.shape[2]
    lib, bindings = _library()
    q, k, v = [x.contiguous() for x in (q, k, v)]
    qi, ki = [torch.empty_like(x, dtype=torch.int8) for x in (q, k)]
    qs = torch.empty((56, (r+15)//16), device=q.device)
    ks = torch.empty((56, (c+63)//64), device=q.device)
    vs = torch.empty_like(ks)  # Shared shader ABI, unused by floating PV.
    vm = torch.empty((56, 128), device=q.device)
    km = torch.empty_like(vm)
    chunks = (c+511)//512
    partial = torch.empty((2, 56, chunks, 32, 4), device=q.device)
    correction = torch.empty((56, r), device=q.device)
    out = torch.empty_like(q)
    lse = torch.empty((56, r), device=q.device)
    tail = torch.empty((1, 56, 64, 128), device=q.device, dtype=q.dtype)
    dims = _dimensions(r, c)
    def dispatch(name, buffers, groups, threads):
        getattr(lib, name)(*[buffers[i] for i in bindings[name]],
                           threads=tuple(g*t for g, t in zip(groups, threads)), group_size=threads)
    means = {0:v, 1:vm, 2:k, 3:km, 4:partial, 21:dims}
    dispatch('compute_v_mean', means, (chunks, 112, 1), (256, 1, 1))
    if chunks > 1:
        dispatch('finalize_v_mean', means, (1, 112, 1), (32, 1, 1))
    dispatch('quantize_q', {0:q, 1:qi, 2:qs, 3:km, 4:correction, 21:dims},
             ((r+15)//16, 56, 1), (128, 1, 1))
    dispatch('quantize_k', {0:k, 1:ki, 2:ks, 3:km, 4:v, 5:tail, 21:dims},
             ((c+63)//64, 56, 1), (256, 1, 1))
    dispatch('int8_attention', {0:qi, 1:ki, 2:v, 3:out, 4:lse, 5:tail, 10:qs,
                              11:ks, 12:vs, 14:vm, 21:dims, 22:correction},
             (((r+63)//64)*56, 1, 1), (128, 1, 1))
    return out


def attention(q, k, v, *, tau=0.0, video_start=0, scale=None, exact_prefix=True,
              schedule='linear', memory_budget_gib=0.0, topk_ratio=0.0):
    if not supported():
        raise ValueError('H3 floating PV requires M5 and torch.mps.compile_shader')
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
            or q.shape[:2] != (1, 56) or q.shape[-1] != 128
            or any(x.device.type != 'mps' or x.dtype != torch.bfloat16 for x in (q,k,v))):
        raise ValueError('H3 floating PV requires matching MPS BF16 [1,56,S,128] inputs')
    tokens = q.shape[2]
    if not (0 < tokens <= 65536 and 0 <= video_start <= tokens):
        raise ValueError('H3 floating PV supports 1–65,536 tokens and a valid prefix')
    if scale is not None and (not math.isfinite(scale) or abs(scale - 128**-.5) > 1e-10):
        raise ValueError('H3 floating PV requires the standard D128 attention scale')
    if topk_ratio != 0 or schedule != 'linear':
        raise ValueError('H3 floating PV uses dense linear traversal without top-k')
    check_workspace(tokens, memory_budget_gib)
    end = min(tokens, ((video_start+63)//64)*64) if exact_prefix else 0
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    if end < tokens:
        out[:,:,end:].copy_(_video_attention(q[:,:,end:], k, v))
    if end:
        out[:,:,:end].copy_(torch.nn.functional.scaled_dot_product_attention(q[:,:,:end], k, v, scale=scale))
    blocks = (tokens+63)//64
    counts = torch.full((56, blocks), blocks, device=q.device, dtype=torch.int64)
    return out, counts, 56*blocks*blocks
