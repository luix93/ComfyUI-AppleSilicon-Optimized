"""Opt-in MOSS dense attention specialization for M5 Ultra.

The exact tile32 route uses hash-checked PyTorch source. The two dense INT8
routes adapt the pack's attributed CCV shaders (BSD-3-Clause) to MOSS's BNHD
physical layout; int8_v3 uses the pack's general INT8 attention kernel. All
INT8 routes retain all keys but are NOT exact versus PyTorch.
The exact tile32 route is the default; the INT8 routes remain opt-in.
No H3 dispatcher is modified.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
from pathlib import Path
import re

import torch
from . import torch_mpp_qcache as mpp
from . import h3_dense_mps as dense

MODES = ('exact', 'tile32', 'dense_int8', 'dense_int8_floatpv', 'int8_v3')
_dead_modes = set()
_HASHES = {
    'h3_dense_shaders/int8.metal': '14201ec68f8e0800fbbd58216c287a19df1d665ec6cb2a51b2f12da91a08fc0d',
    'h3_floatpv_shaders/int8_qk_float_pv.metal': 'b5f22b6a2dea5e3a216c381fbdbf73a16860a1d4c23cc8d85801fe71e5e4020e',
}


def selected_mode():
    # tile32 is exact (byte-identical to the 64-row query-cache kernel) and faster.
    value = os.environ.get('ASFP8_MOSS_ATTN_MODE', 'tile32')
    return value if value in MODES else 'exact'


def _replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError('MOSS attention source changed; retaining exact attention')
    return source.replace(before, after)


def _read(name):
    data = (Path(__file__).parent / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != _HASHES[name]:
        raise RuntimeError('MOSS attention shader differs from the reviewed source')
    return data.decode()


def _int8_source(tokens, v_stride=3072):
    source = _read('h3_dense_shaders/int8.metal')
    for i, (name, value) in enumerate((('N', 1), ('T', tokens), ('H', 24), ('QB', 64))):
        source = _replace(source, f'constant uint SOL_{name} [[function_constant({i})]];',
                          f'constant constexpr uint SOL_{name} = {value};')
    # Read the original BNHD-backed views directly. Quantized scratch stays BHND.
    for before, after in (
        ('(ulong(n) * SOL_H + h) * SOL_T * 32 + t * 32 + vec_dim',
         '(ulong(n) * SOL_T + t) * SOL_H * 32 + h * 32 + vec_dim'),
        ('((ulong(n) * SOL_H + h) * SOL_T + j * 64 + r) * 128 + tid',
         '((ulong(n) * SOL_T + j * 64 + r) * SOL_H + h) * 128 + tid'),
        ('((ulong(n) * SOL_H + h) * SOL_T + t) * 128 + tid',
         '((ulong(n) * SOL_T + t) * SOL_H + h) * 128 + tid'),
        ('(ulong(h) * SOL_T + row + idx[1]) * 128 + idx[0]',
         '((ulong(row) + idx[1]) * SOL_H + h) * 128 + idx[0]'),
        ('  float qsum[4] = {0, 0, 0, 0}; float ksum = 0, vsum = 0;', ''),
        ('    qsum[r / SOL_QB] += float(Q[index]); ksum += float(K[index]); vsum += float(V[index]);', ''),
    ):
        source = _replace(source, before, after)
    source = _replace(source, '(ulong(n) * SOL_T + t) * SOL_H * 32 + h * 32 + vec_dim',
                      f'(ulong(n) * SOL_T + t) * {v_stride//4} + h * 32 + vec_dim')
    source = _replace(source, 'V[index]',
                      f'V[(ulong(n) * SOL_T + j * 64 + r) * {v_stride} + h * 128 + tid]')
    source = _replace(source, 'V[src]',
                      f'V[(ulong(n) * SOL_T + t) * {v_stride} + h * 128 + tid]')
    start = source.index('  const uint JP =', source.index('kernel void sol_quantize'))
    end = source.index('  threadgroup float partials[24];', start)
    source = source[:start] + source[end:]  # Omit unused sparse summary writes.
    # Dense traversal: visit the same K blocks in the same order, without
    # route-bit/word iteration retained from the sparse ancestor.
    start = source.index('    const uint words =', source.index('kernel void sol_attention'))
    end = source.index('    #pragma clang loop unroll(full)', start)
    source = source[:start] + '    for (uint c = 0; c < SOL_T; c += 64) {\n' + source[end:]
    start = source.index('    // Keep SIMD groups near', source.index('kernel void sol_attention'))
    end = source.index('  #pragma clang loop unroll(full)', start)
    source = source[:start] + '    }\n  }\n\n' + source[end:]
    start = source.index('  const uint2 tile = morton_decode_rectangular_2d(group.x,',
                         source.index('kernel void sol_attention'))
    end = source.index(';', start) + 1
    return (source[:start] + '  const uint2 tile = uint2(group.x % query_groups, group.x / query_groups);'
            + source[end:])


def _floatpv_source():
    source = _read('h3_floatpv_shaders/int8_qk_float_pv.metal')
    for before, after in (
        ('constant uint Hq = 56;', 'constant uint Hq = 24;'),
        ('constant uint Hk = 56;', 'constant uint Hk = 24;'),
        ('tgid.y >= 56', 'tgid.y >= 24'),
        ('(head * sequence + start + row) * 128 + dim',
         '((start + row) * heads + head) * 128 + dim'),
        ('auto O = O_buf + tgid.y * R * 128 + tgid.x * (16 * 128);',
         'auto O = O_buf + tgid.y * 128 + tgid.x * (16 * 128 * Hq);'),
    ):
        source = _replace(source, before, after)
    source = _replace(source, '(R + 16 * 4 - 1) / (16 * 4)', '(R + 32 - 1) / 32')
    source = _replace(source, 'tgid.x = tgid.x * 4 + sgid;', 'tgid.x = tgid.x * 2 + sgid;')
    source = _replace(source, '(head * QUANTIZE_KV_SEQUENCE + row) * 128);',
                      '(operand ? (row * QUANTIZE_KV_HEADS + head) : (head * QUANTIZE_KV_SEQUENCE + row)) * 128);', 2)
    source = _replace(source, 'idx[1] * 128]', 'idx[1] * 128 * Hq]', 8)
    constants = {'QUANTIZE_Q_HEADS': 24, 'QUANTIZE_KV_HEADS': 24,
                 'QUANTIZE_Q_TILE_SIZE': 16, 'QUANTIZE_KV_TILE_SIZE': 64,
                 'V_mean_batch_stride': 24 * 128,
                 'dO_batch_stride': 0, 'dV_batch_stride': 0, 'dK_batch_stride': 0,
                 'dQ_batch_stride': 0, 'dO_scale_batch_stride': 0}
    return re.sub(r'constant uint (\w+) \[\[function_constant\(\d+\)\]\];',
                  lambda m: f'constant uint {m[1]} = {constants[m[1]]};', source)


def _bindings(source):
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
    return source, bindings


@functools.lru_cache(maxsize=8)
def _library(mode, tokens=0, v_stride=3072):
    if mode == 'tile32':
        source = _replace(mpp._source(), 'bfloat, 64, 32, 128, 4, 1, false, false, true, bfloat',
                          'bfloat, 32, 32, 128, 2, 1, false, false, true, bfloat')
        return torch.mps.compile_shader(source), {}
    source, bindings = _bindings(_int8_source(tokens, v_stride) if mode == 'dense_int8' else _floatpv_source())
    return torch.mps.compile_shader(source), bindings


@functools.lru_cache(maxsize=16)
def _dimensions(n):
    return torch.tensor([n, n, n*3072, n*3072, n*3072, n*3072,
                         0, 0, 0, 0, 24*((n+15)//16), 24*((n+63)//64),
                         24*((n+63)//64)], device='mps', dtype=torch.int32)


def _run(q, k, v, mode):
    n = q.shape[2]
    if mode == 'int8_v3':
        # The general M5 INT8 kernel with Kitchen-style numerics (rotation, per-tile P).
        from . import int8_attention_mps
        out = torch.empty((1, n, 24, 128), dtype=q.dtype, device=q.device).transpose(1, 2)
        return int8_attention_mps.attention(q, k, v, out=out, numerics='tiled')
    lib, bindings = _library(mode, n if mode == 'dense_int8' else 0, v.stride(2) if mode == 'dense_int8' else 3072)
    out = torch.empty((1, n, 24, 128), dtype=q.dtype, device=q.device).transpose(1, 2)
    if mode == 'tile32':
        p = mpp._parameters(tuple(q.shape), tuple(k.shape), q.stride(), k.stride(), v.stride(), 128**-.5, True)
        lib.custom_attn(q, k, v, out, p, p, p, threads=(((n+31)//32)*64, 24, 1), group_size=(64, 1, 1))
        return out
    def alloc(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device=q.device)
    def dispatch(name, buffers, groups, threads):
        getattr(lib, name)(*[buffers[i] for i in bindings[name]],
                           threads=tuple(g*t for g, t in zip(groups, threads)), group_size=threads)
    blocks = (n+63)//64
    if mode == 'dense_int8':
        # Dummy common-ABI carriers are never read or written by these stages.
        dummy = alloc((1,))
        quantized = [alloc((1, blocks*64, 24, 128), torch.int8) for _ in range(3)]
        qs = alloc((24, blocks*4)); ks, vs = [alloc((24, blocks)) for _ in range(2)]
        vm = alloc((24, 128))
        buffers = [q, k, v, out, dummy, dummy, dummy, dummy, dummy,
                   *quantized, qs, ks, vs, vm, dummy, dummy, dummy, dummy, dummy,
                   dense._parameters(n, 24, 0, 128**-.5, 0.)]
        dispatch('sol_v_mean', buffers, (512, 1, 1), (256, 1, 1))
        dispatch('sol_quantize', buffers, (blocks, 24, 1), (128, 1, 1))
        dispatch('sol_attention', buffers, (blocks*24, 1, 1), (128, 1, 1))
    else:
        v = v.contiguous()  # PV still benefits from head-contiguous rows.
        qi, ki = [alloc((1, n, 24, 128), torch.int8) for _ in range(2)]
        qs = alloc((24, (n+15)//16)); ks = alloc((24, blocks)); vs = alloc((24, blocks))
        vm, km = [alloc((24, 128)) for _ in range(2)]
        chunks = (n+511)//512
        partial = alloc((2, 24, chunks, 32, 4))
        correction, lse = [alloc((24, n)) for _ in range(2)]
        tail = alloc((1, 24, 64, 128), q.dtype); dims = _dimensions(n)
        means = {0:v, 1:vm, 2:k, 3:km, 4:partial, 21:dims}
        dispatch('compute_v_mean', means, (chunks, 48, 1), (256, 1, 1))
        dispatch('finalize_v_mean', means, (1, 48, 1), (32, 1, 1))
        dispatch('quantize_q', {0:q, 1:qi, 2:qs, 3:km, 4:correction, 21:dims},
                 ((n+15)//16, 24, 1), (128, 1, 1))
        dispatch('quantize_k', {0:k, 1:ki, 2:ks, 3:km, 4:v, 5:tail, 21:dims},
                 (blocks, 24, 1), (256, 1, 1))
        dispatch('int8_attention', {0:qi, 1:ki, 2:v, 3:out, 4:lse, 5:tail, 10:qs,
                                  11:ks, 12:vs, 14:vm, 21:dims, 22:correction},
                 (((n+31)//32)*24, 1, 1), (64, 1, 1))
    return out


def maybe_attention(q, k, v):
    mode = selected_mode()
    if mode == 'exact' or mode in _dead_modes or torch.is_grad_enabled():
        return None
    if (torch.__version__.split('+')[0] != '2.14.1' or not mpp._torch_build_supported()
            or not mpp._hardware() or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'):
        return None
    if any(type(t) is not torch.Tensor or t.ndim != 4 or not t.is_mps
           or t.dtype != torch.bfloat16 for t in (q, k, v)):
        return None
    n = q.shape[2]
    if (q.shape != k.shape or q.shape != v.shape or q.shape[:2] != (1, 24)
            or q.shape[3] != 128 or not 4096 <= n <= 6656
            or any(t.stride()[1:] != (128, 3072, 1) for t in (q, k))
            or v.stride(1) != 128 or v.stride(3) != 1 or v.stride(2) not in (3072, 9216)):
        return None
    try:
        if mode != 'tile32':
            dense.check_workspace(n, 24, layout_copies=3)
        return _run(q, k, v, mode)
    except dense.WorkspaceBudgetExceeded:
        return None  # A later, smaller request may be eligible.
    except Exception as exc:
        _dead_modes.add(mode)
        logging.warning('[AppleSilicon-Optimized/MOSS] %s unavailable; retaining exact attention: %s', mode, exc)
        return None
