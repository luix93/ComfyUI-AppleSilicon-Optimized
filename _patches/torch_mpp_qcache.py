"""Exact query-load hoisting for PyTorch 2.14 MPP attention on M5 Ultra.

Uses the installed, hash-checked PyTorch headers. Only Q loads move outside
its key loop; tile sizes, MMA, softmax, barriers and reductions stay intact.
The bounded parameter cache holds at most 16 KiB, never activations.
"""
import functools
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess

import torch

_TORCH_GIT = '08187d9e0fba026dc8217405802ab5381dc88d90'
_SOURCE_HASH = '823c0df0b306a2a82b4c8903439e2b4b70bedb86c28b8f720adb92fb22892cff'
_library = None
_dead = False
_calls = 0


@functools.lru_cache(maxsize=1)
def _hardware():
    try:
        return 'M5 Ultra' in subprocess.check_output(
            ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True,
            timeout=2).strip()
    except (OSError, subprocess.SubprocessError):
        return False


def _source():
    root = Path(torch.__file__).parent / 'include'
    seen = set()
    def expand(name):
        if name in seen:
            return ''
        seen.add(name)
        source = (root / name).read_text()
        if name.endswith('MppAttention.h'):
            source = source[:source.index('#define instantiate_mpp_attn_arch')] + '\n#endif\n#endif\n'
        if name.endswith('PrefillAttention.h'):
            source = source[:source.index('#define instantiate_attn(')] + '\n#endif\n'
        source = re.sub(r'#include <((?:ATen|c10)/[^>]+)>', lambda m: expand(m[1]), source)
        return source.replace('#pragma once', '')
    source = ('#include <metal_stdlib>\nusing namespace metal;\n'
              + expand('c10/metal/utils.h')
              + expand('ATen/native/mps/kernels/MppAttention.h'))
    if hashlib.sha256(source.encode()).hexdigest() != _SOURCE_HASH:
        raise RuntimeError('PyTorch MPP headers differ from the validated 2.14 build')
    start = source.index('prefill_attention_mpp(')
    prefix, body = source[:start], source[start:]
    before = '  for (int kb = 0; kb < kb_lim; kb++) {'
    load = '''  MPPTile<T, 1, 1, QFrag_t> Qcache[TD];
  MPP_PRAGMA_UNROLL
  for (short id = 0; id < TD; id++) {
    if (!align_Q && is_last_q) Qcache[id].load_rows(Q + id*kU,q_seq_stride,lim_rows_q);
    else Qcache[id].load(Q + id*kU,q_seq_stride);
  }
'''
    old_load = '''          if (!align_Q && is_last_q) {
            Qtile.load_rows(Q + Q_load_off, q_seq_stride, lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, q_seq_stride);
          }
'''
    assert body.count(before) == body.count(old_load) == 1
    body = body.replace(before, load + before).replace(
        '          MPPTile<T, 1, 1, QFrag_t> Qtile;',
        '          thread auto& Qtile = Qcache[id];').replace(old_load, '')
    return (prefix + body + '\ninstantiate_kernel("custom_attn", prefill_attention_mpp, '
            'bfloat, 64, 32, 128, 4, 1, false, false, true, bfloat);\n')


@functools.lru_cache(maxsize=128)
def _parameters(shape, keys, qstride, kstride, vstride, scale):
    b, h, seq, dim = shape
    output_stride = (h * seq * dim, seq * dim, dim)
    raw = struct.pack('=6i2f12q', b, h, dim, seq, keys[2], h // keys[1],
                      scale, 1., *qstride[:3], *kstride[:3], *vstride[:3], *output_stride)
    return torch.tensor(list(raw), dtype=torch.uint8, device='mps')


def _supports(q, k, v, kwargs):
    if (_dead or os.environ.get('ASFP8_MPP_QCACHE', '1') != '1'
            or os.environ.get('ASFP8_ATTN_BACKEND') != 'pytorch'
            or torch.is_grad_enabled()
            or torch.__version__.split('+')[0] != '2.14.0'
            or torch.version.git_version != _TORCH_GIT):
        return False
    if (any(type(t) is not torch.Tensor or not t.is_mps or t.dtype != torch.bfloat16
            or t.ndim != 4 or t.stride(-1) != 1 for t in (q, k, v))
            or q.shape[0] != 1 or q.shape[-1] != 128
            or not 2048 <= q.shape[2] <= 32768 or k.shape != v.shape
            or k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]
            or (q.shape[1], k.shape[1]) not in ((30, 30), (48, 12), (56, 56))
            # Batch/head pointers use int64 strides. Only the in-tile row
            # offset is narrowed to int32 by the underlying MPP kernel.
            or any(s < 0 for t in (q, k, v) for s in t.stride())
            or any(t.stride(2) >= 2**31 // 64 for t in (q, k, v))):
        return False
    if (kwargs.keys() - {'attn_mask', 'dropout_p', 'is_causal', 'scale', 'enable_gqa'}
            or kwargs.get('attn_mask') is not None or kwargs.get('dropout_p', 0.) != 0.
            or kwargs.get('is_causal', False)
            or (q.shape[1] != k.shape[1] and not kwargs.get('enable_gqa', False))):
        return False
    scale = kwargs.get('scale')
    return (scale is None or type(scale) in (float, int) and math.isfinite(scale)) and _hardware()


def maybe_attention(q, k, v, **kwargs):
    global _library, _dead, _calls
    if not _supports(q, k, v, kwargs):
        return None
    try:
        if _library is None:
            _library = torch.mps.compile_shader(_source())
            print('[AppleSilicon-Optimized/mpp_qcache] exact PyTorch 2.14 query-load hoisting enabled.')
        scale = kwargs.get('scale')
        params = _parameters(tuple(q.shape), tuple(k.shape), q.stride(), k.stride(), v.stride(),
                             128**-.5 if scale is None else float(scale))
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        _library.custom_attn(q, k, v, out, params, params, params,
                            threads=(((q.shape[2] + 63) // 64) * 128, q.shape[1], 1),
                            group_size=(128, 1, 1))
        _calls += 1
        return out
    except Exception as exc:
        _dead = True
        _parameters.cache_clear()
        print(f'[AppleSilicon-Optimized/mpp_qcache] disabled ({exc!r}); using PyTorch SDPA.')
        return None
