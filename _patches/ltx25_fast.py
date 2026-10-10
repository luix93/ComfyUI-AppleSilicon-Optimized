"""LTX 2.5 fusions for the validated M5 Ultra / PyTorch 2.14 runtime.

The current ComfyUI rotation API supplies (rotation_matrix, split_mode),
including a joint Q/K entry point. Preserve Kitchen's BF16 store rounding.
Older interfaces, training, other layouts and dtypes keep their original path.
"""
import functools
import os
from pathlib import Path

import torch
from . import torch_mpp_qcache

_installed = False
_library = None
_dead = False
_calls = 0


def _lib():
    global _library, _dead
    if _library is None and not _dead:
        try:
            _library = torch.mps.compile_shader(Path(__file__).with_suffix('.metal').read_text())
        except Exception as exc:
            _dead = True
            print(f'[AppleSilicon-Optimized/ltx25] kernel unavailable ({exc!r}); keeping original operations.')
    return _library


def _supports(x, pe):
    if (os.environ.get('ASFP8_LTX25_FAST', '1') != '1'
            or os.environ.get('ASFP8_LTX25_ROPE', '1') != '1'
            or torch.is_grad_enabled() or _dead
            or type(x) is not torch.Tensor or not x.is_mps
            or x.dtype != torch.bfloat16 or x.ndim != 3
            or not x.is_contiguous() or not 0 < x.numel() < 2**31
            or not isinstance(pe, (tuple, list)) or len(pe) != 2 or pe[1] is not True):
        return False
    r = pe[0]
    return (type(r) is torch.Tensor and r.is_mps and r.dtype == x.dtype
            and r.is_contiguous() and r.ndim == 6 and r.shape[-2:] == (2, 2)
            and r.shape[:2] == x.shape[:2] and r.shape[2] == 32
            and r.shape[3] in (32, 64) and x.shape[-1] == r.shape[2] * r.shape[3] * 2)


def _rope(q, k, pe):
    global _calls
    oq = torch.empty_like(q)
    ok = torch.empty_like(k) if k is not None else oq
    _library.split_rope(q, q if k is None else k, pe, oq, ok, pe.shape[3] * 2,
                        threads=(q.numel(), 1 if k is None else 2), group_size=(256, 1))
    _calls += 1
    return oq if k is None else (oq, ok)


def install():
    global _installed
    if (_installed or os.environ.get('ASFP8_DIT_EXT') != '1'
            or os.environ.get('ASFP8_LTX25_FAST', '1') != '1'
            or os.environ.get('PYTORCH_MPS_FAST_MATH', '0') != '0'
            or not torch_mpp_qcache._torch_build_supported()
            or not torch_mpp_qcache._hardware() or not torch.backends.mps.is_available()):
        return
    try:
        import comfy.ldm.lightricks.model as model
    except ImportError:
        return
    if not hasattr(model, 'apply_rotary_emb_qk'):
        return
    single, joint = model.apply_rotary_emb, model.apply_rotary_emb_qk

    @functools.wraps(single)
    def rope(input_tensor, freqs_cis):
        x, pe = input_tensor, freqs_cis
        if (not model.comfy.model_management.in_training and _supports(x, pe)
                and _lib() is not None):
            return _rope(x, None, pe[0])
        return single(x, pe)

    @functools.wraps(joint)
    def rope_qk(q, k, freqs_cis):
        pe = freqs_cis
        if (not model.comfy.model_management.in_training and _supports(q, pe)
                and _supports(k, pe) and q.shape == k.shape and _lib() is not None):
            return _rope(q, k, pe[0])
        return joint(q, k, pe)

    model.apply_rotary_emb, model.apply_rotary_emb_qk = rope, rope_qk
    from . import ltx25_vae
    ltx25_vae.install()
    _installed = True
    print('[AppleSilicon-Optimized/ltx25] current split-RoPE API fusion enabled.')
