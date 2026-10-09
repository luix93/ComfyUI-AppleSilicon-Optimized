"""Apple Silicon learned MiniMax H3 latent upscaling (2D / 3D checkpoints).

Model architecture derives from LBH-123-AI's MIT-licensed implementation.
This adapter adds MPS execution, bounded memory, strict safe loading, and H3
nested video/audio support. It does not perform a second diffusion pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import math
from pathlib import Path
import threading
import time

import torch
import torch.nn.functional as F

_FOLDER = 'latent_upscale_models'
_LOCK = threading.RLock()
_DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def _paths():
    import folder_paths
    if _FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(_FOLDER, str(Path(folder_paths.models_dir) / _FOLDER))
    return folder_paths


@lru_cache(maxsize=128)
def _is_h3_file(path, size, mtime):
    # Read only the safetensors header; unrelated LTX upscalers share this folder.
    from safetensors import safe_open, SafetensorError
    try:
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in ('conv_in.weight', 'resizer.conv_in.weight',
                        'upscaler.conv_in.weight', 'upscaler.resizer.conv_in.weight'):
                if key in f.keys():
                    shape = f.get_slice(key).get_shape()
                    return len(shape) in (4, 5) and shape[1] == 24
    except (OSError, ValueError, RuntimeError, SafetensorError):
        pass
    return False


def model_names():
    paths = _paths()
    result = []
    for name in paths.get_filename_list(_FOLDER):
        path = Path(paths.get_full_path_or_raise(_FOLDER, name))
        if path.suffix.lower() != '.safetensors':
            continue
        stat = path.stat()
        if _is_h3_file(str(path), stat.st_size, stat.st_mtime_ns):
            result.append(name)
    return result or ['(place H3 safetensors in models/latent_upscale_models)']


@lru_cache(maxsize=1)
def _load_model_cpu(path, size, mtime):
    """One CPU model, never an unbounded cache of GPU/dtype combinations."""
    from safetensors.torch import load_file
    sd = load_file(path, device='cpu')
    if any(k.startswith('upscaler.') for k in sd):
        sd = {k.removeprefix('upscaler.'): v for k, v in sd.items() if k.startswith('upscaler.')}
    if 'conv_in.weight' in sd and sd['conv_in.weight'].ndim == 5:
        from . import h3_upscale_network_3d as net
        cls, kind = net.LatentResizer3D, '3d'
    elif 'resizer.conv_in.weight' in sd and sd['resizer.conv_in.weight'].ndim == 4:
        from . import h3_upscale_network_2d as net
        cls, kind = net.VideoLatentResizer, '2d'
    else:
        raise ValueError('This is not a supported MiniMax H3 latent-upscaler checkpoint')
    cfg = net._detect_arch(sd)
    if cfg['in_channels'] != 24:
        raise ValueError('H3 video latents must have 24 channels')
    with torch.device('meta'):
        model = cls(**cfg)
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, kind, net


def target_size(height, width, mode, scale, target_width, target_height, megapixels):
    if mode == 'scale':
        factor = float(scale)
        h, w = height * factor, width * factor
    elif mode == 'dimensions':
        h, w = float(target_height) / 16, float(target_width) / 16
        factor = (h / height + w / width) / 2
    elif mode == 'megapixels':
        factor = math.sqrt(float(megapixels) * 1024 * 1024 / (height * width * 256))
        h, w = height * factor, width * factor
    else:
        raise ValueError('Unknown upscale mode')
    if not all(math.isfinite(v) for v in (h, w, factor)) or not 1 <= factor <= 4:
        raise ValueError('H3 latent upscaling supports factors from 1 to 4')
    if mode == 'scale' and factor == 1:
        return height, width, factor
    h, w = max(2, round(h / 2) * 2), max(2, round(w / 2) * 2)
    if h < height or w < width or h > height * 4 or w > width * 4:
        raise ValueError('Each spatial dimension must remain between 1× and 4× the input')
    return h, w, factor


def working_bytes(model, shape, target_hw, dtype, kind, chunk_size):
    """Conservative admission estimate; runtime checks remain necessary."""
    b, _, t, _, _ = shape
    h, w = target_hw
    channels = model.conv_in.out_channels if kind == '3d' else model.resizer.conv_in.out_channels
    # Upstream overlap can add four kernel widths to an interior chunk.
    temporal = next((m.dwconv.kernel_size[0] for m in model.modules() if hasattr(m, 'dwconv')), 0)
    frames = min(t + 2 * temporal, chunk_size + 4 * temporal) if kind == '3d' and chunk_size and t > chunk_size else t
    elements = b * channels * frames * h * w
    if elements >= 2**29:
        raise ValueError('Upscale feature tensor is too large for safe MPS indexing; reduce size/chunk length')
    element_size = torch.empty((), dtype=dtype).element_size()
    # Include the cached checkpoint, transient CPU copy and active typed weights.
    weights = sum(p.numel() * (element_size + 2 * p.element_size()) for p in model.parameters())
    return weights + elements * element_size * 16 + b * 24 * t * h * w * element_size * 4


@lru_cache(maxsize=1)
def _memory_provider():
    from . import psutil_vmstat
    # The project already patches psutil on affected macOS versions.
    psutil_vmstat.install()
    import psutil
    return psutil


def _available_bytes():
    return _memory_provider().virtual_memory().available


@contextmanager
def _bounded_execution(model, device):
    """Drain completed command buffers per block, not once after the whole video."""
    handles = []
    last_check = 0.0
    def after_block(module, args, output):
        nonlocal last_check
        if device.type == 'mps':
            torch.mps.synchronize()
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
        now = time.monotonic()
        if now - last_check >= 1:
            last_check = now
            if _available_bytes() < 4 * 2**30:
                raise RuntimeError('H3 upscaler stopped: less than 4 GiB system memory remains')
    try:
        for module in model.modules():
            if module.__class__.__name__ in ('ResBlockEmb', 'ResBlockEmb3D', 'TemporalConv'):
                handles.append(module.register_forward_hook(after_block))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _resize_mask(mask, size):
    if getattr(mask, 'is_nested', False):
        from comfy.nested_tensor import NestedTensor
        parts = list(mask.unbind())
        return NestedTensor([_resize_mask(parts[0], size), *parts[1:]])
    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        raise ValueError('Unsupported noise mask for H3 upscaling')
    if mask.shape[-2:] == (1, 1):
        return mask
    shape = mask.shape
    return F.interpolate(mask.reshape(-1, 1, *shape[-2:]).float(), size=size,
                         mode='nearest').to(mask.dtype).reshape(*shape[:-2], *size)


class AppleSiliconH3LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'latent': ('LATENT',),
            'model_name': (model_names(),),
            'mode': (['scale', 'dimensions', 'megapixels'],),
            'scale': ('FLOAT', {'default': 2.0, 'min': 1.0, 'max': 4.0, 'step': .05}),
            'target_width': ('INT', {'default': 1728, 'min': 32, 'max': 8192, 'step': 32}),
            'target_height': ('INT', {'default': 960, 'min': 32, 'max': 8192, 'step': 32}),
            'megapixels': ('FLOAT', {'default': 1.0, 'min': .1, 'max': 16.0}),
            'precision': (['fp16', 'bf16', 'fp32'], {'default': 'fp16'}),
            'device': (['mps', 'cpu'], {'default': 'mps'}),
            'temporal_chunk': ('INT', {'default': 32, 'min': 0, 'max': 128,
                'tooltip': '3D only. 0 = full clip; chunking changes normalization/context and is not bit-exact to full inference.'}),
            'memory_budget_gib': ('FLOAT', {'default': 8.0, 'min': 1.0, 'max': 32.0,
                'tooltip': 'Conservative working-memory admission budget. Reduce dimensions/chunk length if refused.'}),
        }}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'upscale'
    CATEGORY = 'Apple Silicon/Minimax H3'
    DESCRIPTION = ('Learned H3 latent upscaling on Metal. Preserves audio and frame count. '
                   'Use before a refinement sampler or video VAE decode. FP16 recommended.')

    @torch.inference_mode()
    def upscale(self, latent, model_name, mode='scale', scale=2.0,
                target_width=1728, target_height=960, megapixels=1.0,
                precision='fp16', device='mps', temporal_chunk=32, memory_budget_gib=8.0):
        src = latent['samples']
        nested = bool(getattr(src, 'is_nested', False))
        parts = list(src.unbind()) if nested else [src]
        video = parts[0]
        if not isinstance(video, torch.Tensor) or video.ndim not in (4, 5) or video.shape[1] != 24:
            raise ValueError('Expected H3 video latents [B,24,T,H,W] or [B,24,H,W]')
        if nested and (len(parts) != 2 or parts[1].shape[1] != 32):
            raise ValueError('Expected H3 nested video (24 channels) and audio (32 channels)')
        h, w, factor = target_size(*video.shape[-2:], mode, scale, target_width, target_height, megapixels)
        if (h, w) == video.shape[-2:]:
            return (latent,)
        if precision not in _DTYPES or device not in ('mps', 'cpu'):
            raise ValueError('Unsupported precision or device')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('Metal/MPS is unavailable; select CPU or use a PyTorch build with MPS')
        if not 0 <= temporal_chunk <= 128 or not 1 <= memory_budget_gib <= 32:
            raise ValueError('Invalid memory budget or temporal chunk length')
        mask = _resize_mask(latent['noise_mask'], (h, w)) if 'noise_mask' in latent else None
        path = Path(_paths().get_full_path_or_raise(_FOLDER, model_name))
        stat = path.stat()
        dtype, dev = _DTYPES[precision], torch.device(device)
        shape = video.shape if video.ndim == 5 else (*video.shape[:2], 1, *video.shape[2:])
        with _LOCK:
            model, kind, net = _load_model_cpu(str(path), stat.st_size, stat.st_mtime_ns)
            estimate = working_bytes(model, shape, (h, w), dtype, kind, temporal_chunk)
            if estimate > memory_budget_gib * 2**30:
                raise ValueError(f'H3 upscaler estimates {estimate / 2**30:.1f} GiB, above the '
                                 f'{memory_budget_gib:g} GiB budget. Reduce resolution or 3D temporal_chunk.')
            import comfy.model_management as mm
            mm.free_memory(estimate + 2 * 2**30, dev)
            if _available_bytes() < estimate + 4 * 2**30:
                raise RuntimeError('Insufficient free unified memory for H3 latent upscaling')
            # A fresh typed copy leaves the cached checkpoint in its original precision.
            # No keep-on-GPU cache can pin this model across sampling/decoding.
            import copy
            active = copy.deepcopy(model).to(device=dev, dtype=dtype)
            try:
                active.chunk_size = temporal_chunk or shape[2]
                value = video.to(device=dev, dtype=dtype)
                if video.ndim == 4:
                    value = value.unsqueeze(2)
                mean, std = net._make_norm_tensors(dev, dtype)
                value = (value - mean) / std
                with _bounded_execution(active, dev):
                    if kind == '3d':
                        value = active(value, scale=factor, target_size=(shape[2], h, w),
                                       enable_chunking=bool(temporal_chunk))
                    else:
                        value = active(value, scale=factor, target_hw=(h, w))
                value = value * std + mean
                if not torch.isfinite(value).all().item():
                    raise RuntimeError('Non-finite upscaler output; try FP32 precision')
                if video.ndim == 4:
                    value = value.squeeze(2)
                value = value.to(device='cpu', dtype=video.dtype)
            finally:
                del active
                if dev.type == 'mps':
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        result = dict(latent)
        if nested:
            from comfy.nested_tensor import NestedTensor
            result['samples'] = NestedTensor([value, *parts[1:]])
        else:
            result['samples'] = value
        if mask is not None:
            result['noise_mask'] = mask
        return (result,)


NODE_CLASS_MAPPINGS = {'AppleSiliconH3LatentUpscale': AppleSiliconH3LatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {'AppleSiliconH3LatentUpscale': 'H3 Latent Upscale · Apple Silicon'}
