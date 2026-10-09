"""Optional Veda integration, isolated from the existing H3 accelerators."""

import logging
import math
from pathlib import Path

import torch

DEFAULT_PREDICTOR = 'minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors'


def predictor_options():
    try:
        import folder_paths
    except ImportError:
        return [DEFAULT_PREDICTOR]
    folder = str(Path(folder_paths.models_dir) / 'veda')
    if 'veda' not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths['veda'] = ([folder], {'.safetensors'})
    names = folder_paths.get_filename_list('veda')
    return names + ([] if DEFAULT_PREDICTOR in names else [DEFAULT_PREDICTOR])


def apply_veda(model, *, backend, predictor, sparsity, reference_sparsity,
               dense_steps, dense_layers, dense_leading_blocks, start_percent,
               end_percent, window_mode, min_sequence_tokens, memory_budget_gib,
               verbose):
    import folder_paths
    import comfy.patcher_extension
    from . import attn_router_mps
    from .h3_runtime.veda import comfy_patch, settings, hardware
    from .h3_runtime.veda.backends import base
    from .h3_runtime.veda.core import bundle
    from .h3_runtime.veda.runtime import AppleVedaEngine, BoundedMlxBackend

    predictor_options()
    path = folder_paths.get_full_path('veda', predictor)
    if path is None:
        raise ValueError(f'Veda predictor {predictor!r} is missing. Put it in models/veda and restart ComfyUI.')
    loaded = bundle.load_bundle(path)
    diffusion = model.get_model_object('diffusion_model')
    if type(diffusion).__name__ != 'MiniMaxH3Model':
        raise ValueError('Veda acceleration requires a MiniMax H3 model')
    attn = diffusion.blocks[0].attn
    if (len(diffusion.blocks), attn.heads, attn.head_dim) != (loaded.num_layers, loaded.num_heads, loaded.head_dim):
        raise ValueError('The selected Veda predictor does not match this H3 model')
    if backend not in ('veda_mlx', 'veda_metal'):
        raise ValueError(f'Unsupported Veda backend: {backend}')
    if window_mode not in ('sigma_percent', 'step_fraction'):
        raise ValueError(f'Unsupported Veda schedule: {window_mode}')
    if not (0 <= start_percent <= end_percent <= 1):
        raise ValueError('Veda start/end fractions must satisfy 0 <= start <= end <= 1')
    if not math.isfinite(memory_budget_gib) or memory_budget_gib < 0:
        raise ValueError('Veda temporary memory must be finite and non-negative')
    config = settings.VedaSettings(
        generated=settings.parse_sparsity(f'{float(sparsity):g}%', 'Veda sparsity'),
        reference=settings.parse_sparsity(f'{float(reference_sparsity):g}%', 'Veda reference sparsity'),
        dense_layers=settings.parse_index_list(dense_layers, 'Veda dense layers') | frozenset(range(dense_leading_blocks)),
        dense_steps=settings.parse_index_list(dense_steps, 'Veda dense steps'), verbose=verbose)

    class Patch(comfy_patch.VedaPatch):
        def make_override(self, previous):
            return super().make_override(previous or attn_router_mps.dense_without_override)

        def _engine(self, device):
            key = str(device)
            if key not in self._engines:
                if device.type != 'mps':
                    raise base.BackendUnavailable('Apple Silicon Veda requires MPS')
                if backend == 'veda_metal':
                    from .h3_veda_mps import MetalBackend
                    impl = MetalBackend()
                else:
                    from .h3_runtime.veda.backends import mlx_gather
                    upstream = mlx_gather.create(hardware.describe(device))
                    impl = BoundedMlxBackend(upstream.mx)
                try:
                    base.self_test(impl, device)
                except Exception:
                    if backend == 'veda_mlx':
                        impl.close()
                    raise
                self._engines[key] = AppleVedaEngine(self.bundle, config.generated, config.reference, impl, device,
                    route=backend, memory_budget_gib=memory_budget_gib)
            return self._engines[key]

        def _dense_reason(self, q, options):
            reason = super()._dense_reason(q, options)
            if reason is not None:
                return reason
            if q.shape[2] < min_sequence_tokens:
                return 'short sequence'
            if window_mode == 'step_fraction':
                step = self._step(options)
                schedule = options.get('sample_sigmas')
                if step is None or schedule is None or len(schedule) < 2:
                    return 'missing sampler schedule'
                count = len(schedule) - 1
                first = math.ceil(start_percent * count - 1e-9)
                last = math.floor(end_percent * count + 1e-9)
                if not first <= step < last:
                    return 'protected step'
            else:
                sigma = options.get('sigmas')
                if sigma is None:
                    return 'missing sampler sigma'
                sampling = model.get_model_object('model_sampling')
                start, end = sampling.percent_to_sigma(start_percent), sampling.percent_to_sigma(end_percent)
                value = float(sigma.flatten()[0])
                if not end <= value <= start:
                    return 'protected step'
            return None

        def on_cleanup(self):
            summary = self._summary()
            if summary:
                logging.info('[AppleSilicon-Optimized/H3Veda] %s', summary)
            for e in self._engines.values():
                e.reset()
                if backend == 'veda_mlx':
                    e.backend.close()
            self._engines.clear()
            self._steps.clear()
            self.run = comfy_patch._Run()

    patch = Patch(loaded, config, None)
    patched = model.clone()
    patch.install(patched.model_options.setdefault('transformer_options', {}))
    callbacks = comfy.patcher_extension.CallbacksMP
    patched.add_callback_with_key(callbacks.ON_PREPARE_STATE, 'as_h3_veda',
        lambda mp, timestep, options: patch.install(options['transformer_options']))
    patched.add_callback_with_key(callbacks.ON_CLEANUP, 'as_h3_veda', lambda mp: patch.on_cleanup())
    return patched
