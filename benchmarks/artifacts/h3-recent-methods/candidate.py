import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
stats = {}
settings = {}
hooks = []


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def install():
    sys.path.insert(0, str(ROOT / 'upstream/spectrum'))
    sys.path.insert(0, str(ROOT / 'upstream/speed'))


def set_mode(mode):
    stats.clear()
    stats['settings'] = dict(settings)


def count(name):
    def hook(*args):
        stats[name] = stats.get(name, 0) + 1
    return hook


def configure(guider, mode):
    global settings, hooks
    for hook in hooks:
        hook.remove()
    blocks = guider.model_patcher.model.diffusion_model.blocks
    hooks = [blocks[i].register_forward_pre_hook(count('block_' + str(i))) for i in (0, 49)]
    settings = {}
    patched = guider.model_patcher
    if mode == 'ui-motion':
        import nodes
        mod = module('h3_motion_candidate', ROOT / 'upstream/motioncache/motion_cache.py')
        nodes.NODE_CLASS_MAPPINGS['MiniMaxH3MotionCache'] = mod.MiniMaxH3MotionCache
        unified = next(m for n,m in sys.modules.items() if n.endswith('._patches.minimax_h3_unified'))
        settings = {k:v[1]['default'] for k,v in unified.AppleSiliconH3Acceleration.INPUT_TYPES()['required'].items() if k != 'model'}
        settings.update(cache_mode='motion_cache', token_reduction=False, sparse_attention=False)
        patched = unified.apply_h3_acceleration(patched, **settings)
    elif mode.startswith('ui-spectrum'):
        import nodes
        from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3
        nodes.NODE_CLASS_MAPPINGS['SpectrumApplyMiniMaxH3'] = SpectrumApplyMiniMaxH3
        unified = next(m for n,m in sys.modules.items() if n.endswith('._patches.minimax_h3_unified'))
        settings = {k:v[1]['default'] for k,v in unified.AppleSiliconH3Acceleration.INPUT_TYPES()['required'].items() if k != 'model'}
        settings.update(cache_mode=mode[3:].replace('-', '_'), token_reduction=False, sparse_attention=False)
        patched = unified.apply_h3_acceleration(patched, **settings)
    elif mode.startswith('spectrum'):
        from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3
        settings = {k:v[1].get('default', v[0][0] if isinstance(v[0],list) else None)
                    for group in SpectrumApplyMiniMaxH3.INPUT_TYPES().values()
                    for k,v in group.items() if k != 'model'}
        settings.update(debug=True, history_storage='vram', offline_archive_storage='vram')
        if mode == 'spectrum-ram':
            settings.update(history_storage='system_ram', offline_archive_storage='system_ram')
        if mode == 'spectrum-single':
            settings['offline_smoothing_replay'] = False
        if mode == 'spectrum-safe':
            settings.update(warmup_steps=4, tail_actual_steps=2, bootstrap_first_forecast=False)
        patched = SpectrumApplyMiniMaxH3().apply(patched, **settings)[0]
    elif mode.startswith('motion'):
        mod = module('h3_motion_candidate', ROOT / 'upstream/motioncache/motion_cache.py')
        settings = dict(reuse_threshold=float(mode[6:]), motion_strength=1.0,
                        warmup_steps=4, max_consecutive_skips=2,
                        start_percent=.15, end_percent=.95,
                        subsample_factor=8, verbose=True)
        patched = mod.MiniMaxH3MotionCache.execute(patched, **settings).result[0]
    elif mode.startswith('speed') or mode in ('exact', 'harvest'):
        pass
    else:
        raise ValueError(mode)
    clone = copy.copy(guider)
    clone.model_patcher = patched
    clone.model_options = patched.model_options
    return clone


def sample(mode, **kw):
    if mode == 'harvest':
        mod = module('h3_harvest_candidate', ROOT / 'upstream/speed/nodes/sampler_sigma_harvest_node.py')
        kw.pop('sampler')
        report, latent = mod.MiniMaxH3HarvestToConfig().harvest(**kw, sampler_name='res_multistep')
        parsed = json.loads(report)
        stats['harvest'] = parsed
        (ROOT / 'harvest-latest.json').write_text(report)
        assert 'error' not in parsed, parsed
        return latent, latent
    if mode.startswith('speed'):
        mod = module('h3_speed_candidate', ROOT / 'upstream/speed/nodes/sampler_node.py')
        stages = int(mode[5])
        kw.pop('sampler')
        calibration = {}
        if '-cal' in mode:
            parsed = json.loads((ROOT / 'harvest-latest.json').read_text())
            calibration = {k:parsed[k] for k in ('noise_amplitude', 'noise_decay_exponent')}
            calibration['Tolerance (Delta)'] = .01 if mode.endswith('01') else .005
        stats['speed_calibration'] = calibration
        return mod.MiniMaxH3SPEEDSampler().sample(
            **kw, sampler_name='res_multistep', stages=stages,
            noise_policy='coupled_full_grid', **calibration)
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    return SamplerCustomAdvanced.execute(**kw).result
