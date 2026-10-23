"""Parity against pinned sources and lifecycle regressions in owned runtimes."""
import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch


def assert_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _motion_modules():
    # Load the original state verbatim, without importing its ComfyUI node API.
    # Arithmetic parity tests also work on machines without ComfyUI installed.
    import ast
    source = Path(__file__).parents[1] / '_patches/_vendor/motioncache/motion_cache.py'
    original = types.ModuleType('motioncache_parity_reference')
    original.torch = torch
    original.logging = importlib.import_module('logging')
    tree = ast.parse(source.read_text())
    state = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'MiniMaxH3MotionCacheState')
    exec(compile(ast.Module(body=[state], type_ignores=[]), str(source), 'exec'), original.__dict__)
    return importlib.import_module('_patches.h3_runtime.motion.runtime'), original


def _comfy_patcher():
    pytest.importorskip('comfy.cli_args')
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
    return importlib.import_module('comfy.model_patcher')


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('threshold', [0.0, 0.15, 0.35])
def test_motioncache_decisions_scores_and_outputs_match_pinned_runtime(device, threshold):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    modules = _motion_modules()
    states = [m.MiniMaxH3MotionCacheState(threshold, 1.0, 4, 2, .15, .95, 2)
              .prepare(types.SimpleNamespace(percent_to_sigma=lambda p: 1 - p)) for m in modules]
    gen = torch.Generator().manual_seed(987)
    video = torch.randn(1, 4, 5, 8, 8, generator=gen).to(device)
    audio = torch.randn(1, 4, 20, generator=gen).to(device)
    for i in range(20):
        inputs = [video * (1 - i * .012), audio * (1 - i * .009)]
        outputs = []
        decisions = []
        sigma = torch.tensor([1 - i * .045], device=device)
        for state in states:
            state.check_metadata(*inputs, ['cond'])
            state.step += 1
            state.total_calls += 1
            reuse, score = state.should_reuse(*inputs, sigma)
            decisions.append((reuse, score, state.accumulated_change))
            if reuse:
                outputs.append(state.apply_residual(*inputs))
            else:
                actual = [inputs[0] * .24 + .2, inputs[1] * .17 - .1]
                state.update(*inputs, *actual, sigma)
                outputs.append(actual)
        assert decisions[0] == decisions[1]
        for a, b in zip(*outputs):
            assert_bytes(a, b)
    assert states[0].total_skipped == states[1].total_skipped
    if threshold:
        assert states[0].total_skipped > 0


def test_motion_restores_options_when_preparation_fails(monkeypatch):
    motion, _ = _motion_modules()
    original = {'transformer_options': {}}
    class Broken:
        def clone(self): return self
        def prepare(self, sampling): raise RuntimeError('prepare failed')
    cloned = {'transformer_options': {motion.PATCH_KEY: Broken()}}
    monkeypatch.setattr(_comfy_patcher(), 'create_model_options_clone', lambda x: cloned)
    guider = types.SimpleNamespace(model_options=original,
        model_patcher=types.SimpleNamespace(model=types.SimpleNamespace(model_sampling=None)))
    executor = types.SimpleNamespace(class_obj=guider)
    with pytest.raises(RuntimeError, match='prepare failed'):
        motion.motioncache_sample_wrapper(executor)
    assert guider.model_options is original


def test_motion_keyword_inputs_and_cleanup_on_executor_error(monkeypatch):
    motion, _ = _motion_modules()
    state = motion.MiniMaxH3MotionCacheState(.15, 1, 4, 2, .15, .95, 2)
    options = {motion.PATCH_KEY: state, 'uuids': ['cond'], 'sigmas': torch.tensor([1.])}
    inputs = [torch.zeros(1, 4, 3, 4, 4), torch.zeros(1, 4, 10)]
    out = motion.motioncache_forward_wrapper(lambda **kw: [t + 1 for t in kw['x']],
                                             x=inputs, transformer_options=options)
    assert all(torch.equal(x, torch.ones_like(x)) for x in out)
    original = {'transformer_options': {motion.PATCH_KEY: state}}
    monkeypatch.setattr(_comfy_patcher(), 'create_model_options_clone',
                        lambda x: {'transformer_options': dict(x['transformer_options'])})
    guider = types.SimpleNamespace(model_options=original, model_patcher=types.SimpleNamespace(
        model=types.SimpleNamespace(model_sampling=types.SimpleNamespace(percent_to_sigma=lambda p: 1-p))))
    class Executor:
        class_obj = guider
        def __call__(self):
            run = guider.model_options['transformer_options'][motion.PATCH_KEY]
            run.previous_video_input = inputs[0]
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        motion.motioncache_sample_wrapper(Executor())
    assert guider.model_options is original
    assert state.previous_video_input is not None  # template state was not touched


@pytest.mark.parametrize('protected', [False, True])
def test_spectrum_owned_config_and_forecasts_match_pinned_runtime(protected):
    from _patches.h3_runtime.spectrum.api import make_config
    from _patches.h3_runtime.spectrum.forecast import HistoryWeightForecaster
    from _patches._vendor.spectrum_h3.forecast import HistoryWeightForecaster as Original
    from _patches._vendor.spectrum_h3.config import SpectrumH3Config
    config = make_config(protected=protected, blend_weight=.35)
    original = SpectrumH3Config(**dataclasses.asdict(config)).validate()
    assert dataclasses.asdict(config) == dataclasses.asdict(original)
    new, old = [cls(degree=1, ridge_lambda=.1, max_history=8, history_storage='vram')
                for cls in (HistoryWeightForecaster, Original)]
    gen = torch.Generator().manual_seed(816)
    for i in range(10):
        feature = torch.randn(1, 40, 128, generator=gen).to(torch.bfloat16)
        for runtime in (new, old):
            runtime.update(float(i), feature, anchor_id=i)
        if i:
            assert_bytes(new.combined_weights(i+.5, .35), old.combined_weights(i+.5, .35))
            assert_bytes(new.predict(i+.5, .35), old.predict(i+.5, .35))
            assert new.persistent_tensor_bytes == old.persistent_tensor_bytes
    new.reset(); old.reset()
    assert new.persistent_tensor_bytes == old.persistent_tensor_bytes == 0


def test_owned_runtime_has_no_vendor_import_or_public_node_registration():
    root = Path(__file__).parents[1] / '_patches'
    import ast
    paths = list((root / 'h3_runtime').rglob('*.py'))
    paths += [root / name for name in ('minimax_h3_unified.py', 'h3_tile_attention_mps.py', 'h3_vsa_mps.py')]
    for path in paths:
        text = path.read_text()
        assert '._vendor' not in text, path
        if 'h3_runtime' in path.parts:
            assert 'NODE_CLASS_MAPPINGS' not in text, path
    provenance = json.loads((root / 'h3_runtime/ORIGINS.json').read_text())
    import hashlib
    for entry in provenance['derived_files'].values():
        reference = root / '_vendor' / entry['package'] / entry['source']
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == entry['source_sha256']
