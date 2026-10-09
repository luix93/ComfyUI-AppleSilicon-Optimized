"""Parity against pinned sources and lifecycle regressions in owned runtimes."""
import dataclasses
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from _patches.h3_runtime.veda.core import tiling, predictor, selection, bundle
from _patches._vendor.veda.core import tiling as original_tiling
from _patches._vendor.veda.core import predictor as original_predictor
from _patches._vendor.veda.core import selection as original_selection
from _patches._vendor.veda.core import bundle as original_bundle


def assert_bytes(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('references', [False, True])
def test_veda_scores_selection_and_scatter_match_pinned_runtime(device, references):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    generator = torch.Generator().manual_seed(3817)
    layouts = []
    for mod in (tiling, original_tiling):
        spans = [mod.TiledSpan(35, (5, 5, 5), mod.TileShape(4, 4, 8))]
        if references:
            spans.insert(0, mod.TiledSpan(0, (1, 3, 3), mod.TileShape(1, 8, 16)))
        layouts.append(mod.build_tile_layout(spans, 180, device))
    x = torch.randn(3, 180, 3, 128, generator=generator).to(device, torch.bfloat16)
    hs = torch.tensor([2, 0], device=device)
    qkv = [[mod.gather_tiles(t, layout, hs) for t in x]
           for mod, layout in zip((tiling, original_tiling), layouts)]
    for new, old in zip(*qkv):
        assert_bytes(new, old)
    projections = torch.randn(2, 2, 384, 128, generator=generator).to(device)
    logits = []
    selected = []
    for pool, sel, layout, values in zip((predictor, original_predictor),
            (selection, original_selection), layouts, qkv):
        q, k, v = values
        logits.append(pool.tile_logits(pool.pool_video_tiles(q, layout),
            pool.pool_video_tiles(k, layout), *projections))
        budgets = sel.column_blocks(layout, sel.Budget.sparsity(80), sel.Budget.sparsity(50))
        selected.append(sel.select(logits[-1], layout, budgets))
    assert_bytes(*logits)
    for new, old in zip(*selected):
        assert_bytes(new, old)
    masks = [mod.block_mask(*chosen, layout) for mod, chosen, layout in
             zip((selection, original_selection), selected, layouts)]
    assert_bytes(*masks)
    outputs = []
    for mod, layout, values in zip((tiling, original_tiling), layouts, qkv):
        out = torch.zeros(181, 3, 128, dtype=x.dtype, device=device)
        mod.scatter_tiles_(out, values[2], layout, hs)
        outputs.append(out)
    assert_bytes(*outputs)


def test_veda_fp8_checkpoint_dequantization_is_unchanged(tmp_path):
    from safetensors.torch import save_file
    # One layer/head with a valid exported geometry plan.
    plan = {'geometry': 'test', 'grid': [1, 2, 2], 'shapes': ['4x4x8'], 'head_shape': [[0]]}
    metadata = {'format': bundle.FORMAT, 'num_layers': '1', 'num_heads': '1',
                'head_dim': '4', 'keep_ratio': '0.1', 'dtype': 'float8_e4m3fn',
                'plans': json.dumps({'test': plan})}
    gen = torch.Generator().manual_seed(118)
    tensors = {}
    for name in ('proj_q', 'proj_k'):
        key = 'layers.0.' + name
        tensors[key] = torch.randn(1, 12, 4, generator=gen).to(torch.float8_e4m3fn)
        tensors[key + '.__scale'] = torch.tensor([0.017])
    path = tmp_path / 'predictor.safetensors'
    save_file(tensors, str(path), metadata=metadata)
    new, old = bundle.load_bundle(str(path)), original_bundle.load_bundle(str(path))
    assert new.metadata == old.metadata
    for a, b in zip(new.proj_q + new.proj_k, old.proj_q + old.proj_k):
        assert_bytes(a, b)


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
    paths += [root / name for name in ('minimax_h3_unified.py', 'minimax_h3_veda.py', 'h3_veda_mps.py')]
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


def test_veda_projection_cache_reuses_weights_and_clears_at_run_end():
    from _patches.h3_runtime.veda.runtime import AppleVedaEngine
    gen = torch.Generator().manual_seed(443)
    weights = [torch.randn(2, 384, 128, generator=gen).to(torch.bfloat16) for _ in range(2)]
    model = types.SimpleNamespace(proj_q=weights, proj_k=weights)
    runtime = AppleVedaEngine(model, selection.Budget.sparsity(90), selection.Budget.sparsity(0),
        None, torch.device('cpu'), route='veda_metal', memory_budget_gib=1)
    first = runtime._weights(0)
    assert runtime._weights(0) is first
    for value in first:
        assert_bytes(value, weights[0].float())
    assert runtime.projection_bytes == 2 * weights[0].numel() * 4
    runtime._weights(1)
    assert runtime.projection_bytes == 4 * weights[0].numel() * 4
    runtime.reset()
    assert runtime.projections == {} and runtime.projection_bytes == 0
    assert runtime._tile_layouts == {}


def test_bounded_mlx_restores_the_original_cache_limit():
    from _patches.h3_runtime.veda.runtime import BoundedMlxBackend
    class Mx:
        limit = 1234
        clears = 0
        def set_cache_limit(self, value):
            old, self.limit = self.limit, value
            return old
        def clear_cache(self): self.clears += 1
    mx = Mx()
    backend = BoundedMlxBackend(mx)
    assert backend.chunk_bytes == 128 * 2**20
    assert mx.limit == 256 * 2**20
    backend.close()
    assert mx.limit == 1234 and mx.clears == 2


@pytest.mark.parametrize('interrupt', [False, True])
def test_veda_runtime_dense_fallback_and_interrupt_propagation(interrupt):
    pytest.importorskip('comfy.patcher_extension')
    from _patches.h3_runtime.veda import comfy_patch, settings
    config = settings.VedaSettings(generated=selection.Budget.sparsity(90),
                                   reference=selection.Budget.sparsity(0))
    model = types.SimpleNamespace(num_layers=1, num_heads=2, head_dim=128)
    class InterruptProcessingException(Exception): pass
    calls = []
    class Broken(comfy_patch.VedaPatch):
        def _sparse(self, *args):
            calls.append('sparse')
            if interrupt: raise InterruptProcessingException()
            raise RuntimeError('workspace unavailable')
    patch = Broken(model, config, None)
    def dense(func, q, k, v, heads, **kwargs):
        calls.append('dense'); return torch.ones_like(q)
    override = patch.make_override(dense)
    q = torch.zeros(1, 2, 256, 128)
    options = {'block_index': 0, 'minimax_h3_layout': types.SimpleNamespace(seq_len=256)}
    def invoke():
        return override(None, q, q, q, 2, skip_reshape=True, transformer_options=options)
    if interrupt:
        with pytest.raises(InterruptProcessingException): invoke()
        assert patch.run.failed is None and calls == ['sparse']
    else:
        assert_bytes(invoke(), torch.ones_like(q))
        assert_bytes(invoke(), torch.ones_like(q))
        assert calls == ['sparse', 'dense', 'dense']
        assert 'workspace unavailable' in patch.run.failed
