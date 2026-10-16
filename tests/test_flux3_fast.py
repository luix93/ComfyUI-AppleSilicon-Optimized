import inspect
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from _patches import flux3_fast as patch


@pytest.fixture
def fake_cast_module(monkeypatch):
    comfy, ops = ModuleType('comfy'), ModuleType('comfy.ops')
    comfy.ops = ops
    ops.cast_bias_weight = lambda value: value
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.ops', ops)
    monkeypatch.setattr(patch, '_cast_users', 0)
    monkeypatch.setattr(patch, '_active_cast', None)
    return ops


def test_cast_hook_lives_only_during_active_phases(fake_cast_module):
    ops = fake_cast_module
    original = ops.cast_bias_weight
    release_a = patch._acquire_cast()
    wrapped = ops.cast_bias_weight
    release_b = patch._acquire_cast()
    assert wrapped is not original and ops.cast_bias_weight is wrapped
    assert wrapped('unrelated thread') == 'unrelated thread'
    release_a()
    assert ops.cast_bias_weight is wrapped
    release_b()
    assert ops.cast_bias_weight is original and patch._cast_users == 0


def test_cast_cleanup_preserves_foreign_replacement(fake_cast_module):
    ops = fake_cast_module
    release = patch._acquire_cast()
    wrapped = ops.cast_bias_weight
    foreign = lambda value: wrapped(value)
    ops.cast_bias_weight = foreign
    release()
    assert ops.cast_bias_weight is foreign and foreign('safe') == 'safe'


def test_scope_finish_restores_cast_and_releases_weights(fake_cast_module):
    ops = fake_cast_module
    original = ops.cast_bias_weight
    state = dict(cache={'weight': torch.ones(1)}, hits=0, misses=1, bytes=4,
                 budget=4096, part='WEIGHT_CACHE')
    token = patch._scope.set(state)
    state['release_cast'] = patch._acquire_cast()
    with pytest.raises(RuntimeError, match='cancelled'):
        try:
            raise RuntimeError('cancelled')
        finally:
            patch._finish(state, token, {})
    assert ops.cast_bias_weight is original and not state['cache']
    assert patch._scope.get() is None


def test_cast_dispatch_bypasses_cache_outside_flux_scope(monkeypatch):
    original = lambda value, *, flag=False: (value, flag)
    calls = []
    def cache(*args, **kwargs):
        calls.append((args, kwargs))
        return 'scoped'
    monkeypatch.setattr(patch, '_cast_cached', cache)
    wrapped = patch._wrap_cast(original)
    assert wrapped('unchanged', flag=True) == ('unchanged', True)
    assert not calls
    token = patch._scope.set({'part': 'WEIGHT_CACHE'})
    try:
        assert wrapped('input') == 'scoped'
        assert len(calls) == 1
    finally:
        patch._scope.reset(token)


def setup_cache(monkeypatch, budget=4096):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 2**30)
    module = SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(2, 3)), bias=None,
                             weight_function=[], bias_function=[])
    state = dict(modules={id(module)}, cache={}, budget=budget, hits=0, misses=0,
                 bytes=0, part='WEIGHT_CACHE')
    calls = []
    def cast(s, input=None, dtype=None, device=None, bias_dtype=None,
             offloadable=False, compute_dtype=None, want_requant=False):
        calls.append(s)
        result = (s.weight.detach().clone(), None)
        return (*result, 'original lease') if offloadable else result
    def invoke(**kw):
        return patch._cast_cached(cast, inspect.signature(cast), module,
                                  device=torch.device('mps'), dtype=torch.float32, **kw)
    return module, state, calls, invoke


def test_owned_copy_and_parameter_revision(monkeypatch):
    module, state, calls, invoke = setup_cache(monkeypatch)
    token = patch._scope.set(state)
    try:
        first = invoke(offloadable=True)
        hit = invoke(offloadable=True)
        assert first[2] == 'original lease' and hit[2] is None
        first[0].fill_(9)
        assert torch.equal(hit[0], torch.zeros(2, 3))
        with torch.no_grad():
            module.weight.add_(1)
        assert torch.equal(invoke()[0], torch.ones(2, 3))
        assert len(calls) == 2
    finally:
        patch._scope.reset(token)
    invoke()
    assert len(calls) == 3


def test_quantized_storage_and_scale_revisions_invalidate(monkeypatch):
    module, state, calls, invoke = setup_cache(monkeypatch)
    module.weight._qdata = torch.zeros(2, 3, dtype=torch.int8)
    module.weight._params = SimpleNamespace(scale=torch.ones(2), convrot=True,
                                            convrot_groupsize=256, transposed=False)
    token = patch._scope.set(state)
    try:
        invoke(); invoke()
        module.weight._qdata.add_(1)
        invoke()
        module.weight._params.scale.mul_(2)
        invoke()
        module.weight._params.convrot = False
        invoke()
        assert len(calls) == 4
    finally:
        patch._scope.reset(token)


@pytest.mark.parametrize('reason', ['budget', 'functions', 'lowvram', 'vbar', 'disabled', 'training'])
def test_cache_fallbacks_keep_original_lease(monkeypatch, reason):
    module, state, calls, invoke = setup_cache(monkeypatch, 0 if reason == 'budget' else 4096)
    if reason == 'functions':module.weight_function = [lambda x: x]
    if reason == 'lowvram':module.weight_lowvram_function = object()
    if reason == 'vbar':module._v = object()
    if reason == 'disabled':monkeypatch.setenv('ASFP8_FLUX3_FAST', '0')
    if reason == 'training':monkeypatch.setattr(torch, 'is_grad_enabled', lambda: True)
    token = patch._scope.set(state)
    try:
        for _ in range(2):assert invoke(offloadable=True)[2] == 'original lease'
        assert len(calls) == 2 and not state['cache']
    finally:
        patch._scope.reset(token)


def test_sampling_scope_clears_on_interrupt(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(patch, '_budget', lambda prompt=False: 4096)
    class Flux:
        def modules(self):return []
    guider = SimpleNamespace(inner_model=SimpleNamespace(diffusion_model=Flux()),
                             model_patcher=SimpleNamespace(load_device='mps'),
                             model_options={}, conds={})
    captured = []
    def interrupted(*a, **kw):
        state = patch._scope.get(); captured.append(state)
        state['cache']['test'] = torch.ones(1)
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        patch._sample(interrupted, Flux, object, guider)
    assert patch._scope.get() is None and not captured[0]['cache']


def test_conditioning_hooks_disable_sampling_cache(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    class Flux:pass
    guider = SimpleNamespace(inner_model=SimpleNamespace(diffusion_model=Flux()),
                             model_patcher=SimpleNamespace(load_device='mps'),
                             model_options={}, conds={'positive':[{'hooks':object()}]})
    def original(*a, **kw):return patch._scope.get()
    assert patch._sample(original, Flux, object, guider) is None


def test_budget_limits_and_invalid_values(monkeypatch):
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda:100*2**30)
    monkeypatch.setattr(patch.psutil, 'virtual_memory', lambda:SimpleNamespace(available=20*2**30))
    assert patch._budget(prompt=True) == 4*2**30
    for value in ('0', '-1', 'nan', 'inf', 'invalid'):
        monkeypatch.setenv('ASFP8_FLUX3_CACHE_GIB', value)
        assert patch._budget() == 0


def test_inference_parameter_replacement_changes_key():
    with torch.inference_mode():
        a = torch.ones(3); b = a.clone()
    assert patch._weight_key(a) != patch._weight_key(b)


def test_cpu_rope_and_activation_fall_back():
    x = torch.randn(1, 2, 24, 128, dtype=torch.bfloat16)
    pe = torch.zeros(1, 2, 1, 64, 2, 2)
    assert not patch._rope_supported(x, x, pe)
    original = lambda value: value
    assert patch._activation(original, x) is x


def test_weight_cache_switch_keeps_activation_scope(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda:False)
    monkeypatch.setenv('ASFP8_FLUX3_WEIGHT_CACHE', '0')
    monkeypatch.setattr(patch, '_budget', lambda prompt=False:0)
    class Flux:
        def modules(self):return []
    guider = SimpleNamespace(inner_model=SimpleNamespace(diffusion_model=Flux()),
                             model_patcher=SimpleNamespace(load_device='mps'),
                             model_options={}, conds={})
    def original(*a, **kw):
        assert patch._scope.get()['part'] == 'WEIGHT_CACHE'
        assert not patch._enabled('WEIGHT_CACHE') and patch._enabled('SWIGLU')
        return 'unchanged result'
    assert patch._sample(original, Flux, object, guider) == 'unchanged result'
    assert patch._scope.get() is None


def test_prompt_scope_clears_on_failure(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda:False)
    monkeypatch.setattr(patch, '_budget', lambda prompt=False:4096)
    class TestMpsTensor(torch.Tensor):
        @property
        def is_mps(self):return True
    embeds = torch.ones(1).as_subclass(TestMpsTensor)
    model = SimpleNamespace(model_type='qwen35_9b', model=SimpleNamespace(modules=lambda:[]))
    states = []
    def interrupted(*a, **kw):
        state = patch._scope.get();states.append(state)
        state['cache']['test'] = torch.ones(1)
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        patch._generate(interrupted, model, embeds)
    assert patch._scope.get() is None and not states[0]['cache']
