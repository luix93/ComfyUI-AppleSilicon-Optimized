"""Compact recipes remain independent and share the advanced implementation."""
import pytest
from _patches import minimax_h3_presets as presets
from _patches import minimax_h3_unified as unified


def test_small_schema_and_default():
    schema = presets.AppleSiliconH3QuickAcceleration.INPUT_TYPES()
    assert list(schema['required']) == ['model', 'cache', 'sparse', 'tokens']
    assert list(schema['optional']) == ['log_activity', 'approximate_references']
    assert not schema['optional']['log_activity'][1]['default']
    settings = presets.resolve_presets('MotionCache · balanced', presets.OFF, presets.OFF)
    assert settings['cache_mode'] == 'motion_cache'
    assert settings['motion_reuse_threshold'] == .15
    assert not settings['sparse_attention'] and not settings['token_reduction']


def test_off_returns_original_model(monkeypatch):
    monkeypatch.setattr(unified, 'apply_motion_cache_preset', lambda *a, **k: pytest.fail('cache enabled'))
    model = object()
    assert presets.AppleSiliconH3QuickAcceleration().apply(model, 'Off', 'Off', 'Off') == (model,)


@pytest.mark.parametrize('cache', presets.CACHE_PRESETS)
@pytest.mark.parametrize('sparse', presets.SPARSE_PRESETS)
@pytest.mark.parametrize('tokens', presets.TOKEN_PRESETS)
def test_every_combination_resolves_and_does_not_mutate_recipes(cache, sparse, tokens):
    cfg = presets.resolve_presets(cache, sparse, tokens, False)
    assert cfg['cache_mode'] in unified.CACHE_MODES
    assert cfg['sparse_attention'] == (sparse != 'Off')
    assert cfg['token_reduction'] == (tokens != 'Off')
    assert not cfg['cache_verbose'] and not cfg['sparse_verbose'] and not cfg['token_verbose']
    cfg['sparse_dense_layers'] = 'changed'
    assert presets.resolve_presets(cache, sparse, tokens)['sparse_dense_layers'] != 'changed'


def test_recipe_calls_existing_runtime(monkeypatch):
    captured = {}
    def apply(model, **settings):
        captured.update(settings)
        return 'patched'
    monkeypatch.setattr(unified, 'apply_h3_acceleration', apply)
    result = presets.AppleSiliconH3QuickAcceleration().apply('model',
        'FirstBlockCache · protected', 'SOL Metal · balanced', 'Light · experimental')
    assert result == ('patched',)
    assert captured['cache_reuse_threshold'] == .10
    assert captured['sparse_tau'] == .5
    assert captured['token_detail_restore_block'] == 12
    assert captured['token_coarse_restore_block'] == 20


@pytest.mark.parametrize('field', ['cache', 'sparse', 'tokens'])
def test_unknown_recipe_fails_explicitly(field):
    values = dict(cache='Off', sparse='Off', tokens='Off')
    values[field] = 'nonexistent'
    with pytest.raises(ValueError, match='Unknown H3'):
        presets.resolve_presets(**values)


def test_legacy_fast_is_protected_without_a_duplicate_menu_entry():
    old = presets.resolve_presets('Off','Off',presets.LEGACY_FAST)
    protected = presets.resolve_presets('Off','Off',presets.FAST_PROTECTED)
    assert old == protected
    assert old['token_coarse_restore_block'] == 33
    assert old['token_phase_policy'] == 'evaluation_count'
    assert presets.LEGACY_FAST not in presets.TOKEN_PRESETS
    assert presets.AppleSiliconH3QuickAcceleration.VALIDATE_INPUTS(presets.LEGACY_FAST) is True
    assert isinstance(presets.AppleSiliconH3QuickAcceleration.VALIDATE_INPUTS('invalid'),str)


def test_linked_preset_is_validated_when_its_value_is_available():
    assert presets.AppleSiliconH3QuickAcceleration.VALIDATE_INPUTS(None) is True
    with pytest.raises(ValueError, match='Unknown H3 tokens'):
        presets.resolve_presets('Off', 'Off', 'invalid linked value')


def test_aggressive_is_opt_in_and_bounds_the_deeper_phase():
    cfg = presets.resolve_presets('Off','Off',presets.FAST_AGGRESSIVE)
    assert cfg['token_coarse_restore_block'] == 40
    assert cfg['token_detail_restore_block'] == 30
    assert cfg['token_phase_policy'] == 'capped_steps'
    assert cfg['token_early_evaluations'] == 10
    assert cfg['token_early_step_fraction'] == .5


@pytest.mark.parametrize("old", [k for k in presets.SPARSE_ALIASES if k.startswith(("SOL INT8 · ", "Dense INT8 v3 · "))
                                 and k != "SOL INT8 · hybrid balanced"])
def test_renamed_metal_presets_resolve_to_the_same_recipe(old):
    new = presets.SPARSE_ALIASES[old]
    assert new.startswith(("SOL Metal · ", "Dense Metal · ")) and new in presets.SPARSE_PRESETS
    assert presets.resolve_presets("Off", old, "Off") == presets.resolve_presets("Off", new, "Off")
    assert presets.AppleSiliconH3QuickAcceleration.VALIDATE_INPUTS(tokens="Off", sparse=old) is True
