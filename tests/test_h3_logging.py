"""Quiet logs retain truthful work accounting and actionable fallbacks."""
import logging
from types import SimpleNamespace, ModuleType
import contextlib
import importlib
import sys

import pytest
import torch

from _patches.h3_logging import call_summary, block_summary, spectrum_summary
from _patches.h3_runtime.spectrum.runtime import RuntimeStats
from _patches import minimax_h3_sparse as sol
from _patches import minimax_h3_structure as tokens
from _patches import minimax_h3_presets as presets


def test_call_and_block_estimates_keep_first_block_cost():
    assert '1.25x DiT' in call_summary('MotionCache', 4, 20)
    # Four reused tails still execute four first blocks: 20*50 / (20*50 - 4*49).
    assert '196/1000' in block_summary('FirstBlockCache', 4 * 49, 20 * 50)
    assert '1.24x block-compute' in block_summary('FirstBlockCache', 196, 1000)
    assert 'unavailable' in call_summary('Cache', 0, 0)


def test_spectrum_replay_and_discarded_work_do_not_inflate_gain():
    stats = RuntimeStats(actual_transformer_calls=8, forecast_model_calls=12,
                         offline_replay_model_calls=20)
    assert 'saved 12/20' in spectrum_summary(stats)
    assert '2.50x' in spectrum_summary(stats)
    stats.actual_transformer_calls += 3
    stats.discarded_actual_calls = 2
    stats.replayed_transformer_calls = 1
    stats.forecast_model_calls += 4
    stats.speculative_forecast_calls = 4
    assert 'saved 9/20' in spectrum_summary(stats)
    assert '1.82x' in spectrum_summary(stats)


def test_verbose_is_optional_and_preserved():
    off = presets.resolve_presets('MotionCache · balanced', 'Off', 'Off')
    on = presets.resolve_presets('MotionCache · balanced', 'Off', 'Off', True)
    for key in ('cache_verbose', 'sparse_verbose', 'token_verbose'):
        assert off[key] is False and on[key] is True


def test_quiet_sparse_still_warns_once_for_kernel_failure(caplog):
    holder = sol.H3SparseHolder(verbose=False)
    with caplog.at_level(logging.INFO):
        holder.note('protected_step')
        holder.note('kernel_unavailable (memory budget)')
        holder.note('kernel_unavailable (memory budget)')
        holder.note('latched_off')
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert 'memory budget' in caplog.text



@pytest.mark.parametrize('verbose', [False, True])
def test_token_sampler_quiet_summary_and_option_restoration(monkeypatch, caplog, verbose):
    import comfy
    patcher = ModuleType('comfy.model_patcher')
    patcher.create_model_options_clone = lambda opts: {'transformer_options': dict(opts['transformer_options'])}
    monkeypatch.setitem(sys.modules, 'comfy.model_patcher', patcher)
    monkeypatch.setattr(comfy, 'model_patcher', patcher, raising=False)
    state = tokens.H3StructureHolder(1, 10, 4, 12, 20, verbose)
    original = {'transformer_options': {tokens.STATE_KEY: state}}
    guider = SimpleNamespace(model_options=original)

    class Executor:
        class_obj = guider
        def __call__(self):
            active = guider.model_options['transformer_options'][tokens.STATE_KEY]
            active.evaluations = active.reduced_evaluations = 20
            active.last_row_counts = (1000, 550)
            return torch.tensor([7])

    with caplog.at_level(logging.INFO):
        output = tokens.structure_sample_wrapper(Executor())
    assert output.item() == 7 and guider.model_options is original
    assert 'reduced 20/20' in caplog.text
    assert ('enabled scale=' in caplog.text) is verbose
    assert ('bypasses=' in caplog.text) is verbose
    assert len(caplog.records) == (3 if verbose else 1)
