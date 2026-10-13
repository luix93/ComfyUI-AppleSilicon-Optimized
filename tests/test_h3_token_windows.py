"""Protected token recipes select sampler steps, keeping repeated CFG calls together."""
import pytest
import torch
from _patches import minimax_h3_structure as token
from _patches.h3_schedule import step_window


@pytest.mark.parametrize("steps,expected", [(20,(4,18,20)),(8,(2,7,8)),(4,(1,3,4))])
def test_protection_rounds_outward(steps, expected):
    sigmas = torch.linspace(1, 0, steps+1)
    holder = token.H3StructureHolder(1,10,4,8,12,start_percent=.2,end_percent=.9)
    holder.prepare_timesteps(sigmas)
    assert holder.step_range == expected
    for i, sigma in enumerate(sigmas[:-1]):
        for cfg_call in range(2):
            holder.begin_evaluation(None,None,{"sigmas":sigma.reshape(1)})
            assert holder.eligible_evaluation == (expected[0] <= i < expected[1])


def test_existing_all_step_path_needs_no_schedule():
    holder = token.H3StructureHolder(1,10,4,12,20).prepare_timesteps(None)
    holder.begin_evaluation(None,None,{})
    assert holder.eligible_evaluation and holder.step_range is None


def test_clone_keeps_configuration_and_resets_run_state():
    holder = token.H3StructureHolder(1,10,4,8,12,start_percent=.2,end_percent=.9)
    holder.prepare_timesteps(torch.linspace(1,0,21));holder.evaluations=20
    clone = holder.clone()
    assert clone.start_percent == .2 and clone.end_percent == .9
    assert clone.step_range is None and clone.evaluations == 0


@pytest.mark.parametrize("sigmas", [None,torch.tensor([1.,1.,0.]),torch.tensor([1.,float('nan'),0.])])
def test_invalid_schedule_rejected(sigmas):
    with pytest.raises(ValueError,match="sigma"):
        step_window(sigmas,.2,.9)


def test_empty_short_window_never_reduces():
    holder = token.H3StructureHolder(1,10,4,8,12,start_percent=.4,end_percent=.6)
    holder.prepare_timesteps(torch.tensor([1.,0.]))
    holder.begin_evaluation(None,None,{"sigmas":torch.tensor([1.])})
    assert not holder.eligible_evaluation


@pytest.mark.parametrize('steps,coarse_steps', [(1,0),(4,2),(6,3),(8,4),(10,5),(20,10),(40,10)])
def test_capped_phase_tracks_sampler_steps_not_cfg_calls(steps, coarse_steps):
    sigmas = torch.linspace(1, 0, steps+1).square()
    holder = token.H3StructureHolder(1,10,4,30,40,phase_policy='capped_steps')
    holder.prepare_timesteps(sigmas)
    assert holder.coarse_step_range == (0,coarse_steps,steps)
    for index, sigma in enumerate(sigmas[:-1]):
        for _ in range(2):
            holder.begin_evaluation(None,None,{'sigmas':sigma.reshape(1)})
            assert holder.coarse_evaluation == (index < coarse_steps)


def test_cached_steps_do_not_extend_aggressive_phase():
    sigmas = torch.linspace(1,0,21)
    holder = token.H3StructureHolder(1,10,4,30,40,phase_policy='capped_steps')
    holder.prepare_timesteps(sigmas)
    for index in (0,3,8,12,18):
        holder.begin_evaluation(None,None,{'sigmas':sigmas[index:index+1]})
        assert holder.coarse_evaluation == (index < 10)
    assert holder.evaluations == 5
    assert not holder.coarse_evaluation


def test_capped_phase_drives_actual_restore_boundary_and_clone():
    holder = token.H3StructureHolder(1,10,4,30,40,phase_policy='capped_steps')
    holder.prepare_timesteps(torch.linspace(1,0,11))
    for sigma, restore in ((1.,40),(.2,30)):
        holder.begin_evaluation(None,None,{'sigmas':torch.tensor([sigma])})
        holder.current_geometry = (1,1,4,1)
        args = {'img':torch.zeros(6,2), 'rope_freqs':torch.eye(2).reshape(1,1,1,1,2,2).repeat(1,6,1,1,1,1),
                'mod_segments':[(0,2,0),(2,6,1)], 'transformer_options':{}}
        holder.enter_reduction(args)
        assert holder.reduction.restore_block == restore
        holder.end_evaluation()
    clone = holder.clone()
    assert clone.phase_policy == 'capped_steps' and clone.early_step_fraction == .5
    assert clone.coarse_step_range is None and not clone.coarse_evaluation


def test_missing_current_sigma_never_enables_aggressive_phase():
    holder = token.H3StructureHolder(1,10,4,30,40,phase_policy='capped_steps')
    holder.prepare_timesteps(torch.linspace(1,0,11))
    holder.begin_evaluation(None,None,{})
    assert not holder.coarse_evaluation
