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
