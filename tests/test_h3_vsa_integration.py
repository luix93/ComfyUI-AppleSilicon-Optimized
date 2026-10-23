"""Payload geometry and node composition boundaries for trained VSA."""
import types
import sys

import pytest
import torch

from _patches import minimax_h3_vsa as vsa
from _patches import minimax_h3_presets as presets
from _patches import minimax_h3_unified as unified


def test_reference_video_geometry_is_kept_separate_from_images_and_audio():
    payload = {"refs": [dict(kind="image"), dict(kind="audio"),
        dict(kind="video_audio", latent_t=3, latent_h=6, latent_w=10)]}
    grids = vsa.reference_grids(payload)
    assert grids == (None, (3, 3, 5))
    layout = types.SimpleNamespace(signature=(7, 2, 6, 10, 2), seq_len=138,
        segments=[(0,7,"text"), (7,40,"ref_img"), (40,44,"ref_audio"),
                  (44,89,"ref_img"), (89,108,"audio"), (108,138,"video")])
    assert vsa.layout_segments(layout, grids) == (7,33,4,(3,3,5),19,(2,3,5))
    with pytest.raises(ValueError, match="reference metadata"):
        vsa.layout_segments(layout, ())
    with pytest.raises(ValueError, match="does not match"):
        vsa.layout_segments(layout, (None,(3,3,6)))


def test_wrapper_passes_native_payload_and_scopes_reference_metadata():
    patch = vsa.VSAPatch(tile_size=128, keep_percent=10, reference_keep_percent=10,
        start_sigma=1., end_sigma=0., min_tokens=0, dense_layers=frozenset(),
        head_chunk=4, workspace_mib=256, verbose=False)
    options = {"unrelated": 1}
    payload = {"refs": [dict(kind="video", latent_t=2, latent_h=4, latent_w=6)]}
    called = {}
    def execute(x, t, context, transformer_options, **kwargs):
        called.update(options=transformer_options, kwargs=kwargs)
        return x
    assert patch.model_wrapper(execute, "x", "t", "context", options, payload) == "x"
    assert called["options"][vsa.STATE_KEY+"_refs"] == ((2,2,3),)
    assert called["kwargs"]["minimax_payload"] is payload
    assert options == {"unrelated": 1}
    patch.tile_size = 64
    patch.model_wrapper(execute, "x", "t", "context", options, payload)
    assert called["options"][vsa.STATE_KEY+"_refs"] == (None,)


@pytest.mark.parametrize("label,tile", [("VSA Metal · FastH3 Omni (128)",128), ("VSA Metal · ComfyUI (64)",128)])
def test_vsa_presets_preserve_the_exported_sparse_window(label, tile):
    # The removed 64-token recipe now runs the Omni (128) recipe.
    cfg = presets.resolve_presets("Off", label, "Off")
    assert cfg["sparse_backend"] == "vsa_metal"
    assert cfg["vsa_tile_size"] == tile
    assert cfg["vsa_keep_percent"] == 10
    assert cfg["sparse_start_percent"] == 0 and cfg["sparse_end_percent"] == 1
    assert cfg["sparse_min_sequence_tokens"] == 12288
    assert cfg["sparse_dense_layers"] == "" and cfg["sparse_dense_leading_blocks"] == 0
    assert not cfg["token_reduction"] and cfg["cache_mode"] == "disabled"


@pytest.mark.parametrize("cache,tokens", [("MotionCache · balanced", "Off"), ("Off", "Light · experimental")])
def test_vsa_rejects_unvalidated_cache_and_token_combinations_before_patching(cache, tokens):
    cfg = presets.resolve_presets(cache, "VSA Metal · FastH3 Omni (128)", tokens)
    with pytest.raises(ValueError, match="Temporal cache and Token reduction"):
        unified.apply_h3_acceleration(object(), **cfg)


def test_cleanup_releases_plans_and_retains_only_scalar_summary():
    patch = vsa.VSAPatch(tile_size=128, keep_percent=10, reference_keep_percent=10,
        start_sigma=1., end_sigma=0., min_tokens=0, dense_layers=frozenset(),
        head_chunk=4, workspace_mib=256, verbose=False)
    patch.plans["temporary"] = object()
    patch.stats["vsa_calls"] = 400
    patch.cleanup()
    assert not patch.plans and not patch.stats
    assert patch.last_stats == {"vsa_calls":400}


def test_attention_receives_normalized_qk_original_values_and_raw_checkpoint_gate(monkeypatch):
    comfy = types.ModuleType('comfy')
    comfy.model_management = types.SimpleNamespace()
    comfy.quant_ops = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    patch = vsa.VSAPatch(tile_size=128, keep_percent=10, reference_keep_percent=10,
        start_sigma=1., end_sigma=0., min_tokens=0, dense_layers=frozenset(),
        head_chunk=4, workspace_mib=256, verbose=False)
    layout = types.SimpleNamespace(signature=(2,1,2,6,0), seq_len=5,
        segments=[(0,2,'text'),(2,5,'video')])
    x = torch.randn(5,384, generator=torch.Generator().manual_seed(23))
    attn = types.SimpleNamespace(heads=3, head_dim=128,
        qkv_proj=lambda value: torch.cat((value,2*value,3*value),-1),
        q_norm=lambda value:value+1, k_norm=lambda value:value+2,
        to_gate_compress=lambda value:4*value, out_proj=lambda value:2*value)
    seen = {}
    def operator(q,k,value,gate,plan,**kwargs):
        seen.update(q=q,k=k,v=value,gate=gate)
        return value
    monkeypatch.setattr(vsa.h3_vsa_mps,'attention',operator)
    out = patch.attention(attn,0,x,None,{'minimax_h3_layout':layout})
    assert torch.equal(seen['q'],(x+1).view(5,3,128))
    assert torch.equal(seen['k'],(2*x+2).view(5,3,128))
    assert torch.equal(seen['v'],(3*x).view(5,3,128))
    assert torch.equal(seen['gate'],(4*x).view(5,3,128))
    assert torch.equal(out,6*x)
