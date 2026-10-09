import contextlib
import sys
import types
import pytest
import torch
from safetensors.torch import save_file
from _patches import h3_latent_upscaler as up
from _patches.h3_upscale_network_3d import LatentResizer3D


@pytest.fixture
def environment(tmp_path, monkeypatch):
    # Minimal real checkpoint: exercise strict loading and complete execution.
    model = LatentResizer3D(channels=32, in_blocks=1, out_blocks=1, temporal_every=0)
    path = tmp_path / 'h3.safetensors'
    save_file(model.state_dict(), path)
    paths = types.SimpleNamespace(get_full_path_or_raise=lambda *a: str(path))
    monkeypatch.setattr(up, '_paths', lambda: paths)
    monkeypatch.setattr(up, '_available_bytes', lambda: 64 * 2**30)
    mm = types.ModuleType('comfy.model_management')
    mm.free_memory = lambda *a, **k: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    comfy = types.ModuleType('comfy'); comfy.model_management = mm
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    up._load_model_cpu.cache_clear()
    yield path
    up._load_model_cpu.cache_clear()


def test_noop_preserves_object_without_model_loading():
    latent = {'samples': torch.randn(1, 24, 2, 3, 5), 'metadata': object()}
    assert up.AppleSiliconH3LatentUpscale().upscale(latent, 'missing', scale=1)[0] is latent


def test_cpu_full_forward_preserves_frames_metadata_and_resizes_mask(environment):
    latent = {'samples': torch.randn(1, 24, 2, 2, 2),
              'noise_mask': torch.tensor([[[0., 1.], [1., 0.]]]), 'batch_index': [99]}
    result = up.AppleSiliconH3LatentUpscale().upscale(latent, 'h3', device='cpu', precision='fp32')[0]
    assert result['samples'].shape == (1, 24, 2, 4, 4)
    assert torch.isfinite(result['samples']).all()
    assert result['batch_index'] is latent['batch_index']
    assert result['noise_mask'].shape == (1, 4, 4)
    assert set(result['noise_mask'].unique().tolist()) == {0., 1.}
    assert latent['samples'].shape[-2:] == (2, 2)


def test_memory_admission_happens_before_device_copy(environment, monkeypatch):
    monkeypatch.setattr(up, 'working_bytes', lambda *a: 100 * 2**30)
    with pytest.raises(ValueError, match='above'):
        up.AppleSiliconH3LatentUpscale().upscale({'samples': torch.randn(1,24,2,2,2)}, 'h3', device='cpu')


def test_rejects_wrong_family_and_large_or_invalid_targets():
    with pytest.raises(ValueError, match='24'):
        up.AppleSiliconH3LatentUpscale().upscale({'samples': torch.zeros(1,16,2,2,2)}, 'x')
    for scale in (0.5, float('nan'), 5):
        with pytest.raises(ValueError):up.target_size(30,54,'scale',scale,0,0,0)
    assert up.target_size(30,54,'scale',2,0,0,0) == (60,108,2)


def test_picker_header_accepts_h3_and_rejects_ltx(environment, tmp_path):
    st = environment.stat()
    assert up._is_h3_file(str(environment),st.st_size,st.st_mtime_ns)
    other = tmp_path/'ltx.safetensors';save_file({'conv_in.weight':torch.zeros(32,128,3,3,3)},other)
    assert not up._is_h3_file(str(other),other.stat().st_size,other.stat().st_mtime_ns)


def test_partial_model_is_rejected(environment):
    from safetensors.torch import load_file
    weights = load_file(environment);weights.pop('conv_out.bias');save_file(weights,environment)
    with pytest.raises(RuntimeError, match='Missing key'):
        up._load_model_cpu(str(environment),environment.stat().st_size,environment.stat().st_mtime_ns)


def test_2d_checkpoint_architecture_and_frame_count(environment):
    from _patches.h3_upscale_network_2d import VideoLatentResizer
    network=VideoLatentResizer(channels=32,in_blocks=1,out_blocks=1,temporal_every=2)
    save_file(network.state_dict(),environment)
    result=up.AppleSiliconH3LatentUpscale().upscale(
        {'samples':torch.randn(1,24,2,2,2)},'h3',device='cpu',precision='fp32')[0]
    assert result['samples'].shape==(1,24,2,4,4)
    assert torch.isfinite(result['samples']).all()


def test_chunked_3d_preserves_temporal_length(environment):
    network=LatentResizer3D(channels=32,in_blocks=1,out_blocks=1,temporal_every=2)
    save_file(network.state_dict(),environment)
    result=up.AppleSiliconH3LatentUpscale().upscale(
        {'samples':torch.randn(1,24,9,2,2)},'h3',device='cpu',precision='fp32',temporal_chunk=4)[0]
    assert result['samples'].shape==(1,24,9,4,4)
    assert torch.isfinite(result['samples']).all()


def test_corrupt_checkpoint_does_not_break_model_picker(tmp_path):
    p=tmp_path/'partial.safetensors';p.write_bytes(b'incomplete')
    assert not up._is_h3_file(str(p),p.stat().st_size,p.stat().st_mtime_ns)
