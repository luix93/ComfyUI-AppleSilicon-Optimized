from types import SimpleNamespace as NS
import pytest
import torch
from _patches import moss_fast as owner, moss_ffn_mps as f


def test_default_and_explicit_disable_preserve_original(monkeypatch):
    sentinel = object()
    for value in (None, '0'):
        if value is None:
            monkeypatch.delenv('ASFP8_MOSS_FFN_EPILOGUE', raising=False)
        else:
            monkeypatch.setenv('ASFP8_MOSS_FFN_EPILOGUE', value)
        assert owner._block(lambda *a: sentinel, None, None, None, None, None) is sentinel


def test_no_scope_never_runs_block_or_compiles(monkeypatch):
    monkeypatch.setenv('ASFP8_MOSS_FFN_EPILOGUE', '1')
    monkeypatch.setattr(f, '_ready', lambda: pytest.fail('unexpected compile'))
    sentinel = object()
    with torch.inference_mode():
        assert owner._block(lambda *a: sentinel, None, None, None, None, None) is sentinel


def test_compile_failure_latches_without_repeated_attempts(monkeypatch):
    calls = []
    def compile(source):
        calls.append(1)
        raise RuntimeError('test unsupported compiler')
    monkeypatch.setattr(f, '_dead', False)
    monkeypatch.setattr(f, '_library', None)
    monkeypatch.setattr(torch.mps, 'compile_shader', compile)
    assert not f._ready()
    assert not f._ready()
    assert len(calls) == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires MPS')
def test_fused_projection_activation_tail_and_scale_order():
    from _patches import int8_linear_kernel_mps as i8
    if not owner.torch_mpp_qcache._hardware():
        pytest.skip('requires reviewed M5 Ultra')
    with torch.inference_mode():
        assert owner._ready() and f._ready()
        kernel = i8._load_kernel()
        assert kernel is not None
        torch.manual_seed(917)
        q = torch.randint(-127, 128, (129, 3072), device='mps', dtype=torch.int8)
        w = torch.randint(-127, 128, (16384, 3072), device='mps', dtype=torch.int8)
        rs = torch.rand(129, device='mps') * .001
        cs = torch.rand(16384, device='mps') * .005
        wide = kernel.i8_matmul2d_nt_fused_cs(q, w, rs, cs, None)
        reference = owner._situ_tensor(wide.reshape(1, 129, 16384)).reshape(129, 8192)
        result = f.project(q, rs, w, cs, owner._tables)
        assert torch.equal(result.view(torch.uint8), reference.view(torch.uint8))

@pytest.mark.parametrize('attribute', [
    'weight_function', 'bias_function', 'weight_lowvram_function',
    'bias_lowvram_function', '_forward_hooks', '_forward_pre_hooks',
    '_full_precision_mm', 'comfy_force_cast_weights', '_asfp8_runtime_lora',
])
def test_custom_linear_behavior_is_not_bypassed(monkeypatch, attribute):
    from _patches import int8_linear_kernel_mps as i8
    monkeypatch.setattr(i8, '_kernel', object())
    layer = NS(bias=None, **{attribute: [object()]})
    assert not f._eligible_layer(layer, (16384, 3072), torch.device('mps'))


def test_foreign_forward_not_recognized():
    assert not f._known_forward(lambda x: x)
    assert not f._known_forward(None)
