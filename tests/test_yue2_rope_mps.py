from types import SimpleNamespace

import pytest
import torch

from _patches import yue2_rope_mps as patch
from _patches import yue2_fast as fast


def config():
    return SimpleNamespace(hidden_size=2048, num_attention_heads=16, num_key_value_heads=8)


def test_runtime_guard_and_unknown_config(monkeypatch):
    from _patches import torch_mpp_qcache as qc
    monkeypatch.setattr(qc, '_hardware', lambda: True)
    monkeypatch.setattr(patch, '_sources_supported', lambda: True)
    monkeypatch.setattr(patch, '_dead', False)
    monkeypatch.setattr(torch, '__version__', '2.14.0')
    monkeypatch.setattr(torch.version, 'git_version', qc._TORCH_GIT)
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')
    assert patch.prepare(config()) is not None
    assert patch.prepare(SimpleNamespace()) is None
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '1')
    assert patch.prepare(config()) is None
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH', '0')
    monkeypatch.setattr(patch, '_sources_supported', lambda: False)
    assert patch.prepare(config()) is None
    monkeypatch.setattr(patch, '_sources_supported', lambda: True)
    monkeypatch.setattr(torch.version, 'git_version', 'unvalidated')
    assert patch.prepare(config()) is None


def test_cpu_inputs_and_outside_scope_fall_back():
    q = torch.zeros(1, 16, 1, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16)
    assert patch.maybe_rope(q, k, None, None) is None
    assert patch.maybe_rope(q, k, None, {'rope_state': {}}) is None


def test_independent_switch_and_rope_only_scope(monkeypatch):
    monkeypatch.setattr(torch, 'is_grad_enabled', lambda: False)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(fast, '_budget', lambda: 0)
    monkeypatch.setattr(fast, '_head_supported', lambda *a: False)
    monkeypatch.setattr(patch, 'prepare', lambda c: dict(calls=0, matrices=0))
    monkeypatch.setenv('ASFP8_YUE2_WEIGHT_CACHE', '0')
    monkeypatch.setenv('ASFP8_YUE2_VOCAB_HEAD', '0')
    class Head:
        pass
    head = Head()
    model = SimpleNamespace(lm_head=head, modules=lambda: [head])
    encoder = SimpleNamespace(model=model, execution_device=torch.device('mps'),
                              config=SimpleNamespace(vocab_size=184704))
    scopes = []
    def generate(*a, **kw):
        state = fast._scope.get()
        scopes.append(None if state is None else state['rope_state'])
        return [1], False
    monkeypatch.setenv('ASFP8_YUE2_ROPE', '1')
    fast._generate(generate, encoder, [], 0, 128, 'abc', torch.bfloat16)
    assert scopes[-1] is not None and fast._scope.get() is None
    monkeypatch.setenv('ASFP8_YUE2_ROPE', '0')
    fast._generate(generate, encoder, [], 0, 128, 'abc', torch.bfloat16)
    assert scopes[-1] is None


def gpu_runtime():
    if not torch.backends.mps.is_available():
        pytest.skip('requires MPS')
    if patch.prepare(config()) is None:
        pytest.skip('requires validated M5 Ultra / Torch / Comfy / Kitchen sources')


def bytes_equal(a, b):
    return torch.equal(a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8))


@pytest.mark.parametrize('seed', range(8))
def test_rope_bytes_storage_offsets_and_positions(seed):
    gpu_runtime()
    from comfy.text_encoders.llama import rope_matrix, precompute_freqs_cis
    from comfy_kitchen.backends.eager.rope import apply_rope_split_half
    with torch.inference_mode():
        torch.manual_seed(seed)
        # Mimic the real merged QKV projection, including K's nonzero offset.
        qkv = torch.randn(1, 1, 4096, device='mps', dtype=torch.bfloat16)
        q, k, v = qkv.split((2048, 1024, 1024), dim=-1)
        q = q.reshape(1, 1, 16, 128).transpose(1, 2)
        k = k.reshape(1, 1, 8, 128).transpose(1, 2)
        assert k.storage_offset() == 2048
        for position in (0, 13, 254, 2000, 10000, 24575):
            freqs = precompute_freqs_cis(128, torch.tensor([[position]], device='mps'),
                                        1000000., device='mps')
            matrix = rope_matrix(freqs)
            expected = apply_rope_split_half(q, k, matrix)
            state = {'rope_state': patch.prepare(config())}
            got = patch.maybe_rope(q, k, freqs, state)
            assert got is not None and all(bytes_equal(a, b) for a, b in zip(got, expected))
            again = patch.maybe_rope(q, k, freqs, state)
            assert all(bytes_equal(a, b) for a, b in zip(again, expected))
            assert state['rope_state']['calls'] == 2 and state['rope_state']['matrices'] == 1
            next_freqs = tuple(x.clone() for x in freqs)
            assert patch.maybe_rope(q, k, next_freqs, state) is not None
            assert state['rope_state']['matrices'] == 2


def test_layout_dtype_and_backend_fallback(monkeypatch):
    gpu_runtime()
    from comfy.text_encoders.llama import precompute_freqs_cis
    from comfy_kitchen.registry import registry
    with torch.inference_mode():
        q = torch.zeros(1, 16, 1, 128, device='mps', dtype=torch.bfloat16)
        k = torch.zeros(1, 8, 1, 128, device='mps', dtype=torch.bfloat16)
        freqs = precompute_freqs_cis(128, torch.tensor([[0]], device='mps'), 1000000., device='mps')
        state = {'rope_state': patch.prepare(config())}
        assert patch.maybe_rope(q.float(), k, freqs, state) is None
        assert patch.maybe_rope(q.expand(2, -1, -1, -1), k, freqs, state) is None
        backing = torch.zeros(1, 16, 1, 256, device='mps', dtype=torch.bfloat16)
        assert patch.maybe_rope(backing[..., ::2], k, freqs, state) is None
        monkeypatch.setattr(registry, 'get_implementation', lambda *a, **kw: lambda: None)
        assert patch.maybe_rope(q, k, freqs, state) is None
        assert state['rope_state']['calls'] == 0


def test_kernel_failure_latches_fallback(monkeypatch):
    gpu_runtime()
    from comfy.text_encoders.llama import precompute_freqs_cis
    with torch.inference_mode():
        q = torch.zeros(1, 16, 1, 128, device='mps', dtype=torch.bfloat16)
        k = torch.zeros(1, 8, 1, 128, device='mps', dtype=torch.bfloat16)
        freqs = precompute_freqs_cis(128, torch.tensor([[0]], device='mps'), 1000000., device='mps')
        state = {'rope_state': patch.prepare(config())}
        calls = []
        def fail(*a):
            calls.append(1)
            raise RuntimeError('compile unavailable')
        monkeypatch.setattr(patch, '_dead', False)
        monkeypatch.setattr(patch, '_fused', fail)
        assert patch.maybe_rope(q, k, freqs, state) is None
        assert patch.maybe_rope(q, k, freqs, state) is None
        assert calls == [1] and patch._dead


def test_every_bf16_encoding_including_nonfinite_values():
    gpu_runtime()
    from comfy.text_encoders.llama import rope_matrix, precompute_freqs_cis
    from comfy_kitchen.backends.eager.rope import apply_rope_split_half
    with torch.inference_mode():
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to('mps')
        for position in (0, 13, 2000, 24575):
            matrix = rope_matrix(precompute_freqs_cis(
                128, torch.tensor([[position]], device='mps'), 1000000., device='mps'))
            for offset in range(0, 65536, 2048):
                q = bits[offset:offset + 2048].reshape(1, 16, 1, 128)
                k = bits[:1024].reshape(1, 8, 1, 128)
                expected = apply_rope_split_half(q, k, matrix)
                got = patch._fused(q, k, matrix)
                assert all(bytes_equal(a, b) for a, b in zip(got, expected))
