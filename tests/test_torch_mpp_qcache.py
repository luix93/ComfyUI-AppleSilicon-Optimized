"""Byte-level regression coverage for the guarded 2.14 MPP query cache."""
import pytest
import torch
import torch.nn.functional as F

from _patches import torch_mpp_qcache as patch


def _available():
    return (torch.backends.mps.is_available()
            and patch._torch_build_supported()
            and patch._hardware())


@pytest.mark.skipif(not _available(), reason='requires validated M5 Ultra / PyTorch 2.14')
@pytest.mark.parametrize('heads,kv,seq', [(30,30,2048),(48,12,2051),(56,56,2111),
                                       (48,12,8306),(56,56,15458)])
@pytest.mark.parametrize('scale', [None, 0., .2])
def test_ragged_strided_gqa_and_scales(monkeypatch, heads, kv, seq, scale):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('ASFP8_MPP_QCACHE', '1')
    with torch.inference_mode():
        # Interleaved Q/K/V-style storage, nonzero offsets and padded rows.
        torch.manual_seed(seq)
        q = torch.randn(1,seq+2,heads,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2)
        k, v = [torch.randn(1,seq+2,kv,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2)
                for _ in range(2)]
        kw = dict(scale=scale,enable_gqa=heads!=kv)
        expected = F.scaled_dot_product_attention(q,k,v,**kw)
        got = patch.maybe_attention(q,k,v,**kw)
        assert got is not None and not patch._dead
        assert torch.equal(got.cpu().contiguous().view(torch.uint8),expected.cpu().contiguous().view(torch.uint8))
        assert patch._parameters.cache_info().maxsize == 128
        for unsupported in [dict(attn_mask=torch.ones(seq,seq,device='mps',dtype=torch.bool)),
                            dict(is_causal=True), dict(dropout_p=.1)]:
            assert patch.maybe_attention(q,k,v,**dict(kw,**unsupported)) is None
        if heads != kv:
            assert patch.maybe_attention(q,k,v) is None
        monkeypatch.setenv('ASFP8_MPP_QCACHE', '0')
        assert patch.maybe_attention(q,k,v,**kw) is None


def test_cpu_falls_back(monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    q = torch.zeros(1,30,2048,128,dtype=torch.bfloat16)
    with torch.inference_mode():
        assert patch.maybe_attention(q,q,q) is None


def test_unsupported_build_falls_back(monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setattr(torch.version,'git_version','unvalidated')
    with torch.inference_mode():
        assert patch.maybe_attention(None,None,None) is None


@pytest.mark.parametrize('version,revision', list(patch._TORCH_BUILDS.items()))
def test_validated_build_requires_matching_revision(monkeypatch, version, revision):
    monkeypatch.setattr(torch, '__version__', version + '+local')
    monkeypatch.setattr(torch.version, 'git_version', revision)
    assert patch._torch_build_supported()
    monkeypatch.setattr(torch.version, 'git_version', 'unvalidated')
    assert not patch._torch_build_supported()


def test_unknown_version_falls_back_even_with_known_revision(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.14.2')
    monkeypatch.setattr(torch.version, 'git_version', patch._TORCH_GIT)
    assert not patch._torch_build_supported()


@pytest.mark.skipif(not _available(), reason='requires validated M5 Ultra / PyTorch 2.14')
@pytest.mark.parametrize('keys', [8192, 8217, 12288])
def test_qwen_edit_layout_and_optouts(monkeypatch, keys):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('ASFP8_MPP_QCACHE', '1')
    monkeypatch.setenv('ASFP8_QWEN21_FAST', '1')
    monkeypatch.setenv('ASFP8_QWEN21_ATTN', '1')
    with torch.inference_mode():
        torch.manual_seed(keys)
        q = torch.randn(1,4098,32,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2)
        k,v = [torch.randn(1,keys+2,32,128,device='mps',dtype=torch.bfloat16)[:,1:-1].transpose(1,2)
               for _ in range(2)]
        expected = F.scaled_dot_product_attention(q,k,v)
        for layout in (False, True):
            got = patch.maybe_attention(q,k,v,output_bshd=layout)
            assert got is not None and not patch._dead
            assert torch.equal(got.cpu().contiguous().view(torch.uint8), expected.cpu().contiguous().view(torch.uint8))
            if layout:
                assert got.transpose(1,2).is_contiguous()
        assert patch.maybe_attention(q,k[:,:,:4104],v[:,:,:4104]) is None
        assert patch.maybe_attention(q,k,v,is_causal=True) is None
        for flag in ('ASFP8_QWEN21_ATTN', 'ASFP8_QWEN21_FAST'):
            monkeypatch.setenv(flag, '0')
            assert patch.maybe_attention(q,k,v,output_bshd=True) is None
            monkeypatch.setenv(flag, '1')


@pytest.mark.skipif(not _available(), reason='requires validated M5 Ultra / PyTorch 2.14')
@pytest.mark.parametrize('seq', [32769, 44600, 65536])
def test_h3_long_interleaved_attention_bytes(monkeypatch, seq):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('ASFP8_MPP_QCACHE', '1')
    with torch.inference_mode():
        torch.manual_seed(seq)
        storage = torch.randn(1, seq + 1, 3, 56, 128,
                              device='mps', dtype=torch.bfloat16)
        q, k, v = [storage[:, 1:, i].transpose(1, 2) for i in range(3)]
        expected = F.scaled_dot_product_attention(q, k, v)
        actual = patch.maybe_attention(q, k, v)
        assert actual is not None and not patch._dead
        assert torch.equal(actual.cpu().contiguous().view(torch.uint8),
                           expected.cpu().contiguous().view(torch.uint8))


@pytest.mark.skipif(not _available(), reason='requires validated M5 Ultra / PyTorch 2.14')
def test_extended_limit_is_h3_only(monkeypatch):
    monkeypatch.setenv('ASFP8_ATTN_BACKEND', 'pytorch')
    monkeypatch.setenv('ASFP8_MPP_QCACHE', '1')
    with torch.inference_mode():
        small = torch.zeros(1, 1, 1, 128, device='mps', dtype=torch.bfloat16)
        for heads, seq, expected in [(56, 65536, True), (56, 65537, False),
                                     (30, 32768, True), (30, 32769, False)]:
            q = small.expand(1, heads, seq, 128)
            assert patch._supports(q, q, q, {}) == expected
