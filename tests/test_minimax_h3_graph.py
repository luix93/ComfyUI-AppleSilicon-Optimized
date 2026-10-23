import types

import torch

from _patches import minimax_h3_graph as graph


def test_rewriter_retains_native_pdd_output_heads_and_sampling_arguments():
    class FinalLayer:
        def forward(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
            return self.video_out(self.norm(x)), self.audio_out(self.adaln_proj(t_emb)), sigma, sample_sigmas, shifts

    original = FinalLayer.forward
    result = graph.rewrite_final_layer(types.SimpleNamespace(FinalLayer=FinalLayer),
        types.SimpleNamespace(rms_mod_seg_f32=lambda *args: None), lambda *args: None)
    assert not result.applied
    assert "PDD" in result.reason
    assert FinalLayer.forward is original


def test_rewriter_rejects_an_unknown_final_graph():
    class FinalLayer:
        def forward(self, x, t, video, audio):
            return x

    result = graph.rewrite_final_layer(
        types.SimpleNamespace(FinalLayer=FinalLayer),
        types.SimpleNamespace(rms_mod_seg_f32=lambda *args: None),
        lambda *args: None,
    )
    assert result.applied is False
    assert "unrecognized" in result.reason


def test_rewriter_is_idempotent_and_cpu_falls_through():
    class FinalLayer:
        def adaln_proj(self, value):
            return value, value

        def norm(self, value):
            return value

        def video_out(self, value):
            return value

        def audio_out(self, value):
            return value

        def forward(self, x, t_emb, video_seg, audio_seg):
            shift, scale = self.adaln_proj(t_emb)
            return self.video_out(self.norm(x)), self.audio_out(scale + shift)

    module = types.SimpleNamespace(FinalLayer=FinalLayer)
    kernel = types.SimpleNamespace(rms_mod_seg_f32=lambda *args: None)
    first = graph.rewrite_final_layer(
        module, kernel, lambda *args: None)
    wrapped = FinalLayer.forward
    second = graph.rewrite_final_layer(
        module, kernel, lambda *args: None)
    assert first.applied and second.applied
    assert FinalLayer.forward is wrapped

    instance = FinalLayer()
    x = torch.ones(3, 2)
    video, audio = instance.forward(x, torch.ones(1, 2), (1, 3, 0), (0, 1, 0))
    assert torch.equal(video, x)
    assert torch.equal(audio, torch.full((1, 2), 2.0))
