# M5 Ultra: native INT8 ConvRot MiniMax H3 video VAE

2026-09-24. M5 Ultra, 80 GPU cores, 256 GB; macOS 27, PyTorch 2.13.0,
ComfyUI `b33e2b55` / 0.37.0, Comfy Kitchen 0.2.35. Baseline is the accepted
exact stack and optional FP16 relaxed preset at `fb73a5e`.

The downloaded `minimax_h3_video_vae_int8_convrot.safetensors` loads on MPS,
but initially decodes more slowly than our FP16 VAE. The guarded native route
reduces complete full-clip decode from **22.29 to 11.28 seconds**. With separately
optional dense attention it takes **8.41 seconds**, versus **10.11 seconds**
for our previous fastest FP16 VAE: **1.70 seconds / 16.8% less decode time**.
The paired complete workflow averages **154.645 → 152.800 seconds E2E**, saving **1.85 seconds**.

## What was missing

The checkpoint contains 144 INT8 decoder matrices with per-output-channel FP32
scales and ConvRot group size 256. Its file is 2,811,065,184 bytes, versus
5,207,808,496 bytes for the FP16 checkpoint. These are file sizes, not measured
peak-memory savings.

Kijai's [core integration](https://github.com/Comfy-Org/ComfyUI/pull/16187)
was already installed. Its newer `linear_input_act` calls carry RMSNorm parameters
and residual epilogues. Our sampling-oriented INT8 wrapper preserved these
extended calls but did not accelerate them. A counted short decode called
`F.linear` 876 times and never entered `ck.int8_linear`: the core MPS path
dequantized the weights and used W8A16 matrix multiplication. Our FP16 VAE
wrapper also deliberately excluded quantized layers.

Routing these calls through the existing unfused INT8 backend alone was slower:
3.61 seconds versus 3.18 seconds for the short decoder. It materialized large
INT32 matrices, then ran scaling, conversion and bias passes separately. That
variant is retained in diagnostics and is not the production path.

## Implementation and selection

The new, independently named Metal entry points keep the existing INT8×INT8
integer accumulation, and emit FP16 directly with per-row activation and
per-channel weight scales. The scaled result rounds to FP16 before bias; bias
addition then rounds again. This removes the INT32 output allocation and extra
memory passes without changing the tested unfused W8A8 output bytes. Existing
BF16 sampling kernels are unchanged.

An H3-specific wrapper handles native FP16 activation rotation/row quantization,
these matrix operations, RMSNorm/SwiGLU inputs and residual arguments. It reuses
our exact Q/K normalization, RoPE, layout and SwiGLU kernels. Normalization and
rotation remain separate operations here; this is not a claim that all of
Kijai's [CUDA fusions](https://github.com/Comfy-Org/comfy-kitchen/pull/167) were
ported. Attention remains FP16. No INT8 attention, reduced tiling/overlap,
FP16 accumulation, caching or sampling approximation is introduced by this VAE
change.

Load the INT8 VAE with `ASFP8_NATIVE=1`. Native W8A8 is on for eligible H3
quantized layers. `ASFP8_MINIMAX_VAE_INT8=0` restores the previous core path.
Selecting the FP16 checkpoint retains all existing exact/relaxed FP16 choices.

`ASFP8_MINIMAX_RELAXED=1` selects the existing combined sampling/VAE preset and
now works with this INT8 VAE. To enable only its VAE attention choices, leave
the master preset unset and set `ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1` plus
`ASFP8_MINIMAX_VAE_ATTN_TUNING=1`. Explicit individual settings still take
precedence. The FP16 QKV matrix kernel is bypassed for quantized weights.
Restart after changing environment settings. No flag changes the selected VAE
checkpoint automatically.

CPU/offloaded tensors, custom subclasses/forwards, hooks, runtime adapters,
weight/bias transformations, training, explicit full-precision requests and
unsupported shapes fall back to core. A kernel failure logs once and latches
back to core for the process. Source installation preserved previous files in
the diagnostic backup and did not change launch flags or user workflows.

## Complete decoder measurements

Each case repeatedly decodes identical saved latents. Sampling is absent.
Two warmups per mode precede four alternating forward/reverse measured rounds.
Timers synchronize MPS and include allocations, rotation, quantization, layout,
all decoder blocks and final frame assembly. Host copies, hashes, image metrics
and review encoding occur outside the decoder timer. GPU jobs run serially.

Short clips: 864×480, 22 frames. Times below are median seconds.

| Case | FP16 exact | FP16 relaxed | INT8 core | INT8 native | INT8 + dense |
|---|---:|---:|---:|---:|---:|
| reference | 1.8713 | 1.4520 | 3.1776 | 1.6155 | 1.2034 |
| scarf | 1.8758 | 1.4546 | 3.1846 | 1.6170 | 1.2021 |
| butterfly | 1.8851 | 1.4561 | 3.1868 | 1.6195 | 1.2035 |

Full clip: 864×480, 124 frames.

| Mode | Median decode (s) | Measured range (s) |
|---|---:|---:|
| FP16 exact | 13.1006 | 13.0950–13.1598 |
| FP16 relaxed | 10.1091 | 10.0964–10.1288 |
| INT8 before adaptation | 22.2948 | 22.2851–22.3008 |
| Native INT8 | 11.2835 | 11.2615–11.2927 |
| Native INT8 + dense attention | 8.4079 | 8.4007–8.4115 |

The production fused epilogue matches unfused outputs on all eight captured
VAE layer shapes (four projections, batches one/four). The complete short
native INT8 frame tensor also matches the unfused W8A8 prototype byte for byte,
both with and without dense attention. This is equivalence to W8A8 arithmetic,
not to the prior W8A16 fallback or FP16 checkpoint.

All five modes reproduce their own complete frame bytes on all four
repetitions, across three short scenes and the full clip. Exact and relaxed FP16
both reproduce their earlier saved reference frames. The INT8 disable switch
reproduces the pre-change INT8 reference frames.

## Quality measurements and visual inspection

INT8 checkpoint weights and activation quantization change decoded output.
The short-scene INT8+dense PSNR range is 60.73–65.26 dB against exact FP16.
These metrics quantify differences; they are not a perceptual-quality guarantee.

| Fixed-latent comparison | PSNR (dB) | Sampled luminance SSIM | Temporal difference error RMS |
|---|---:|---:|---:|
| reference | 65.26 | 0.9999515 | 0.0005786 |
| scarf | 63.86 | 0.9999130 | 0.0007418 |
| butterfly | 60.73 | 0.9999324 | 0.0009176 |
| full | 63.15 | 0.9999237 | 0.0008576 |

For the full clip, INT8+dense pixel RMSE is 0.0006961, maximum absolute error
0.047906, and mean RGB biases are below 0.000033. About 9.94% of RGB8 channel
values differ, by at most 12 levels. The reference/candidate frame-difference
RMS is 0.0648526/0.0648423. Native INT8 without dense attention has PSNR 63.20 dB;
the prior weight-only INT8 path has 66.48 dB. The extra activation quantization
therefore adds measurable error beyond quantizing checkpoint weights alone.

Inspected enlarged crops retain paper folds/edges, ripples/reflections, scarf
weave, hair strands, butterfly wing markings and flower detail at the displayed
scale. Red, teal and orange/purple colors look matched. The enlarged worst-pixel
water-reflection and butterfly-wing crops show no obvious structural damage.
Twelve consecutive boat frames around the largest temporal error, and the
corresponding scarf/butterfly sequences, show matched motion without obvious
new flicker in these inspected examples. This limited review does not establish
quality equivalence for all content. Raw FP32 frames, lossless PNG comparisons
and silent side-by-side MP4 review copies remain available locally.

## Complete workflow confirmation

864×480, 124 frames, 24 fps, 20 steps; seed 812764035, identical boat prompt,
res_multistep/simple/denoise 1. Both modes use the previously accepted optional
tuned sampling attention. Only the video VAE checkpoint/path changes. Each
receives a full-shape two-step warmup before the FP16/INT8/INT8/FP16 measured
order. E2E includes audio decode and video export; post-run frame saving and
numerical comparison are outside execution timestamps.

| Run / VAE | Sampling (s) | Video VAE (s) | E2E (s) |
|---|---:|---:|---:|
| 0 / fp16 | 143.269 | 10.231 | 154.614 |
| 1 / int8 | 143.100 | 8.436 | 152.638 |
| 2 / int8 | 143.379 | 8.457 | 152.961 |
| 3 / fp16 | 143.389 | 10.199 | 154.676 |
| **FP16 mean** | **143.329** | **10.215** | **154.645** |
| **INT8 mean** | **143.239** | **8.447** | **152.800** |

All initial noise and complete joint video/audio sampler-latent tensors match
between the four runs. They also match the earlier combined-preset reference.
Each VAE repeats its own frame bytes, and the FP16 controls reproduce the prior
combined-preset frame bytes. Sampling differences in wall time are measurement
variation, not a sampling optimization from changing the VAE.

## Validation and retained evidence

112 distinct kernel/routing/VAE tests pass. One existing CPU-reference test
required `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing Torch; its rerun passed.
The final custom-subclass guard test also passed after the guard was tightened.
Tests cover irregular matrix boundaries and storage offsets, integer reference
accumulation, both FP16 rounding stages, input validation, fallbacks and the
unchanged BF16 INT8 paths. Fresh installed-node processes verify native, relaxed
and disabled INT8 modes against saved frame bytes.

Machine-readable evidence: [m5-ultra-int8-vae.json](data/m5-ultra-int8-vae.json).
The workspace's `diagnostics/int8-vae-sprint5/` retains source snapshots,
checkpoint header, captured real layer inputs, all comparison scripts, timings,
raw frames/latents/noise, workflow exports, install backups and the visual review
gallery. Model files are not modified or committed.
