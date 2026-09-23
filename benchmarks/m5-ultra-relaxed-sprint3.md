# M5 Ultra: optional relaxed kernels, sprint 3

Date: 2026-09-23. M5 Ultra, 80 GPU cores, 256 GB unified memory; macOS 27,
PyTorch 2.13.0, ComfyUI `b33e2b55` / 0.37.0, Comfy Kitchen 0.2.35.
The exact baseline is commit `723573f`, pushed before this sprint. Its accepted
VAE fusions, sampler, INT8 projections and long-attention schedule remain the
default. No cache, sparsity, token reduction, overlap or step-count change is
part of these experiments.

**Outcome:** retain dense FP16 VAE attention as a default-off option for user
review. It saves about 2.6 seconds end to end on the full workflow. The tail
candidate saves about 3.1 seconds sampling at full resolution, but stays
independently selectable in the benchmark harness: no consistent small-clip
gain and substantial same-seed composition/motion changes. Neither result
reaches 7 seconds per sampling step. Softmax and GEMM experiments are not
promoted. No relaxed candidate is enabled by the native preset.

## Optional dense FP16 VAE attention

`ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1` selects the existing dense FP16
FlashAttention kernel for the supported H3 video-decoder attention geometry
(32 heads, head dimension 64). This changes floating-point reduction and
rounding, and therefore decoded frame bytes. It does not approximate attention
by removing tokens. Projection GEMMs, normalization/RoPE, SwiGLU, spatial
tiling, overlap and temporal chunking retain the accepted exact implementation.

The switch is **off by default**, including under `ASFP8_NATIVE=1`.
Set it to `1` in the environment used to start ComfyUI and restart to enable it.
Set it to `0` (or remove it) and restart to restore the exact decoder. It
requires the native VAE path (`ASFP8_NATIVE=1` or `ASFP8_DIT_EXT=1`, with
`ASFP8_MINIMAX_VAE_FAST` enabled) and the dense Metal attention extension.
Unsupported calls keep the existing path; a dense-kernel failure logs the
fallback and disables that candidate for the process.

### Complete decoder on identical saved latents

Each path receives two warmups. Alternating synchronized repetitions time
the entire `VAE.decode`, including layout conversion, padding, allocations,
tiling and output handling. Numerical checks and review-media creation happen
after timing. The short exact result matches the saved sprint-one frame tensor.

| Saved input | Pairs | Exact median (s) | Dense median (s) | Time saved |
|---|---:|---:|---:|---:|
| 864×480, 22 frames | 6 | 1.885658 | 1.531765 | 18.77% |
| 864×480, 124 frames | 4 | 13.222633 | 10.667718 | 19.32% / 2.555 s |
| 640×352, 22 frames, boat | 4 | 0.755891 | 0.608088 | 19.55% |
| 640×352, 22 frames, scarf | 4 | 0.756770 | 0.609283 | 19.49% |
| 640×352, 22 frames, butterfly | 4 | 0.757026 | 0.607761 | 19.72% |

The latter three cases exercise the production optional wrapper. Switching
back to exact reproduces every reference frame byte in each case. Separate
fresh processes validate default-off selection and enabled selection against
their saved outputs. All 42 VAE/kernel/native-switch regression tests pass,
including simulated kernel failure, fallback latching and both output layouts.

### Numerical and visual assessment

For the full 124-frame boat clip, dense versus exact FP32 frames in [0,1]:

| Metric | Result |
|---|---:|
| Maximum absolute pixel error | 0.00481248 |
| RMSE | 0.00009623 |
| PSNR | 80.33 dB |
| Mean sampled-frame luminance SSIM | 0.99999727 |
| Largest absolute mean RGB-channel bias | <0.0000004 |
| Temporal-difference error RMS | 0.00012754 |
| Exact / dense temporal-difference RMS | 0.06485255 / 0.06485248 |

Lossless side-by-side detail crops and six matched frames spanning each clip
were inspected. Paper folds, water ripples, reflections and warm red color
look unchanged at the displayed scale. Additional scarf and butterfly checks
show matching fine fabric/hair/skin texture, wing markings and flower detail.
Sampled motion positions match, with no visible new flicker in those sequences.
The three short content cases have PSNR 79.40–84.13 dB and sampled SSIM above
0.999997. These observations and temporal metrics support keeping this as an
optional candidate for user review; they do not prove perceptual equivalence
for every prompt, frame or model.

Further inspection targets the largest absolute-error location in the full
boat clip and the two additional scarf/butterfly references, rather than only
central crops. Those enlarged crops still show matching detail at the displayed
scale. Twelve consecutive boat frames around the largest temporal error also
show matching motion. After direct RGB8 rounding, **1.0964% of channel values**
in the full clip change, all by at most **one 8-bit level**. The largest
frame-to-frame difference-error RMSE is 0.00015088. The additional short scarf
and butterfly clips likewise differ by at most one RGB8 level in these checks.

Raw FP32 frame tensors and lossless PNG crops are retained locally. Synchronized
side-by-side MP4s are review copies encoded at H.264 CRF 12; their compression
must not be used to establish numerical equivalence.

## Full five-second workflow confirmation

The reference is 864×480, 124 frames at 24 fps, 20 steps; seed 812764035,
`res_multistep` / `simple` / denoise 1. It uses the same red-paper-boat prompt,
INT8 convrot H3 model and Qwen text encoder, FP16 video VAE and FP32 audio VAE
as sprint two. Each mode first receives a two-step warmup at this full shape.
The six measured runs execute in the order below. VAE and tail candidates are
tested separately; their combined performance is not claimed.

| Run | Sampling (s) | Video decode (s) | E2E (s) |
|---|---:|---:|---:|
| Exact 1 | 147.145325 | 13.173888 | 161.406 |
| Tail 1 | 144.002574 | 13.157319 | 158.231 |
| Dense VAE 1 | 146.889117 | 10.703627 | 158.658 |
| Dense VAE 2 | 146.856070 | 10.709505 | 158.622 |
| Tail 2 | 143.868849 | 13.166184 | 158.113 |
| Exact 2 | 146.897692 | 13.140778 | 161.079 |
| **Exact mean** | **147.021509** | **13.157333** | **161.2425** |
| **Dense VAE mean** | **146.872594** | **10.706566** | **158.6400** |
| **Tail mean** | **143.935712** | **13.161752** | **158.1720** |

Dense VAE saves **2.450767 s video decode** in the live workflow and
**2.6025 s E2E (1.61%)**. Its nominal 0.149 s sampling difference is ordinary
variation, not a sampler improvement. The stronger isolated decoder test above
gives 2.555 s / 19.32% less decode time. Tail specialization saves
**3.085797 s sampling (2.10%)**, or **7.3511 → 7.1968 s/step**, and 3.0705 s
E2E. The two mode ranges do not overlap their exact controls.

Timers include the complete sampling loop and synchronized video decode;
E2E uses ComfyUI execution timestamps and includes audio decode and MP4 export.
All modes share benchmark instrumentation, including initial-noise capture.
Post-run frame saving, comparisons and review rendering are outside E2E.
GPU jobs run serially in an isolated server; other desktop applications remain
open. These are matched comparisons, not a claim against the fastest old run.

All six captured initial-noise tensors match. Both exact controls and both
dense-VAE runs reproduce all four previous-production sampler latent tensors
byte for byte; decoded exported audio also matches. Exact raw frames reproduce
the previous-sprint SHA-256
`538f33de59395e714e0fb87ade8011dfe61f4131d11f34ba5c142d72dd71d5be`.
Dense-VAE frames match its saved-latent decoder reference byte for byte.
Each of the three modes reproduces its own complete frame bytes on repetition.
No VAE fallback occurred. Tail changes video and audio latents and decoded audio.

## Attention tail specialization

The candidate peels the ragged final key block from the main loop. It retains
the algorithm and operand precision, but Metal code generation changes the
resulting output bytes. A long-H3-only gate is essential: the ungated version
is approximately 27% slower at 2,968 tokens and 15% slower at 7,873 tokens.
The gated version is unchanged at those shapes and saves about 3.9% on the
captured 15,458-token attention input, including padding and output allocation.

Sampling comparisons use identical captured initial noise, not only equal
seeds. Each candidate is independent of VAE relaxation and uses the exact
decoder. Three 20-step, 640×352, 124-frame cases are warmed at two steps first:

| Prompt / seed | Exact sampling (s) | Tail sampling (s) | Video latent relative L2 |
|---|---:|---:|---:|
| Red paper boat / 812764035 | 67.2346 | 66.8967 | 17.56% |
| Woman in teal scarf / 20260924 | 67.1436 | 67.5774 | 46.02% |
| Monarch butterfly / 20260925 | 67.5672 | 67.6298 | 34.40% |

The smaller workflow shows no consistent sampling benefit. Tiny operator
differences accumulate during denoising: frame PSNR becomes 25.60, 15.37 and
16.95 dB; sampled SSIM becomes 0.892, 0.620 and 0.678. Side-by-side crops and
frame sequences show changed portrait framing, scarf folds/weave, flower
placement and butterfly wing positions. The outputs remain plausible, but
they are visibly different, including motion. This is not evidence of equal
quality merely because the single-attention error is small.

The full-resolution tail video has **48.64% video-latent relative L2**,
**12.86% audio-latent relative L2**, frame PSNR **15.65 dB**, sampled SSIM
**0.6112** and temporal-difference error RMS **0.08315**. Inspection shows
changed boat orientation, drift trajectory, water reflections and ripple
positions. Fine folds and ripples remain plausible, and these observations do
not prove a systematic aesthetic-quality loss. They do establish a material
content/motion change, with equivalent temporal quality still unestablished.
For a roughly 2% full-shape sampling gain, this remains benchmark-only pending
further perceptual review. The frozen candidate and side-by-side clips are kept.

## Independent softmax experiments

The software-exponential and deferred-rescaling ideas follow the corresponding
sections of [FlashAttention-4](https://arxiv.org/html/2603.05451v1). These are
separate Metal experiments, not an assumption that its NVIDIA results transfer.

Three candidates replace one quarter, one half, or all score exponentials with
a centered degree-four polynomial and exponent-bit reconstruction. Two further
candidates retain the softmax anchor until the next maximum exceeds four or
eight bits of headroom, then rescale through a SIMD-uniform branch. Accumulator
types remain FP32. No candidate is combined with tail specialization.

All fail the speed gate. At 15,458 tokens, quarter/half/all polynomial variants
take approximately **64%, 76%, and 12% more time**, while the deferred anchors
take approximately **34% more time**. Outputs are finite but changed. Slower
candidates do not advance to expensive sampling or perceptual acceptance and
are not exposed as production modes.

## FP16 VAE GEMMs with fused output operations

Custom TensorOps GEMMs retain FP16 inputs and FP32 accumulation. Five tile
schedules screen the output projection plus bias/residual, FC2 plus residual,
and paired FC1 gate/up projections plus SwiGLU. Timings include allocation and
adjacent output operations; inputs come from the real decoder at batch sizes
one and four. An initial descriptor-layout bug was corrected before these
results; the invalid prototype is excluded from every speed/quality claim.

| Complete 22-frame decode | Exact median (s) | Candidate median (s) |
|---|---:|---:|
| Output projection | 1.878502 | 1.881089 |
| FC2 | 1.879540 | 1.879904 |
| FC1 + SwiGLU | 1.887272 | 1.874466 |
| Output projection + FC2 | 1.891691 | 1.892507 |

Five alternating pairs per candidate show only a 12.81 ms / 0.68% FC1 gain.
Full saved-clip confirmation gives **13.200401 → 13.109929 s** across three
alternating pairs: **90.47 ms / 0.69%**. All tested complete frame bytes match,
but limited matching examples do not prove arbitrary-input GEMM bit-exactness.
The small complete-decoder return does not justify another production matrix
backend. These candidates remain archived experiments.

## Reproduction and local evidence

Workspace diagnostics are in `diagnostics/relaxed-sprint3/`, one directory
above this repository. `exact-baseline/` freezes all 79 starting Python/Metal
sources, with a hash manifest. `modes.json` keeps tail, exponential and rescale
candidates independently selectable; `attention_variants.py` creates separate
extension builds without editing the installed kernels. GEMM candidates are
individually selected in `vae_gemm_decode.py`.

Key evidence: `vae-dense-short.json`, `vae-dense-full.json`, `vae-more.json`,
`tail-probe.json`, `tail-quality-results.json`, `softmax-probe.json`,
`vae-gemm-probe.json`, `vae-gemm-decode.json`, `vae-gemm-decode-full.json`,
and `runtime-default.json` / `runtime-dense.json`. Raw latents, initial noise,
workflow JSON, frame tensors and review media are retained there. Full-workflow
timings and byte/audio checks are in `full-confirm-results.json`,
`full-summary.json` and `full-verification.json`. Small machine-readable
measurements are published with this report in
[`data/m5-ultra-relaxed-sprint3.json`](data/m5-ultra-relaxed-sprint3.json).

Only `_patches/minimax_vae_fast.py` is updated in the local installed node;
the other 78 frozen source files remain unchanged. Its original file is backed
up in the diagnostics directory. Launch environment settings are untouched,
so dense VAE attention remains off until explicitly enabled. The source audit
is recorded in `installed-source-verification.json`. Fresh-process checks from
the installed node reproduce saved reference bytes with the switch off and
with it on (`installed-runtime-default.json` / `installed-runtime-dense.json`).
The isolated benchmark server is stopped after verification.
