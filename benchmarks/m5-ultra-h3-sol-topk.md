# H3 SOL top-k and protected token presets

October 7, 2026 · M5 Ultra, 80 GPU cores, 256 GB · macOS 27 · PyTorch 2.14.

These optional modes start from the best exact optimized implementation. The
reference here is **not stock**. Existing defaults, tau-based SOL recipes, Veda,
Light/Fast token recipes and exact inference remain available.

## What changed

A native Metal router ranks pooled Q/K video-block scores with a deterministic
bitonic sort. It selects a fixed top-k video fraction instead of tau thresholding.
This gives a predictable video-block budget as the sequence grows. Global
conditioning/audio blocks and immediate local neighbors are additional to that
budget; the existing pooled tail represents the remaining blocks. Protected
prefix queries retain exact PyTorch attention. Attention, INT8 arithmetic and
centroid-tail accumulation are unchanged. No external custom node is required.

Two additional sparse presets are available in **H3 Acceleration · Simple** and
as starting points in the advanced node:

| Preset | Video top-k | Dense steps at 20 | Dense layers |
|---|---:|---|---|
| Top-k Wide | 15% | First 2, last 1 | 0,1,33,34,35; two leading blocks |
| Top-k All-step Draft | 10% | None | None |

“10%” does not mean exactly 90% of all work disappears. Global/local blocks are
additional, prefix queries stay dense, pooled-tail work remains, and matrix
multiplies, normalization, text/audio, VAE decode and export still run.

The advanced **Top-k video fraction** defaults to 0, preserving tau routing.
Positive values override tau. Sorting supports up to **262,144 packed tokens**
(4,096 blocks / 32 KiB shared memory); larger shapes fall back with a reason.
This limit is specific to the new router. Existing tau and Auto/manual temporary
memory controls are unchanged.

## Complete workflow measurements

20 steps, `res_multistep` / `simple`, no Turbo LoRA, 124 frames at 24 fps. Models,
unchanged conditioning and model setup are warm. The final exact control at each
shape is the timing reference. Initial noise and conditioning bytes match across
modes. End-to-end includes sampling, both VAE decoders and video/audio export;
CPU diagnostic tensor saving happens afterwards. Runs use authored prompts and
generated noise, never private input media.

| Mode | 864×480 sampling | 864×480 E2E | 1344×768 sampling | 1344×768 E2E | E2E saving vs exact |
|---|---:|---:|---:|---:|---:|
| Exact optimized | 129.77 s | 140.85 s | 547.56 s | 569.58 s | — |
| Previous SOL14, tau 1.3 | 102.15 s | 113.16 s | 355.31 s | 375.32 s | 19.7% / 34.1% |
| New Top-k Wide | 95.51 s | 106.55 s | 312.14 s | 332.14 s | 24.3% / 41.7% |
| New All-step Draft | 82.77 s | 93.79 s | 227.46 s | 247.46 s | 33.4% / 56.6% |

Wide and Draft each have two complete warm repeats per resolution. The previous
SOL and exact 1 MP measurements are single full runs after smaller screening.
At 480p, exact controls were 145.11 s initially and 140.85 s fully cached; the
latter is used above. Existing SOL measured 113.16 s.

The larger gain is substantially from increasing the eligible steps/layers;
fixed-budget routing helps control sparse density, rather than independently
accounting for all of the end-to-end savings.


The 864×480 test contains 15,458 packed attention tokens; 1344×768 contains
37,768. Sparse helps more at the larger shape because dense attention grows
approximately quadratically while many other operations grow linearly.

The second authored scene/seed is a red toy car passing fixed plants and
bookshelves (864×480, 73 decoded frames, 9,215 packed attention tokens;
70 frames requested and rounded by H3). Sampling measured **62.17 s Exact,
54.10 s previous SOL, 51.22 s Wide and 47.22 s Draft**, single full runs.
The first exact run includes new prompt/shape setup, so its 70.58 s E2E is not
used to claim a precise cached end-to-end saving. The variants retain the
left-to-right action but change foliage, framing and car markings. No catastrophic
background morphing is visible at the sampled points; this is not a broad
quality qualification. On its one-second token test, sampling measured
17.03 / 16.46 / 14.50 s for Exact / Protected / Balanced.

## Token reduction

Horizontal pairing, original RoPE behavior and per-pair detail restoration are
retained. New sigma-derived windows keep full tokens on protected sampler steps,
including repeated CFG calls at the same timestep. Existing Light/Fast remain
all-step and retain their previous restore blocks.

| New preset | Early/later restore block | Reduced block count | Full-token steps at 20 |
|---|---|---|---|
| Protected | 12 / 8 | 8 / 4 | First 4, final 2 |
| Balanced | 30 / 20 | 26 / 16 | First 2, final 1 |

| Preset | 864×480 / 124 frames sampling | E2E | Less E2E time vs exact |
|---|---:|---:|---:|
| Protected | 123.92 s | 134.93 s | 4.2% |
| Balanced | 103.39 s | 114.42 s | 18.8% |

Full-clip token timings above are single runs. One-second 20-step repeats
produced identical outputs: Protected 17.788 / 17.794 s E2E; Balanced
15.954 / 15.974 s, with the final exact control at 18.482 s. The initial exact
short run was 18.104 s; small speed differences are therefore less conclusive
than the larger Balanced gain. A second authored prompt/seed also completed.

Protection rounds outward on shorter schedules. “Protected” describes retained
compute, not identical output or guaranteed quality. Each token preset can be
selected independently of sparse attention and temporal cache. The combined
runtime is smoke-tested, but these measurements evaluate individual presets;
speed and quality of arbitrary combinations are not established.

## Quality and reproducibility

All new presets are **non-bit-exact** relative to exact inference. Byte-identical
repeats demonstrate reproducibility of a preset, not identical quality between
presets. Existing Exact, Light and Fast produced identical noise, conditioning,
latents, decoded frames and audio to the saved pre-change reference runs.

For the 1 MP teapot clip, complete-output errors against Exact were:

| Mode | Video latent relative L2 | Decoded frame RMSE (0–1) | Frame PSNR |
|---|---:|---:|---:|
| Previous SOL14, tau 1.3 | 0.3179 | 0.1043 | 19.63 dB |
| New Top-k Wide | 0.5160 | 0.1659 | 15.60 dB |
| New All-step Draft | 0.8579 | 0.2603 | 11.69 dB |

These errors are significant, not evidence of nearly identical images. Audio
also changes through cross-stream interactions, despite protected prefix
attention; the JSON includes its errors.

The teapot scene retained coherent pouring in Wide, with changes to framing and
kitchen geometry. Draft changed the backsplash, camera framing and visibility of
the face substantially. Sampled background strips did not show catastrophic
morphing, but sparse frame inspection cannot rule out flicker or establish quality
across prompts. Protected token reduction produced smaller visual changes;
Balanced changed geometry more. These are useful experimental choices, not
quality-preserving substitutes for Exact.

**Subsequent user feedback · October 7:** Top-k Wide morphed the background,
while SOL INT8 Fast remained stable in the user's comparison. That clip has not
been inspected or independently reproduced. The earlier sampled-frame checks
did not establish temporal stability. Wide remains an experimental speed option;
Fast is the better choice for the reported workflow. Fixed-budget routing and
the wider step window are both possible contributors; they have not been
isolated in a matched quality comparison. Timings above are retained as speed
measurements, not evidence that Wide preserves quality.

[480p sparse comparison](artifacts/h3-sol-topk/sol-480p.mp4) ·
[1 MP sparse comparison](artifacts/h3-sol-topk/sol-1mp.mp4) ·
[Token comparison](artifacts/h3-sol-topk/tokens-480p.mp4) ·
[Second prompt/seed](artifacts/h3-sol-topk/sol-car.mp4).
Comparisons are muted and resized to 432 pixels per column; diagnostic outputs
retain native frames and audio. Numerical errors, mean
color, spatial-detail and frame-delta diagnostics are recorded in the
[measurement JSON](data/m5-ultra-h3-sol-topk.json). Frame-delta error also includes
composition changes; it is not a perceptual temporal-stability score.

## Research and rejected experiments

The upstream [comfy-kitchen API](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/__init__.py)
exposes top-k routing, pooled tail and optional token augmentation. Its compiled
SOL implementation is CUDA/HIP; it is not a ready-made MPS kernel. The
[SOL paper](https://arxiv.org/abs/2607.24027) reports different hardware and
workloads; its speedup does not establish quality or timing on this machine.

The public comparison studied earlier used 8-step Turbo v4, Euler/simple and
longer sequences. Our full results here use 20-step non-Turbo sampling. More
sparse steps/layers and longer token sequences explain much of the larger gain;
absolute timings and quality are not directly comparable.

Individual-token augmentation is not shipped. Simply adding 256 exact token
contributions on top of the existing pooled tail would double-count them.
A bounded PyTorch ranking prototype was expensive (15.1 / 59.0 ms per block at
15,458 / 32,768 tokens, before route filtering and attention). Grouping four
blocks cut ranking to 5.7 / 17.7 ms, but a useful native implementation also
needs consistent residual-tail replacement. This prototype does not rule out a
fused Metal port; it is not a lower bound on such a kernel.

A head-major INT8 scratch-layout prototype matched existing sparse output
bytes at both operator sizes. At 32,768 tokens it reduced the top-k 15% operation
from 56.87 to 55.37 ms (2.6%); at 15,458 tokens, 17.92 to 17.62 ms (1.6%).
It was not promoted: this small operator gain has no complete-decoder/sampler
qualification. The shipped scratch layout remains unchanged.

Four-step non-Turbo 1 MP screens also measured 130.0 s exact versus 66.1 s
all-step top-k 10%. These short runs were used only to choose candidates, not to
claim converged output quality. Synthetic 32,768-token operator screens extend a
saved diagnostic Q/K/V capture; they are explicitly not complete long clips.

## Validation and memory safeguards

- Core CPU tests: 396 passed, 37 device-dependent skips.
- Related runtime tests with ComfyUI imports: 26 passed, 5 skips.
- UI tests: 17 passed. Unified controls: 18 passed after adding a backend-switch
  regression check (hidden INT8 top-k settings do not affect BF16 mode).
- Real MPS router tests: 8 passed, including ties, ragged/global boundaries and
  maximum 4,096-block shared-memory allocation.
- Default tau operation bytes are unchanged at both operator-screen sizes.
- Isolated server: 110 GiB RSS ceiling, 48 GiB minimum available RAM, bounded
  MPS watermarks and timeout. It stops if the user's server starts processing.

No stock benchmark table values were replaced by these optional-mode timings.
4/6/8-step Turbo quality has not been qualified for the new presets.
