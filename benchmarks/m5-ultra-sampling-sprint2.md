# M5 Ultra: bit-exact sampling sprint 2

Date: 2026-09-23. Hardware: M5 Ultra, 80 GPU cores, 256 GB unified memory.
Runtime: macOS 27, PyTorch 2.13.0, ComfyUI `b33e2b55` / 0.37.0,
Comfy Kitchen 0.2.35. The accepted FP16 VAE kernels were frozen throughout.

The acceptance gate is byte-identical intermediate results, complete sampler
latents and decoded outputs, plus a repeatable complete-sampler improvement.
Operator speed alone does not pass that gate. Allocation, layout conversion,
rotation and quantization are included in operator-chain timings. Workflow
timings include their costs as well; loading and compilation are warmed first.

**Outcome: no new sampling change is promoted.** The final-layer repair is
byte-exact but does not demonstrate a repeatable complete-sampler improvement.
The quantization prototype fails exactness; attention and INT32 scheduling
experiments fail the performance gate. Candidate code/tests are archived for
reproduction. All 79 frozen installed Python/Metal source files still match
their pre-sprint hashes, including the accepted VAE implementation.

## Workflow and measurements

The short reference uses 864×480, 22 frames, 20 steps. The final reference uses
864×480, 124 frames at 24 fps, 20 steps. Additional screening uses two steps
and a 640×320, 124-frame clip. The full reference is approximately five seconds
(124 / 24 = 5.167 s), matching the previous production benchmark.

Both use seed 812764035, `res_multistep`, `simple`, denoise 1,
`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, the Qwen3VL 32B INT8 convrot
text encoder, the FP16 video VAE and FP32 audio VAE. The unchanged prompt is:

> A cinematic close-up of a red paper boat floating on a shallow pond. The boat
> drifts gently to the right as a small ripple passes underneath. Warm morning
> sunlight sparkles on the water. A single continuous shot with a fixed camera.
> Audio: soft water lapping and distant birdsong.

The sampler timer surrounds the synchronized complete step loop. E2E uses
ComfyUI execution timestamps and includes audio/video decode and MP4 export.
Post-run comparisons and media verification are outside those timers. GPU
experiments run serially in an isolated ComfyUI server; other desktop apps
remain open. Baseline and candidate runs share instrumentation and VAE code.

## Full five-second confirmation

Both modes were warmed at the full shape before these four 20-step runs.
Order is baseline, candidate, candidate, baseline; only final-layer fusion
changes. All four runs use the accepted VAE kernels.

| Run | Sampling (s) | Sampling per step (s) | Video decode (s) | E2E (s) |
|---|---:|---:|---:|---:|
| Baseline 1 | 147.564618 | 7.378231 | 13.305225 | 161.922 |
| Final-layer candidate 1 | 147.542385 | 7.377119 | 13.224798 | 161.801 |
| Final-layer candidate 2 | 147.663511 | 7.383176 | 13.228992 | 161.877 |
| Baseline 2 | 147.861775 | 7.393089 | 13.232175 | 162.174 |
| **Baseline mean** | **147.713196** | **7.385660** | **13.268700** | **162.048** |
| **Candidate mean** | **147.602948** | **7.380147** | **13.226895** | **161.839** |

The nominal sampling difference is 0.110249 s, or 0.0746%, with overlapping
ranges. The stronger short-workflow comparison is likewise within variation.
These measurements do not establish a repeatable sampling gain or 7 s/step.
The E2E difference includes decode variation despite identical decoder code.

All four complete latent tensors per run match the earlier production
reference byte for byte. Raw FP32 frame tensors match the current reference
byte for byte and match the previous VAE-sprint SHA-256:
`538f33de59395e714e0fb87ade8011dfe61f4131d11f34ba5c142d72dd71d5be`.
Decoded exported video and audio also match the earlier production output.
The fused final-layer kernel ran without a fallback. The isolated server was
stopped after validation; the installed node was not modified.

## Final-layer fusion

The earlier graph guard skipped the current final layer because normalization
moved into a nested helper. Current ComfyUI also added `sigma`, `sample_sigmas`
and `shifts`, per-token modulation rows and PDD output-head banks. Simply
removing the guard would be incorrect.

The implemented wrapper recognizes the current signature and nested helper,
forwards all arguments on fallback, and excludes unsupported per-token/PDD
graphs. The new kernel matches native RMSNorm's reduction order, preserves
intermediate BF16 rounding and FP32 modulation, and keeps existing head GEMMs.
The checkpoint uses FP32 modulation despite stored BF16 AdaLN weights.

Captured intermediate tensors and complete final-layer outputs match byte for
byte. Ten alternating operator rounds reduced median complete-layer time from
0.921465 to 0.404285 ms. Thirty-one kernel/wrapper and MiniMax regression tests
passed, including widths across the native reduction-loop threshold, both
modulation dtypes, storage offsets, new arguments and fallback behavior.

In the stronger one-second workflow check, three warm runs per mode after two
warmups gave median sampling times of **19.273755 s baseline and 19.267357 s
candidate**. The 6.398 ms difference (0.033%) is within run-to-run variation.
All sampler latents and raw decoded frames matched. This establishes an exact
layer optimization, not a repeatable complete-sampler improvement.

## Normalization/modulation into INT8 quantization

This checkpoint rotates activations through a BF16 Hadamard matmul before
rowwise INT8 quantization. The prototype combines normalization/modulation,
optional residual update, rotation and quantization, returns the rounded
activation needed by fallback paths, and caches INT8 rows/scales for QKV/FC1.

Complete preprocessing-chain medians improved from 0.2511 to 0.2244 ms for
normalization and from 0.3788 to 0.3424 ms for residual plus normalization.
Two-step screening passed 200 intermediate calls and complete latents/frames.
The full 20-step test failed: **six INT8 values differed across 2,000 calls**,
at calls 874, 900, 1549 and 1968. Residuals, normalized/modulated BF16 values
and FP32 scales remained exact. The first divergence is in rotation rounding
when replacing the existing MPS matmul accumulation with a butterfly transform.

Greater mathematical accuracy is insufficient: quantization boundaries must
match the existing bytes. Sequential FP32 summation also fails a saved boundary
case. The prototype is rejected. Independent intermediate verification feeds
reference values onward after checking each call; its matching final latents
must not be mistaken for validation of the uncorrected candidate.

## Attention registers and temporary lifetimes

Four variants preserve QK/PV arithmetic, softmax arithmetic and reduction order:
compiler thread limits of 512/1024, reuse of max/sum reduction scratch, and
explicit constant score-fragment indices. All match captured output bytes.

Tests use 2,968-token and 15,458-token tensors, with eight alternating rounds
of three repetitions and padding/output allocations included. At the long
shape, measured savings are +0.179%, +0.014%, −0.081% and −0.424%, respectively;
short-shape results are neutral or slower. No consistent gain is established.

Apple documents the thread-limit control in
[MTLComputePipelineDescriptor](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup)
and discusses occupancy in
[Metal Compute on MacBook Pro](https://developer.apple.com/videos/play/tech-talks/10580/).
Available public counters exposed timestamps, not actual register/spill counts.
These experiments do not prove that register pressure is the limiting factor.

## INT32 partial-accumulation schedules

The prior profile assigns approximately 42% of instrumented sampling to INT8
projections/feed-forward processing. Two schedules keep independent INT32
partials over alternating 16-wide fragments or alternating 256-wide K blocks,
combine them in INT32, then retain the existing dequantization/bias epilogue.
The tested K range cannot overflow INT32, even at the maximum INT8 product.

Both match output bytes at `(M,K,N)` shapes `(2968,5376,5376)`,
`(2968,14336,5376)`, `(8192,5376,28672)` and `(15458,14336,5376)`.
Striped partials are roughly 11–22% slower; block partials are roughly 88–102%
slower. Timers include output allocation and rescaling. Neither is promoted.

## Why the earlier sampler was faster

The earlier production run took **146.129493 s sampling / 163.650 s E2E**.
The warm VAE-sprint result took **148.335996 s sampling / 162.666 s E2E**.
Faster video decode offset the extra 2.206502 s in sampling.

The prompt, seed, model settings, resolution, frames, sampler, scheduler and
sampling operation names/shapes/counts match. Sampling instrumentation also
matches. The final-layer fusion was skipped in both runs. Core versions and
model storage policies match; attention and INT8 source hashes match the older
tested sources. Loading is excluded from these warm comparisons.

A controlled test of the earlier compiled attention binary against the current
one gives identical bytes and effectively identical timings: **79.671 ms old
vs 79.489 ms current** on the saved long input. Fresh two-step full-workflow
controls are likewise approximately 14.9 s with either binary.

There is no evidence here for a prompt change or a kernel/source regression.
Session-dependent runtime variation is plausible, but historical clock, power
and background-load telemetry is missing. No specific clock, thermal or
contention cause can be proved retrospectively.

## Local evidence

The accompanying workspace diagnostics are in `diagnostics/sampling-sprint2/`
(one directory above this repository), including frozen installed source hashes,
candidate sources, captured inputs/divergences, paired timings, workflow JSON,
raw reference frames and latent tensors. The main evidence files are
`final-probe.json`, `final-confirm-1s-results.json`,
`normquant-intermediate-verification.json`, `attention-probe.json`,
`int32-probe.json`, `workflow-audit.json` and `timing-audit.json`.
Full validation is recorded in `final-full5s-results.json`,
`full-verification.json`, `full-summary.json` and
`installed-source-verification.json`. `final-candidate.patch` and
`final-candidate/` preserve the wrapper, kernel and tests. The standalone
`test_final_candidate.py` replays the archived tests without modifying
production sources. `probe_final.py` rebuilds and benchmarks the archived
candidate; `experiment.py` loads it for the isolated workflow harness.
