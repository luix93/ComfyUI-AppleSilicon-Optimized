# PyTorch 2.14 production clone: stock and native kernels

Measured 2026-09-25 on Apple M5 Ultra (80 GPU cores, 256 GB), macOS 27.0
(26A428). The separate `ComfyUI-AppleSilicon-2.14` installation uses PyTorch
2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`), torchvision 0.29.0,
Python 3.13.12, and ComfyUI `b33e2b55cae074eca5aec96283cceac19aa249ba`.
Torchaudio remains 2.11.0: a matching 2.14 wheel was unavailable, its import
works, and the H3 audio path is included in the end-to-end tests. The original
2.13 installation, model files and ComfyUI core source were not modified.

## Final measurements

| Workflow | Stock 2.14 | Optimized 2.14 | Less total time | Optimized sampling |
|---|---:|---:|---:|---:|
| SDXL, 896×1152 / 12 steps | 2.598 s | **2.416 s** | 7.0% | 2.210 s |
| Z-Image, 1024² / 8 steps | 7.942 s | **3.443 s** | 56.6% | 3.227 s |
| Krea2, 1448² / 8 steps + LoKr | 28.036 s | **13.688 s** | 51.2% | 13.084 s |
| Krea2, same / no LoKr | 27.922 s | **13.798 s** | 50.6% | 13.186 s |
| H3, 864×480 / 124 frames / 20 steps | 236.752 s | **142.693 s** | 39.7% | 131.504 s |

Values are arithmetic means. Per-run values, ranges, warmups, complete workflow graphs
and tensor hashes are in the [machine-readable results](data/m5-ultra-torch214-production.json).

These are warm render times, including sampling, VAE decode and file export.
Unchanged model loaders and text encoding are cached, as in repeated ComfyUI
renders. Cold runs are retained in the data and excluded from the summaries.
Image confirmations discard two renders and measure three; the full H3 test
discards one complete render and measures two. Only one GPU job runs at a time.
The observation harness synchronizes MPS at sampler callbacks and stage boundaries.
Tensor hashing and reference-tensor saves happen after execution timing.

**Stock means the acceleration package is absent**, with an assertion that
none of its patch modules were imported. Both configurations use the actual
same 2.14 installation and `--use-pytorch-cross-attention`. Optimized uses the
installed custom node with `ASFP8_NATIVE=1`, `ASFP8_ATTN_BACKEND=pytorch`,
`MTLFLASHATTN_SDPA=off`, and the existing relaxed H3 preset off. Observation
nodes are present in both. Earlier reports comparing “our attention” against
“PyTorch attention” kept the other optimizations active and are not stock
baselines.

## Frozen inputs

The machine-readable data includes the complete API graphs, source hashes,
checkpoint names and full-file SHA-256 hashes, seeds, raw timings, tensor hashes and environment metadata.
The Z-Image, Krea2 and H3 graphs retain the inputs in the
[earlier frozen workflows](data/m5-ultra-torch214-workflows.json).

| Model | Workload | Seed / sampler | Checkpoint / adapter |
|---|---|---|---|
| SDXL | 896×1152, 12 steps, CFG 1.5 | 137740203719215; Euler ancestral / Karras | `SDXL/intorealismUltra_v90.safetensors`; original embedded PNG workflow |
| Z-Image Turbo | 1024×1024, 8 steps, CFG 1 | 630651083309421; res_multistep / simple | `Z-Image-Turbo/z_image_turbo_int8_convrot.safetensors` |
| Krea2 Turbo | 1448×1448, 8 steps, CFG 1 | 535122089142682; Euler / simple | `Krea2/krea2_turbo_int8_convrot.safetensors`; `Krea2/snofs_krea_v1_3D.safetensors`, strength 0.8 |
| Krea2 without LoKr | Same graph and prompt | Same seed / sampler | Adapter strength 0; separately labelled |
| MiniMax H3 | 864×480, 124 frames at 24 fps, 20 steps | 812764035; res_multistep / simple | Original INT8 convrot diffusion/text checkpoints; FP16 video VAE, FP32 audio VAE |

H3 screens use 22 frames and then the full step count; long-shape verification
uses 124 frames with two steps before the final complete workload. No prompt,
resolution, sampler, step-count or approximation shortcut is used to improve
its reported final time. The prompt is the same red-paper-boat scene as the
previous H3 comparison. Krea2 prompt expansion stays off.

## Installed changes

- Native extensions now follow PyTorch's compiler-standard default: C++20 on
  2.14. Build directories and Torch-library links are isolated by runtime
  version, build and installation path, preventing collisions with 2.13.
- Explicit 2.14 attention routing bypasses the optional mtlflashattn wrapper,
  preserves attention overrides, and retains the node's INT8 projections,
  Q/K normalization, RoPE, layouts and other model-specific fusions. Krea2
  uses PyTorch's grouped heads directly. Unsupported calls retain fallback.
- The accepted Krea2 normalization/elementwise fusions are enabled and
  validated on 2.14. Disabling them was a real regression in the first
  isolated version comparison.
- SDXL's new FP16 GEGLU pass preserves the projection and uses PyTorch's
  evaluated FP16 GELU table before multiplication. This eliminates an
  intermediate pass without changing rounding. The table is 128 KiB;
  `ASFP8_SDXL_GEGLU=0` disables it. Three alternating series found roughly
  0.03–0.07 s saved per render. All 840 affected operations per image and
  complete sampler/decoder outputs passed byte comparisons.

The new MPP attention specialization loads each query fragment once and keeps
it in registers across the key loop. It uses the installed PyTorch 2.14
implementation, with the same BQ=64/BK=32 tiles, matrix multiply, barriers,
softmax arithmetic and reduction order. It validates the Torch build and the
combined header hash, and is limited to the tested M5 Ultra dense BF16 D=128
model geometries. It supports native grouped K/V heads and last-dimension
contiguous views. A 128-entry parameter cache holds 16 KiB of buffer payload;
activations are not cached. Unsupported inputs fall back to PyTorch.
`ASFP8_MPP_QCACHE=0` disables it independently.

Whole-workflow alternating screens, before final confirmation:

| Workload | Original 2.14 sampling | Query cache | Byte validation |
|---|---:|---:|---|
| Z-Image, 8 steps | 3.305 s | 3.232 s | 256 attention calls; final latents and frames |
| Krea2 + LoKr, 8 steps | 13.614 s | 13.066 s | 224 attention calls; final latents and frames |
| H3, 22 frames / 20 steps | 17.435 s | 17.162 s | full latents, video and audio; 100 attention calls verified separately at two steps |
| H3, 124 frames / 2 steps | 14.313 s | 13.380 s | 100 attention calls; final latents, video and audio |

Verification renders run both attention implementations and copy results for
byte comparison; their timings are excluded. Ordinary timing renders include
output allocation, parameter lookup, layout, quantization and all other
sampler costs. Isolated long H3 attention measured about 73 → 63 ms, but the
full sampler improvement is the acceptance criterion.

The final full H3 comparison confirms **152.195 → 142.693 s end to end**,
with sampling **140.995 → 131.504 s** (**7.050 → 6.575 s/step**).
The two control renders took 152.191 / 152.199 s; the two query-cache renders
took 142.824 / 142.562 s. All video/audio sampler latents, decoded frame
tensors and decoded audio tensors match byte for byte, including the warmup.
Each enabled render dispatches 1,000 cached-query attention calls. The new
kernel saves about **9.5 seconds / 6.2% total time** over the otherwise
unchanged optimized 2.14 stack, without enabling a relaxed H3 mode.

## Experiments not promoted

- Changing MPP attention query/key tile sizes: larger key tiles changed
  bytes and were substantially slower. The byte-exact 32-row query variant
  did not improve complete Z-Image renders repeatably.
- Changing INT8 accumulation scheduling/fences: Z-Image projection outputs
  and final renders stayed exact, but a confirming four-pair run improved
  only about 0.004 s (0.1%). The Krea2 tensorwise variant was ineligible:
  this checkpoint uses column scales. Its zero-call result is a control,
  not evidence of a kernel win.
- Fusing Krea2 residual addition with post-normalization: intermediate,
  latent and frame bytes matched, with and without LoKr. Complete sampling
  changed by only about 0.02–0.03 s (0.2%), overlapping run variation.
- Removing attention barriers and specializing tails: all tested outputs
  matched in this 2.14 implementation, but speed results were mixed or
  negative. Removing the threadgroup scheduling barrier alone nearly
  doubled the long H3 attention time. These variants are not installed.

## FP16 H3 VAE on 2.14

Repeated decode of identical saved short latents (no sampling; alternating
modes, two warmup rounds and three measured rounds):

| VAE attention / projection | Complete decode, median | Final-frame comparison to PyTorch |
|---|---:|---|
| PyTorch 2.14 + accepted exact VAE fusions | 1.454 s | Reference |
| Same + existing optional QKV projection | 1.390 s | Identical for this saved latent |
| Existing optional dense attention | 1.502 s | Max error 0.00236; RMSE 0.0000985 |
| Existing dense attention + QKV | 1.438 s | Same measured error as dense attention |

The previous dense-attention VAE option is slower than the new default here.
It is still selectable, but is not enabled by the 2.14 settings. QKV remains
an independent optional route; one exact reference does not reclassify its
numerical contract for every input.

On the full saved 124-frame latent, alternating complete decodes measured
**10.162 → 9.739 s** with QKV enabled: about **0.424 s / 4.2%** of decode,
and about 0.3% of total generation time. Both full decoded tensors matched
exactly for this reference too. This uses `ASFP8_MINIMAX_VAE_QKV=1` with
`ASFP8_MINIMAX_RELAXED=0` and `ASFP8_MINIMAX_VAE_DENSE_ATTENTION=0`.
The table's default optimized timings do **not** enable this option.

## Numerical scope and validation

Stock and optimized **are not byte-identical stacks**. Both start from the
same workflow text and initial noise, but stock text encoding runs on CPU
while the node enables MPS and quantized text kernels. Conditioning hashes
therefore differ. Existing W8A8 arithmetic also differs from stock fallback.
Switching from the previous attention backend to PyTorch 2.14 changes bytes.
This report does not relabel those changes as exact.

New exact candidates are compared within the otherwise unchanged optimized
stack, using identical conditioning and noise. Checks cover affected
attention/GEGLU operations, complete sampler latents, decoded video frames
and audio tensors where applicable. Floating-point tensors are compared as
raw bytes, not just with `allclose`; repeated renders have their hashes
recorded. CPU reference tensors are saved locally for review.

**91 targeted tests passed; 4 were skipped.** Coverage includes every finite
FP16 GELU input bit pattern, ragged projection widths, actual full-size
Krea2/H3 attention shapes, strided views with nonzero offsets, GQA, scale zero,
masks/causality/dropout fallbacks, build guards, Krea2 normalization/elementwise
ordering, and third-party attention overrides. Complete rendering tests use
normal ComfyUI startup without the diagnostic MPS pre-initialization helper.
All warm repeats within each configuration reproduce the reference bytes.

All 12 checkpoint/adapter files have full SHA-256 hashes in the data, together
with sizes and modification timestamps. Runtime patch source hashes are
recorded per series. Krea2's final with-LoKr confirmation was repeated after
correcting an overly restrictive batch-stride guard; both adapter variants
confirm that the installed query-cache route is active. Removing LoKr did
not yield a useful inference-speed gain in these tests.

The stock/native image pairs retain broadly similar composition and color,
but fine detail and label lettering differ. The H3 samples also differ in
boat shape/position and water reflections. Those are differences between
the existing complete stacks, not differences introduced by the new exact
kernels. Six sampled video frames do not establish temporal quality parity.
The numerical differences below are measurements, not quality guarantees.

| Stock vs optimized decoded frames | RMSE, float [0,1] | Maximum absolute difference |
|---|---:|---:|
| SDXL | 0.00840 | 0.26758 |
| Z-Image | 0.02701 | 0.60254 |
| Krea2 + LoKr | 0.04675 | 0.83398 |
| Krea2 without LoKr | 0.05919 | 0.90430 |
| H3, all 124 frames | 0.08031 | 0.91634 |

[Image comparisons](data/m5-ultra-torch214-production-images.jpg) and
[H3 sampled-frame comparisons](data/m5-ultra-torch214-production-video.jpg)
are saved with this report. Full PNG/MP4 files and floating-point tensors
remain in the local diagnostic output directory.

The benchmark process is capped at 110 GiB RSS and requires at least 48 GiB
available system RAM; the watchdog checks throughout runs. Output artifacts
and diagnostic kernels are under `diagnostics/torch214-production` in the
local workspace. The original shared output files are preserved.

## Runtime controls

```text
ASFP8_NATIVE=1
ASFP8_ATTN_BACKEND=pytorch
MTLFLASHATTN_SDPA=off
```

Launch ComfyUI with `--use-pytorch-cross-attention`. These settings are saved
on the clone in Comfy Desktop. `ASFP8_SDXL_GEGLU=0` disables the new SDXL
fusion; `ASFP8_MPP_QCACHE=0` disables query caching. The existing `ASFP8_MINIMAX_RELAXED` preset and individual VAE relaxed
flags remain independent; the clone's default does not turn them on.
Removing `ASFP8_ATTN_BACKEND` restores the existing router, not the arithmetic
or output bytes of PyTorch 2.13.
