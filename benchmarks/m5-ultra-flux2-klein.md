# Flux.2 Klein 9B on M5 Ultra

**2026-09-26 · M5 Ultra / 80 GPU cores / 256 GB · PyTorch 2.14.0.**
The saved text-to-image workflow averages **5.59 → 2.25 s**;
the saved edit averages **9.12 → 4.66 s**.
Most of the gain was already present in our native stack. This follow-up adds
small, byte-exact activation and layout improvements; it does not change steps,
sampler, seeds, prompts, reference image or VAE settings.

## Baseline and frozen workflows

Completely unmodified ComfyUI fails to load this checkpoint with
`Unknown quantization format for layer double_blocks.0.img_attn.qkv`.
The file's per-layer INT8 ConvRot descriptors omit `format`. The existing
`quant_meta_fix` supplies `int8_tensorwise` for the identified INT8+scale layout.
Stock measurements load **only that metadata repair**, independently of the
acceleration package. No weights are rewritten and no native arithmetic is
enabled. The original failure and repaired stock runtime are retained in the
raw data. The installed custom node already included this compatibility fix.

Both workflows use:

- `flux/Flux2-Klein-9B-True-V3-INT8-ConvRot.safetensors`.
- `qwen_3_8b_fp8mixed.safetensors`, CLIP type `flux2`, device `default`.
- `flux2-vae.safetensors`; the recorded VAE compute dtype is BF16.
- **1024×1024, 4 Euler steps, Flux2Scheduler, CFG 1**.
- PyTorch 2.14 attention, preview disabled, ordinary PNG export.

`image_flux2_text_to_image_9b.json` uses seed **174953945942269** and the saved
vintage-motorcycle / sunset-diner prompt, including the “KLEIN” sign.
`flux.2 klein edit.json` uses seed **766258942932248**, the original
`z-image-turbo_00003_.png` reference, and **“Make the cat in image 1 orange.”**
The image is already 1024×1024; its one-megapixel resize retains that geometry.
Workflow JSON, input hash, model header hashes and file sizes are frozen locally
and recorded in the raw data. The original shared workflows remain unchanged.

ComfyUI commit `b33e2b55cae074eca5aec96283cceac19aa249ba`, Torch git
`08187d9e0fba026dc8217405802ab5381dc88d90`. Only one benchmark GPU job runs
at a time. Servers are isolated from the user's normal installation session,
with RSS/free-memory watchdogs. Two warmup runs are discarded from primary
tables. Timings include allocation, layout conversion, quantization and export.
Profiling, tensor hashing and intermediate verification are excluded.

## Warm full-workflow results

Model loading and unchanged text/reference encoding are cached here.

| Workflow / configuration | End to end | Sampling | VAE decode |
|---|---:|---:|---:|
| Text-to-image / stock + metadata repair | 5.585 s | 5.216 s | 0.322 s |
| Text-to-image / previous native, paired control | 2.277 s | 2.081 s | 0.159 s |
| **Text-to-image / new exact fusions** | 2.254 s | 2.043 s | 0.170 s |
| Edit / stock + metadata repair | 9.116 s | 8.719 s | 0.354 s |
| Edit / previous native, paired control | 4.702 s | 4.501 s | 0.164 s |
| **Edit / new exact fusions** | 4.655 s | 4.414 s | 0.200 s |

Text-to-image paired control: [2.279, 2.284, 2.269]; new:
[2.231, 2.302, 2.228]. Edit control:
[4.678, 4.732, 4.696]; new: [4.622, 4.665, 4.678].
The earlier separate native baselines were 2.313 s and 4.772 s. Use the paired
controls to attribute the small new improvement, rather than treating all
variation between separate runs as a kernel gain.

The new changes save about **0.024 s end to end / 0.038 s sampling** for
text-to-image, and **0.047 s / 0.088 s** for editing in these paired runs.
The larger stock-to-native difference comes from existing INT8 projections,
fused Q/K normalization–RoPE, attention query caching and text-encoder placement.
Both Flux block fast paths and the existing query-cache attention route engage
on the original workflows: 32 heads, dimension 128, 4608 tokens for text-to-image
and 8704 tokens when the reference image is included.

## Re-encoding the image and prompt

A separate test forces the same input image and text through encoding every
run, with model loading still warm. This measures the work hidden by ordinary
ComfyUI caching without changing the prompt or image.

| Edit configuration | End to end | Text encode | VAE encode |
|---|---:|---:|---:|
| Stock + metadata repair | 12.518 s | 3.182 s | 0.142 s |
| Installed native + exact fusions | 5.917 s | 1.172 s | 0.101 s |

The existing text-encoder placement patch runs the FP16 text encoder on MPS;
stock selects CPU. These figures include repeated encoding, not just sampling.

## Accepted changes and exactness

- Fuse BF16 SiLU lookup and gate multiplication in the double-stream MLPs.
  The 128 KiB lookup table contains the running PyTorch SiLU result for every
  BF16 input bit pattern, retaining intermediate rounding.
- In single-stream blocks, fuse that activation with concatenation of the
  attention output. Read the strided MLP view directly from the first linear
  projection and write the next projection's input in one pass.
- Keep all projections, quantization, normalization, attention arithmetic and
  residual arithmetic unchanged. Unsupported calls and custom activation hooks
  retain the original path.

Both original workflows preserve conditioning, initial noise, complete sampler
latents and final FP32 pixels byte for byte against the previous native stack.
Direct comparisons cover **all 8 double-stream and 24 single-stream blocks**
in each case. Two additional cases also pass: a blue-teapot text-to-image prompt
with seed 42, and a red-motorcycle edit with seed 901 and a different reference
image. Across the four cases, this verifies **32 double-stream and 96
single-stream block outputs**, as well as all complete sampler/decoder outputs.
The edit replay forces reference encoding. **17 tests pass**, covering finite
BF16 gate values, strided inputs/storage offsets, merged layouts, hooks,
unsupported calls and the existing Flux block regressions.

The new files are installed in `ComfyUI-AppleSilicon-2.14`; fresh-process replays
of both workflows match the previous native reference. Restart ComfyUI to load
the update. No new node or workflow change is needed. Switches:

```
ASFP8_FLUX2_GLUE=0        # disable both new additions
ASFP8_FLUX2_SWIGLU=0      # disable standalone SwiGLU fusion
ASFP8_FLUX2_MLP_LAYOUT=0  # disable single-block activation/concatenation fusion
```

The defaults are enabled through the existing Flux fast path on the validated
M5 Ultra / Torch 2.14 build. **Exactness applies to the new additions relative
to the existing native implementation.** Stock and native stacks already use
different quantized computation, fused operations and text-encoder placement;
they are not byte-identical to each other.

## Rejected experiments

| Candidate | Result |
|---|---|
| Disable existing query-load caching | Exact but slower on the edit (5.060 s versus roughly 4.7–4.8 s); retain query caching. |
| Short INT8 scheduling on the edit | Exact; no improvement over the current long-row schedule. |
| INT8 workgroup-order / compiler-fence sweep | All 60 matrix variants match baseline output bytes. The best promising 4608×36864×4096 candidate was neutral/slower in complete renders: 2.261 s control versus 2.274 s candidate; not installed. |
| Channels-last VAE encoding and decoding | Encoding matched, but final decoded pixels changed. Decoder costs did not justify the change; rejected. |
| Encoder-only layout conversion | All complete outputs matched, but paired fresh-input renders were 4.730 s control versus 4.780 s candidate. Inspection confirms the encoder input is already channels-last, so this conversion is a no-op. No encoder change installed. |

Synchronized profiling identifies INT8 projections as the largest remaining
cost, followed by attention, especially in the edit. The VAE is already a small
part of the cached workflow. These experiments do not establish a hardware
ceiling; they establish which tested changes improve this workload exactly.

## Saved images and data

| Text-to-image / stock | Text-to-image / optimized |
|---|---|
| ![Stock text-to-image](data/m5-ultra-flux2-t2i-stock.png) | ![Optimized text-to-image](data/m5-ultra-flux2-t2i-optimized.png) |

| Edit / stock | Edit / optimized |
|---|---|
| ![Stock edit](data/m5-ultra-flux2-edit-stock.png) | ![Optimized edit](data/m5-ultra-flux2-edit-optimized.png) |

Both render the requested scene/edit without gross corruption in the inspected
outputs. The new exact fusions preserve the optimized images; the stock/native
comparison is not a claim of identical quality. Numerical stock/native deltas
are included in the [raw measurements](data/m5-ultra-flux2-klein.json), alongside
all runs, workflow graphs, profiles, hashes, runtime settings, source hashes,
test output and memory guards. Local diagnostics retain all rejected candidates.
