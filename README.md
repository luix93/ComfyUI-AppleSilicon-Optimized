<img width="1684" height="934" alt="banner" src="https://github.com/luix93/ComfyUI-AppleSilicon-Optimized/blob/main/banner.png?raw=true" />

# ComfyUI-AppleSilicon-Optimized

## Latest results — M5 Ultra (80 GPU cores)

**README updated 2026-10-09 · measurements through October 8 · 256 GB unified memory · macOS 27.0 · PyTorch 2.14.0**

<!-- Canonical current benchmark table: update the date, measurements and linked
     report together after validated runs. Keep earlier results below as history. -->

| Workflow | Before (stock) | After (optimized) | Less total time |
|---|---:|---:|---:|
| SDXL, 896×1152 / 12 steps | 2.60 s | **2.42 s** | 7.0% |
| Z-Image Turbo, 1024×1024 / 8 steps | 7.94 s | **3.44 s** | 56.6% |
| Flux.2 Klein 9B, text-to-image / 1024×1024 / 4 steps | 5.59 s† | **2.25 s** | 59.6% |
| Flux.2 Klein 9B, image edit / 1024×1024 / 4 steps | 9.12 s† | **4.66 s** | 48.9% |
| Ideogram 4, 1024×1024 / 12 steps / dual-model CFG | 43.89 s | **18.23 s** | 58.5% |
| Krea2 Turbo, 1448×1448 / 8 steps + LoKr | 28.04 s | **13.69 s** | 51.2% |
| Krea2 Turbo, same workflow without LoKr | 27.92 s | **13.80 s** | 50.6% |
| Qwen-Image 2.1, 1024×1024 / 25 steps | 29.28 s | **12.59 s** | 57.0% |
| Qwen-Image 2.1 edit, 1024×1024 / 40 steps | 50.77 s | **23.08 s** | 54.5% |
| MiniMax H3, 864×480 / 124 frames / 20 steps | 236.75 s | **142.69 s** | 39.7% |
| LTX 2.5, diffusion VAE / 1280×704 / 121 frames / 8+3 steps | 91.42 s | **46.38 s** | 49.3% |
| LTX 2.5, convolutional VAE / same clip and original tiling | 51.71 s | **32.34 s** | 37.5% |
| Wan 2.2 I2V 14B×2, 512×512 / 81 frames / 20 steps / CFG 3.5 | 423.62 s | **302.65 s** | 28.6% |
| Wan 2.2 I2V 14B×2, same clip / 4-step LoRAs / CFG 1 | 45.76 s | **32.36 s** | 29.3% |
| SeedVR2 7B INT8, 2400×1600 → 7200×4800 / 1 step | 74.02 s‡ | **49.01 s** | 33.8% |
| YuE2 3B INT8, controlled 8 s / 128 ABC tokens / 32 steps | 124.05 s§ | **5.60 s** | 95.5% |

Warm end-to-end averages, including sampling, VAE decode and file export;
model loading and unchanged text/reference encoding are cached, except YuE2,
which regenerates score/music tokens and conditioning on every run. Both configurations use **PyTorch
2.14 attention** and matching workflows, prompts and seeds. **Stock** has this
custom node absent; **optimized** enables its native kernels and exact fusions.
The optional relaxed H3 preset is **off**. H3 sampling averages **6.58 s/step**.

**YuE2 · full-song results:** the saved `audio_yue2_text2music.json` now
produces the same **83.2-second song in 33.48 seconds** on this M5 Ultra.
The latest paired warm test improves **47.95 → 33.48 s**, **30.2% less time**,
against the preceding exact GPU implementation. These are complete workflow
times, regenerating score/music tokens, conditioning, sampling, decode and FLAC.

| YuE2 matched GPU workflow | Before | After | Less total time |
|---|---:|---:|---:|
| Full saved graph, October 7 first exact pass | 354.18 s | 48.46 s | 86.3% |
| Same full graph, October 8 paired RoPE follow-up | 47.95 s | **33.48 s** | **30.2%** |
| Mothersuperior AR + real-audio v4 NAR adapters, 20 s cap | 30.95 s | **21.18 s** | **31.6%** |

The historical first pass and latest follow-up are separate experiments; the
full-song rows compare GPU implementations, **not a measured stock CPU song**.
Both accepted passes preserve the earlier GPU output bytes. Phase-local reuse
of dequantized/cast weights removes repeated ConvRot/casts; vocabulary slicing
keeps the permitted logits and full CPU sampling/RNG vector. Scoped acoustic
query reuse removes repeated loads and an output-layout pass. Fused single-token
Q/K RoPE preserves native FP32 intermediate rounding and reuses the frequency
matrix across the 28 layers. Original matrix math, PyTorch autoregressive
attention, CPU RNG and sampling remain. Score/music tokens, conditioning, latents,
waveform and **saved FLAC bytes** match. The original **120 s maximum** and
8192-token score allowance remain; this seed ends naturally at **83.2 s**.
Peak measured MPS driver memory: **13.67 GiB**. Enabled by default on the validated
M5 Ultra / PyTorch 2.14 build; `ASFP8_YUE2_ROPE=0` disables just the new route.
[Latest paired benchmarks and screened candidates](benchmarks/m5-ultra-yue2-followup.md).
The [October 7 first pass](benchmarks/m5-ultra-yue2.md) took the preceding GPU
stack from **354.18 to 48.46 s**; those earlier measurements remain documented.
For instrumental music, a [Mothersuperior two-LoRA workflow](docs/yue2-mothersuperior.md)
is available with the AR planner in CLIP and the real-audio NAR adapter in MODEL.
The native loaders work with our INT8 checkpoint; paired 20-second checks now
improve **30.95 → 21.18 s**, with all adapter weights loaded and identical audio bytes.

**Optional H3 attention update · October 8:** matched 20-step / 124-frame
(5.17 s) tests now include a camera push-in with rigid background objects,
at both 864×480 and 1344×768. These are comparisons with **our exact optimized
stack**, not new vanilla baselines; cache, token reduction and relaxed VAE are off.

| H3 attention mode | 864×480 E2E | 1344×768 E2E | Less time vs exact optimized |
|---|---:|---:|---:|
| Exact optimized reference | 143.20 s | 587.48 s | — |
| Previous SOL INT8 Fast | 114.40 s | 378.92 s | 20.1% / 35.5% |
| SOL INT8 Fast, new layout | **113.74 s** | **372.87 s** | 20.6% / 36.5% |
| New SOL INT8 Hybrid Fast | **113.04 s** | **365.41 s** | 21.1% / 37.8% |
| New Dense INT8 Protected | 137.86 s | 493.25 s | 3.7% / 16.0% |
| New Dense INT8 Full Draft | 133.58 s | 443.89 s | 6.7% / 24.4% |

Each new mode repeats at both sizes. The SOL layout refinement preserves the
previous SOL's complete output bytes; an alternating same-server 480p test
measured **114.78 → 113.67 s**, a small 1.11 s gain. SOL remains non-bit-exact
relative to native attention. Hybrid adds about **13.5 s** over previous Fast
at the larger size, with changed fine detail. Dense Protected retains the
reference composition more closely in this study but is slower than Fast;
**Dense Full Draft changed window placement and picture count** at the larger
size. All new modes are optional, and defaults remain unchanged.
Every attention preset received a temporal audit. No gross abrupt background
replacement was reproduced in this scene; **Top-k Wide still retains its
user-reported morphing warning**. These tests do not establish quality for every
prompt, Turbo, RefMod or combined accelerator configuration.
[Full timings, numerical errors, all-preset visual comparisons and rejected experiments](benchmarks/m5-ultra-h3-kitchen-attention.md).

**Also included in this update:** [Prompt Studio](docs/prompt-studio.md) sends
connected references with the brief, offers a single Generate on queue toggle,
and caches discovered models for the canvas model picker. [H3 controls](docs/h3-acceleration.md)
include the Simple recipe node, independent cache/attention/token controls,
pack-owned MotionCache/Spectrum/Veda runtimes and configurable attention memory.
[H3 LoRA](docs/h3-acceleration.md) combines ordinary and Turbo adapters in an
ordered list with per-entry enable/weight controls. [SeedVR2's video safeguard](benchmarks/m5-ultra-seedvr2-video-memory.md)
retains the saved 7B model and full context, completing the tested 5/10-second
upscales without the prior accumulation of temporary memory. Workflows,
benchmark evidence and bundled-source licenses ship alongside the runtime.

**Optional H3 SOL/token update · October 7:** two new native Metal top-k
presets reduce complete 20-step / 124-frame inference on this **M5 Ultra 80 GPU**:

| H3 optional mode | 864×480 E2E | 1344×768 E2E | Less time vs exact optimized |
|---|---:|---:|---:|
| Exact optimized reference | 140.85 s | 569.58 s | — |
| Previous SOL14 | 113.16 s | 375.32 s | 19.7% / 34.1% |
| New Top-k Wide | **106.55 s** | **332.14 s** | 24.3% / 41.7% |
| New All-step Draft | **93.79 s** | **247.46 s** | 33.4% / 56.6% |

Wide/Draft each repeat twice at both sizes. **The sparse modes are non-bit-exact**;
Wide changes geometry/detail and Draft changes composition more substantially.
**Quality feedback · October 7:** Top-k Wide morphed the background in a user
test where **SOL INT8 Fast** remained stable. Wide remains experimental; use
Fast for that workflow. The sampled benchmark frames did not establish temporal
stability across clips.
Global/local blocks and the pooled tail remain. New token **Protected / Balanced**
recipes add dense step protection; individually they measured
**134.93 / 114.42 s** at 864×480 (single full runs,
with repeated short checks). Existing defaults, tau SOL, Veda and Light/Fast
remain available. These are optional comparisons against exact optimized,
not new stock baselines; Turbo quality and arbitrary combinations are not qualified.
[Settings, complete timings, numerical errors and generated comparisons](benchmarks/m5-ultra-h3-sol-topk.md).

**H3 runtime ownership · October 5:** MotionCache, Spectrum and Veda now use
locally maintained internal runtimes with direct APIs for our combined node.
The existing presets and algorithms are retained; upstream node wrappers and
unused media-analysis tools are removed from execution. Pinned sources remain
parity references, with original attribution and licenses preserved.
[Runtime details](docs/h3-acceleration.md#internal-runtime-ownership)
· [Byte parity and matched timings](benchmarks/m5-ultra-h3-owned-runtime.md).

**New optional Veda attention · October 5:** matched 864×480 / 124-frame /
20-step H3 tests on this M5 Ultra average **144.72 s Exact → 122.42 s Veda
protected (15.4% less)** or **108.46 s Veda fast (25.1% less)**. Both are
repeated warm complete-clip measurements against the current exact optimized
stack. Eight-step Turbo: **70.37 s Exact → 55.88 s Veda fast (20.6% less)**.
**Veda fast + MotionCache: 84.37 s (41.7% less)**, two full repeats;
cache reuse compounds the approximation.
The native Metal route fuses tile conversion/statistics and avoids upstream
MLX's expanded K/V copies. The fast mode changed color, geometry and motion
substantially; protected retained this reference more closely. Existing
defaults and exact/SOL/cache routes remain available. The two optional Veda
starting points live in the existing H3 acceleration node and require the
released predictor in `models/veda`. Short sequences stay dense by default.
[Results, errors and generated comparisons](benchmarks/m5-ultra-h3-veda.md)
· [Setup and controls](docs/h3-acceleration.md#veda-learned-sparse-attention-optional).
The stock/optimized table above retains its previously published measurements; these
are additional optional-mode comparisons, not a fresh stock baseline.

**New H3 methods · September 29:** on the same 864×480 / 124-frame / 20-step
workflow, today’s exact optimized reference averages **142.95 s**. Optional
**Spectrum fast: 84.13 s (41.1% less time)**; **Spectrum protected: 97.31 s
(31.9% less)**; **MotionCache: 103.53 s (27.6% less)**; calibrated **SPEED:
110.66 s (22.6% less)**. Each is a repeated warm full-clip measurement. All
are non-bit-exact. Fast Spectrum changes composition; protected Spectrum and
MotionCache retain this reference more closely. Spectrum fast is only about
4.2% faster than the earlier 87.83 s FirstBlockCache preset below, not another
41% improvement on top of it. The existing H3 node now offers optional
Spectrum/MotionCache implementations bundled in the node, with independent
SOL and token-reduction switches; exact remains available.
[Updated controls and bundling details](docs/h3-acceleration.md). [Results, visual comparisons, new-method survey and workflows](benchmarks/m5-ultra-h3-recent-methods.md).

**Optional H3 acceleration update:** against a matched **144.80 s** warmed exact reference,
14-step INT8 SOL averages **115.23 s**; SOL14 tau 0.5 plus limited FirstBlockCache
averages **98.44 s (32.0% less time)**; FirstBlockCache 0.20 alone
averages **87.83 s (39.3% less)**. The latter two are repeated full-clip
measurements. The all-three combination measured **79.08 s once**, with substantially larger
composition changes. All are **non-bit-exact, independently selectable** and preserve the
exact option. Cache/token execution on current ComfyUI is fixed; the new step-based window
protects exactly the first four and final two of 20 steps. [Results, controls and visual tradeoffs](benchmarks/m5-ultra-h3-sol-aggressive.md).
The [previous conservative seven-step SOL results](benchmarks/m5-ultra-h3-sol-hybrid.md) remain available.

**H3 Latent Upscale · Apple Silicon:** a new MPS node supports the learned
2D/3D checkpoint formats and preserves nested H3 audio. The tested **3D FP16**
checkpoint upscales **864×480 → 1728×960** latents in **0.43 s for one second**
or **1.85 s for five seconds with full context**, excluding VAE decode.
The port matches upstream FP16 output bytes under the same temporal settings.
FP16 is recommended; the same checkpoint takes **0.44 / 1.69 s** in BF16 / FP32
compute on the one-second test. [Setup, precision and memory controls](docs/h3-latent-upscaler.md).

**15-second H3 update:** extending the exact query-cache limit reduces sampling
from **41.66 to 36.85 s/step** (11.5%) at 864×480 / 362 frames. This is an
alternating **two-step** comparison against the previous optimized stack,
not a new 20-step total. Latents, video and audio are byte-identical.
[Long-clip test and token-limit fix](benchmarks/m5-ultra-h3-long-qcache.md).

**SeedVR2 video memory fix:** the saved **7B INT8**, 2× upscale to **832×1472**
now completes **5 s / 120 frames in 138.41 s** and **10 s / 240 frames in
278.60 s** warm, including video/audio export. Maximum sampled Metal memory
was **36.61 / 47.53 GiB**. Previously, the five-second encode crossed our
48 GiB safety cutoff before its first spatial tile finished. The new waits
inside video tiles are enabled by default; **Split Latent remains off**.
The short before/after test preserves output bytes, and both long clips repeat
identically. These are memory-fix results, not new full-length stock baselines.
[Video memory investigation and results](benchmarks/m5-ultra-seedvr2-video-memory.md).

**Optional Apple Silicon Super Resolution:** new IMAGE and streamed VIDEO nodes
run the MLX-DLSS VSR **2× / 8-bit SDR** model. A **2400×1600 → 4800×3200** image
takes **2.22 s** for the complete warm node call, with **2.40 GiB** peak active
MLX memory. The 416×736 → 832×1472 H3 video source takes **1.88 s for five seconds**
and **3.11 s for ten seconds**, including preparation and HEVC/audio export,
with **0.61 GiB** peak active MLX memory. This is a separate, frame-independent upscaler with different quality
and scale from SeedVR2; it is not another row in the stock/optimized diffusion
comparison. [Setup, limits and validation](docs/super-resolution.md).

**Optional VSR → DLSS neural rendering:** the same nodes now support detail
enhancement after upscaling, while keeping the final 2× dimensions. On this
**M5 Ultra 80 GPU**, a **1024² → 2048²** image takes **0.71 s VSR-only**, **1.40 s
with NR at internal scale 1**, or **3.07 s with NR at internal scale 2**. The
five-second H3 source above takes **44.11 s** with temporal NR at internal scale 2,
using **3.09 GiB** peak active MLX memory. These are complete warm node timings;
NR is an optional appearance-changing stage, disabled in existing workflows.
[Settings and example workflows](docs/super-resolution.md#optional-neural-rendering-vsr-first-detail-second)
· [Measurements](benchmarks/data/m5-ultra-vsr-neural-rendering.json).

† Klein's checkpoint omits its INT8 format metadata. Its stock baseline uses
only the existing metadata repair needed to load it, with acceleration absent.

‡ SeedVR2 stock retains only the tiled-VAE synchronization safeguard against
temporary-memory accumulation. Its timing includes recomputing input encoding
on every run. The new exact VAE cleanup saves only about 0.2 s over our
previous optimized stack; the main gains were already implemented.

§ YuE2 stock uses ComfyUI's default CPU FP32 encoder; optimized uses the existing
MPS BF16 encoder plus the new exact patches. The controlled short screen truncates
the score at 128 tokens and generates 8 s of music, rerunning all stages. It is
not the full saved song or a quality example. Stock CPU and GPU arithmetic differ;
the new patches are bit-exact against the preceding GPU stack. No full stock CPU
song timing is claimed.

The complete stock and optimized stacks are **not bit-exact to each other**;
new exact fusions are validated against the otherwise unchanged optimized
stack. See the [full benchmark, numerical checks and saved workflows](benchmarks/m5-ultra-torch214-production.md),
the [Qwen 2.1 results](benchmarks/m5-ultra-qwen21.md),
the [Flux.2 Klein results](benchmarks/m5-ultra-flux2-klein.md),
the [40-step image-edit investigation](benchmarks/m5-ultra-qwen21-edit.md),
the [LTX 2.5 video benchmark](benchmarks/m5-ultra-ltx25.md),
the [further exact kernels and convolutional VAE comparison](benchmarks/m5-ultra-ltx25-followup.md),
the [Wan 2.2 original and four-step LoRA results](benchmarks/m5-ultra-wan22.md),
the [Ideogram 4 benchmark and exact attention changes](benchmarks/m5-ultra-ideogram4.md),
the [SeedVR2 2.14 retest and memory checks](benchmarks/m5-ultra-seedvr2-torch214.md),
and [startup settings](#pytorch-214-on-m5-ultra). This table tracks the latest
validated results; earlier measurements remain below for reference.

## TL;DR

**What this is:** one ComfyUI custom node that makes modern quantized models
(FP8 / INT8) actually work on a Mac — and then makes them fast. Drop it into
`custom_nodes`, restart, done. No model conversion, no forked ComfyUI, no
settings to learn.

**What it fixes:** out of the box, most FP8/INT8 checkpoints either crash on
MPS or fall back to paths so slow they're unusable. The compatibility layer
(automatic, any M1→M5) makes them load and render correctly.

**What it speeds up:** with one environment variable (`ASFP8_NATIVE=1`),
custom Metal kernels take over the heavy math — attention, the quantized
matrix multiplies, and each model family's glue code. Image models render in
roughly **half** their stock time; long video generation becomes genuinely
viable on a laptop. On an M5 Max, Z-Image runs ~7.6 s, Krea2 ~14 s,
Ideogram 4 ~41 s, and an 81-frame Wan 2.2 video ~111 s — matching or beating
the fastest native Mac apps (full table below). These are measured results
for specific hardware, software and kernel designs; the tuning experiments
below do not establish an absolute hardware limit.

**What to expect:** identical or visually identical output (most fast paths
are bit-exact or within one rounding step of stock math — see
[Numerics policy](#numerics-policy)), silent per-call fallback to stock
behavior for anything unrecognized, a **Free RAM** button in the top bar to
hand memory back to the system, and honest engineering notes: things that
didn't work are documented alongside the things that did.

---

**A full Apple Silicon inference stack for ComfyUI: quantized-model
compatibility plus custom Metal 4.1 kernels — flash attention, int8/fp8
matmuls, per-architecture DiT block fusions, and video-VAE kernels — tuned on
M5 Max and M5 Ultra.**

Private continuation of
[ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8).
What began as an FP8 compatibility shim has diverged into a general
optimization layer: 34 runtime patches and seven JIT-compiled Metal extensions
that take image models to roughly half their stock render time and make long
video generation on a MacBook competitive with — and in places faster than —
the fastest native Mac apps.

No model conversion, no forked ComfyUI: one custom node applies everything at
startup. Every fast path checks its inputs per call and **falls back to the
eager implementation** on anything it doesn't recognize. Most native routes
are bit-exact or exact-or-1-ulp; the deliberate H3 W8A8 seam and every
workflow-level approximation are called out separately in the
[Numerics policy](#numerics-policy).

## PyTorch 2.14 on M5 Ultra

The separate `ComfyUI-AppleSilicon-2.14` installation now runs PyTorch 2.14.0
and torchvision 0.29.0. The original installation remains on 2.13.
The clone keeps native INT8/DiT/VAE kernels and selects PyTorch's new MPS
attention with these startup settings:

```text
ASFP8_NATIVE=1
ASFP8_ATTN_BACKEND=pytorch
MTLFLASHATTN_SDPA=off
```

Use `--use-pytorch-cross-attention` in ComfyUI's launch arguments. The explicit
backend setting routes supported calls directly to 2.14 SDPA, including Krea2's
native grouped heads, while retaining fused Q/K normalization, RoPE, layouts,
quantization and projections. Third-party attention overrides retain priority.
Unsupported shapes retain the existing router. Removing the backend setting
restores that router; it does not restore 2.13 arithmetic or output bytes.
The earlier H3 relaxed preset remains separate and off by default.

Native extensions use PyTorch's compiler-standard default (C++20 on 2.14),
with build caches separated by PyTorch version/build and installation path.
Krea2's accepted exact normalization and elementwise fusions are validated
on 2.14 as well as 2.13. A new exact FP16 GEGLU fusion removes a memory pass
from SDXL; `ASFP8_SDXL_GEGLU=0` disables it. Its 128 KiB lookup table contains
PyTorch's evaluated GELU for every FP16 input bit pattern, retaining the
intermediate rounding before multiplication.

On the validated M5 Ultra / 2.14.0 build, dense BF16 D=128 attention also
keeps query tiles in registers across the key loop. This retains PyTorch's
arithmetic and reduction order. H3's BF16 56-head geometry is validated through
**65,536 tokens**, covering the 44,600-token 15-second reference clip; other
families retain their 32,768-token limit. `ASFP8_MPP_QCACHE=0` restores the unmodified
PyTorch kernel. The specialization checks both the Torch build and installed
header hash, and falls back on unsupported inputs or initialization failure.

The current stock-versus-optimized measurements are at the top of this README.
Query caching alone saves **9.5 s** on H3
against the otherwise unchanged optimized 2.14 stack (152.20 → 142.69 s),
with byte-identical sampler latents, decoded frames and audio. The default
keeps the earlier relaxed H3 modes off. **91 tests passed; 4 skipped.**

Full stock-versus-optimized validation and the rejected experiments are recorded
in the [2.14 production benchmark](benchmarks/m5-ultra-torch214-production.md).
An additional [H3 Sage-style attention screen](benchmarks/m5-ultra-h3-sage.md)
replayed the saved 15,458-token attention inputs: the best tested qlip-style
Metal port took **184.6 ms** with normal compiler settings, versus **64.0 ms**
for the current exact path and **73.2 ms** for unmodified PyTorch 2.14. A
separate fast-math control improved the port to **158.4 ms**, still slower
than stock. All five workgroup sizes lost; the port remains a diagnostic
and is not enabled in the H3 preset.
Stock here means the acceleration node is absent, with PyTorch attention
selected in both configurations. The complete stacks are **not bit-exact to
each other**: quantized compute and text-encoder placement differ, and changing
attention backends also changes output bytes. New exact fusions are compared
against the otherwise unchanged optimized stack.

## LTX 2.5 on M5 Ultra

The saved two-pass text-to-video workflow averages **91.42 s stock → 46.38 s
optimized**, including audio, video decoding and MP4 export. It retains
121 frames, 24 fps, the original prompt/seeds and **8+3 sampling steps**.
The spatial selector and latent rounding produce **1280×704** final frames.

That is **0.901 MP**, 14.1% fewer pixels than the earlier M5 Max's
1024×1024 LTX 2.3 benchmark (86 s). The versions also use different decoder
architectures and attention backends: this workflow uses LTX 2.5's heavier
diffusion VAE. These totals do not establish Max-to-Ultra hardware scaling
or an optimization ceiling; see the comparison notes in the report.

The first exact round reduced the previous native stack from **74.52 s to
48.84 s**. A follow-up adds exact SwiGLU/modulation fusions and combined
neighborhood copies: **48.66 s → 46.38 s** in a fresh paired comparison.
Video decoding falls from **21.88 s to 19.59 s**; the earliest native decoder
took **45.69 s**. A Metal mask kernel removes repeated GPU readbacks,
independent attention tiles are batched within a bounded workspace, and fused
Q/K normalization–RoPE removes memory passes. Sampling gains come from
restoring the current split-RoPE interface and extending exact PyTorch
query-load caching. Matrix arithmetic and attention reduction order stay intact.

All full native runs matched byte for byte in both sampler outputs, audio and
final frames; direct sampler/decoder block checks also passed. These additions
are enabled under `ASFP8_NATIVE=1` on the validated M5 Ultra / PyTorch 2.14 build.
`ASFP8_LTX25_FAST=0` restores the preceding native implementation;
`ASFP8_LTX25_VAE=0` disables the decoder routes. The latest additions can be
disabled independently with `ASFP8_LTX25_VAE_SWIGLU=0`,
`ASFP8_LTX25_VAE_MODULATE=0` and `ASFP8_LTX25_VAE_LAYOUT=0`; setting all
three to zero restores this round's 48.66 s paired control.

The optional `ltx-2.5-video-vae-conv-bf16.safetensors` works with the standard
Load VAE node and the original tiling: **51.71 s stock → 32.34 s optimized**.
It is a different decoder with different output pixels, not a bit-exact
replacement for the diffusion VAE. Each decoder repeats its own output bytes.
Using regular **VAE Decode** instead of tiling takes **30.18 s** with the
convolutional VAE; it needs more memory and also changes boundary pixels.
A separate **1024×1024 / 121-frame / 8+3-step** run with the convolutional VAE
and original tiling averages **37.82 s**. It is a measured resolution comparison;
the historical M5 Max LTX 2.3 run still differs in model and backend.
The [initial report](benchmarks/m5-ultra-ltx25.md) and
[follow-up comparison](benchmarks/m5-ultra-ltx25-followup.md) record timings,
numerical checks, optional settings, rejected experiments and raw results.

## Flux.2 Klein 9B on M5 Ultra

The original four-step Euler / Flux2Scheduler workflows average **5.59 → 2.25 s**
for text-to-image and **9.12 → 4.66 s** for a reference-image edit at 1024×1024.
Warm timings include sampling, VAE decoding and PNG export; model loading and
unchanged input/text encoding are cached. The stock comparison needs only the
existing missing-format metadata repair for the supplied INT8 ConvRot checkpoint.
When the image and prompt are re-encoded on every run, the edit averages
**12.52 → 5.92 s**, with model loading still warm.

Most of the gain comes from the existing native INT8 projections, fused Flux
Q/K normalization–RoPE, and exact PyTorch attention query caching. New BF16
SwiGLU and activation/concatenation kernels give a smaller additional gain:
**2.277 → 2.254 s** for text-to-image and **4.702 → 4.655 s** for editing in
paired runs. The new kernels preserve the preceding native outputs byte for
byte; the full stock/native stacks already differ numerically.

`ASFP8_FLUX2_GLUE=0` disables the new additions. Standalone activation and
single-block activation/layout fusion can be disabled independently with
`ASFP8_FLUX2_SWIGLU=0` and `ASFP8_FLUX2_MLP_LAYOUT=0`. They default on only
under the existing Flux fast path on the validated M5 Ultra / PyTorch 2.14 build.
The [full report](benchmarks/m5-ultra-flux2-klein.md) includes fresh-input costs,
exactness checks, saved images and rejected matrix/VAE experiments.

## Qwen-Image 2.1 on M5 Ultra

The saved 1024×1024 / 25-step workflow now averages **29.28 s stock → 12.59 s
optimized**. The initial exact fusion pass reduced the repaired native stack
from **15.29 s to 13.15 s**; the subsequent exact VAE scheduling improvement
brings it to **12.59 s**, preserving the same complete latents and decoded pixels.

The INT8 ConvRot text encoder now works with the node's MPS placement patch.
ComfyUI's embedding cast can return an already-dequantized BF16 tensor;
its INT8 embedding path previously attempted to dequantize that tensor again,
raising `NoCapableBackendError: dequantize_int8_embedding`.
The default compatibility repair gathers those floating rows directly, while
retaining ordinary INT8 row decoding when the quantized layout survives.
Checkpoint files, scales, inverse ConvRot and output dtype are preserved.
Vanilla ComfyUI keeps this encoder on CPU and does not encounter that transfer bug.

With native kernels enabled on the validated M5 Ultra / PyTorch 2.14.0 build,
Qwen-Image 2.1 also gets exact **Q/K RMSNorm + RoPE**, **modulation**, and
**gated residual** fusions. LayerNorm, matrix multiplies, attention and ComfyUI's
existing prefix cache retain their arithmetic and behavior. The new fusions
preserve the repaired native stack's complete latents and decoded pixels;
the stock and native stacks themselves are not bit-exact.

The saved **1024×1024 / 40-step image edit** now takes **23.08 s** with the
native stack. New exact attention query caching/output layout and VAE
normalization scheduling save **1.66 s / 6.7%** against the previous native
stack's paired **24.74 s** control. Complete sampler latents and decoded
pixels match byte for byte. Repeated decoding of identical saved latents
improved from **0.788 s to 0.343 s**; decode inside full renders averages
**0.441 s**. The Qwen VAE schedule also covers reference-image encoding.

The tested qlip-style Metal Sage attention port was slower than 2.14 SDPA
at this edit's actual shapes, so it is not installed. Comfy Kitchen 0.2.35
INT8 attention has CUDA/HIP implementations and rejects MPS tensors;
using it for Qwen or H3 on a Mac would require a new Metal backend. The
[image-edit report](benchmarks/m5-ultra-qwen21-edit.md) records timings,
compiler controls, numerical checks, scope and rejected experiments.

`ASFP8_QWEN21_FAST=0` disables these new fusions while keeping the encoder
repair and existing generic native kernels. The individual `ASFP8_QWEN21_ROPE`,
`ASFP8_QWEN21_MODULATE`, `ASFP8_QWEN21_RESIDUAL`, `ASFP8_QWEN21_ATTN` and
`ASFP8_QWEN21_VAE` switches can each be set to
`0`. Other hardware, Torch builds, dtypes, shapes and attention hooks retain
the original path. The [Qwen 2.1 benchmark](benchmarks/m5-ultra-qwen21.md)
records the unchanged workflow, warm measurements, byte checks and rejected
attention, feed-forward and VAE experiments.

## Results

Earlier measurements and experiments follow. The current M5 Ultra / PyTorch
2.14 stock-versus-optimized comparison is at the top of this README.

Measured on an M5 Max (128 GB), macOS 27 dev beta, warm model cache. The
"before" column is this node's compatibility layer alone (already several
times faster than a crash, which is the stock experience for most of these
models); "after" is with `ASFP8_NATIVE=1`.

| Workflow | Before | After | Reference point |
|---|---|---|---|
| Z-Image (Lumina) int8, 1MP | 18.0 s | **7.6 s** | — |
| Flux2 Klein 9B int8, 1MP | ~24 s | **~13 s** | — |
| Krea2 Turbo int8 convrot, 1MP / 8 steps | ~28 s | **~14.3 s** | DrawThings ~13.5–14 s |
| Ideogram 4 int8 convrot (8.7B), 1MP / 12 steps | ~50 s† | **~41 s** | DrawThings ~42 s |
| LTX 2.3 video, 512² × 5 s | 63 s | **22 s** | DrawThings ~35 s |
| LTX 2.3 video, 1024² × 5 s | 295 s | **86 s** | DrawThings ~120 s |
| Wan 2.2 i2v 14B×2 int8, 512² × 81 f / 4-step LoRAs / CFG 1 | 445 s | **111 s** | DrawThings-class ≈ 100 s |
| MiniMax H3 int8, 864×480 × 124 f / 20 steps | 490.4 s‡ | **461.3 s** | exact paths only; workflow approximations off |

**SDXL on M5 Ultra / PyTorch 2.13 (2026-09-25):** the embedded 896×1152, 12-step workflow
replays in **3.1–3.2 s warm**, matching the original PNG pixels. Its FP16 UNet
uses a different path from the INT8 models below. Exact projection/scheduling
experiments were neutral. A diagnostic FlashAttention route measured
**3.12 → 2.90 s**, but changed latents and images and remains uninstalled.
See the [SDXL investigation](benchmarks/m5-ultra-sdxl-investigation.md) for
profiling, controls, numerical errors and the limits of the Max/Ultra comparison.
An isolated follow-up with **PyTorch 2.14** measured **3.15 → 2.77 s** with
the current router and **2.39 s** with raw PyTorch SDPA, approximately **24%
less total time**. Both 2.14 routes change the reference output bytes; the
original environment remains on 2.13; the separate 2.14 clone is described above. See the [version comparison](benchmarks/m5-ultra-sdxl-investigation.md#follow-up-pytorch-213-vs-214)
for repeatability, runtime metadata and the routing change needed to reach 2.39 s.

**Matched PyTorch 2.14 tests on M5 Ultra (2026-09-25):** with the original
workflows, prompts, seeds and warmup policy, raw PyTorch SDPA measured
**Z-Image 1024²/8 steps: 3.82 → 3.51 s**, **Krea2 1448²/8 steps:
15.05 → 14.01 s**, and **H3 864×480/124 frames/20 steps: 163.00 → 152.32 s**
end to end against fresh 2.13 controls. H3 sampling averages **7.05 s/step**;
FP16 video decode improves from **13.21 to 10.26 s**. Krea2's result required
re-enabling its version-gated fusions in the isolated 2.14 copy; operator and
complete-output checks passed. The production clone now includes that change.
Without that compatibility change, 2.14/raw SDPA takes **15.90 s** for Krea2.
The isolated native extensions also required **C++20** instead of the older
loaders' C++17 setting. Switching attention changes output bytes and visible
details. That initial investigation left defaults unchanged; the separate
2.14 clone described above now uses the tested stack. See the
[complete comparison](benchmarks/m5-ultra-torch214-models.md) and
[frozen API workflows](benchmarks/data/m5-ultra-torch214-workflows.json).

**M5 Ultra validation (2026-09-22):** Z-Image Turbo INT8 convrot, 1024×1024,
8 res_multistep/simple steps, improved from **7.33 s to 4.29 s** end to end in
warm, alternating runs on macOS 27.0 (26A428), PyTorch 2.13.0, ComfyUI
`b33e2b55`. Two fixes are required: override ComfyUI's newly added MPS INT8
hardware exclusion for our native implementation, and use a 16-wide inner
GEMM step on M5 Ultra. The latter preserves every output pixel against the
previous native INT8 schedule; other chips retain their existing schedule.
Restoring INT8 routing resumes the existing W8A8 numerical policy, rather than
the W8A16 fallback that newer ComfyUI had selected. Explicit full-precision
requests are still honored. This does not establish whether the kernel
slowdown originated in a macOS update or Ultra-specific compiler behavior.

Further Ultra tuning reduced warm Z-Image runs from **4.23 s to 3.75 s**
end to end. A fixed **864×480, 22-frame, 20-step MiniMax H3** benchmark
reduced sampling from **21.4 s to 19.4 s**. The Ultra schedule now uses
BK=256 with two-way unrolling, small-row GEMM through 128 rows, and
SIMD-local Q/K normalization with the original reduction/rounding order.
Attention pads Q to each active 16-row slice, avoiding unnecessary copies.
Video/audio latents, decoded video/audio, and Z-Image pixels matched the
previous native build exactly. Longer H3 operator benchmarks improved by
20–28%; a full 5-second clip was deliberately not benchmarked in this round.

The subsequent **124-frame, 864×480, 20-step H3** benchmark reduced warm
sampling from **193.99 s to 172.66 s** (**9.70 → 8.63 s/step**), and total
execution from **211.57 s to 190.34 s**. Long H3 attention on M5 Ultra now
interleaves query tiles and heads in Morton order, with shape specialization
for ragged lengths. Aligned lengths retain dynamic dimensions to avoid a
compiler rounding change. The long H3 feed-forward bridge also uses larger,
capped scratch tiles. All video/audio latent bytes and decoded frames matched
the preceding native build exactly; **106 tests passed**. Z-Image remained
at **3.81 s** warm with identical pixels. These are another **11% less sampling
time**, not an additional halving; first-run model loading is excluded from
the warm comparison.

A subsequent exact attention synchronization pass reduced a fresh, warmed H3
control from **8.34 to 7.68 s/step**: **166.75 → 153.51 s** sampling and
**184.38 → 171.08 s** total. Periodic threadgroup barriers are enabled only for
the long Ultra H3 path, with full query-tile padding so inactive tail groups
participate safely. All latent bytes and decoded video/audio frames matched;
34 attention boundary tests passed. This pass saves another **13.29 s** in the
matched workflow; it does not establish 7.0 s/step or another 30–50 s reduction.

Further long-sequence query ordering reduced a fresh warm control from
**7.48 to 7.31 s/step**: **149.53 → 146.13 s** sampling and
**167.09 → 163.65 s** total. For H3 sequences of at least 12,288 tokens,
up to 64 adjacent query tiles run together before the remaining query/head
bits are interleaved; shorter sequences retain the previous traversal.
All latent bytes and decoded video/audio frames matched, and **42 attention
checks passed**, including both dtypes and batch size two above the new cutoff.
This saves another **3.44 s** end to end in the matched 20-step workflow;
7.0 s/step remains unachieved. Alternative attention layouts, larger TensorOps,
and a single-SIMD feed-forward bridge were slower or neutral and were not retained.

The next Ultra research pass tested **19 attention/INT8/feed-forward variants**
without retaining a new production change. FC1/SwiGLU fusion, shared activation
loads, smaller tiles, shared query storage and interleaved FC1 output layouts
were exact but slower; attention loop unrolling was neutral. Separating the
ragged attention tail gained roughly 3–4% in operator tests but changed output
bytes, so it was rejected. Lossless INT8 weight packing improved alternating
warm two-step H3 probes from **7.489 to 7.445 s/step** (0.58%), but required an
additional **17.94 GiB** for 200 packed weights. All latent bytes matched, but
that tradeoff was not promoted. These short probes are not a new full 20-step
result: the last verified installed result remains **7.31 s/step** above.
The public Metal counter API exposed timestamps only, so register pressure and
memory-stall explanations remain hypotheses rather than counter measurements.

The FP16 H3 video-VAE sprint then reduced **decode-only** time from **2.351 s
to 1.875 s** (**20.3% less time**) on a fixed **864×480, 22-frame** latent
reference. These are medians of eight alternating warmed comparisons; model
loading and compilation are excluded. The accepted Metal kernels fuse the
decoder's head-interleaved FP16 QKV normalization/RoPE/layout conversion,
standalone SwiGLU with its intermediate FP16 rounding, and attention-output
head merging with `nan_to_num`. Existing matrix multiplies, the selected
attention backend, spatial tiling, overlap and temporal chunking are preserved.
All output bytes matched for Q/K/V, 72 captured decoder-block cases, 72 further
batch/sequence cases, and every complete decoded FP32 frame tensor. SwiGLU
was also checked across all 63,488 finite FP16 bit patterns with five multipliers,
and the layout/sanitization pass across all 65,536 FP16 bit patterns.
A residual-plus-following-RMSNorm fusion was exact but had no repeatable
incremental decode gain in ten alternating comparisons, so it was not retained.
Those initial measurements used a short decode. The decoder fast path is
enabled under `ASFP8_NATIVE=1` / `ASFP8_DIT_EXT=1`; set
`ASFP8_MINIMAX_VAE_FAST=0` to restore the core decoder forwards. Unsupported
dtypes, layouts, training calls and unrecognized decoder/rotary APIs fall back.

Full **124-frame, 864×480, 20-step** validation then measured **166.55 →
162.67 s end to end**, with video decode falling from **16.71 → 13.24 s**
(**20.8% less decode time**). Both paths were warmed before this matched pair;
the two optimized full executions took **162.82 and 162.67 s**, with **13.22
and 13.24 s** decodes. Only the exact VAE fusions were toggled. Sampling was
**148.70 → 148.34 s** (about **7.42 s/step** in this session); that small
variation is not a new sampling improvement. The measured saving is **3.88 s
end to end (2.3%)**, of which **3.48 s** comes from video decode. Every raw
FP32 frame byte matched the baseline, and all video/audio latents plus decoded
exported video/audio matched the previous production output. E2E includes
audio decode and MP4 export; post-run byte checks are outside the timer.

The next **bit-exact sampling sprint** kept those VAE changes frozen. An
adapted final-layer wrapper and kernel matched current ComfyUI's FP32
modulation and new arguments, reducing a captured complete layer from
**0.921 to 0.404 ms**. However, alternating one-second, 20-step workflow
medians were **19.274 vs 19.267 s**—within run-to-run variation. The candidate
is archived rather than enabled as a claimed sampler speedup; **31 regression
checks passed**. Fused normalization/modulation/rotation/INT8 quantization
improved preprocessing by about 10%, but the 20-step check found **six
different INT8 values across 2,000 calls**, so it was rejected. Four exact
attention register/compiler variants were neutral or slower; two exact INT32
partial-accumulation schedules were slower. See the
[sprint-two benchmark report](benchmarks/m5-ultra-sampling-sprint2.md).

Four warmed **124-frame, 864×480, 20-step** confirmation runs averaged
**147.713 s sampling / 162.048 s E2E** for the frozen baseline and
**147.603 s sampling / 161.839 s E2E** for the final-layer candidate. That
nominal **0.075% sampling difference** is within observed variation, so no
new sampling change was promoted. All latent bytes, raw FP32 frames and
decoded exported video/audio matched the previous production reference.

The **relaxed-kernel sprint** adds an independently optional FP16 H3 VAE
attention path. On identical saved latents, dense FlashAttention reduces the
complete **124-frame, 864×480 decoder from 13.223 to 10.668 s** (four warmed,
alternating pairs: **19.3% less decode time / 2.555 s saved**). Three further
boat/scarf/butterfly short references improve by about 19–20%. Sampling and
projection GEMMs retain their existing implementation. This changes decoded
frame bytes: full-clip pixel RMSE is **0.0000962**, PSNR **80.33 dB** and sampled
SSIM **0.9999973**. Inspected lossless crops and sampled sequences show no
visible detail/color change at the displayed scale; temporal-difference
measurements are close. That scoped assessment is not a guarantee for every
prompt or frame.

**Exact remains the default.** Set `ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1` in
the environment used to launch ComfyUI, then restart to try it. Set it to `0`
or remove it and restart to restore exact attention. Native mode does not
enable this relaxation implicitly. The switch requires the native FP16 VAE
fast path and dense Metal attention extension; failures fall back to the
existing backend. Default/enabled startup checks reproduce their saved outputs,
and **42 VAE/native regression tests pass**. See the
[sprint-three report](benchmarks/m5-ultra-relaxed-sprint3.md) for complete
measurements, visual-assessment limits and the independent tail, softmax and
FP16 GEMM experiments. The latter softmax variants were slower; FC1/SwiGLU
GEMM fusion saved only 0.09 s on the full decoder and was not promoted.

Six warmed **124-frame, 864×480, 20-step** runs confirm **161.24 → 158.64 s
E2E** for the optional dense VAE (**2.60 s / 1.61% saved**). Live video decode
averages **13.157 → 10.707 s**; all sampler latents and decoded audio remain
byte-identical. Both exact controls reproduce the previous raw-frame hash.
The separately tested tail specialization averages **147.022 → 143.936 s
sampling (7.351 → 7.197 s/step)**, but changes same-seed composition, motion
and audio and gives no consistent smaller-workflow gain. In that sprint it stayed
benchmark-only; the two candidates were not combined.

**Optional H3 relaxed preset.** The next kernel round adds
`ASFP8_MINIMAX_RELAXED=1`, set before starting ComfyUI alongside
`ASFP8_NATIVE=1`. It selects the experimental long-H3 attention route, dense
FP16 VAE attention with tuned tiles, and a fused FP16 QKV projection with FP32
accumulation, normalization, RoPE and layout conversion. It does not change
steps, caches, sparsity, tokens, VAE tiling or other model families.
Warmed full-workflow means are **161.13 s exact →
153.35 s with the preset**, versus **155.93 s**
for the previous tail-plus-dense combination. Sampling averages
**7.11 s/step** and video decode **10.15 s**.
This saves **7.78 s E2E** against the matched exact control,
including **2.58 s beyond the earlier experiments**; it does not yet reach
7.0 s/step. See the [sprint-four report](benchmarks/m5-ultra-relaxed-sprint4.md).

Every component remains independently selectable, with explicit settings taking
precedence over the preset:

```bash
export ASFP8_NATIVE=1
export ASFP8_MINIMAX_RELAXED=1
# Optional overrides:
# export ASFP8_MINIMAX_ATTN_MODE=exact       # exact | tail | tuned
# export ASFP8_MINIMAX_VAE_DENSE_ATTENTION=0
# export ASFP8_MINIMAX_VAE_ATTN_TUNING=0     # retain the original dense tiles
# export ASFP8_MINIMAX_VAE_QKV=0             # retain the existing projection
```

**All of these changes remain off by default.** Native mode alone retains the
accepted exact stack. To restore it, remove the relaxed preset and individual
relaxed overrides, then restart. To try only the earlier dense VAE path, leave
the preset unset and set `ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1`. The earlier tail
experiment remains available with `ASFP8_MINIMAX_ATTN_MODE=tail`.

The sampling changes can visibly alter same-seed composition, motion and audio.
They are experimental choices for review, not a claim of perceptual equivalence.
The exact route stays available. Previous rejected kernel experiments remain
separately runnable in the diagnostic harness; slower candidates are not enabled
by the preset.

**INT8 ConvRot video VAE on M5.** Loading
`minimax_h3_video_vae_int8_convrot.safetensors` with `ASFP8_NATIVE=1` now selects
a guarded native W8A8 decoder path. The checkpoint alone previously used
ComfyUI's MPS weight-only fallback, repeatedly dequantizing weights into FP16.
The new route rotates and quantizes activations, keeps INT32 matrix accumulation,
and fuses channelwise scaling and two separately rounded FP16 bias/output stages
into the matrix kernel. The existing VAE Q/K normalization, RoPE, layout and
SwiGLU kernels also accept these eligible quantized layers.

This is **not bit-exact to the FP16 checkpoint or to the prior weight-only
fallback**. `ASFP8_MINIMAX_VAE_INT8=0` restores that previous INT8-checkpoint path;
selecting the FP16 VAE retains the existing exact/relaxed FP16 choices.
Dense attention stays independently optional:
`ASFP8_MINIMAX_RELAXED=1` enables the combined preset, or set just
`ASFP8_MINIMAX_VAE_DENSE_ATTENTION=1` and
`ASFP8_MINIMAX_VAE_ATTN_TUNING=1` to relax VAE attention while retaining the
existing sampler. The FP16 QKV projection override is bypassed for quantized
weights. Restart after changing environment settings.

Custom/offloaded layers, runtime weight/bias transforms, hooks, explicit
full-precision requests and unsupported shapes retain the core path. Optional
kernel failures latch back to that path. The model selector determines which
checkpoint loads; the relaxed flag never changes the VAE file automatically.

On the M5 Ultra, repeated complete decodes of the same **864×480, 124-frame**
latents take **13.10 s exact FP16**, **11.28 s native INT8**, and **8.41 s INT8
with dense attention**. The previous relaxed FP16 mode takes **10.11 s**:
the new fastest VAE saves **1.70 s / 16.8%** against it. The unadapted INT8
checkpoint takes **22.29 s** on this MPS setup. All are warm medians of four
alternating repetitions, including allocation, rotation, quantization and layout
costs. In the full 20-step workflow, alternating FP16/INT8/INT8/FP16 runs average
**154.65 s → 152.80 s E2E**, saving **1.85 s** against the previous relaxed
preset. Complete video/audio sampler latents remain byte-identical; this change
speeds up decoding, not sampling. See the
[INT8 VAE report](benchmarks/m5-ultra-int8-vae.md) for individual runs, numerical
differences and validation scope.

The earlier **163.65 s E2E / 146.13 s sampling** result used the same prompt,
seed and workflow. Sampling instrumentation, operation shapes/counts and kernel
sources match the later VAE-sprint run, and controlled old/current attention
binary tests show essentially identical performance. The later **162.67 s**
result is only **0.98 s faster than that historical total**; its 3.88 s saving
above is against its fresh matched control. Historical GPU-clock and background
load telemetry is missing, so the extra sampling time cannot be attributed to
a particular thermal, clock or contention cause.

† generic patches already on — Ideogram 4 shipped after the generic layer, so
its "before" is the pre-#32 state, not compatibility-alone.
‡ H3 isolates the latest exact final-layer graph rewrite against the preceding
already-native build; it is not a compatibility-only comparison. The matched
sampler portion fell from about 445.2 s to 418.2 s.

The LTX numbers beat DrawThings outright at both resolutions; Ideogram 4 edges
past it; Krea2 is within ~5%; Wan 2.2 went 445 s → 111 s (4×), putting an
81-frame 14B×2 video generation on a laptop within ~10% of a DGX Spark-class
result.

**Read benchmarks from run 2.** The first run of any workflow pays JIT kernel
compilation and cold-allocator growth (Wan: ~142 s vs 111 s steady). Sustained
load also drifts these numbers ±10% by thermal state alone — bigger than most
individual optimizations here — so only same-session A/B comparisons are
meaningful (a heat-soaked chassis can drift far past ±10%; let the machine
cool before trusting absolute numbers).

## The two layers

**1. Compatibility (automatic, any M1→M5).** FP8/INT8 checkpoints load and
render instead of crashing: LUT-based bit-exact FP8 decode, CPU reroutes for
the casts MPS can't do, `torch._int_mm` on GPU, flash attention for
long-context correctness/OOM, text encoders moved off the CPU, plus fixes for
macOS-beta psutil, the PiD ≥2048px black image, and WanVideo block swap. No
env vars needed. See [the compatibility layer](#the-compatibility-layer-patches-116).

**2. Native kernels (opt-in, `ASFP8_NATIVE=1`).** Seven ObjC++/Metal
extensions, JIT-built on first use against **Metal 4.1**. Six production
feature groups sit behind the master switch; the seventh (`sol_ext`) builds
only when the experimental H3 sparse-attention node is applied. Developed on
the macOS 27 dev beta; the int8 and fp8 matmuls additionally need an
**M5**-class GPU for cooperative TensorOps / the fp8 format type. Production
routes that cannot build or do not apply fall back silently, so the master
switch is safe to leave on; the explicit sparse node instead refuses to apply
when its required kernel is unavailable.

```bash
xcode-select --install        # Metal toolchain
pip install ninja
export ASFP8_NATIVE=1         # before launching ComfyUI
```

`ASFP8_NATIVE=1` implies `ASFP8_FP8_EXT`, `ASFP8_FP8_SGEMM`,
`ASFP8_INT8_EXT`, `ASFP8_ATTN_ROUTER`, `ASFP8_DIT_EXT`, `ASFP8_CONV3D`.
Explicit per-feature values always win (e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`).

For MiniMax H3, that one switch also activates the eligible native subpaths
behind the INT8 and DiT extensions: direct QKV emission, Q/K RMSNorm + RoPE,
the FC1/SwiGLU/FC2 bridge, cross-block glue, allocation-lifetime releases, and
the guarded final-layer graph rewrite. There is no second H3 master switch.
An adapter is never selected automatically: apply it with `H3 LoRA (Apple Silicon)` in `fast_runtime`
mode, after which the exact LoRA-aware QKV/MLP routes engage per eligible
layer. Cache, token reduction, sparse attention, reduced VAE overlap, and VAE
tile batching remain explicit workflow/experiment choices because they either
change output or have workload-dependent tradeoffs.

## The native fast paths

### flash41 — Metal 4.1 flash attention (`mfa4_ext`)

The workhorse. A cooperative-matmul flash-attention forward kernel written
for Metal 4.1 TensorOps. The kernel skeleton comes from
[mtlflashattn](https://github.com/pawel-mazurkiewicz/mtlflashattn) (MIT); we
port it to a compile path that works on macOS 27, roughly double its speed,
and extend it to fp16/video/GQA — see [Credits](#credits) for the split.

- **Compiles at all on macOS 27.** mtlflashattn JITs through
  `torch.mps.compile_shader`, which no longer exposes `mpp::tensor_ops` there,
  so its fast tiers silently degrade to a fallback *slower than sub-quadratic
  attention*. We compile the kernel through `newLibraryWithSource` at an
  explicit `MTLLanguageVersion4_1` inside an ObjC++ extension, dispatched on
  torch's own MPS stream (no `COMMIT_AND_WAIT` — forcing a sync per call
  serializes the pipeline and was worth seconds per render).
- **Dtype-templated bf16/fp16 I/O** with fp32 math inside — Wan computes in
  fp16, the flux family in bf16; both take the same kernel. Low-precision
  score tiles by default (`ASFP8_FA41_F32S=1` restores fp32 tiles).
- **mergeP**: the P (softmax) tile overwrites the score tile in threadgroup
  memory, halving threadgroup memory so twice as many threadgroups stay
  resident — this is what makes the wide-tile geometries pay.
- **Wide KV tiles + static reduction dims**: BC=128 (vs the original 32) feeds
  the matmul units properly; declaring the reduction extents compile-time
  constant where provably safe removes the dynamic-extent overhead. Together
  with mergeP: 69 ms → 33.7 ms at Flux shapes.
- **Sequence-adaptive tiling**: BR=32 below ~12.8k tokens, BR=64 above, where
  K/V re-reads dominate — worth ~15% at Wan's 21.5k tokens.
- **GQA-native grouped KV**: Krea2's 48Q/12KV heads run without
  `repeat_interleave`, bit-identical to the repeated form.
- **~28–30 TF/s** at Flux shapes, **~26–28 TF/s** at video length, measured
  same-session — the cooperative kernel's practical ceiling (see below), which
  is why the current production forward is the `na` kernel:

**Attention v2 — the `na` kernel.** A structural rewrite of the forward
following DrawThings' NAX recipe (liuliu/ccv `NAAttentionKernel`, BSD-3 — see
[Credits](#credits)): 16 *independent* simdgroups × 16 rows each instead of
one cooperating threadgroup, `execution_simdgroups<1>` tensor ops, K/V read
directly from device memory through static-extent slices, two 64-column
blocks per softmax pass with row-reduction cooperative tensors +
`fast::exp2`, relaxed-precision enabled, and the P tile register-chained into
the PV matmul. **40+ TF/s at Wan shapes (+46% over the cooperative kernel)** —
this is what took Wan 122 → 111 s and LTX 1024² ~100 → 86 s. Two hard-won
details:

- Static-extent tensor-ops slices **never bounds-check** (that's their speed).
  Ragged sequence tails are handled by host-side zero-padding to tile bounds,
  with padded strides passed separately from the true lengths that drive
  masking. An earlier tail path that read past the buffer end passed every
  parity test (fresh MPS pages are zeroed) and still corrupted ~3 of 4 real
  renders on a dirty allocator pool; the permanent regression test now poisons
  the pool with NaN pages first.
- D=256 heads (Ideogram 4) were tried via a two-block head split and **lost to
  plain bmm** (22.9 ms vs 12.4 ms): the 16-row × 256 × fp32 accumulator costs
  128 registers/lane and spills. Budget rule: 16 × D × fp32 / 32 lanes must
  stay ≲64/lane. Wide heads stay on the bmm route.

**How the ceiling moved.** An earlier sprint tried to beat the cooperative
kernel with bigger tiles and failed; the na rewrite then beat it by changing
structure instead. The failed probe is still worth recording:

- A head-count sweep at fixed geometry showed throughput flat from a 55 MB to
  a 440 MB K/V working set, and configurations that *halve* K/V traffic ran
  slower — so the kernel is compute/latency-bound, not bandwidth-bound. Tiling
  harder is not the lever.
- We ported [Cider](https://github.com/Mininglamp-AI/cider)'s manual
  register-fragment machinery (the structure behind our 105 TF/s int8 GEMM) to
  fp16 as a standalone probe. The fragment register layout *is* valid for fp16
  (parity 2e-5), but at attention's shallow K=128 reductions even **MPS's own
  tuned GEMM only reaches 34.5 TF/s** (manual fragments: 24; at deep K=512 they
  reach 55). The int8 win came from a regime that does not exist inside
  attention. (The na kernel later cleared that "ceiling" anyway — it was a
  bound on that *design family*, not the hardware.)
- `flash_attn_fwd_frag` — a staged-KV per-simdgroup kernel (K/V staged once per
  threadgroup, softmax state in lane registers via `simd_shuffle`, optional
  double-buffered prefetch) — ships in the source, parity-clean across
  fp16/bf16/causal/GQA/tails, but is **not routed**: 22.8 TF/s vs the
  cooperative kernel's 28.5, both since superseded by na.

**Why attention stays BF16/FP16 (the int8 attention verdict).** A full int8
attention forward ships in the source (`flash_attn_fwd_na_i8`, the
DrawThings `NAInt8AttentionKernel` recipe: on-the-fly symmetric int8 quant
of Q/K and mean-centered V with per-tile scales, int8 QK → int32, exact
V-mean addback). It is functional and accurate (~1.2% relative, all shapes,
poisoned-pool clean) — and **deliberately unrouted**, because it loses:
best configuration 0.89×/0.78× the native low-precision `na` kernel at
Wan/LTX shapes. The
root cause is a measured law of this silicon: the int8 tensor units' ~2×
advantage only exists at deep reductions. Our own 105 TF/s int8 GEMM drops
to **25 TF/s at K=128** (104 at K=6144, 93 at K=512, 33 at K=256) — and
attention reduces over `head_dim = 128`, where the BF16/FP16 `na` kernel
already runs 36–40. int8 for the deep-K linears, BF16/FP16 for shallow-K
attention is
not a preference, it's the crossover the hardware dictates — DrawThings,
Cider, and this repo converged on the same split independently.

The same verdict holds for **block-scaled FP8 QK with BF16 PV**. An
experimental native path (`flash_fwd_na_fp8qk`) quantizes each Q/K tile on the
GPU, uses FP8 TensorOps for QK, and leaves softmax and PV in BF16. At the exact
H3 lengths 5,378 / 8,192 / 17,000, its best paired ratios were 0.976× / 1.005×
(thermal-noise territory) / 0.981× versus the shipped BF16 kernel, with maximum
BF16 output deltas of 4.88e-4 / 3.81e-4 / 2.44e-4. Quantization cost plus the
unchanged BF16 PV half erase the faster QK primitive, so this path remains an
unrouted research backend. A 16-candidate BF16 schedule tournament at the same
shapes likewise found no bit-identical improvement worth promoting; the
apparent faster 96-column schedule changes reduction order by one BF16 ULP.

### Attention router (patch #19)

Per-call routing on MPS: unmasked long-sequence attention → flash41; masked
attention above a real-score-size threshold → memory-safe sub-quadratic;
mid-size → batched bmm (gated on the *actual* `Lq×Lk` score bytes, not the
max-sequence square — that distinction is what keeps tiny cross-attention off
the slow path). Stock fused SDPA — silently inaccurate past ~4k tokens and
O(L²) memory — is never taken at length. `ASFP8_ATTN_DEBUG=1` logs each
distinct route taken.

`Attention Backend (Apple Silicon)` exposes the same dense router per MODEL,
without requiring the global `ASFP8_ATTN_ROUTER` switch. Its choices are:

- `automatic/current` — clone the incoming model without changing its backend;
- `apple_silicon_mps_auto` — flash41 / bmm / sub-quadratic per-call routing;
- `pytorch` — ComfyUI's registered PyTorch attention backend;
- `comfy_default` — the attention backend selected when ComfyUI started,
  explicitly bypassing this package's global router.

On ComfyUI builds containing [PR #15479](https://github.com/Comfy-Org/ComfyUI/pull/15479),
the Apple backend is also registered as `apple_silicon_mps` and MiniMax H3 uses
ComfyUI's single-owner Q/K/V containers. Older builds keep the tensor API. In
both cases H3 drops the fused QKV projection, the three attention inputs, and
the attention output at their earliest safe lifetime boundary so those large
allocations do not remain live into the next stage merely because Python still
references them. This is an exact allocation-lifetime change; it does not alter
the selected attention backend's arithmetic.

The backend node composes with H3 sparse attention in either graph order:
sparse attention stays outermost, and every declined call reaches the selected
dense backend. Unknown third-party attention overrides are rejected rather
than silently replaced. Comfy Kitchen INT8 attention is deliberately absent:
its released implementation is CUDA-only, while this repository's native
Metal INT8 attention measured 0.89x/0.78x the fp16 kernel at Wan/LTX shapes.

### Quantized matmuls (patches #3, #17, #18)

- **int8 W8A8** (`ASFP8_INT8_EXT`, M5+): convrot checkpoints run the path the
  format was designed for — rotate the activation online, per-row quantize,
  then a bit-exact INT8×INT8→INT32 kernel (~105 TF/s at diffusion shapes,
  ~1.85× over bf16) with the rescale + bias fused into the store epilogue.
  Ported from [Cider](https://github.com/Mininglamp-AI/cider) (MIT). This is
  the backbone of the Z-Image / Flux2 / Krea2 / Ideogram / Wan int8 numbers
  above. A 1-slot activation cache also skips repeat rotate+quant passes when
  several projections share one input (qkv triples, SwiGLU pairs).
  An earlier M5 Max variant tournament (`dev/tournament_int8.py`) found
  97–107 TF/s from cache-resident to streaming working sets, with no dependable
  gain from the tested schedules under interleaved A/B. The Metal API's larger
  tensor-op descriptors were 2.5–4× slower in those tests. That result applies
  to that kernel family and environment, not an absolute silicon ceiling:
  the subsequent Ultra tests above improved throughput by changing the inner
  step, unrolling and dispatch schedule.
- **Fused o-projection pre-pass** (`ASFP8_INT8_OPROJ_FUSE`, on): when
  `head_dim == convrot groupsize` (Ideogram 4: both 256, so each head is
  exactly one rotation group), the attention output goes straight from its
  native `[B,H,S,D]` layout to quantized GEMM input in one kernel — head-merge
  gather + rowwise quant fused, the 46 MB merge copy and the separate quant
  pass gone (0.48 ms vs 1.29 ms per call; 816 calls per Ideogram run). The
  convrot rotation stays the eager bf16 matmul, applied directly in attention
  layout — measured bit-identical to the merged-layout matmul — so the default
  mode is **bit-exact: same-seed renders are pixel-identical**. `=2` swaps in
  a fully in-kernel FWHT rotation (single pass, rounding-equivalent but not
  pixel-identical — and slightly slower than the matmul anyway; kept as an
  experiment). `=0` disables.
- **fp8-native `_scaled_mm`** (`ASFP8_FP8_EXT`): fp8×fp8 matmuls read fp8
  operands directly on the matrix units — no bf16 materialization —
  bit-exact, ~1.2–2.1× on FLUX/SD3.5-class shapes.
- **W8A16 staged GEMM** (`ASFP8_FP8_SGEMM`): half×fp8 kernel with
  tiled/swizzled geometry and fused bias for standard-fp8 Linears, skipping
  the per-forward weight decode.

### DiT block fusions (`ASFP8_DIT_EXT` — patches #21, #23–#25, #27, #28, #32, #33)

Per-architecture Metal kernels for the elementwise/copy-heavy glue between
matmuls, which on MPS costs as much as the matmuls themselves. Each patch
mirrors eager rounding op-for-op (exact-or-1-ulp) and falls back per call:

- **Lumina / Z-Image (#21)**: fused qkv-split + QK-RMSNorm + RoPE + head
  transpose (~7.5 ms → ~0.5 ms per layer), fused norm+modulate,
  norm+gate+residual, silu-mul.
- **RoPE (#23)**: flux-family `apply_rope` in one pass (~7.3 ms → ~0.7 ms per
  attention at Flux2 shapes).
- **Flux2 (#24)**: double/single-block fused qkv+QKNorm+RoPE+seq-cat — kills
  4 RMSNorms and 3 sequence cats per double block; handles Klein's global
  modulation.
- **Krea2 (#25)**: fused q/k/v norm+RoPE at native GQA head counts feeding
  flash41's grouped-KV path (the `repeat_interleave` and its 4× KV traffic
  are gone). On **M5 Ultra / PyTorch 2.13**, additional exact kernels fuse
  FP32 RMSNorm reduction + BF16 modulation, gated residuals, SwiGLU, and
  sigmoid gating + attention layout. The original reduction order and BF16
  rounding points are retained. At 2MP, the attention kernel also uses the
  Ultra's synchronized Morton traversal, retaining dynamic dimensions and
  the same softmax/accumulation arithmetic. The tested **1448×1448, 8-step
  INT8 ConvRot + LoKr 0.8** workflow falls from **18.50 s to 15.09 s warm**
  including decode/PNG saving (**18.4% less time**), with identical sampler
  latents, decoded tensors and original PNG pixels. Sampling alone falls
  from **18.08 s to 14.67 s**. Set `ASFP8_KREA2_GLUE=0` and
  `ASFP8_KREA2_ATTN_SCHEDULE=0` to restore the previous implementation.
  See the [Krea2 Ultra report](benchmarks/m5-ultra-krea2-exact.md) for
  interleaved measurements, numerical checks and rejected experiments.
- **LTX (#27)**: one-pass split-RoPE (zero transposed copies) + fused
  rms_norm+modulate with per-frame timestep rows.
- **Wan (#28)** (`wan_ext`, fp16/bf16-templated): fused LayerNorm+modulate
  and gated residuals with *per-frame modulation rows* — the eager path
  materializes [B,F,C]→[B,L,C] repeats (220 MB a use, six uses per block);
  here they never exist — plus wide RMSNorm+RoPE written straight into
  attention layout.
- **Ideogram 4 (#32)**: fused qkv-split + per-head affine RMSNorm +
  interleaved-MRoPE split-half rotation (D≤512 via element-pair-per-thread),
  the tanh-gated norm/modulate/residual block glue, MRoPE freqs memoization
  (24 identical per-forward rebuilds → 1), PSO prewarm overlapped with the
  ~9 GB checkpoint loads, and the fused o-projection pre-pass above.
  56.6 → ~41 s at 12 steps 1MP — past DrawThings' 42 s.
- **MiniMax H3 (#33)**: Q/K/V are written directly from the fused projection
  into `[B,H,S,D]` attention layout while Q/K RMSNorm and partial split-half
  RoPE run in the same pass. On an INT8 QKV layer with D=128, each complete
  Q/K head is consumed on chip by the channelwise GEMM epilogue: projection
  dequant/bias, affine RMSNorm, and partial RoPE retain the established BF16
  rounding order, while V is stored directly. This removes both the full
  `[S,3HD]` bf16 QKV allocation and the separate raw-Q/K post-pass without
  changing an output byte. Eligible standard runtime LoRAs use a partitioned
  exact epilogue: PyTorch computes each low-rank Q/K/V delta in BF16, then the
  Metal kernel reproduces PyTorch's scalar and product rounding while adding
  it before the nonlinear Q/K norm and RoPE. Only one `[S,HD]` delta partition
  is live at a time; the full `[S,3HD]` adapter output is never allocated. The
  native attention result then goes straight
  through cross-head convrot + row quantization into the INT8 output projection.
  Segmented gated residuals, RMSNorm and modulation are fused as block glue;
  when no workflow node owns a block boundary, the MLP residual gate also
  produces the next block's attention AdaLN input. Large intermediates are
  released before the following wide allocation. INT8 H3 MLPs additionally
  use a 1,024-row FC1 scratch tile, overwrite its dead left half with the
  SwiGLU + convrot result, and feed per-group maxima into exact row
  quantization for FC2. This removes the full-sequence wide FC1 and separate
  rotated BF16 allocations. The FC2 input bytes and output are bit-exact with
  the existing native SwiGLU/convrot route, so the bridge introduces no new
  quality change. Standard FC1/FC2 runtime LoRAs retain the same ordering in a
  separate exact 2,048-row bridge: `FC1 base + delta → SwiGLU → FC2 base +
  delta`. Partial target sets (FC1-only or FC2-only) are also exact. Synchronized
  rank-64 paired A/B measurements on M5 Max show why both LoRA routes ship:

  | H3 tokens | QKV LoRA speedup | Q/K traffic avoided | MLP LoRA bridge speedup | peak BF16 storage removed |
  |---:|---:|---:|---:|---:|
  | 5,378 | **1.400×** (21.43 → 15.07 ms) | 294.1 MiB | **1.001×** | 329.2 MiB |
  | 8,192 | **1.392×** (32.22 → 23.08 ms) | 448.0 MiB | **1.012×** | 560.0 MiB |
  | 17,000 | **1.374×** (70.41 → 51.21 ms) | 929.7 MiB | **1.069×** | 1,282.5 MiB |

  The MLP column uses the 2,048-row LoRA tile and reports paired ratios rather
  than a ratio reconstructed from unrelated medians. These are isolated hot-path
  measurements, not whole-sampler speedups. For long clips, the
  channelwise INT8 GEMMs use an exact
  long-M compiler schedule and narrowly gated 32-row suffix dispatches; all
  preserve the INT32 reduction and bf16 epilogue byte-for-byte. At 864×480,
  39 frames and 20 steps, the warm saved workflow
  measured 110.4 → 103.5–104.3 s despite sustained thermal throttling
  (sampler 98.6 → 92.1–92.9 s); early cool steps improved from ~4.9 to
  ~3.8–4.1 s. The exported 864×480, 124-frame / 20-step workflow completes
  in 491.20 s: 427.94 s for sampling (21.397 s/step average, 19.19 s first,
  ~21.9 s heat-soaked) and 38.59 s for video decode. A prior already-
  optimized but more heat-soaked run was 536.69 s / 469.86 s sampler; do not
  attribute that full cross-session delta to the new schedule. Controlled
  interleaved GEMM A/B predicts ~1.84 s of exact sampler savings from the
  long-M schedule plus QKV suffix dispatch; the remainder is thermal state.
  A guarded H3 graph rewrite also recognizes the final target-tail
  RMSNorm → scale/shift → fp32-cast chains and executes both video and audio
  segments in one exact Metal pass. With only that rewrite toggled, matched
  1-second steady runs fell from 103.5–104.5 s to 96.6–97.3 s. The promoted
  5-second check fell from 490.39 s to **461.34 s** end to end (1.063×), with
  sampler time approximately 445.2 → 418.2 s. The rewrite is enabled by
  default, validates the module graph and runtime geometry, and falls back to
  the original forward on anything it does not recognize.

### Video VAE and SeedVR2 (patches #26, #29, #31, #36, #37, #38)

- **Conv3d → sliced conv2d (#26, `ASFP8_CONV3D`)**: MPS's conv3d lowering is
  ~10× slower than the equivalent per-slice conv2d. Temporal taps fold into
  conv2d's *channel* dimension, so one conv2d accumulates all taps in fp32
  internally and rounds once — **exactly** conv3d's numerics at full speed.
  (Summing separately-rounded per-tap outputs looks harmless in one decode,
  but recursive causal VAEs like Wan's feed each chunk's output into the next
  chunk's cache, and the per-tap rounding compounds into horizontal banding
  that grows over the clip. Found the hard way.) 7× on LTX's video VAE.
- **Wan 2.1 VAE fast path (#29)**: decode profiling showed the convs already
  at MPS's ~60 TF/s ceiling; the time was in memory patterns around them.
  `chan_rms` runs the channelwise RMS_norm (+fused SiLU) — a 42 MB-strided
  reduction MPS handles at ~80 GB/s — at ~400 GB/s via threadgroup tiling;
  `fold_taps` gathers each causal conv's folded input straight from
  [zeros | cache | x], eliminating the pad-cat/tap-copy/channel-cat chain
  (~14 tensor volumes per conv → ~4); and the causal chunk loops stream 16
  frames per pass instead of 2. Decode 16.8 s → 6.1 s, encode 6.5 s → ~4 s at
  512²×81f, parity 1–2 ulp.
- **VAE slice_attention INT_MAX cap (#31, `ASFP8_VAE_SLICE_FIX`)**: comfy's
  VAE spatial attention sizes its query slices from free memory ALONE, but
  MPSGraph refuses any tensor over INT_MAX elements. At Wan-VAE 1024² video
  decode the score tensor is ~5.6e9 elements — memory says "one slice fits"
  (on 128 GB it does), MPSGraph raises, and comfy's retry loop re-raises
  because the error isn't an OOM. Ironically, the more RAM you have the more
  likely the crash: pessimistic free-memory figures on small machines force
  extra slices by accident, and honest accounting (#30) removes that veil.
  The patch floors the slice count to a per-slice element budget and treats
  the INT_MAX error as retryable. Numerically free — the query rows are merely
  divided into more calls and each still sees the full key length.
- **SeedVR2 VAE tile scheduling (#36, `ASFP8_SEEDVR2_VAE_SYNC`)**: waits for
  MPS completion between spatial encoder/decoder tiles and between residual
  blocks inside video tiles. On the tested M5 Ultra
  setup, the original tile loop grows Metal temporary allocations even while
  live tensor memory stays flat; the same growth occurs without this custom
  node. Waiting lets buffers be reused across tiles. This changes scheduling
  only, preserving precision, arithmetic and tile overlap. It is enabled by
  default on MPS, independent of native/relaxed kernel flags; set
  `ASFP8_SEEDVR2_VAE_SYNC=0` before startup to restore the original scheduling.
  Video tiles contain many internal temporal slices, so spatial-only waits
  are insufficient: the saved five-second video crossed a protected 48 GiB
  cutoff during encoding with only 3.36 GiB of live tensors. The new video
  waits preserve causal caches and output bytes; **Split Latent can stay off**.
  `ASFP8_SEEDVR2_VAE_VIDEO_SYNC=0` restores spatial-only waits for diagnostics.
  See the [video memory investigation](benchmarks/m5-ultra-seedvr2-video-memory.md).
  Non-tiled SeedVR2 calls, other VAEs, CPU and CUDA retain their original calls.
  The **2400×1600 → 7200×4800** INT8 7B image workflow completed in **80.15 s
  first run / 73.41 s warm**, including PNG writing, with **40.26 GiB** maximum
  observed Metal allocation. A smaller complete workflow preserves encoded
  latents, sampler latents and decoded frame bytes with the fix on/off. See the
  [SeedVR2 memory report](benchmarks/m5-ultra-seedvr2-memory.md) for measurement
  scope, numerical checks and the guarded disabled-path reproduction.
  The [SeedVR2 acceleration audit](benchmarks/m5-ultra-seedvr2-acceleration.md)
  records the original INT8/convolution coverage. The subsequent exact stack
  below supersedes its preliminary optimization status.
- **SeedVR2 PyTorch 2.14 retest:** **74.02 s stock + memory safeguard → 49.01 s optimized** at 2400×1600 → 7200×4800. Unit-scale VAE residuals avoid a redundant division; output bytes match the preceding optimized stack. See the [current retest](benchmarks/m5-ultra-seedvr2-torch214.md). The following measurements are historical PyTorch 2.13 results.
- **SeedVR2 exact kernels and bounded attention (#37)**: with
  `ASFP8_NATIVE=1`, cache convolution weight layouts only during each tiled VAE
  call, fuse FP16 SiLU with causal padding/channel packing, fuse BF16
  normalization/modulation and gated residual memory passes, cache RoPE
  trigonometry, and batch matching attention windows. Existing INT8 matrix
  multiplies, attention arithmetic, convolution operands and intermediate
  rounding are retained. At **2400×1600 → 7200×4800**, the complete sequence
  including PNG writing fell from **73.52 s to 50.98 s warm** (30.6% less time), with identical
  encoded latents, sampled latents and final frame bytes. Set
  `ASFP8_SEEDVR2_FAST=0` before startup to disable these performance additions.
  The MPS attention buffer fix remains enabled independently: a small window
  can otherwise fail with `INT_MAX` when its backing feature buffer exceeds
  2,147,483,647 elements. Compact MLP inputs also prevent the same issue from
  forcing INT8 projections onto their fallback path. Very large projections
  use bounded row dispatches, and large window gathers use explicit 64-bit
  offsets. See the
  [SeedVR2 exact optimization report](benchmarks/m5-ultra-seedvr2-exact.md)
  for full-size tests, the 18,720×12,480 reproduction, numerical acceptance,
  and the GroupNorm experiment retained off by default.
  The **6240×4160 → 18720×12480** job now completes: **370.60 s** including
  PNG writing in the first complete final run, with finite output and no
  INT8 fallbacks. Maximum observed Metal allocation was **125.51 GiB**;
  this is sampled allocator memory, not total system RAM or a continuous peak.
- **Raw SeedVR2 7B FP16 loader compatibility (#38)**: the original
  `seedvr2_ema_7b_fp16.safetensors` omits `positive_conditioning` and
  `negative_conditioning`. ComfyUI recognizes its architecture but rejects
  model-class matching without those buffers, reporting **"Could not detect
  model type"**. The default macOS loader patch supplies ByteDance's fixed
  conditioning in memory before matching. The bundled tensors preserve the
  official BF16 bytes and also match the tested normal/sharp INT8 checkpoints.
  All existing tensors and on-disk checkpoint files remain unchanged; models
  with embedded conditioning retain their original loading path. This is
  limited to the original 36-layer, separate-stream 7B architecture.
  No model download, quantization or relaxed-math flag is involved. Disable
  with `ASFP8_SEEDVR2_LOADER=0` before startup. See the
  [conditioning provenance and license](_patches/seedvr2_assets/README.md).
  ComfyUI chooses BF16 computation for these FP16 weights on the tested M5
  setup, so the existing BF16 SeedVR2 fusions apply; dense matrix multiplies
  use MPS, not the INT8 kernels. FP16 and INT8 results need not be identical.
  The complete **2400×1600 → 7200×4800** FP16 workflow passed at **57.45 s
  first / 54.22 s warm**, including PNG saving, with identical first/warm
  latents and frame bytes. That compares with the prior **50.98 s warm**
  INT8 run. See the [FP16 compatibility report](benchmarks/m5-ultra-seedvr2-fp16.md)
  for reference checks, dtype behavior and measurement scope.

### Memory accounting (patch #30)

comfy decides whether to keep a model resident by calling `get_free_memory`,
which on MPS returns `psutil.virtual_memory().available`. That number can't
see torch's MPS allocator pool and swings with OS file-cache noise — so on a
128 GB machine comfy would still **evict a 13.9 GB model per sampler switch**,
re-uploading it every run (the Wan 2.2 high/low-noise pair cost ~14 s/run this
way; text encoders churn the same way on other workflows).

`free_mem_mps` replaces the estimate with one that's true for unified memory:
`free = total RAM − torch's live tensors − an OS reserve`
(`ASFP8_MPS_RESERVE_GB`, default max(16 GB, 15%)). Under genuine OS-level
pressure (available memory below the reserve itself) the estimate is
additionally capped by `available + torch's reclaimable pool`, so a machine
other apps have truly filled is never overcommitted. The cap is deliberately
pressure-gated: macOS *understates* what a busy-but-healthy box can free up
(compressed pages, quittable apps), and steering by that number in normal
operation puts 14 GB models into partial-unload/lowvram churn — the exact
disease this patch cures (measured: an ungated cap pushed a Wan 1024² rerun
20 GB into swap). It also reports the "torch
free" figure as 0 — its sole consumer is a `free_memory` branch that dumps the
entire allocator cache when "torch hoards >25% of free memory," a condition
stock MPS code made *always true*, so every model load was silently discarding
the reuse pool and forcing the next sampler to re-grow multi-GB activation
buffers. Honest, not an override: genuine pressure still evicts. After this,
post-warmup model loads drop from 1.4–13.5 s to **0.1 ms**, and Wan 2.2 holds
a steady 122 s.

One consequence worth knowing: the process footprint now parks at its
high-water mark (~45 GB steady, ~66 GB during the VAE decode peak) because the
MPS allocator never returns freed memory to the OS and we deliberately keep
that pool resident. On a 128 GB machine that is the intended trade — warm
reruns and 0.1 ms model switches. On a smaller machine the same formula
adapts (reserve scales, comfy evicts sooner); set `ASFP8_MPS_RESERVE_GB`
higher to hand memory back to other apps at the cost of more eviction.

### Free RAM button

The pool-parking trade above comes with a manual release valve: a **Free RAM**
button in ComfyUI's top bar (`web/asfp8_topbar.js`) that unloads all models
and empties the cache on click — reclaim the ~45 GB parked footprint before
switching to another heavy app, without an idle timer evicting models behind
your back mid-session. (Measured: +7.5 GB returned immediately; the rest
follows as the allocator drains.)

On explicit free-everything requests the unload also drains the MPS pool
*between* models instead of once at the end (`ASFP8_FREE_INTERLEAVE=0`
disables), bounding the transient footprint to roughly one model. After a
long session you may still see a brief memory-pressure blip while macOS
decompresses long-parked pool pages — it self-resolves within seconds and
touches nothing on disk.

### MLX text generation (patch #14)

Autoregressive prompt-expansion encoders (Krea2's Qwen3-VL, LTX's Gemma3) run
token-by-token in Python on MPS (~1 s/token). With `mlx-vlm` installed, the
generation loop routes through MLX's fused decode loop with GPU KV-cache;
conditioning encode is untouched.

## The compatibility layer (patches #1–#16)

Condensed — each is automatic, MPS-scoped, and a no-op where not needed:

| # | Fixes |
|---|---|
| 1 | psutil `host_statistics64` crash on macOS betas (vm_stat fallback; self-disabling on healthy systems) |
| 2 | comfy_kitchen eager FP8 + NVFP4/MXFP8 dequant (Ideogram 4, LTX's Gemma3 TE) |
| 3 | `torch._scaled_mm` FP8 → LUT decode + bf16 matrix-unit matmul (FLUX, SD3.5, fp8_scaled) |
| 4 | `F.rms_norm` garbage above ~2²² rows — PiD ≥2048px black image |
| 5 | `F.scaled_dot_product_attention` → [mtlflashattn](https://github.com/pawel-mazurkiewicz/mtlflashattn) at length (correctness ≥4k tokens, OOM rescue, 3–4× over fused SDPA) |
| 6 | FP8 `UNETLoader` weight+bias decode in `cast_bias_weight` |
| 7 | LoRA-on-FP8 re-quant (`stochastic_rounding`) via CPU |
| 8 | `torch.Tensor.to` FP8↔float shim (third-party FP8 Linears, WanVideoWrapper) |
| 9 | WanVideo block-swap neutralizer (unified memory makes it pure breakage on MPS) |
| 10 | `F.linear` FP8 operand decode (T5 and friends) |
| 11 | Text/LLM encoders redirected CPU→MPS (autoregressive encoders were ~1.7 s/token on CPU) |
| 12 | `torch._int_mm` on GPU instead of the silent CPU-fallback freeze (routes to the exact patch-#17 int8 kernel where available — parity-gated; fp32 emulation otherwise) |
| 13 | int8-fast wide-batch Linear via native bf16 GEMM (~3.5–4.7×) |
| 14 | MLX-backed TextGenerate (above) |
| 16 | Strided-FP8 materialization (`reshape`/`contiguous`/`clone`) via CPU — fixes NVFP4/MXFP8 block-scale swizzles everywhere at once |

(#15 was retired: wrong seam — ComfyUI's fp8 routes through `_scaled_mm`, not
`F.linear`. #20 and #22 round out the set: fp8 fill/factory ops via CPU, and
defaulting missing `comfy_quant` format metadata so community int8-convrot
checkpoints load. #30 — the MPS memory-accounting fix — is
[described above](#memory-accounting-patch-30).)

### How the FP8 trick works

MPS can't cast to/from `float8_e4m3fn`/`e5m2` on the GPU — but it *can* move
FP8 tensors to the device, bit-view them as `uint8`, and gather. So a
256-entry table maps every FP8 byte to its float value (decoded once on CPU),
and any FP8 tensor decodes as `lut[x.view(uint8)]`: bit-exact with a real
cast, entirely on GPU, feeding MPS's fast float matmuls.

## Numerics policy

Three tiers, stated per patch:

- **Bit-exact**: all FP8 decodes, the int8 W8A8 kernel (including long-M
  scheduling and suffix dispatches), `fold_taps`, the conv3d channel fold
  (single fp32 accumulation = conv3d's own numerics), and the D=256 fused
  o-projection pre-pass in default mode. H3's LoRA-aware QKV partition
  epilogue and tiled FC1/FC2 bridge are byte-exact against their corresponding
  unfused native runtime-LoRA chains.
- **Exact-or-1-ulp**: the DiT fusions and VAE norm kernels mirror eager
  rounding at every point eager rounds (fp32 math between roundings); H3's
  D=128 cross-head convrot and optional fused SwiGLU use a reassociated FWHT
  and can move rare quantized values by one integer step. Images shift
  imperceptibly but deterministically.
- **Quantization-mode change**: H3's SwiGLU down-projection stays in the
  checkpoint's W8A8 path, while ComfyUI's generic `linear_input_act` helper
  dequantizes that seam to W8A16. The measured local error class is about 1.3%;
  `ASFP8_INT8_SWIGLU=0` restores the stock W8A16 route. The tiled MLP and LoRA
  bridges do not add further error: they are byte-exact against the selected
  native W8A8 SwiGLU/convrot chain.
- **Regime-gated**: flash41's low-precision score tiles (default; parity
  ~5e-4, `ASFP8_FA41_F32S=1` for fp32 tiles).

Patch #30 (memory accounting) changes *scheduling*, not numerics — outputs are
identical, only fewer reloads.

**October 9 publication check:** the current CPU regression suite passes
**934 tests**, with **441 GPU tests deliberately skipped**, and all **23 frontend
tests pass**. This check leaves the user's GPU/inference session untouched;
complete GPU parity, memory and workflow timing evidence is retained in the
individual benchmark reports. No new performance measurements were made for
this documentation/publishing pass.

Everything ships with parity tests against its stated reference path. The
historical full suite in the dedicated `ComfyUI-torch213-latest` environment
reported **397 passed, 19 skipped** on PyTorch 2.13.0. The Metal harnesses include a dirty-pool
test that poisons the MPS allocator with NaN pages before checking ragged
attention shapes, because parity on fresh, zeroed allocations structurally
cannot catch out-of-bounds reads.

## Prompt Studio

**Apple Silicon Prompt Studio** generates editable prompts for image, video and
audio workflows through **oMLX, LM Studio, Ollama or an OpenAI-compatible server**.
It outputs `prompt` and `negative_prompt` as `STRING` values. The compact node
opens a workspace with creative controls, editable target-model profiles,
image/video-frame/audio references, a searchable prompt library with favorites,
and JSON backup/restore. Classic nodes and Nodes 2.0 are supported.

Start with **Open Studio → Connections**, save your endpoint, discover models,
then select your LLM. The canvas node now has a **model picker**, using the
saved per-connection discovery list, and one **Generate on queue** toggle.
With it off, workflows reuse the editable saved text; with it on, each execution
requests a fresh prompt. The Generate button sends the brief together with
connected image/video/audio inputs automatically, running only the required
upstream nodes. References require a compatible LLM/server. Endpoint setup stays
in Open Studio; you can switch models directly on the node.
API keys stay in a separate private server file, outside workflows and backups.

[Setup and reference support](docs/prompt-studio.md) ·
[Example workflow](workflows/apple_silicon_prompt_studio.json)

## MiniMax H3 workflow nodes and approximate modes

These are separate from the native fast paths above. They are **off until a
workflow node is connected**, change the result, and should be A/B tested with
the same seed. For a compact interface, use `H3 Acceleration · Simple`;
`H3 Apple Silicon Acceleration` retains all custom settings. Both are
`MODEL → MODEL` nodes, sharing the same acceleration runtime.

| Node | Purpose |
|---|---|
| Attention Backend (Apple Silicon) | Select the model's attention backend. |
| H3 Acceleration · Simple | Three independent recipe selectors: temporal cache, attention acceleration and token reduction, including optional SOL, dense INT8 and hybrid attention recipes. |
| H3 Apple Silicon Acceleration | Advanced cache, token reduction and sparse attention settings. |
| H3 LoRA (Apple Silicon) | Ordered LoRA list, including Turbo, with per-entry enable, weight and validation. |

Two optional [Super Resolution nodes](docs/super-resolution.md) additionally handle
IMAGE and VIDEO upscaling. They are separate from these model controls.

The standalone Cache, Structural and Sparse nodes, the nonfunctional preset
wrapper, and the separate Turbo loader have been removed. Opening a saved
visual workflow automatically migrates supported settings; disabled groups
stay disabled. Old single-LoRA entries become a one-entry list, and Turbo
entries retain strict validation. Failed research modes and linked legacy
widget inputs require manual replacement and produce a warning rather than
silently changing the calculation. API JSON prompts using retired node IDs
must be updated to the nodes above. Native startup patches, relaxed-mode flags
and the Free RAM button are unchanged.

The implementation boundary matters when combining them:

| Mechanism | Implemented as | FirstBlockCache | Spectrum | H3 sparse attention |
|---|---|---|---|---|
| Exact H3 QKV/block/lifetime fusions | startup backend + Metal kernels | yes; cross-block-only fusion disables itself | yes; same boundary rule | yes |
| H3 LoRA with a Turbo adapter | workflow `MODEL → MODEL` node | mechanically yes; few hits remain at 4–8 steps | mechanically yes, not calibrated for Turbo dynamics | mechanically yes; errors compound |
| H3 token reduction | unified workflow `MODEL → MODEL` node | yes; the unified node fixes the internal order | yes; the unified node applies Spectrum before token reduction | yes; reduced-sequence metadata is synchronized |
| FirstBlockCache | unified workflow `MODEL → MODEL` node | — | choose one block predictor, do not stack | yes |

“Mechanically yes” means the hooks do not erase each other; it is not a quality
claim. Newly placed unified nodes use MotionCache alone, the closest content
preservation in our recent 20-step tests. For a quality diagnosis, select Exact
and add mechanisms back one at a time. The exact
backend remains useful under every node, although a block-replacement node
necessarily removes the opportunity for the exact cross-block AdaLN carry at
that boundary.

- **H3 Acceleration · Simple.** A small native-widget node for Nodes 2.0 and
  the classic canvas. Choose **Temporal cache**, **Attention acceleration**, and
  **Token reduction** independently. The existing cache, SOL, Veda and token
  recipes remain, alongside Dense INT8 Protected / Full Draft and SOL Hybrid Fast. Defaults are
  **MotionCache · balanced**, sparse **Off**, tokens **Off**. **Detailed logs** defaults off; each active mechanism prints
  a compact final work/speedup summary. Set all three selectors to Off for pass-through.
  Connect one of the Simple/advanced nodes between the H3 model/LoRA loader
  and the guider; use the advanced node for predictor selection, memory tuning,
  or custom numerical settings. [Recipes and usage](docs/h3-acceleration.md#simple-node)
  · [Node template](workflows/h3_acceleration_simple.json)
  · [Token compatibility and benchmark checks](benchmarks/m5-ultra-h3-presets.md).
  On the authored five-second 864×480, 20-step check, complete execution was
  **141.7 s** with tokens off, **123.8 s** with Light, and **94.7 s** with Fast.
  The token presets change composition/identity and remain experimental and off
  by default; these figures compare against the exact optimized stack, using
  a different prompt from the primary stock/optimized table above.

- **H3 Apple Silicon Acceleration (advanced node).** Newly placed nodes show
  a compact panel compatible with **Nodes 2.0** and the classic canvas.
  **Starting point** applies a recipe. **Temporal cache**, **Attention acceleration**,
  and **Token reduction** have separate cards and collapsible Settings, without
  blank hidden rows. Saved workflows retain their numerical settings; manually
  tuned or combined values display **Custom combination**.

  INT8 SOL's [temporary-memory ceiling](docs/h3-acceleration.md#temporary-memory)
  is configurable under **Attention acceleration → Settings → Temporary memory (GiB)**.
  `0` means Auto (up to **32 GiB on a 256 GiB Mac**, subject to available memory);
  positive values set a manual ceiling. This replaces the hardcoded 4 GiB limit.
  Budget fallbacks report the required memory and no longer disable SOL for the
  rest of the sampling run.

  **MotionCache · recommended** is the new-node default: threshold `0.15`,
  four warmup evaluations, window `0.15–0.95`, maximum two consecutive skips,
  motion strength `1`, subsample factor `8`. Token reduction and sparse
  attention are **off**. This was the closest content preservation among our
  recent tests, not an exact-output guarantee. MotionCache and Spectrum are
  **included**; neither requires another installation or node in the graph.
  Select **Exact · no approximation** to disable this node's approximations
  while keeping the exact startup kernels. Other nodes/environment settings
  still apply.

  Other one-click recipes are **Spectrum · protected**, **Spectrum · fast
  draft**, **FirstBlockCache · 0.20**, **SOL attention · 14 steps**, and
  **SOL + FirstBlockCache**. Each writes the measured recipe, including the
  SOL INT8 backend, protected layers, step-fraction schedule and token guard;
  the hybrid uses its own lower cache threshold and one-skip limit. These are
  20-step, non-Turbo measurements. See [controls and bundled-source licenses](docs/h3-acceleration.md)
  and [timing/quality comparisons](benchmarks/m5-ultra-h3-recent-methods.md).

  Choose **one cache method**, then independently enable SOL and/or token
  reduction, including with MotionCache or Spectrum. These cache predictors
  are alternatives to each other. Token reduction remains experimental;
  combinations need their own quality check. Non-skipping `oracle`/`block_oracle`
  diagnostics and the per-step cache policy remain available. Sparse block size
  `64` and temporal taps `0` remain fixed; layer thinning stays unavailable because
  it damaged H3 audio. [UI and combination validation](benchmarks/m5-ultra-h3-node-v2.md).

  Bundled Spectrum is GPL-3.0-or-later; the combined distribution uses that
  license, preserving existing MIT notices. [Attribution and licenses](THIRD_PARTY_NOTICES.md).

- **Optional dense and hybrid attention.** Dense INT8 Protected keeps the first
  four and final two of 20 steps plus layers 0, 1, 33, 34, 35 exact. Full Draft
  quantizes video-query attention in every layer/step while retaining all key
  tokens. Packed prefix queries stay exact in both, but later video changes can
  affect audio too. Hybrid retains the protected step window, uses SOL Fast in
  ordinary layers and dense INT8 in protected layers inside the active window.
  These are independently selectable starting points, with existing memory
  safeguards and native fallback. Full Draft can change scene composition;
  Hybrid's additional benefit is modest. [Complete timings and temporal audit](benchmarks/m5-ultra-h3-kitchen-attention.md).

- **H3 LoRA (including Turbo).** Click **Add LoRA** for each adapter. Every
  entry has an enable switch, file selector, signed weight, and `standard` or
  `turbo` validation. Remove entries individually; an empty or all-disabled list
  passes the model through. Disabled and zero-weight entries do not load their
  files. Enabled adapters are applied in list order, including when their
  targets overlap. Use one list instead of chaining runtime LoRA nodes.

  `standard` accepts any nonempty, shape-compatible subset of H3 QKV,
  attention-output, FC1, FC2 and AdaLN targets, encoded as linear
  `lora_A`/`lora_B` pairs with optional alpha. `turbo` additionally requires the
  complete released 208-target Turbo layout (50 × 4 DiT targets plus 2 × 4
  token-refiner targets). Both accept the `diffusion_model.` prefix. The
  released adapters work on native and pruned/curve bases; full-width AdaLN
  Turbo adapters require the native base. DoRA/LoHa/LoKr/mid/reshape formats
  are rejected clearly by this loader.

  For Turbo, choose its file and `turbo` validation. The node changes neither
  sampler nor steps: keep the adapter's intended settings (our Turbo workflow
  uses Euler/simple, weight 1.0 and 4–8 steps). Turbo and style adapters can
  share the list; combining them still requires visual assessment.

  The three modes make the quantized-base trade explicit:

  - `fast_runtime` keeps all updates in activation space. The INT8 FC2 and
    attention-output paths explicitly add their low-rank deltas even though
    those optimized calls bypass `Linear.forward`; eligible QKV and FC1/FC2
    layers additionally use the exact LoRA-aware fused/tiled hot paths described
    above. This is the intended quality mode and requires this package's INT8
    backend for an INT8 base.
  - `reference_runtime` uses ComfyUI's normal runtime bypass and leaves
    quantized FC2 updates as ModelPatcher weight patches. It is slower but useful
    for isolating a fast-runtime issue.
  - `merged` uses ordinary ModelPatcher weight patches. It has the lowest
    runtime adapter memory, but small updates can be softened when the base is
    quantized.

  The node changes neither sampler nor step count. At 4–8 steps, benchmark
  Turbo alone before adding a cache:
  cache setup and forced refreshes consume a much larger fraction of the run.

  Single-adapter behavior retains the existing fast kernels. Multiple adapters
  share one reversible ModelPatcher hook per target and add their deltas in
  order, without concatenating ranks or pre-scaling weights. INT8 base GEMMs,
  FC2 activation handling and attention-output acceleration remain available.
  On overlapping targets, stacks use QKV scatter plus the existing normalization
  pass and decline the tiled MLP bridge: screening found BF16 rounding changes
  from tiled LoRA GEMMs. This avoids promising an exact fusion where it failed
  comparison. More runtime adapters cost compute and memory; this consolidation
  is not a new speedup or a claim of bit equality with merged-weight LoRAs.

  API input example (`loras` is a JSON string; `mode` defaults to `fast_runtime`):

  ```json
  [{"enabled":true,"lora_name":"turbo.safetensors","strength":1.0,"validation":"turbo"},
   {"enabled":false,"lora_name":"style.safetensors","strength":0.6,"validation":"standard"}]
  ```

- **Rejected cache experiments (internal research only).** The earlier
  full-model cache learned separate video and audio input-to-output change rates
  from exact evaluations. On a full-model hit it reused H3's previous video
  velocity and analytically corrected the audio velocity for H3's distinct
  shifted sigma schedule; either modality could veto a visually stable hit. A
  learned scalar secant predictor and the earlier residual/output-hold
  predictors remain in the internal research implementation. Sampler-facing
  denoised holding and first-order denoised extrapolation are also internal
  research functions. `cached_suffix_blocks` can instead recompute the model
  prefix exactly and reuse only a short transformer tail.
  The first/last sampling regions were protected, consecutive hits were bounded,
  and state was isolated to one sampling run. The standalone research node has been removed; these failed forecasting
  and suffix modes are not offered by the canonical node.

  No nonzero full-model forecasting preset is recommended. In the matched
  1-second/20-step case, one full-model residual-carry hit reached 0.938 video
  SSIM / 17.3 dB audio SNR. Correctly holding the velocity improved this to
  0.942 / 17.2 dB; correcting the audio schedule reached 0.943 / 17.7 dB; the
  learned secant reached 0.942 / 17.9 dB. A five-block suffix hit was still
  only 0.947 / 15.4 dB and saved no measurable sampler time after its probe and
  cache-maintenance overhead. Two full-model hits fell to 0.908 / 15.9 dB, and
  a sigma-scaled first-order forecast reached 0.931 / 17.5 dB. Moving the cache
  to ComfyUI's sampler-facing denoised boundary was exact at threshold zero,
  but one isolated denoised-hold hit reached only 0.946 / 17.0 dB and linear
  denoised extrapolation reached 0.931 / 17.2 dB. Because even the smallest
  useful 1-second candidates failed the quality gate, none was promoted to a
  5-second run.
- **FirstBlockCache (`cache_mode=first_block_cache` or
  `first_block_cache_per_step`).** Block 0 runs exactly on every evaluation;
  its own residual is the change detector, and when it is stable the cached
  aggregate residual of blocks 1–49 stands in for running them. Unlike
  whole-model forecasting this never predicts the model's output from its own
  history — every step still executes a real transformer block over the real
  state — so the failure mode is a stale tail rather than a diverging
  trajectory. Both modalities can veto a hit, consecutive hits are bounded,
  and threshold `0` calibrates without ever skipping.

  The original `first_block_cache` uses a sampled detector and accumulates
  relative change across consecutive reuses, which makes repeated small
  changes eventually force a refresh. `first_block_cache_per_step` reproduces
  the Discord-style policy: it measures the full target video and audio block-0
  residuals separately and compares only the latest relative-L1 change. The
  matching controls are threshold `0.10`, start `0.00`, end `0.95`, maximum
  consecutive skips `5`, audio weight `1.0`, and suffix blocks `0`. The latter
  is more aggressive and can be faster, but is also more likely to carry a
  stale tail through gradual motion. Not yet promoted to a preset: awaiting the
  calibration matrix.
- **H3 token reduction.** The Structural Acceleration section of the unified
  node exposes the implemented [h3.c](https://github.com/antirez/h3.c)-derived
  target-video token reduction. The Simple node adds **Light** and **Fast**
  recipes; the old 40/30-block policy is Fast. Masked video now retains
  per-token modulation, and differing mask partners use unreduced tokens with
  a logged reason. SOL and Veda packed metadata is reduced/restored together,
  including on interruption. See the [validation report](benchmarks/m5-ultra-h3-presets.md).

  - `token_reduction` runs blocks 0–3 at full sequence length, horizontally
    pairs only the target-video tokens, and leaves text, conditioning,
    reference, and audio token counts unchanged. Their outputs can still change
    through interaction with the reduced video stream. During the first ten
    evaluations it restores
    the full sequence before block 40; later it restores before block 30. Each
    restored token is its original full-resolution value plus its pair's learned
    update, scaled by `reduction_scale`.
  The published h3.c M5 Max result at 512×512×22 was 39.13 → 28.06 seconds of
  denoising for token reduction alone (28.3%), with final video/audio latent
  relative-L2 changes of 5.56%/15.14%. That is a native h3.c measurement, **not
  a predicted ComfyUI/MPS gain**; PyTorch allocation, graph, and attention costs
  differ. This implementation also retains an owned
  full-resolution bypass and pooled baseline, so it reduces middle-block
  attention/MLP work but may briefly raise activation memory at the reduction
  boundary.

  Layer thinning was removed after real 15-second H3 renders damaged audio with
  both the combined configuration and `active_layers=45` alone. A saved legacy
  workflow that combines token reduction with fewer than 50 layers now runs
  token reduction at all 50 layers and logs that the old value was ignored; a
  legacy thinning-only configuration fails with an explicit error instead of
  silently producing bad audio.

  Select Spectrum inside the unified node to apply forecasting before token
  reduction automatically. MotionCache and FirstBlockCache use the same
  controlled ordering. Sparse attention is
  kept logically aligned, but stacking all three approximations compounds
  spatial and temporal error and must be A/B tested. With Turbo's 4–8 steps,
  test the LoRA alone before adding cache or structural approximation.
- **H3 Sparse Attention.** The Sparse Attention section of the unified node is
  corrected Sol-style block-sparse self-attention,
  routed by **NVIDIA's released H3 policy (`diag64`)** adapted to Metal.
  A block-centroid proxy picks the exact KV blocks per head and 64-row query
  block; the `sol_ext` Metal 4.1 kernel computes only those; and every
  *rejected* block is folded back through its centroid with `log(block_length)`
  multiplicity, as extra columns in the same online softmax. Anything
  unsupported delegates to this package's own dense router, so declined calls
  still reach flash41.

  The policy, verbatim:

  - 64-token KV blocks and 64-row query blocks;
  - one **analytic** threshold per query block,
    `mu + tau*sqrt(sigma2 + 1e-6)` in log2 units, where `mu` and `sigma2` come
    from the diagonal covariance of that head's key centroids — *not* from the
    proxy row's own spread;
  - strict `proxy > threshold`;
  - the whole `[0, video_start)` prefix (text + conditioning + audio) is one
    contiguous exact KV sink, rounded outward to a 64-token boundary, and every
    query block overlapping it is fully dense;
  - a local ±1 block band;
  - the first two transformer blocks dense;
  - no Morton reordering, no temporal band.

  **The earlier empirical router was rejected.** It thresholded each 32-row
  compute tile on the mean and variance *of its own proxy row*, which adapts the
  threshold to whatever spread the row happens to have — tightening exactly
  where the row is flat and the approximation is least safe. It reached 13%
  density, but by discarding structure the model needed; the visible symptom was
  static-background morphing (see the rejected-router findings below). It is
  retained for one diagnostic cycle behind `ASFP8_SOL_ROUTER=legacy_empirical`
  and is reachable from no preset and no widget. `tau` is **not comparable**
  between the two routers.

  **Routing and attention execute on the GPU, with the sparse kernels in
  Metal.** The first version kept the route construction and correction in
  PyTorch and merged the correction from an exported log-sum-exp:
  algebraically clean, and measured at 25.5 ms routing + 93.6 ms correction
  against an 85.8 ms kernel and 220 ms dense attention — the scaffolding cost
  more than the kernel saved at *every* density. `sol_diag_route` now emits a
  compact block bitmask once per attention call, and `sol_masked_corrected_fwd`
  consumes it. Both 32-row compute tiles of a query block read the *same* mask
  word, so the TensorOps tiling is free to stay at 32 without the routing policy
  following it. Block means, key-centroid statistics and diagnostic density
  accumulation still use PyTorch/MPS operations around those kernels, and the
  diffusion wrapper reads the runtime sigma once per denoising evaluation.
  Selected-block counts remain on-device and are read once at the end of the
  sampling run rather than once per attention block.

  **Route parity is bit-exact.** The Metal route and the Python reference
  (`route_blocks_diag_reference`) agree on 100.0000000% of entries at every
  tested shape — aligned, ragged query *and* KV tails, sinks on and off a block
  boundary, `tau` from 0.0 to 1.5 — with a 0.0000pp density difference. The
  route runs entirely in fp32, pooled query centroids included, which is why it
  lands exact rather than merely inside the ≥99.99% / 2e-4 tolerance the plan
  allows. Corrected attention under that route matches the *same sparse route's*
  fp32 specification to 2.7e-3 relative error, against 2.3e-3 for the same
  kernel at full density. This establishes kernel arithmetic fidelity; it does
  **not** measure the approximation error against full dense attention or imply
  visual equivalence.

  Measured M5 Max, H=56 D=128 bf16, against the flash41 dense kernel this
  package already ships, on synthetic Gaussian activations. Whole path,
  including block means, the route and padding:

  | seq | dense | **diag64** | density | **vs dense** |
  |---|---|---|---|---|
  | 5378 (1s) | 23.2 ms | **12.0 ms** | 31% | **1.94x** |
  | 8192 | 47.4 ms | **23.3 ms** | 28% | **2.04x** |
  | 17000 (~5s) | 217.7 ms | **86.6 ms** | 22% | **2.51x** |

  (`tau=1.0`, 512-row sink. Densities on real H3 activations will differ —
  attention mass concentrates far more than Gaussian noise does.)

  **`tau=0` is not a promotion candidate.** At `tau=0` the same shapes measure
  53–57% density and **1.00–1.02x** against dense — the centroid pass costs
  exactly what the skipped blocks save. So the calibration ladder
  (`tau` 1.0 → 0.75 → 0.50 → 0.0) has a hard floor: if quality only passes at
  `tau=0`, sparse attention fails the ≥5% speed gate by construction and should
  be left disabled rather than shipped.

  **Status: implemented and route-verified; the 20-step quality calibration has
  not been run.** The defaults below are the plan's starting point, not a
  calibrated result: `tau=1.0`, `start_percent=0.50` (protects the first ten of
  twenty evaluations), `end_percent=0.85` (protects the final three),
  `dense_leading_blocks=2`, `block_size=64`, `temporal_taps=0`,
  `min_sequence_tokens=8192`. The intended split is evaluations 0–9 dense,
  10–16 sparse and 17–19 dense. A scheduler-level regression test uses the real
  shift=12.0 20-step sigma sequence and verifies that exact split. Quality is
  still the gate, and it has not yet been measured on real `diag64` renders; do
  not treat the synthetic speedups above as a shipped quality result.

  `block_size` and `temporal_taps` remain as widgets **only** because ComfyUI
  maps saved-workflow values positionally, so removing them would shift every
  earlier value into the wrong slot. `diag64` *refuses* any value but the
  default for both, rather than silently ignoring them.

  **Findings that rejected the empirical router** (kept because they constrain
  what any router here can do). On a synthetic static shot the frame-to-frame
  RMS of co-located tokens was 1.22x the exact-attention value at 13.8% density;
  the visible symptom is background flicker/morphing on content that should be
  still. A rejected block contributes only its MEAN value vector, so at 20%
  density ~80% of the attention mass arrives as block averages carrying no
  spatial detail — and cross-frame agreement, which is what makes a static
  background hold its shape, is computed through exactly that channel. Five
  interventions were built and measured; none helped:

  | intervention | result |
  |---|---|
  | frame-aligned grid (512x512 divides exactly) | no change |
  | `dense_leading_blocks=10` | no change |
  | temporal band, synthetic | **worse** (1.22x -> 1.64x) |
  | `temporal_taps=2`, real clip | no change |
  | rank-1 structure correction | **1.05x** even at intrinsic rank 8 |

  The rank-1 term (`M_j = sum_t (k_t-kbar)v_t^T ~ u_j w_j^T`, the first-order
  Taylor term Sol appears to use) was prototyped and measured on both random and
  deliberately low-rank activations. It recovers ~5% of the error at best: what
  the centroid discards is genuinely high-rank. The temporal band is why
  `temporal_taps` defaults to `0` and is now refused outright: the spatial band
  is translation-invariant — every tile forces the same relative offsets — while
  a temporal band is position-dependent with edge effects at the first and last
  frames, which is itself a source of frame-to-frame variation.

  What changes under `diag64` is the *threshold*, not the centroid recovery, so
  these findings bound the outcome without predicting it: the analytic threshold
  is expected to buy quality by spending density, and NVIDIA's own H3 numbers
  show only a **1.10x** incremental Sol-Attn gain. Expect a quality-preserving
  Mac result more modest than the discarded 13%-density implementation — which
  is consistent with NVIDIA shipping sparse attention only in their "super
  efficient" preset and disabling it in "quality preserving".

  The correction is what makes the route usable at all: on matched routing,
  folding the rejected centroids back cut the error against full dense
  attention by **7–11x** versus dropping them. That is why this is not a port
  of [KingGore's Blackwell node](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell),
  whose CUDA-gated implementation hard-drops rejected blocks (and indexes
  block 0 ambiguously); only the general routing idea is shared. Adapted from
  [Sol-Attn](https://arxiv.org/abs/2607.24027), not the original fused NVIDIA
  kernel. `hard_drop` and `dense_reference` are diagnostics, not production
  modes.

  **FlexAttention was evaluated and rejected.** torch 2.13 does compile
  `score_mod`/`mask_mod` into a Metal shader on MPS, and captured-buffer
  routing works — but at H3 shapes and 26% block density the generated kernel
  measured **5–9x slower than dense SDPA** (1214 ms vs 140 ms at S=8192), i.e.
  roughly 25x too slow to break even against flash41. `create_block_mask` also
  materializes a dense token mask first and tries to allocate 112 GiB at
  S=16384/H=56. Do not extrapolate PyTorch's headline 12.3x sparse-SDPA figure
  to this workload; it came from a 0.8%-density sliding-window case.
- **Spectrum hidden-feature forecasting (external reference).** The native H3
  [Spectrum adaptation](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
  was tested independently rather than vendored. Its forced-actual wrapper
  reproduced decoded video and audio bit-for-bit and retained 409.3 MiB of MPS
  history for this 1-second shape. One forecast at its default step 5 reduced
  the transformer count from 20 to 19 but reached only 0.657 video SSIM /
  14.6 dB audio SNR. A favorable isolated forecast at step 15, after eight
  retained actual features and before four exact tail steps, reached 0.937 /
  17.7 dB. The external six-forecast preset was therefore not run, and neither
  case was promoted to 5 seconds. The result suggests that H3's post-block
  feature trajectory is not accurately forecast by this uncalibrated
  Chebyshev/local-linear configuration, even though its wrapper and current
  output-head reconstruction are exact when forecasting is disabled.
- **ComfyUI EasyCache node.** Current ComfyUI already includes the upstream
  [EasyCache](https://github.com/H-EmbodVis/EasyCache) algorithm, including
  support for H3's paired video/audio latent.
  Insert it between `UNETLoader` and both model consumers. On the 5-second,
  864×480, 20-step H3 reference workflow, `0.30 / 0.20 / 0.90` skipped 6/20
  model evaluations: sampling 428s -> 292s and end-to-end 491s -> 337s
  (about **1.46x**). This is an aggressive preview preset: a matched short
  render measured video SSIM 0.912 and visibly changed motion/pose. Lower
  thresholds trade most of that gain back for fidelity.
- **Reduced H3 video-VAE tile overlap.** Set
  `ASFP8_MINIMAX_VAE_OVERLAP=32` before startup to reduce the reference
  64-pixel overlap while keeping 256-pixel tiles. At 864×480 this reduces the
  decoder grid from 15 tiles to 8. In a matched 1-second test, the
  post-sampling portion fell from 16.1s to 6.0s, video SSIM was 0.980, and
  audio remained bit-identical. VAE
  decode is a small share of a normal 20-step render, so expect only roughly
  **3-6% end-to-end** on a 5-second clip. Watch tile boundaries, fog, gradients,
  and fast motion before adopting it.
- **Batched H3 video-VAE tile decode (rejected speed experiment).**
  `ASFP8_MINIMAX_VAE_DECODE_BATCH=N` still exposes the exact scheduling probe:
  tiles are stacked only on the batch dimension, while tile order and blending
  are untouched. A decoder-shaped microbenchmark improved 131 → 68 ms at
  batch 5 with a maximum difference of 0.0, but that result did not transfer to
  the real long decode. At 864×480 × 124 frames, sequential decode took
  36.21 s versus 38.43 / 38.50 / 39.03 / 40.25 s at batches 2 / 4 / 5 / 8.
  It therefore remains off by default and is retained for controlled
  experiments, not recommended as an H3 speed setting. Peak decoder memory
  also scales with the batch.
- **Different H3 VAE tile size.** `ASFP8_MINIMAX_VAE_TILE_SIZE` is exposed for
  controlled experiments, not as a recommended speed preset. Larger tiles can
  reduce the grid but make the VAE transformer's attention much more expensive
  and can sharply increase peak memory; smaller tiles do the reverse. Values
  above 320 are rejected because the observed attention/memory cost and tile
  artifacts made them unsuitable as a supported path.

`dev/bench_workflows.py` can inject EasyCache or the unified H3 Acceleration
node and pin duration, steps and megapixels without changing the source
workflow. `--h3cache 0 0.20 0.85 1` runs cache calibration without skipping;
`--h3cache-audio-weight 1.0` sets the audio reuse veto. The predictor defaults
to `first_block_cache`; `--h3cache-predictor first_block_cache_per_step` selects
the per-step detector, while `oracle` and `block_oracle` provide diagnostics.
`--h3-sparse corrected 1.0 0.20 2 8192` enables sparse attention after the cache.
Only the requested groups are enabled by benchmark injection. Retired
forecasting/suffix modes and uncalibrated preset flags have been removed.
Sparse runs require a server reporting torch 2.13+.
Its JSONL record includes prompt IDs and generated artifact paths so matched
runs can be passed to `dev/compare_av.py`. The comparison reports decoded
bit-exactness, per-frame Gaussian-window SSIM, spatial and temporal-delta PSNR,
and audio exactness/SNR/channel correlation. This avoids comparing two
different latent shapes—or judging an audio-video cache from spatial SSIM
alone—by accident.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ASFP8_NATIVE` | off | Master switch: enables the six exact/native feature groups below; it does not apply LoRAs or approximation nodes |
| `ASFP8_FP8_EXT` | off | fp8-native `_scaled_mm` kernel (M5+, Metal 4.1) |
| `ASFP8_FP8_SGEMM` | off | staged half×fp8 W8A16 backend |
| `ASFP8_INT8_EXT` | off | int8 W8A8 kernel + fused epilogue (M5+) |
| `ASFP8_ATTN_ROUTER` | off | per-call attention routing (flash41 / bmm / sub-quad) |
| `ASFP8_ATTN_BACKEND` | unset | `pytorch` selects the tested PyTorch 2.14 MPS attention route; unset retains the existing router |
| `ASFP8_MPP_QCACHE` | on | exact query-load hoisting under the PyTorch attention route; guarded to validated M5 Ultra / 2.14.0 build, headers and shapes; H3 supports up to 65,536 tokens, other families up to 32,768; `0` restores the unmodified PyTorch kernel |
| `ASFP8_ATTN_NA` | 1 | production independent-simdgroup flash41 kernel; `0` restores the older cooperative kernel |
| `ASFP8_DIT_EXT` | off | DiT block-fusion kernels (all families) |
| `ASFP8_INT8_EMBEDDING` | on | preserve already-dequantized INT8 embedding weights across CPU/MPS casts; compatibility repair independent of native kernels |
| `ASFP8_QWEN21_FAST` | on (under DIT_EXT) | exact Qwen-Image 2.1 fusions on the validated M5 Ultra / PyTorch 2.14.0 build; `0` retains the encoder fix and generic native paths |
| `ASFP8_SOL_LAYOUT` | `head_major` | optional SOL physical layout; `legacy` selects the prior wrapper/shaders for diagnosis. New layout preserves prior SOL output bytes, not exact-native attention bytes; restart after changing it |
| `ASFP8_YUE2_FAST` | on | exact phase-local cast-weight reuse, guarded vocabulary projection and single-token BF16 RoPE for the MPS YuE2 encoder; scoped acoustic attention under the existing PyTorch query-cache route; `0` restores the preceding YuE2 paths |
| `ASFP8_YUE2_WEIGHT_CACHE` / `ASFP8_YUE2_VOCAB_HEAD` / `ASFP8_YUE2_ATTN` / `ASFP8_YUE2_ROPE` | on | independent YuE2 switches; exact against the existing GPU stack |
| `ASFP8_YUE2_CACHE_GIB` / `ASFP8_YUE2_VERBOSE` | 8 / off | temporary weight-cache cap (also limited by device memory/headroom) / optional phase counters; cache size 0 disables storage |
| `ASFP8_QWEN21_ROPE` / `ASFP8_QWEN21_MODULATE` / `ASFP8_QWEN21_RESIDUAL` | on (under QWEN21_FAST) | independent switches for Q/K RMSNorm + RoPE, modulation, and gated residual fusions |
| `ASFP8_SDXL_GEGLU` | on (under DIT_EXT) | exact FP16 GELU + multiplication fusion for SDXL on PyTorch 2.14 MPS; `0` restores eager GEGLU |
| `ASFP8_CONV3D` | off | conv3d → sliced conv2d + Wan VAE fast path |
| `ASFP8_FLUX_FAST` / `KREA2_FAST` / `LTX_FAST` / `WAN_FAST` / `WAN_VAE_FAST` / `IDEOGRAM_FAST` / `MINIMAX_FAST` | on (under DIT_EXT/CONV3D) | per-architecture kill switches |
| `ASFP8_KREA2_GLUE` | on | exact Krea2 normalization/modulation, residual, SwiGLU and attention-layout fusions; guarded to M5 Ultra / PyTorch 2.13–2.14 under KREA2_FAST |
| `ASFP8_KREA2_ATTN_SCHEDULE` | on | exact synchronized Morton attention schedule for M5 Ultra BF16 Krea2, B=1, 48Q/12KV, D=128, 8192–12287 tokens; other geometries keep their existing schedule |
| `ASFP8_MINIMAX_ATTN` / `ASFP8_MINIMAX_BLOCK` | 1 | MiniMax direct QKV/attention-output layout and segmented block-glue fusions |
| `ASFP8_MINIMAX_GRAPH` | 1 | exact guarded H3 final-layer graph rewrite (target RMSNorm + modulation + fp32 casts) |
| `ASFP8_MINIMAX_VAE_FAST` | 1 (under DIT_EXT) | FP16 H3 video-decoder QK normalization/RoPE/layout and elementwise kernels; `0` restores core forwards |
| `ASFP8_MINIMAX_VAE_SWIGLU` / `ASFP8_MINIMAX_VAE_MERGE` | 1 | per-feature switches for exact FP16 SwiGLU and attention-output merge/sanitization |
| `ASFP8_MINIMAX_RELAXED` | 0 (off) | explicit experimental H3 kernel preset; independent of `ASFP8_NATIVE`; per-feature settings override it; restart to change |
| `ASFP8_MINIMAX_ATTN_MODE` | exact (tuned under RELAXED) | `exact`, original relaxed `tail`, or `tuned`; the relaxed preset supplies its selected mode when unset; long H3 only, workflow attention overrides take precedence |
| `ASFP8_MINIMAX_VAE_DENSE_ATTENTION` | 0 (1 under RELAXED) | optional non-bit-exact dense FP16 H3 VAE attention; requires native VAE/Metal attention; independent of sampling |
| `ASFP8_MINIMAX_VAE_ATTN_TUNING` | 0 (1 under RELAXED) | tuned batch-dependent tiles within optional dense VAE attention; `0` retains the original dense kernel settings |
| `ASFP8_MINIMAX_VAE_QKV` | 0 (1 under RELAXED) | FP16 QKV projection with FP32 accumulation and fused normalization/RoPE/layout; `0` retains the existing GEMM; independent of dense attention |
| `ASFP8_MINIMAX_VAE_INT8` | 1 (under native INT8/VAE kernels) | native W8A8 for an explicitly loaded H3 INT8 ConvRot VAE; `0` restores the prior weight-only MPS decoder; FP16 VAE and sampler unchanged |
| `ASFP8_INT8_QKV_SCATTER` | 1 | exact H3 INT8 QKV epilogue directly to three attention-layout tensors; `0` restores the full fused-QKV buffer |
| `ASFP8_INT8_QKV_NORM_ROPE` | 1 | exact D=128 H3 INT8 projection + optional standard runtime-LoRA delta + Q/K RMSNorm + partial-RoPE epilogue |
| `ASFP8_MINIMAX_CROSS_BLOCK` | 1 | exact MLP-gate → next-attention-AdaLN carry when no workflow block replacement is active |
| `ASFP8_INT8_ACT_CACHE` | 1 | one-slot reuse of identical rotate + row-quant inputs shared by projection groups |
| `ASFP8_INT8_LONG_SCHEDULE` / `ASFP8_INT8_LONG_QKV_TAIL` | 1 | exact long-M INT8 schedule / exact QKV suffix dispatch (active only at M≥8192) |
| `ASFP8_INT8_OPROJ_FUSE` | 1 | fused o-proj pre-pass: `1` exact D=256 rotation or full cross-head FWHT for D<256, `2` always full FWHT, `0` off |
| `ASFP8_INT8_SWIGLU` | 1 | keep H3's SwiGLU down-projection in native W8A8; `0` restores ComfyUI's stock dequantized W8A16 seam |
| `ASFP8_INT8_SWIGLU_ROT` | 1 | fuse SwiGLU + convrot before H3's INT8 FC2 (rare ±1 quantized-byte drift from FWHT reassociation) |
| `ASFP8_INT8_MLP_BRIDGE` | 1 | tiled H3 FC1 → SwiGLU/convrot/premax quant → FC2 bridge; exact relative to the native SwiGLU/convrot route, `0` restores its materialized tensors |
| `ASFP8_INT8_MLP_LORA_TILE_ROWS` | 2048 | row scratch size for the exact standard-LoRA H3 FC1/SwiGLU/FC2 bridge; minimum/alignment 128 |
| `ASFP8_MINIMAX_VAE_OVERLAP` | unset (reference 64) | approximate H3 video-VAE tile overlap; 32 is the tested faster setting |
| `ASFP8_MINIMAX_VAE_DECODE_BATCH` | 1 (off) | exact experimental tile batching; slower in the real 124-frame test, with peak decoder memory scaling by batch |
| `ASFP8_SOL_EXT` | 1 | set `0` to force-disable the Sol sparse-attention Metal kernel (the H3 sparse node then refuses to apply) |
| `ASFP8_SOL_ROUTER` | `diag64` | set `legacy_empirical` to route H3 sparse attention with the **rejected** 32-row-tile router (proxy-row mean/variance threshold), for diagnostics only. `tau` is not comparable between the two |
| `ASFP8_MINIMAX_VAE_TILE_SIZE` | unset (reference 256) | experimental H3 video-VAE tile size (multiple of 16) |
| `ASFP8_AUTOFIX` | on | self-heal JIT prerequisites at boot (install ninja, restore stripped torch headers) |
| `ASFP8_WAN_EXACT` | on | Wan 14B exact FP16 INT8 output fusion and batched attention query caching on the validated M5 Ultra / PyTorch 2.14 build; `0` restores the preceding native paths |
| `ASFP8_WAN_FP16_GEMM` / `ASFP8_WAN_ATTN` | on | independent switches for the Wan FP16 projection epilogue and exact attention optimization, under WAN_EXACT |
| `ASFP8_WAN_CROSS_ATTN` | on | also use the exact Wan query-cache/layout kernel for 512-token text cross-attention; `0` keeps only its self-attention extension |
| `ASFP8_WAN_VAE_CHUNK` | 16 | Wan VAE frames per causal pass (even, 2–32; lower caps decode memory peak) |
| `ASFP8_FREE_MEM` | on | MPS free-memory accounting fix (patch #30) |
| `ASFP8_VAE_SLICE_FIX` | on | VAE slice_attention INT_MAX cap (patch #31) |
| `ASFP8_SEEDVR2_VAE_SYNC` | on | wait between SeedVR2 VAE spatial tiles and video residual blocks on MPS to bound temporary allocations; scheduling-only, independent of native/relaxed flags |
| `ASFP8_SEEDVR2_VAE_VIDEO_SYNC` | on | include residual-block waits inside SeedVR2 video tiles; setting 0 restores the previous spatial-only schedule and its video memory growth |
| `ASFP8_SEEDVR2_LOADER` | on | supply official missing conditioning for raw original SeedVR2 7B checkpoints on macOS; existing conditioning and weights are preserved |
| `ASFP8_SEEDVR2_FAST` | on | SeedVR2 exact performance additions; native sampling/SiLU kernels require `ASFP8_DIT_EXT=1` (implied by `ASFP8_NATIVE=1`); `0` at startup retains the memory/error safeguards |
| `ASFP8_SEEDVR2_ATTN_FIX` | on | pack MPS attention windows into bounded buffers to avoid oversized-backing-storage `INT_MAX` failures; independent of the native kernel flag |
| `ASFP8_SEEDVR2_ATTN_BATCH` | 8 with native/FAST, otherwise 1 | maximum equal-size windows per call, clamped to 1–16; small safe images keep the original loop |
| `ASFP8_SEEDVR2_VAE_PACK` | on | reuse folded convolution weights within a tiled SeedVR2 VAE call; requires FAST and VAE_SYNC wrappers |
| `ASFP8_SEEDVR2_VAE_SILU` | on | native FP16 single-frame SiLU + causal padding + channel packing, retaining MPS Conv2D |
| `ASFP8_SEEDVR2_MOD` / `ASFP8_SEEDVR2_RMS` / `ASFP8_SEEDVR2_ROPE` | on | independent native SeedVR2 BF16 sampling fusions; RMS retains PyTorch's existing mean/rsqrt reduction |
| `ASFP8_SEEDVR2_VAE_GN` | off | experimental exact GroupNorm + SiLU + packing on tested PyTorch 2.13; whole-decoder timing was neutral, so it is not enabled by default |
| `ASFP8_FREE_INTERLEAVE` | on | drain the MPS pool between models on explicit free-all requests |
| `ASFP8_MPS_RESERVE_GB` | max(16, 15%) | OS memory reserve held back from comfy's estimate |
| `ASFP8_FA41_F32S` | off | fp32 score tiles in flash41 |
| `ASFP8_FP8_EXT_MIN_DIM` | 8192 | fp8 kernel size gate |
| `ASFP8_ENABLE_ONLY` / `ASFP8_DISABLE` | — | bisection: install only / all except these patch modules |
| `ASFP8_ATTN_DEBUG` / `ASFP8_EXEC_TRACE` / `ASFP8_PROFILE` | off | route logging / per-node timing / profiling (stderr) |
| `ASFP8_EXEC_TRACE_SYNC` | off | with EXEC_TRACE: time the GPU tail drain after the sampler loop (splits real overhead from queued GPU work) |
| `APPLESILICON_FP8_PSUTIL` | auto | psutil fix: `auto` / `off` / `force` |
| `MTLFLASHATTN_SDPA*` | — | patch #5 gates (see mtlflashattn docs) |

## Install

```bash
cd <your ComfyUI>/custom_nodes
git clone https://github.com/luix93/ComfyUI-AppleSilicon-Optimized
pip install -r ComfyUI-AppleSilicon-Optimized/requirements.txt
```

Restart ComfyUI. The only required dependency (`mtlflashattn`) installs
automatically; the native kernels additionally need `ninja` + Xcode
command-line tools + torch's C++ headers — and the node **self-heals the
first two Python-side gaps at boot**: it installs `ninja` into the venv and
restores torch's C++ headers from a local cache when the bundled torch ships
without them (ComfyUI Desktop strips them; a healthy venv seeds the cache).
`ASFP8_AUTOFIX=0` opts out. If prerequisites still can't be met, a loud
banner says exactly what's missing instead of silently rendering at half
speed. For ComfyUI Desktop, install into the app's own venv
(`.venv/bin/python3 -m pip ...`) — the bundled torch is the one that matters.

Startup prints one line per active patch
(`[AppleSilicon-Optimized/<patch>] ...`); with `ASFP8_NATIVE=1` you'll also
see the master-switch line listing what it implied. Kernels JIT-build on
first use (a one-time delay per dtype) and print nothing when they fall back
— set the debug vars above to see routing decisions.

> Primary release validation: `ComfyUI-torch213-latest` on M5 Max / 128 GB,
> macOS 27 dev beta, PyTorch 2.13.0, Python 3.13, ComfyUI Desktop. M1–M4 get
> the compatibility
> layer and flash attention; the int8/fp8 matmul kernels need M5-class
> TensorOps and fall back cleanly below that.

## Field notes (what the benchmarks taught us)

A few load-bearing lessons encoded in this repo, for future work:

- **Check the model's compute dtype per family.** Wan runs fp16 where the
  flux family runs bf16; a bf16-only kernel gate silently dropped Wan's 21.5k
  token attention onto a chunked-python path that was ~70% of the render.
  Dtype-gated routes degrade *silently* — `ASFP8_ATTN_DEBUG=1` exists for this.
- **Size gates on actual bytes, not worst-case shapes.** Two separate bugs
  came from gating on max-sequence² instead of the real `Lq×Lk`.
- **Check the sampler for hidden work.** LTX's `euler_ancestral_cfg_pp`
  disables comfy's cfg=1 optimization and double-evaluates the model; plain
  euler is frame-identical and half the cost.
- **"Launch-bound" is a hypothesis, not a diagnosis.** The Wan VAE looked
  launch-bound (dozens of tiny decoder passes); batching them changed
  nothing. Attribution showed strided reductions and cat chains — memory
  patterns, not launches — and the convs already at the platform ceiling.
- **The same trap applies to MiniMax's tiled ViT VAE.** A 124-frame decode
  invokes 105 independent spatial tiles, but batching them is slower on M5:
  sequential 36.21 s versus batch-2 38.43 s, batch-4 38.50 s, batch-5
  39.03 s, and batch-8 40.25 s in the one-step control workflow. All 105
  batch-4 tile outputs were bit-exact, but launch reduction cannot repay the
  decoder's larger working set, so the experiment was rejected as a recommended
  fast path and retained only behind its explicit research switch.
- **MPS traps**: conv2d silently miscomputes with non-contiguous weights;
  per-tap rounding compounds through recursive causal VAE caches; tensor
  prints during execution go to a swallowed stdout (use stderr); the first
  call after a kernel build pays PSO compilation — benchmark run 2.
- **Check the framework's accounting before writing a kernel.** The single
  biggest Wan win of the latest round wasn't a kernel at all — it was noticing
  comfy re-uploaded 13.9 GB models every run because its MPS free-memory
  estimate was wrong (patch #30). ~14 s/run, recovered by fixing arithmetic.
- **Measure the ceiling before optimizing toward it.** The attention rewrite
  was abandoned after a probe showed even MPS's own GEMM couldn't clear
  ~35 TF/s at attention's shallow reductions — the target didn't exist. A
  half-day of kernel work saved by a two-hour measurement.
- **Trust same-session numbers only.** Thermal drift on a sustained run is
  ±10%, larger than most effects being chased; a cross-session "regression"
  turned out to be a warm vs cool machine. A heat-soaked chassis after hours
  of benching degrades far past that (observed 50 → 78 s on identical runs)
  and recovers after a ~2-minute cooldown.
- **Random-data parity does not prove real-data parity.** A rotation kernel
  bit-matched the eager path across 23M random values, then diverged on real
  renders: real activations carry exact zeros and duplicated rows (the
  zeroed-text unconditional pass) that land on fp32-summation ties. The
  pipeline itself is deterministic — same seed + same code = pixel-identical
  PNGs across process restarts — so a same-seed pixel-diff is the cheap,
  decisive bit-exactness test. Use it before shipping any numerics change.
- **Bounds-checking is your job in tensor-ops kernels.** Static-extent slices
  don't bounds-check, fresh test allocations are zeroed so parity tests can't
  see OOB reads, and the failure mode is *intermittent* whole-image
  corruption on dirty pool memory. Zero-pad ragged inputs host-side and test
  under a NaN-poisoned allocator.
- **OS betas can break Metal mid-flight.** A macOS beta changed tensor-ops
  operand address-space rules; every JIT kernel failed to compile, and the
  per-call compile *re-attempts* cost 33 s/it. Fixes that ship: address-space
  scrubbing on operand types, plus a compile-failure latch so a broken kernel
  disables itself for the session instead of retrying per call.
- **The s/it readout lies when submission outruns the GPU.** tqdm's per-step
  time measures CPU submission pace; on Krea2 the GPU actually ran 1.81 s/step
  against a displayed 1.56, with ~1.8 s of queued DiT work draining silently
  after the loop. What looked like 3 s of "overhead" to hunt was just more
  model. `ASFP8_EXEC_TRACE_SYNC=1` splits the two — tail-probe before hunting
  phantom overhead.
- **Block-ordered kernel benches produce phantom wins.** GPU clock ramp gave
  candidate variants +9–29% "improvements" that evaporated under interleaved
  A/B (alternate baseline/candidate per round, median of ratios). Shapes that
  read 51 TF/s in a cold block read 97 under sustained interleaved load. Every
  kernel claim in this repo now requires the interleaved methodology
  (`dev/tournament_int8.py --ab`, `dev/bench_i8_attn.py`).
- **int8's 2× only exists at deep reductions.** Measured on our own GEMM
  kernel: 104 TF/s at K=6144, 93 at K=512, 33 at K=256, 25 at K=128. That one
  curve explains the whole architecture — int8 for the model's linears
  (K = 4608–24576), fp16 for attention (K = head_dim = 128) — and killed two
  otherwise-plausible sprints (int8 attention, wider GEMM descriptors) with
  data instead of intuition.

## Credits

- [ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)
  and [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn)
  (Paweł Mazurkiewicz, MIT) — the original compatibility layer this project
  grew from, and the flash-attention-on-Metal-4-TensorOps kernel design.
  mtlflashattn backs the compatibility-tier SDPA reroute (patch #5) directly;
  our `flash41` extension (patch #19's fast path) is built *around* his kernel
  skeleton — we make it compile on macOS 27, add the wide-tile/mergeP/static-
  reduction/seq-adaptive performance work (~2× over the straight port), and
  extend it to fp16, video sequence lengths, and native GQA. The QK→softmax→PV
  cooperative-tensor structure is his; the compile path, host dispatch, and
  tuning are ours. His MIT copyright is preserved in `flash41.mm`.
- [Cider](https://github.com/Mininglamp-AI/cider) (Mininglamp, MIT) — the
  register-tiled cooperative-TensorOps int8 `matmul2d` that patch #17's GEMM
  is ported from, and the design reference for the staged/swizzled fragment
  work.
- [ccv / NNC](https://github.com/liuliu/ccv) (Liu Liu, BSD-3-Clause) — the
  `NAAttentionKernel` structure DrawThings runs is the recipe our `na`
  attention kernel follows (independent simdgroups, direct-device K/V through
  static-extent slices, row-reduction softmax, relaxed precision). The Metal
  4.1 port, the register-chained P variant, the ragged-tail handling, and the
  torch host integration are ours.
- [h3.c](https://github.com/antirez/h3.c) (Salvatore Sanfilippo) — the released
  MiniMax H3 horizontal token-reduction schedule adapted by the structural
  workflow node. The ComfyUI implementation retains its own full-resolution
  bypass and composes cache and sparse-attention metadata around that schedule.
- [Sol-Attn](https://arxiv.org/abs/2607.24027) and NVIDIA's released H3 routing
  policy — the block-centroid sparse-attention method behind `diag64`. The
  Metal router, corrected sparse kernel integration, ragged handling, and
  ComfyUI ownership/fallback plumbing in this repository are local work.

## License

The combined distribution, including bundled Spectrum, uses
[GPL-3.0-or-later](LICENSE-BUNDLED). Original project code retains its
[MIT notices](LICENSE). See [third-party attribution](THIRD_PARTY_NOTICES.md)
for the licenses and provenance of bundled components.
