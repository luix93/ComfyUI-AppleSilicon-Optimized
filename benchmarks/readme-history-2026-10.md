# Historical results and experiments — README archive

Archived from the README on October 10, 2026. These sections preserve the
original measurements, rejected experiments and engineering observations.
Statements about defaults, hardware support and test counts describe their
original context; use the [current README](../README.md) for release behavior.
No new benchmark was run during this documentation pass.

SeedVR2’s exact VAE follow-up reduces a fresh matched image baseline from **47.44 to 45.15 s**, with approximately **32 GiB** peak Metal memory. Bounded video scheduling reduces the five-second 416×736 → 832×1472 fixture from **128.18 to 124.66 s**, peaking at **33.58 GiB**. Both preserve encoded/sample/frame tensor bytes. See the [convolution, fusion and bounded-scheduling audit](../benchmarks/m5-ultra-seedvr2-vae2141.md).

These are measurements from the isolated `ComfyUI-AppleSilicon-2.14-Update` install. Its actual core revision is **`v0.39.0-33-g1d2ea294`**, rather than the tagged v0.39.2 release. The earlier install remains untouched. Qwen edit uses a new generated reference; confidential inputs and SATA files were not accessed. Flux and SeedVR output bytes change across this stack update even in stock mode; restored fusions preserve their current optimized outputs. The existing INT8 matrix routes are not a blanket guarantee of parity with stock.

[New-stack timings, compatibility repairs and validation](../benchmarks/m5-ultra-torch2141-retest.md). All final retests passed after the compatibility repairs. The legacy full relaxed H3 preset currently measures **155.33 s**, slower than **141.65 s exact optimized**, and remains default off. Its September gains were measured on Torch 2.13, before the faster 2.14 exact route.

Krea2's October 10 **1024×1024 / 8-step** follow-up adds a narrowly selected exact INT8 GEMM tile: **6.24 → 6.13 s** E2E, with **5.88 s sampling**. Twenty native tests, 1120 actual GEMM byte checks and three prompt/seed fixtures preserve the preceding optimized outputs. The best relaxed attention prototype averages **6.00 s**, changes generated details and remains research-only. We have not reliably beaten Draw Things' reported sub-six-second E2E timing. See [the implementation, rejected experiments and image comparisons](../benchmarks/m5-ultra-krea2-oct2026.md).

Ideogram 4's October 10 exact kernel follow-up improves a fresh matched **18.41 → 17.78 s** control with identical sampler and decoded tensor bytes. Independent output-channel attention tiles and projection-specific INT8 schedules pass 71 native tests. The relaxed INT8 attention prototype reaches **15.22 s** against its matched 18.26 s control, but visibly changes image details and remains research-only. No further exact H3 candidate has established a repeatable complete-workflow win; newer dense INT8 attention saves only 2% sampling in a two-step long-clip screen and is not promoted. See [accepted changes, rejected experiments and comparisons](../benchmarks/m5-ultra-ideogram-h3-oct2026.md).

**Current optional H3 attention · 20 steps / 124 frames:** matched camera-scene averages against our exact optimized stack. Cache, token reduction and relaxed VAE are off; SOL remains non-bit-exact. All six mode/size pairs preserve their earlier output bytes.

| Attention mode | 864×480 E2E | 1344×768 E2E |
|---|---:|---:|
| Exact optimized | 146.40 s | 606.72 s |
| SOL INT8 Fast | 114.39 s | 384.59 s |
| SOL INT8 Hybrid Fast | **113.79 s** | **371.75 s** |

Hybrid saves **22.3% / 38.7%** in these matched tests. The large exact reference is 3.3% slower than the earlier batch; output bytes are unchanged. All 21 individual H3 recipes and three short combinations pass. Veda is slower on the small screen, and combining cache/sparse/token controls does not guarantee additive gains. Top-k Wide retains its morphing warning. [Complete mode timings, counters and comparisons](../benchmarks/m5-ultra-torch2141-retest.md).

**SeedVR2 video safety:** the initial stack retest's generated-reference 5/10-second upscales completed in **128.57 / 262.57 s**, peaking at **33.55 / 46.15 GiB** Metal memory, with repeatable outputs and no watchdog triggered. The exact follow-up now measures **124.66 s / 33.58 GiB** for the same five-second fixture; the ten-second case has not been rerun with the new schedule. The old video is missing, so these are new references at matching dimensions, not a strict old/new input comparison.

**YuE2 · current full-song results:** the same 83.2-second song measures **50.46 → 34.12 s** against the matched GPU route with the extra fusions disabled. Mothersuperior AR + real-audio v4 adapters measure **31.35 → 21.94 s** for the 20-second workflow. Paired current-stack tokens, tensors, waveform and FLAC bytes match. Across stacks the encoded audio frames and PCM checksums match; only the exporter vendor comment differs.

**Historical YuE2 measurements · October 7–8:**
 the saved `audio_yue2_text2music.json`
produced the same **83.2-second song in 33.48 seconds** on this M5 Ultra.
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
[Latest paired benchmarks and screened candidates](../benchmarks/m5-ultra-yue2-followup.md).
The [October 7 first pass](../benchmarks/m5-ultra-yue2.md) took the preceding GPU
stack from **354.18 to 48.46 s**; those earlier measurements remain documented.
For instrumental music, a [Mothersuperior two-LoRA workflow](../docs/yue2-mothersuperior.md)
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
[Full timings, numerical errors, all-preset visual comparisons and rejected experiments](../benchmarks/m5-ultra-h3-kitchen-attention.md).

**Also included in this update:** [Prompt Studio](../docs/prompt-studio.md) sends
connected references with the brief, offers a single Generate on queue toggle,
and caches discovered models for the canvas model picker. [H3 controls](../docs/h3-acceleration.md)
include the Simple recipe node, independent cache/attention/token controls,
pack-owned MotionCache/Spectrum/Veda runtimes and configurable attention memory.
[H3 LoRA](../docs/h3-acceleration.md) combines ordinary and Turbo adapters in an
ordered list with per-entry enable/weight controls. [SeedVR2's video safeguard](../benchmarks/m5-ultra-seedvr2-video-memory.md)
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
[Settings, complete timings, numerical errors and generated comparisons](../benchmarks/m5-ultra-h3-sol-topk.md).

**H3 runtime ownership · October 5:** MotionCache, Spectrum and Veda now use
locally maintained internal runtimes with direct APIs for our combined node.
The existing presets and algorithms are retained; upstream node wrappers and
unused media-analysis tools are removed from execution. Pinned sources remain
parity references, with original attribution and licenses preserved.
[Runtime details](../docs/h3-acceleration.md#internal-runtime-ownership)
· [Byte parity and matched timings](../benchmarks/m5-ultra-h3-owned-runtime.md).

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
[Results, errors and generated comparisons](../benchmarks/m5-ultra-h3-veda.md)
· [Setup and controls](../docs/h3-acceleration.md#veda-learned-sparse-attention-optional).
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
[Updated controls and bundling details](../docs/h3-acceleration.md). [Results, visual comparisons, new-method survey and workflows](../benchmarks/m5-ultra-h3-recent-methods.md).

**Optional H3 acceleration update:** against a matched **144.80 s** warmed exact reference,
14-step INT8 SOL averages **115.23 s**; SOL14 tau 0.5 plus limited FirstBlockCache
averages **98.44 s (32.0% less time)**; FirstBlockCache 0.20 alone
averages **87.83 s (39.3% less)**. The latter two are repeated full-clip
measurements. The all-three combination measured **79.08 s once**, with substantially larger
composition changes. All are **non-bit-exact, independently selectable** and preserve the
exact option. Cache/token execution on current ComfyUI is fixed; the new step-based window
protects exactly the first four and final two of 20 steps. [Results, controls and visual tradeoffs](../benchmarks/m5-ultra-h3-sol-aggressive.md).
The [previous conservative seven-step SOL results](../benchmarks/m5-ultra-h3-sol-hybrid.md) remain available.

**H3 Latent Upscale · Apple Silicon:** a new MPS node supports the learned
2D/3D checkpoint formats and preserves nested H3 audio. The tested **3D FP16**
checkpoint upscales **864×480 → 1728×960** latents in **0.43 s for one second**
or **1.85 s for five seconds with full context**, excluding VAE decode.
The port matches upstream FP16 output bytes under the same temporal settings.
FP16 is recommended; the same checkpoint takes **0.44 / 1.69 s** in BF16 / FP32
compute on the one-second test. [Setup, precision and memory controls](../docs/h3-latent-upscaler.md).

**15-second H3 update:** extending the exact query-cache limit reduces sampling
from **41.66 to 36.85 s/step** (11.5%) at 864×480 / 362 frames. This is an
alternating **two-step** comparison against the previous optimized stack,
not a new 20-step total. Latents, video and audio are byte-identical.
[Long-clip test and token-limit fix](../benchmarks/m5-ultra-h3-long-qcache.md).

**SeedVR2 video memory fix:** the saved **7B INT8**, 2× upscale to **832×1472**
now completes **5 s / 120 frames in 138.41 s** and **10 s / 240 frames in
278.60 s** warm, including video/audio export. Maximum sampled Metal memory
was **36.61 / 47.53 GiB**. Previously, the five-second encode crossed our
48 GiB safety cutoff before its first spatial tile finished. The new waits
inside video tiles are enabled by default; **Split Latent remains off**.
The short before/after test preserves output bytes, and both long clips repeat
identically. These are memory-fix results, not new full-length stock baselines.
[Video memory investigation and results](../benchmarks/m5-ultra-seedvr2-video-memory.md).

**Optional Apple Silicon Super Resolution:** new IMAGE and streamed VIDEO nodes
run the MLX-DLSS VSR **2× / 8-bit SDR** model. A **2400×1600 → 4800×3200** image
takes **2.22 s** for the complete warm node call, with **2.40 GiB** peak active
MLX memory. The 416×736 → 832×1472 H3 video source takes **1.88 s for five seconds**
and **3.11 s for ten seconds**, including preparation and HEVC/audio export,
with **0.61 GiB** peak active MLX memory. This is a separate, frame-independent upscaler with different quality
and scale from SeedVR2; it is not another row in the stock/optimized diffusion
comparison. [Setup, limits and validation](../docs/super-resolution.md).

**Optional VSR → DLSS neural rendering:** the same nodes now support detail
enhancement after upscaling, while keeping the final 2× dimensions. On this
**M5 Ultra 80 GPU**, a **1024² → 2048²** image takes **0.71 s VSR-only**, **1.40 s
with NR at internal scale 1**, or **3.07 s with NR at internal scale 2**. The
five-second H3 source above takes **44.11 s** with temporal NR at internal scale 2,
using **3.09 GiB** peak active MLX memory. These are complete warm node timings;
NR is an optional appearance-changing stage, disabled in existing workflows.
[Settings and example workflows](../docs/super-resolution.md#optional-neural-rendering-vsr-first-detail-second)
· [Measurements](../benchmarks/data/m5-ultra-vsr-neural-rendering.json).

Klein's checkpoint omits its INT8 format metadata. Its stock baseline uses
only the existing metadata repair needed to load it, with acceleration absent.

SeedVR2 stock retains the tiled-VAE synchronization safeguard against temporary-memory accumulation and recomputes input encoding on every run. The updated-core exact fusions were restored with source/build guards; the initial 2.14.1 retest measured 66.29 → 47.59 s. The subsequent exact VAE follow-up reaches **45.15 s**, preserving current optimized output bytes.

YuE2 stock uses ComfyUI's default CPU FP32 encoder; optimized uses the existing
MPS BF16 encoder plus the new exact patches. The controlled short screen truncates
the score at 128 tokens and generates 8 s of music, rerunning all stages. It is
not the full saved song or a quality example. Stock CPU and GPU arithmetic differ;
the new patches are bit-exact against the preceding GPU stack. No full stock CPU
song timing is claimed.

The complete stock and optimized stacks are **not bit-exact to each other**;
new exact fusions are validated against the otherwise unchanged optimized
stack. See the [full benchmark, numerical checks and saved workflows](../benchmarks/m5-ultra-torch214-production.md),
the [Qwen 2.1 results](../benchmarks/m5-ultra-qwen21.md),
the [Flux.2 Klein results](../benchmarks/m5-ultra-flux2-klein.md),
the [40-step image-edit investigation](../benchmarks/m5-ultra-qwen21-edit.md),
the [LTX 2.5 video benchmark](../benchmarks/m5-ultra-ltx25.md),
the [further exact kernels and convolutional VAE comparison](../benchmarks/m5-ultra-ltx25-followup.md),
the [Wan 2.2 original and four-step LoRA results](../benchmarks/m5-ultra-wan22.md),
the [Ideogram 4 benchmark and exact attention changes](../benchmarks/m5-ultra-ideogram4.md),
the [SeedVR2 2.14 retest and memory checks](../benchmarks/m5-ultra-seedvr2-torch214.md),
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

**What it speeds up:** automatically on macOS with an available MPS device,
custom Metal kernels take over the heavy math — attention, the quantized
matrix multiplies, and each model family's glue code. Image models render in
roughly **half** their stock time; long video generation becomes genuinely
viable on a laptop. No acceleration flag or extra workflow node is required.
Set `ASFP8_NATIVE=0` to disable the implied native feature groups; individual
feature overrides retain priority. On an M5 Max, Z-Image runs ~7.6 s, Krea2 ~14 s,
Ideogram 4 ~41 s, and an 81-frame Wan 2.2 video ~111 s — matching or beating
the fastest native Mac apps (full table below). These are measured results
for specific hardware, software and kernel designs; the tuning experiments
below do not establish an absolute hardware limit.

**What to expect:** identical or visually identical output (most fast paths
are bit-exact or within one rounding step of stock math — see
[Numerics policy](../README.md#numerics-policy)), silent per-call fallback to stock
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
[Numerics policy](../README.md#numerics-policy).

## PyTorch 2.14 on M5 Ultra

The isolated `ComfyUI-AppleSilicon-2.14-Update` installation runs PyTorch 2.14.1
and torchvision 0.29.1. The earlier `ComfyUI-AppleSilicon-2.14` installation
remains on 2.14.0 / 0.29.0 and is unchanged.
The clone keeps native INT8/DiT/VAE kernels and selects PyTorch's new MPS
attention with these settings, now selected automatically on the reviewed
M5 Ultra builds when no explicit override is supplied:

```text
ASFP8_NATIVE=1
ASFP8_ATTN_BACKEND=pytorch
MTLFLASHATTN_SDPA=off
```

The benchmark launch also uses `--use-pytorch-cross-attention`; it can still be
used to select ComfyUI's default for calls outside our router. The automatic
backend setting routes supported calls directly to 2.14 SDPA, including Krea2's
native grouped heads, while retaining fused Q/K normalization, RoPE, layouts,
quantization and projections. Third-party attention overrides retain priority.
Unsupported shapes retain the existing router. Set `ASFP8_ATTN_BACKEND=` explicitly
to retain the earlier router; it does not restore 2.13 arithmetic or output bytes.
The earlier H3 relaxed preset remains separate and off by default.

Native extensions use PyTorch's compiler-standard default (C++20 on 2.14),
with build caches separated by PyTorch version/build and installation path.
Krea2's accepted exact normalization and elementwise fusions are validated
on 2.14 as well as 2.13. A new exact FP16 GEGLU fusion removes a memory pass
from SDXL; `ASFP8_SDXL_GEGLU=0` disables it. Its 128 KiB lookup table contains
PyTorch's evaluated GELU for every FP16 input bit pattern, retaining the
intermediate rounding before multiplication.

On the pinned, validated M5 Ultra / 2.14.0 and 2.14.1 builds, dense BF16 D=128 attention also
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
in the [2.14 production benchmark](../benchmarks/m5-ultra-torch214-production.md).
An additional [H3 Sage-style attention screen](../benchmarks/m5-ultra-h3-sage.md)
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
The [initial report](../benchmarks/m5-ultra-ltx25.md) and
[follow-up comparison](../benchmarks/m5-ultra-ltx25-followup.md) record timings,
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
The [full report](../benchmarks/m5-ultra-flux2-klein.md) includes fresh-input costs,
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
[image-edit report](../benchmarks/m5-ultra-qwen21-edit.md) records timings,
compiler controls, numerical checks, scope and rejected experiments.

`ASFP8_QWEN21_FAST=0` disables these new fusions while keeping the encoder
repair and existing generic native kernels. The individual `ASFP8_QWEN21_ROPE`,
`ASFP8_QWEN21_MODULATE`, `ASFP8_QWEN21_RESIDUAL`, `ASFP8_QWEN21_ATTN` and
`ASFP8_QWEN21_VAE` switches can each be set to
`0`. Other hardware, Torch builds, dtypes, shapes and attention hooks retain
the original path. The [Qwen 2.1 benchmark](../benchmarks/m5-ultra-qwen21.md)
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
See the [SDXL investigation](../benchmarks/m5-ultra-sdxl-investigation.md) for
profiling, controls, numerical errors and the limits of the Max/Ultra comparison.
An isolated follow-up with **PyTorch 2.14** measured **3.15 → 2.77 s** with
the current router and **2.39 s** with raw PyTorch SDPA, approximately **24%
less total time**. Both 2.14 routes change the reference output bytes; the
original environment remains on 2.13; the separate 2.14 clone is described above. See the [version comparison](../benchmarks/m5-ultra-sdxl-investigation.md#follow-up-pytorch-213-vs-214)
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
[complete comparison](../benchmarks/m5-ultra-torch214-models.md) and
[frozen API workflows](../benchmarks/data/m5-ultra-torch214-workflows.json).

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
[sprint-two benchmark report](../benchmarks/m5-ultra-sampling-sprint2.md).

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
[sprint-three report](../benchmarks/m5-ultra-relaxed-sprint3.md) for complete
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
7.0 s/step. See the [sprint-four report](../benchmarks/m5-ultra-relaxed-sprint4.md).

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
[INT8 VAE report](../benchmarks/m5-ultra-int8-vae.md) for individual runs, numerical
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
