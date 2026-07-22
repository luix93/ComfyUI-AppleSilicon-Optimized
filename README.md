<img width="1684" height="934" alt="banner" src="https://github.com/luix93/ComfyUI-AppleSilicon-Optimized/blob/main/banner.png?raw=true" />

# ComfyUI-AppleSilicon-Optimized

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
the fastest native Mac apps (full table below). Several of these numbers are
measured *hardware floors*: the kernels run at the silicon's own limits, and
the receipts for that claim are in this README.

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
M5 Max.**

Private continuation of
[ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8).
What began as an FP8 compatibility shim has diverged into a general
optimization layer: 32 runtime patches and six JIT-compiled Metal extensions
that take image models to roughly half their stock render time and make long
video generation on a MacBook competitive with — and in places faster than —
the fastest native Mac apps.

No model conversion, no forked ComfyUI: one custom node applies everything at
startup. Every fast path checks its inputs per call and **falls back to the
eager implementation** on anything it doesn't recognize, and every kernel is
held to **exact-or-1-ulp parity** against the eager math it replaces.

## Results

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
| Wan 2.2 i2v 14B×2 int8, 512² × 81 f | 445 s | **111 s** | DrawThings-class ≈ 100 s |

† generic patches already on — Ideogram 4 shipped after the generic layer, so
its "before" is the pre-#32 state, not compatibility-alone.

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

**2. Native kernels (opt-in, `ASFP8_NATIVE=1`).** Six ObjC++/Metal
extensions, JIT-built on first use against **Metal 4.1** (developed on the
macOS 27 dev beta; the int8 and fp8 matmuls additionally need an **M5**-class
GPU for cooperative TensorOps / the fp8 format type). Anything that can't
build or doesn't apply falls back silently, so the switch is safe to leave on
everywhere.

```bash
xcode-select --install        # Metal toolchain
pip install ninja
export ASFP8_NATIVE=1         # before launching ComfyUI
```

`ASFP8_NATIVE=1` implies `ASFP8_FP8_EXT`, `ASFP8_FP8_SGEMM`,
`ASFP8_INT8_EXT`, `ASFP8_ATTN_ROUTER`, `ASFP8_DIT_EXT`, `ASFP8_CONV3D`.
Explicit per-feature values always win (e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`).

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

**Why attention stays fp16 (the int8 attention verdict).** A full int8
attention forward ships in the source (`flash_attn_fwd_na_i8`, the
DrawThings `NAInt8AttentionKernel` recipe: on-the-fly symmetric int8 quant
of Q/K and mean-centered V with per-tile scales, int8 QK → int32, exact
V-mean addback). It is functional and accurate (~1.2% relative, all shapes,
poisoned-pool clean) — and **deliberately unrouted**, because it loses:
best configuration 0.89×/0.78× the fp16 na kernel at Wan/LTX shapes. The
root cause is a measured law of this silicon: the int8 tensor units' ~2×
advantage only exists at deep reductions. Our own 105 TF/s int8 GEMM drops
to **25 TF/s at K=128** (104 at K=6144, 93 at K=512, 33 at K=256) — and
attention reduces over `head_dim = 128`, where the fp16 na kernel already
runs 36–40. int8 for the deep-K linears, fp16 for shallow-K attention is
not a preference, it's the crossover the hardware dictates — DrawThings,
Cider, and this repo converged on the same split independently.

### Attention router (patch #19)

Per-call routing on MPS: unmasked long-sequence attention → flash41; masked
attention above a real-score-size threshold → memory-safe sub-quadratic;
mid-size → batched bmm (gated on the *actual* `Lq×Lk` score bytes, not the
max-sequence square — that distinction is what keeps tiny cross-attention off
the slow path). Stock fused SDPA — silently inaccurate past ~4k tokens and
O(L²) memory — is never taken at length. `ASFP8_ATTN_DEBUG=1` logs each
distinct route taken.

### Quantized matmuls (patches #3, #17, #18)

- **int8 W8A8** (`ASFP8_INT8_EXT`, M5+): convrot checkpoints run the path the
  format was designed for — rotate the activation online, per-row quantize,
  then a bit-exact INT8×INT8→INT32 kernel (~105 TF/s at diffusion shapes,
  ~1.85× over bf16) with the rescale + bias fused into the store epilogue.
  Ported from [Cider](https://github.com/Mininglamp-AI/cider) (MIT). This is
  the backbone of the Z-Image / Flux2 / Krea2 / Ideogram / Wan int8 numbers
  above. A 1-slot activation cache also skips repeat rotate+quant passes when
  several projections share one input (qkv triples, SwiGLU pairs).
  **This rate is the silicon ceiling, not a tuning artifact**: a dedicated
  variant tournament (`dev/tournament_int8.py`) showed the kernel flat at
  97–107 TF/s from fully cache-resident to streaming working sets
  (issue-bound, not memory-bound), every schedule variant a wash under
  interleaved A/B, and the Metal API's larger tensor-op descriptors
  2.5–4× *slower* — 16×32×16 maps 1:1 onto the native op. GEMM-bound int8
  models are at their hardware floor.
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

### DiT block fusions (`ASFP8_DIT_EXT` — patches #21, #23–#25, #27, #28, #32)

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
  are gone).
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

### Video VAE (patches #26, #29)

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

### Memory accounting (patch #30)

comfy decides whether to keep a model resident by calling `get_free_memory`,
which on MPS returns `psutil.virtual_memory().available`. That number can't
see torch's MPS allocator pool and swings with OS file-cache noise — so on a
128 GB machine comfy would still **evict a 13.9 GB model per sampler switch**,
re-uploading it every run (the Wan 2.2 high/low-noise pair cost ~14 s/run this
way; text encoders churn the same way on other workflows).

`free_mem_mps` replaces the estimate with one that's true for unified memory:
`free = total RAM − torch's live tensors − an OS reserve`
(`ASFP8_MPS_RESERVE_GB`, default max(16 GB, 15%)). It also reports the "torch
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
| 12 | `torch._int_mm` on GPU instead of the silent CPU-fallback freeze |
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

- **Bit-exact**: all FP8 decodes, the int8 W8A8 kernel, `fold_taps`, the
  conv3d channel fold (single fp32 accumulation = conv3d's own numerics), the
  fused o-projection pre-pass in its default mode (verified by same-seed
  pixel-identical renders).
- **Exact-or-1-ulp**: the DiT fusions and VAE norm kernels mirror eager
  rounding at every point eager rounds (fp32 math between roundings); images
  shift imperceptibly but deterministically.
- **Regime-gated**: flash41's low-precision score tiles (default; parity
  ~5e-4, `ASFP8_FA41_F32S=1` for fp32 tiles).

Patch #30 (memory accounting) changes *scheduling*, not numerics — outputs are
identical, only fewer reloads.

Everything ships with parity tests against the eager path (116 pass on torch
2.13 with the native kernels enabled, plus kernel parity harnesses for the
Metal extensions — including a dirty-pool test that poisons the MPS allocator
with NaN pages before checking ragged attention shapes, because parity on
fresh, zeroed allocations structurally cannot catch out-of-bounds reads).

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ASFP8_NATIVE` | off | Master switch: enables all six below |
| `ASFP8_FP8_EXT` | off | fp8-native `_scaled_mm` kernel (M5+, Metal 4.1) |
| `ASFP8_FP8_SGEMM` | off | staged half×fp8 W8A16 backend |
| `ASFP8_INT8_EXT` | off | int8 W8A8 kernel + fused epilogue (M5+) |
| `ASFP8_ATTN_ROUTER` | off | per-call attention routing (flash41 / bmm / sub-quad) |
| `ASFP8_DIT_EXT` | off | DiT block-fusion kernels (all families) |
| `ASFP8_CONV3D` | off | conv3d → sliced conv2d + Wan VAE fast path |
| `ASFP8_FLUX_FAST` / `KREA2_FAST` / `LTX_FAST` / `WAN_FAST` / `WAN_VAE_FAST` / `IDEOGRAM_FAST` | on (under DIT_EXT/CONV3D) | per-architecture kill switches |
| `ASFP8_INT8_OPROJ_FUSE` | 1 | fused o-proj pre-pass: `1` bit-exact merge+quant, `2` full FWHT (rounding-equivalent), `0` off |
| `ASFP8_AUTOFIX` | on | self-heal JIT prerequisites at boot (install ninja, restore stripped torch headers) |
| `ASFP8_WAN_VAE_CHUNK` | 16 | Wan VAE frames per causal pass (even, 2–32; lower caps decode memory peak) |
| `ASFP8_FREE_MEM` | on | MPS free-memory accounting fix (patch #30) |
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

> Developed and tested on: M5 Max / 128 GB, macOS 27 dev beta, PyTorch
> 2.10 & 2.13, Python 3.12, ComfyUI Desktop. M1–M4 get the compatibility
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

## License

MIT — see [LICENSE](LICENSE).
