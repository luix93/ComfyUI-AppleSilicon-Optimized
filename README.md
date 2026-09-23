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
M5 Max.**

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
| MiniMax H3 int8, 864×480 × 124 f / 20 steps | 490.4 s‡ | **461.3 s** | exact paths only; workflow approximations off |

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
An adapter is never selected automatically: apply it with `H3 LoRA (Apple
Silicon, Exact Runtime)` or `H3 Turbo LoRA (Apple Silicon)` in `fast_runtime`
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

### Video VAE (patches #26, #29, #31)

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

Everything ships with parity tests against the eager path. The latest full
suite in the dedicated `ComfyUI-torch213-latest` environment reports **397
passed, 19 skipped** on PyTorch 2.13.0. The Metal harnesses include a dirty-pool
test that poisons the MPS allocator with NaN pages before checking ragged
attention shapes, because parity on fresh, zeroed allocations structurally
cannot catch out-of-bounds reads.

## MiniMax H3 workflow nodes and approximate modes

These are separate from the native fast paths above. They are **off until a
workflow node is connected**, change the result, and should be A/B tested with
the same seed. New workflows should use the single `H3 Apple Silicon
Acceleration` `MODEL → MODEL` node. The former H3Cache, Structural
Acceleration, and Sparse Attention nodes remain registered as deprecated
loaders so saved workflows do not lose nodes.

The implementation boundary matters when combining them:

| Mechanism | Implemented as | FirstBlockCache | Spectrum | H3 sparse attention |
|---|---|---|---|---|
| Exact H3 QKV/block/lifetime fusions | startup backend + Metal kernels | yes; cross-block-only fusion disables itself | yes; same boundary rule | yes |
| H3 Turbo LoRA | workflow `MODEL → MODEL` node | mechanically yes; few hits remain at 4–8 steps | mechanically yes, not calibrated for Turbo dynamics | mechanically yes; errors compound |
| H3 token reduction | unified workflow `MODEL → MODEL` node | yes; the unified node fixes the internal order | yes **only when the unified node is applied after Spectrum** | yes; reduced-sequence metadata is synchronized |
| FirstBlockCache | unified workflow `MODEL → MODEL` node | — | choose one block predictor, do not stack | yes |

“Mechanically yes” means the hooks do not erase each other; it is not a quality
claim. The unified defaults reproduce the supplied three-node chain. For a
quality diagnosis, disable groups and add them back one at a time. The exact
backend remains useful under every node, although a block-replacement node
necessarily removes the opportunity for the exact cross-block AdaLN carry at
that boundary.

- **H3 Apple Silicon Acceleration (canonical node).** The controls are grouped
  in execution order—cache, token reduction, sparse attention—and settings for
  a disabled group collapse in the ComfyUI frontend. Adding the node enables
  all three approximate groups by default. The values reproduce the supplied
  experimental workflow; they are starting points for matched A/B testing,
  **not a quality-calibrated recommendation**:

  | Group | Default | Initial values |
  |---|---|---|
  | Cache | `first_block_cache` | threshold `0.20`, window `0.15–0.95`, maximum `2` consecutive skips, audio veto weight `1.0` |
  | Token reduction | enabled | update scale `1.0`, first `10` evaluations restore at block `40`, later evaluations restore at block `30` |
  | Sparse attention | enabled | `corrected`, tau `1.0`, window `0.20–0.85`, first `2` blocks dense, minimum sequence `0` |

  Cache can instead select `first_block_cache_per_step`, or the non-skipping
  `oracle` and `block_oracle` diagnostics. Failed full-model forecasting, suffix caching,
  block Taylor, and sparse hard-drop are intentionally absent from the canonical
  menu. Sparse block size `64` and temporal taps `0` are fixed properties of the
  route-verified `diag64` implementation, not decorative controls. Layer thinning is also
  absent: real H3 renders showed broken audio whenever layers were omitted.

- **H3 Turbo LoRA.** `H3 Turbo LoRA (Apple Silicon)` is this package's strict
  loader for the released 208-target ComfyUI MiniMax H3 Turbo adapters (50 × 4
  DiT targets plus 2 × 4 token-refiner targets). It accepts the standard
  `diffusion_model.` key prefix, validates every A/B pair and target shape, and
  works with both the native and pruned/curve H3 bases because these adapters do
  not target the 51 collapsed AdaLN projections. A hypothetical full-width
  AdaLN adapter remains native-base-only. Use the current ComfyUI H3 workflow,
  stock Euler sampler, `simple`
  scheduler, strength `1.0`, and 4–8 steps. Four steps is the maximum-speed
  setting; 6–8 is the safer quality range for fast motion. Against a 20-step
  base workflow those counts remove 80%, 70%, or 60% of denoiser evaluations,
  respectively, before LoRA overhead; an MPS end-to-end number has not yet been
  measured here.

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

  `H3 LoRA (Apple Silicon, Exact Runtime)` is the checkpoint-agnostic companion
  node. It accepts any nonempty subset of H3 QKV, attention-output, FC1, FC2,
  and shape-compatible AdaLN targets encoded as standard `lora_A`/`lora_B`
  linear pairs, with arbitrary rank, alpha, and strength. It is not specialized
  to a particular LoRA file. One runtime LoRA may own a target at a time;
  stacked runtime adapters and DoRA/LoHa/LoKr/mid/reshape formats should use
  ComfyUI's ordinary loader, which makes these specialized routes decline and
  preserves the normal correct path, or use merged mode where appropriate.

- **Rejected cache experiments (deprecated research node).** The earlier
  full-model cache learned separate video and audio input-to-output change rates
  from exact evaluations. On a full-model hit it reused H3's previous video
  velocity and analytically corrected the audio velocity for H3's distinct
  shifted sigma schedule; either modality could veto a visually stable hit. A
  learned scalar secant predictor and the earlier residual/output-hold
  predictors remain selectable for controlled comparison. Sampler-facing
  denoised holding and first-order denoised extrapolation are also available as
  research controls. `cached_suffix_blocks` can instead recompute the model
  prefix exactly and reuse only a short transformer tail.
  The first/last sampling regions were protected, consecutive hits were bounded,
  and state was isolated to one sampling run. The deprecated standalone node
  retains these failed forecasting and suffix modes only for old workflow
  loading; they are not offered by the canonical node.

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
  target-video token reduction:

  - `token_reduction` runs blocks 0–3 at full sequence length, horizontally
    pairs only the target-video tokens, and leaves text, conditioning,
    references, and audio exact. During the first ten evaluations it restores
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

  Apply Spectrum first and the unified node second so the coordinator can
  compose the existing Spectrum block wrappers. FirstBlockCache and token
  reduction are ordered correctly inside the unified node. Sparse attention is
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
- **H3 Acceleration (preset node).** `H3 Acceleration (Apple Silicon)` wraps
  the above behind `exact` / `balanced` / `aggressive`. `exact` is the default
  and adds no approximation. `balanced` and `aggressive` currently **refuse to
  run**: their constants are placeholders until the calibration matrix fills
  them in, and shipping unvalidated numbers behind a friendly name is worse
  than an error message. Use the low-level nodes until then.
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

`dev/bench_workflows.py` can inject EasyCache or H3Cache and pin duration, step
count, and megapixels without modifying the source workflow. For H3Cache,
`--h3cache 0 0.20 0.85 1` is the exact-compute calibration configuration;
`--h3cache-audio-weight 1.0` controls the audio reuse veto.
`--h3cache-predictor velocity_hold` selects the H3-aware full-model predictor,
`--h3cache-predictor denoised_hold` or `denoised_linear` selects the
sampler-facing research controls, `--h3cache-predictor first_block_cache`
selects the cumulative block-0 detector, and `--h3cache-predictor
first_block_cache_per_step` selects the Discord-style per-step detector.
`--h3cache-suffix-blocks 5` changes compatible full-model modes to a five-block
tail; it must remain `0` for either first-block mode.
`--h3-sparse corrected 1.0 0.20 2 8192` injects the sparse attention node and
composes with either cache flag (it is applied after the cache node, matching
the preset order); `--h3-preset exact|balanced|aggressive` injects the combined
node instead. Sparse runs are refused unless the server reports torch 2.13+.
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
| `ASFP8_ATTN_NA` | 1 | production independent-simdgroup flash41 kernel; `0` restores the older cooperative kernel |
| `ASFP8_DIT_EXT` | off | DiT block-fusion kernels (all families) |
| `ASFP8_CONV3D` | off | conv3d → sliced conv2d + Wan VAE fast path |
| `ASFP8_FLUX_FAST` / `KREA2_FAST` / `LTX_FAST` / `WAN_FAST` / `WAN_VAE_FAST` / `IDEOGRAM_FAST` / `MINIMAX_FAST` | on (under DIT_EXT/CONV3D) | per-architecture kill switches |
| `ASFP8_MINIMAX_ATTN` / `ASFP8_MINIMAX_BLOCK` | 1 | MiniMax direct QKV/attention-output layout and segmented block-glue fusions |
| `ASFP8_MINIMAX_GRAPH` | 1 | exact guarded H3 final-layer graph rewrite (target RMSNorm + modulation + fp32 casts) |
| `ASFP8_MINIMAX_VAE_FAST` | 1 (under DIT_EXT) | FP16 H3 video-decoder QK normalization/RoPE/layout and elementwise kernels; `0` restores core forwards |
| `ASFP8_MINIMAX_VAE_SWIGLU` / `ASFP8_MINIMAX_VAE_MERGE` | 1 | per-feature switches for exact FP16 SwiGLU and attention-output merge/sanitization |
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
| `ASFP8_WAN_VAE_CHUNK` | 16 | Wan VAE frames per causal pass (even, 2–32; lower caps decode memory peak) |
| `ASFP8_FREE_MEM` | on | MPS free-memory accounting fix (patch #30) |
| `ASFP8_VAE_SLICE_FIX` | on | VAE slice_attention INT_MAX cap (patch #31) |
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

MIT — see [LICENSE](LICENSE).
