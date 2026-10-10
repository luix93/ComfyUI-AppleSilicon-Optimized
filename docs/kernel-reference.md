# Kernel and compatibility reference

Detailed architecture and kernel notes moved from the README. Historical
timings below are measurements of their named stack, not new release
benchmarks. See the [current performance table](../README.md#latest-results--m5-ultra-80-gpu-cores)
and [settings reference](settings.md) for the current summary and switches.

## The two layers

**1. Compatibility (automatic, any M1→M5).** FP8/INT8 checkpoints load and
render instead of crashing: LUT-based bit-exact FP8 decode, CPU reroutes for
the casts MPS can't do, `torch._int_mm` on GPU, flash attention for
long-context correctness/OOM, text encoders moved off the CPU, plus fixes for
macOS-beta psutil, the PiD ≥2048px black image, and WanVideo block swap. No
env vars needed. See [the compatibility layer](#the-compatibility-layer-patches-116).

**2. Native kernels (automatic on macOS/MPS).** Seven ObjC++/Metal
extensions, JIT-built on first use against **Metal 4.1**. Six production
feature groups sit behind the master switch; the seventh (`sol_ext`) builds
only when the experimental H3 sparse-attention node is applied. Developed on
macOS 27 and benchmarked on M5 Max/Ultra. Older Apple Silicon also attempts the
enabled routes, retaining existing build/runtime fallbacks. Their performance
and numerical parity have not been validated on M1–M4. M5 Ultra-specific
source/build/hardware guards remain intact. The explicit sparse node instead
refuses to apply when its required kernel is unavailable.

```bash
xcode-select --install        # Metal toolchain
pip install ninja
```

With no launch flag, macOS/MPS sets `ASFP8_NATIVE=1` and implies `ASFP8_FP8_EXT`, `ASFP8_FP8_SGEMM`,
`ASFP8_INT8_EXT`, `ASFP8_ATTN_ROUTER`, `ASFP8_DIT_EXT`, `ASFP8_CONV3D`.
Explicit per-feature values always win (e.g. `ASFP8_NATIVE=1 ASFP8_DIT_EXT=0`).
`ASFP8_NATIVE=0` disables this automatic implication while keeping compatibility
and memory fixes. Explicitly enabled individual features still run. Linux,
Windows and macOS without MPS receive no automatic native settings.

On pinned tested PyTorch 2.14.0/2.14.1 builds and M5 Ultra, the pack also defaults
to `ASFP8_ATTN_BACKEND=pytorch` and `MTLFLASHATTN_SDPA=off`; explicit modern or
legacy settings retain priority. Other hardware/builds keep the existing router.
H3 relaxed kernels, sparse attention, caches and token reduction remain optional.

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
and extend it to fp16/video/GQA — see [Credits](../README.md#credits) for the split.

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
[Credits](../README.md#credits)): 16 *independent* simdgroups × 16 rows each instead of
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
- In the earlier Torch 2.13 attention experiment, D=256 heads (Ideogram 4)
  were tried via a two-block head split and **lost to
  plain bmm** (22.9 ms vs 12.4 ms): the 16-row × 256 × fp32 accumulator costs
  128 registers/lane and spills. Budget rule: 16 × D × fp32 / 32 lanes must
  stay ≲64/lane. That historical implementation kept wide heads on bmm.
  The current Torch 2.14 exact Ideogram route uses MPP attention; the
  [October 10 follow-up](../benchmarks/m5-ultra-ideogram-h3-oct2026.md) splits its output channels while retaining full-width Q/K arithmetic.

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
  See the [Krea2 Ultra report](../benchmarks/m5-ultra-krea2-exact.md) for
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
  MPS completion between spatial encoder/decoder tiles and bounds pending residual
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
  The reviewed 2.14.1/current-core/M5 Ultra FP16 path now queues at most four
  residual blocks, fencing earlier at 2 GiB driver growth or 40 GiB total driver
  allocation. The latter is an early-fence threshold, not a hard allocation cap.
  Spatial-tile fences remain unchanged. Set `ASFP8_SEEDVR2_VAE_BLOCK_BATCH=1`
  to restore every-block waits; FP32 and unsupported builds retain that policy.
  `ASFP8_SEEDVR2_VAE_VIDEO_SYNC=0` restores spatial-only waits for diagnostics.
  See the [video memory investigation](../benchmarks/m5-ultra-seedvr2-video-memory.md).
  Non-tiled SeedVR2 calls, other VAEs, CPU and CUDA retain their original calls.
  The **2400×1600 → 7200×4800** INT8 7B image workflow completed in **80.15 s
  first run / 73.41 s warm**, including PNG writing, with **40.26 GiB** maximum
  observed Metal allocation. A smaller complete workflow preserves encoded
  latents, sampler latents and decoded frame bytes with the fix on/off. See the
  [SeedVR2 memory report](../benchmarks/m5-ultra-seedvr2-memory.md) for measurement
  scope, numerical checks and the guarded disabled-path reproduction.
  The [SeedVR2 acceleration audit](../benchmarks/m5-ultra-seedvr2-acceleration.md)
  records the original INT8/convolution coverage. The subsequent exact stack
  below supersedes its preliminary optimization status.
- **SeedVR2 PyTorch 2.14.1 exact VAE follow-up:** the reviewed M5 Ultra path fuses FP16 GroupNorm, SiLU and causal channel folding in two dispatches, retaining FP32 statistics/reduction order and both FP16 rounding boundaries. Fresh full-image comparison: **47.44 → 45.15 s**; fixed-latent complete decode: **22.86 → 21.66 s**. Bounded four-block video scheduling gives **128.18 → 124.66 s** on the five-second fixture. All encoded latents, sampled latents and decoded frame bytes are identical; memory fences remain active. Set `ASFP8_SEEDVR2_VAE_GN_SPLIT=0` or `ASFP8_SEEDVR2_VAE_BLOCK_BATCH=1` to restore the respective preceding routes. See the [follow-up report](../benchmarks/m5-ultra-seedvr2-vae2141.md).
- **SeedVR2 PyTorch 2.14 retest:** **74.02 s stock + memory safeguard → 49.01 s optimized** at 2400×1600 → 7200×4800. Unit-scale VAE residuals avoid a redundant division; output bytes match the preceding optimized stack. See the [current retest](../benchmarks/m5-ultra-seedvr2-torch214.md). The following measurements are historical PyTorch 2.13 results.
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
  [SeedVR2 exact optimization report](../benchmarks/m5-ultra-seedvr2-exact.md)
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
  [conditioning provenance and license](../_patches/seedvr2_assets/README.md).
  ComfyUI chooses BF16 computation for these FP16 weights on the tested M5
  setup, so the existing BF16 SeedVR2 fusions apply; dense matrix multiplies
  use MPS, not the INT8 kernels. FP16 and INT8 results need not be identical.
  The complete **2400×1600 → 7200×4800** FP16 workflow passed at **57.45 s
  first / 54.22 s warm**, including PNG saving, with identical first/warm
  latents and frame bytes. That compares with the prior **50.98 s warm**
  INT8 run. See the [FP16 compatibility report](../benchmarks/m5-ultra-seedvr2-fp16.md)
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
