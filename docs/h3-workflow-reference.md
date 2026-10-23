# H3 workflow controls and experiment reference

Detailed controls, combination experiments and migration notes moved from
the README. For the current compact-node setup and presets, start with
[H3 acceleration](h3-acceleration.md). Timings retain their original stack
and baseline; historical external-node tests do not make those nodes required
by the bundled MotionCache/Spectrum runtime.

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

Two optional [Super Resolution nodes](../docs/super-resolution.md) additionally handle
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
  **Token reduction** independently. The existing cache, SOL and token
  recipes remain, alongside Dense INT8 v3 and SOL Hybrid Fast. Defaults are
  **MotionCache · balanced**, sparse **Off**, tokens **Off**. **Detailed logs** defaults off; each active mechanism prints
  a compact final work/speedup summary. Set all three selectors to Off for pass-through.
  Connect one of the Simple/advanced nodes between the H3 model/LoRA loader
  and the guider; use the advanced node for predictor selection, memory tuning,
  or custom numerical settings. [Recipes and usage](../docs/h3-acceleration.md#simple-node)
  · [Node template](../workflows/h3_acceleration_simple.json)
  · [Token compatibility and benchmark checks](../benchmarks/m5-ultra-h3-presets.md).
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

  INT8 SOL's [temporary-memory ceiling](../docs/h3-acceleration.md#temporary-memory)
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
  draft**, **FirstBlockCache · 0.20**, **SOL Metal · fast**, and
  **SOL Metal + FirstBlockCache**. Each writes the measured recipe, including the
  SOL INT8 backend, protected layers, step-fraction schedule and token guard;
  the hybrid uses its own lower cache threshold and one-skip limit. These are
  20-step, non-Turbo measurements. See [controls and bundled-source licenses](../docs/h3-acceleration.md)
  and [timing/quality comparisons](../benchmarks/m5-ultra-h3-recent-methods.md).

  Choose **one cache method**, then independently enable SOL and/or token
  reduction, including with MotionCache or Spectrum. These cache predictors
  are alternatives to each other. Token reduction remains experimental;
  combinations need their own quality check. Non-skipping `oracle`/`block_oracle`
  diagnostics and the per-step cache policy remain available. Sparse block size
  `64` and temporal taps `0` remain fixed; layer thinning stays unavailable because
  it damaged H3 audio. [UI and combination validation](../benchmarks/m5-ultra-h3-node-v2.md).

  Bundled Spectrum retains GPL-3.0-or-later; the combined distribution uses
  GPL-3.0, preserving existing MIT notices. [Attribution and licenses](../THIRD_PARTY_NOTICES.md).

- **Optional dense and hybrid attention.** Dense INT8 Protected keeps the first
  four and final two of 20 steps plus layers 0, 1, 33, 34, 35 exact. Full Draft
  quantizes video-query attention in every layer/step while retaining all key
  tokens. Packed prefix queries stay exact in both, but later video changes can
  affect audio too. Hybrid retains the protected step window, uses SOL Fast in
  ordinary layers and dense INT8 in protected layers inside the active window.
  These are independently selectable starting points, with existing memory
  safeguards and native fallback. Full Draft can change scene composition;
  Hybrid's additional benefit is modest. [Complete timings and temporal audit](../benchmarks/m5-ultra-h3-kitchen-attention.md).

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
  a logged reason. SOL packed metadata is reduced/restored together,
  including on interruption. See the [validation report](../benchmarks/m5-ultra-h3-presets.md).

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
- **Spectrum hidden-feature forecasting (historical external screen).** The native H3
  [Spectrum adaptation](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
  was initially tested independently; the pinned runtime is now bundled as
  described in Credits. Its forced-actual wrapper
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
