# Release notes

## October 24, 2026 — Dense Metal on longer sequences

- Dense Metal (v3) now accepts up to **262,144 keys**; it was 65,536. That cap
  only protects v2, which accumulates P·V in INT32 across every key tile. v3
  merges each 64-key tile into FP32. Long or high-resolution H3 clips that
  previously fell back to exact attention for the whole run now use Dense
  Metal. On random BF16 attention the relative error rises smoothly, with no
  break at the old limit: 0.0245 at 60k keys, 0.0248 at 70k, 0.0285 at 131k and
  0.0307 at 200k. `dense_metal_v2` keeps its 65,536-key limit.
- The fallback warning now names Dense Metal and states the token count.

## October 23, 2026 — H3 attention presets trimmed, Veda removed

- **H3 Acceleration · Simple:** ten attention presets were removed. v3 beats
  Dense INT8 v1/v2 on error at equal or better speed. Dense QK INT8 / PV BF16
  was only 1.05–1.08× faster than exact. Hybrid balanced was never benchmarked
  on a full clip. VSA ComfyUI (64) was a comparison recipe, and SOL beat Veda
  at the same distance from exact. Ten remain: SOL conservative / balanced /
  fast / top-k wide / top-k all-step draft / hybrid fast, Dense INT8 v3
  protected / balanced / full experimental, and VSA FastH3 Omni (128).
- Saved workflows keep loading. Removed names map to the nearest remaining
  recipe, with a log line, on both the server and the frontend.
- **Veda removed entirely:** the backend, vendored runtime, predictor setup,
  workflow template, the `veda-mlx` extra, and its credits and notices. The
  tile-list Metal attention kernel VSA uses (our PyTorch MPP adaptation) moved
  to `_patches/h3_tile_attention_mps.py`. On the advanced node, saved Veda
  selections run SOL INT8. Workflows saved with the four Veda inputs have those
  positional values dropped on load.
- The advanced node keeps the Dense INT8 v1, v2 and floating-PV backends
  selectable; only their presets were removed.
- Renamed to show the Metal kernels: **SOL Metal** · conservative / balanced /
  fast / hybrid fast / top-k wide / top-k all-step draft, and **Dense Metal** ·
  protected / balanced / full experimental (the INT8 v3 kernel). VSA Metal ·
  FastH3 Omni (128) is unchanged. The former `SOL INT8 · …` and
  `Dense INT8 v3 · …` names still load and run the same recipes. The advanced
  panel's presets follow the same names.
- **Attention Backend (Apple Silicon):** `int8_dense_accurate` is renamed
  `dense_metal`, the same v3 kernel as the Dense Metal presets and the Metal
  counterpart to Comfy Kitchen's INT8 attention, and `int8_dense_approximate`
  is renamed `dense_metal_v2`. `dense_metal` is listed first. Saved workflows
  using the former names still load, and the frontend updates the values.
- **SOL Metal · hybrid fast** now runs its protected layers on the Dense Metal
  (v3) kernel instead of v1. On H3 at 480p it took 113.51 s (v1: 113.90 s, SOL
  fast: 114.32 s, exact: 145.50 s), with frame PSNR against exact of 17.43 dB
  (v1: 17.37 dB). Set `ASFP8_H3_HYBRID_DENSE=v1` to get the previous bytes back.
  [Report](benchmarks/m5-ultra-h3-hybrid-v3.md)

## October 23, 2026 — regression on nightly `7a283320`

- 35 workflow/mode/preprocessor cases run twice on the new dev instance (core
  `v0.42.0-19-g7a283320`, Kitchen 0.2.39). All are byte-identical to the
  October 19 regression, or for Fibo to its October 23 benchmark.
  [Report](benchmarks/m5-ultra-comfy042-nightly-oct23.md)
- Repair: YuE2's exact single-token RoPE stood down after upstream's top-p fix
  changed `llama.py`. The reviewed hash is now admitted (Attention/RoPE
  unchanged). 8 s control: **7.65 → 5.83 s**, identical bytes.
- H3 coverage this pass: exact, SOL INT8 fast and Dense INT8 v3 balanced. Wan:
  four-step only, with the 20-step time scaled from it.

## October 23, 2026 — Fibo 2.0

- New exact Fibo 2.0 route (Dev and Edit Turbo, INT8 ConvRot):
  - Joined CFG rows: both branches run as one batch-2 forward when the
    prompts share a gist count. The Perceiver, text cross-attention and
    modulation stay per branch.
  - Fused RMSNorm+modulate and RMSNorm+gate+residual passes, in PyTorch's
    RMSNorm reduction order.
  - tile32 attention for its 28 heads.
- Dev 1024², 30 steps, CFG 5: **88.49 s stock → 38.47 s (previous pack) →
  31.87 s**. Edit Turbo 1024², 4 steps: **16.19 → 7.38 → 6.61 s**. Latents
  and decoded frames are byte-identical to the previous pack.
- Roll back with `ASFP8_FIBO2_FAST=0`, or switch parts off with
  `ASFP8_FIBO2_CFG_JOIN`, `ASFP8_FIBO2_GLUE` and `ASFP8_FIBO2_ATTN`.
  [Report](benchmarks/m5-ultra-fibo2.md) · [Controls](docs/fibo2.md)

## October 23, 2026 — faster exact MOSS route

- MOSS-Image now joins the positive and negative CFG branches through every INT8
  projection. Attention, positions and text stay per branch. It also uses a
  128-row INT8 tile order for MOSS shapes, and 32-row exact attention tiles by
  default.
- SFT, 25 steps: 1024² **20.43 → 19.31 s** (6.2%), 1248² **32.39 → 31.37 s**
  (3.2%). Every latent and decoded frame is byte-identical to the previous
  route.
- Roll back with `ASFP8_MOSS_CFG_JOIN=0`, `ASFP8_MOSS_GEMM_SCHEDULE=0` and
  `ASFP8_MOSS_ATTN_MODE=exact`.
- Beta-tester contributions, re-verified here, all byte-identical:
  - MOSS: exact RMS/modulation memory-pass fusion and fused Q/K RMS + RoPE. On
    top of the route above, SFT 1024² **19.47 → 17.48 s** and 1248²
    **31.65 → 28.43 s**. The total since October 22 is 20.43 → 17.48 s (−14%).
  - Krea2 INT8: paired gate/up projections with a direct SwiGLU store
    (about 1% on the Turbo + LoRA fixture).
  - BF16-only MOSS FC1/SiTU pairing and 32-row Krea2 attention tiles. These
    need BF16 checkpoints, which we did not have for end-to-end tests here.
  - Switches: `ASFP8_MOSS_QK_FUSION`, `ASFP8_MOSS_RMS_MODE=original`,
    `ASFP8_KREA2_FFN_PAIR`, `ASFP8_KREA2_TILE32` and `ASFP8_MOSS_BF16_FFN_PAIR`.
- `ASFP8_MOSS_ATTN_MODE=int8_v3` adds the general INT8 attention as an opt-in
  MOSS mode. At 1248², 50 steps it saves about 2% (25 dB PSNR vs exact). The
  existing `dense_int8_floatpv` saves about 4% (28 dB). Both are off by default.
- An exact fused ConvRot rotate/quantize kernel was also built. It is
  byte-identical only on the matrix units, and slower, so it was not adopted.
  [Measurements](benchmarks/m5-ultra-moss-exact-combo.md).

## October 22, 2026 — opt-in INT8 attention for long sequences

- **v3 / `int8_dense_accurate`:** Kitchen-style numerics (Hadamard Q/K rotation,
  per-tile probabilities) at v2's end-to-end speed; real-activation error
  0.0100 vs 0.0238 (v2) at 44.7k tokens, level with the Kitchen policy. H3 15 s
  at full coverage 799.65 → 588.57 s; Dense INT8 v3 presets added, including
  **Dense INT8 v3 · balanced** (first 2 and last 1 of 20 steps exact).
- H3 dense INT8 presets no longer compute the exact text/reference/audio prefix
  in INT8 before recomputing it. Output is byte-identical. On OmniRef, 41k
  tokens, each preset call is about 30 ms faster: balanced now saves 0.14 s per
  quantized call, level with the Attention Backend node.
- New general M5 INT8 attention kernel: every key retained; 255-level
  probabilities, per-channel V scales, INT32 accumulation across key tiles with
  in-place integer rescaling. Never a default; exact routes are unchanged.
- Select it with **Attention Backend (Apple Silicon) → int8_dense_approximate**
  (any model using ComfyUI attention; unmasked D=128 calls with ≥8192 keys) or
  the H3 presets **Dense INT8 v2 · protected / full experimental**.
- MiniMax H3 864×480 / 20 steps, warm end to end: 15-second clip
  **799.65 → 604.97 s** (full) / 666.40 s (protected); 5-second clip
  143.9 → 129.4 s (full, two-run means). Real-activation operator error is about
  one third lower than the existing Dense INT8 kernel at equal or better speed.
- Wan 2.2 I2V 14B×2, 512×512 / 81 frames with the backend node: 20 steps
  **285.72 → 225.52 s**, four-step LoRAs 31.6 → 26.6 s, frames visually close
  to exact (cosine 0.994).
  [Measurements, quality and limits](benchmarks/m5-ultra-int8-attention.md).

## October 19, 2026 — updated-core compatibility repairs

- Restore the reviewed MOSS path on core `8f04fd6e` / Kitchen 0.2.39, adapting
  the new positional-options interface and preserving custom RoPE controls.
  The matched 1024² / 25-step probe falls from 34.39 to 22.53 s with identical
  captured output bytes; the final installed-copy check takes 22.30 s.
- Admit reviewed Kitchen 0.2.39 for Flux.3 and the changed Llama source for
  YuE2 RoPE. No new approximation or default attention change.
- Add cache-interface/passthrough tests; 69 focused tests pass against the
  actual updated core. Record restored SDXL/Wan cases and cross-version
  numerical/input limitations in the [regression report](benchmarks/m5-ultra-comfy0421-smoke.md).

## October 18, 2026 — MOSS CFG batching experiment closed

- Record byte-identical grouped CFG projection tests: 2.55% E2E improvement
  at 1024² and an inconsistent 0.19% at 1248². No runtime option was added.
- Keep the final joined-state screen separate from complete-render results;
  current defaults remain unchanged. [Benchmark report](benchmarks/m5-ultra-moss-cfg-projections.md).

## October 18, 2026 — optional MOSS dense attention tuning

- Add MOSS-specific dense INT8 QK/PV and INT8 QK/BF16 PV paths with direct
  input/output addressing, reduced preparation and smaller floating-PV query
  groups. All keys remain; no SOL. Existing exact attention remains the default.
- Add an independent exact 32-row query-tile option. Verify final latents/frames
  against exact, and verify each tuned INT8 mode against its preceding dense
  algorithm. Guard both packed and interleaved QKV value layouts.
- Reject larger floating key tiles, FP16 conversion and inconsistent direct-V
  gains. [Full-render results and visual comparisons](benchmarks/m5-ultra-moss-attention-tuning.md).

## October 18, 2026 — optional exact MOSS FFN fusion

- Add an opt-in FC1/SiTU matrix-output kernel with direct paired-column weight
  reads, unchanged BF16 rounding and no packed-weight cache. Full SFT renders
  retain identical latents/frames; gains are small and the default is unchanged.
- Reject the larger FC1/SiTU/ConvRot fusion after slower operator timings and
  numerical differences. [Measurements](benchmarks/m5-ultra-moss-ffn-epilogue.md).

## October 18, 2026 — Kitchen-style Metal research

- Added a disabled-by-default shared D128 BF16 INT8 attention prototype and an
  independent fused MOSS SiTU/ConvRot/quantization experiment. Both are
  non-bit-exact and rejected as accelerations after complete 25-step tests.
  Exact defaults and existing H3 presets remain unchanged.
- Pinned Kitchen 0.2.38 provenance, added local quantization/oracle tests and a
  standalone CUDA reference exporter/comparator. CUDA byte parity is unverified.
  [Implementation and results](benchmarks/m5-ultra-moss-kitchen-int8.md).

## October 18, 2026 — MOSS FFN dispatch audit

- Remove redundant FFN weight preparation on the reviewed MOSS path while
  retaining the same SiTU and INT8 module arithmetic. Known wrapper identities
  and core-source guards preserve fallback for unsupported combinations.
- Eliminate the unused 1.97 GiB prepared-weight cache on this path. Warm 1024²
  SFT results are 22.21 / 42.71 s at 25 / 50 steps; the change is below 1% E2E,
  so this is not a material speedup claim. Full captures remain byte-identical;
  72 focused tests pass. [Audit](benchmarks/m5-ultra-moss-execution.md).

## October 18, 2026 — MOSS SFT 1.5 MP

- Extend validated exact MOSS query reuse through 6,656 tokens. The user's
  1248×1248 workflow previously used ordinary PyTorch attention at this size.
- Warm optimized-to-optimized means: **35.27 → 34.07 s** at 25 steps and
  **68.41 → 66.73 s** at 50 steps, CFG 2.5. All complete sampler and decoded
  tensors match; direct SDPA checks cover real branch lengths and the new boundary.
- Matrix tile experiments remain uninstalled. The tested changes do not achieve
  the user-reported Blackwell timings. [Full report](benchmarks/m5-ultra-moss-15mp.md).

## October 18, 2026 — nightly regression and compatibility guards

- Retested 29 available workflow/mode/preprocessor cases twice on ComfyUI
  nightly `c06bf5f5` (after v0.41.3), PyTorch 2.14.1 and Kitchen 0.2.38.
  SDXL and Wan checkpoints were unavailable. SeedVR memory guards remained
  enabled; no memory watchdog tripped.
- Restored YuE2's exact single-token RoPE on the reviewed nightly source:
  full-workflow warm time recovered from 49.10 to 34.83 seconds, with identical
  tensors. Enabled the existing exact Flux.3 fusions/caches on reviewed Kitchen
  0.2.38: distilled/base T2I 5.79/33.38 seconds, edit 12.80/70.59 seconds.
  Repairs preserve the pre-repair nightly bytes; 46 focused tests passed.
- LTX and Flux.3 differ numerically from the previous core. Diagnostic replays
  of its quantized-linear dispatch restore the old bytes on LTX convolutional
  VAE and Flux.3 distilled T2I. That rollback is not installed. Other cases
  match their references. [Results and scope](benchmarks/m5-ultra-comfy0413-nightly-smoke.md).

## October 18, 2026 — MOSS-Image

- Added automatic MOSS-Image SFT/Base exact BF16 SiTU, split-half RoPE and
  modulation fusions, dense PyTorch query reuse, and bounded sampling-local
  weight/text/position caches. Full-step captures preserve the preceding
  optimized output bytes; standalone MOSS-only tests also preserve stock bytes.
- Sampling improves by about 37% over the preceding pack at 1024² / 25 steps.
  [Complete results and exactness boundaries](benchmarks/m5-ultra-moss-image.md).
- Tested on nightly ComfyUI `c06bf5f5`, PyTorch 2.14.1 and Kitchen 0.2.38 on
  M5 Ultra; the subsequent regression is documented above. Source/build guards
  retain existing paths on unreviewed stacks, explicit backend overrides or
  unsupported cases.
- New caches are phase-local and released on completion/interruption/failure;
  the tested prepared weights use about 1.97 GiB. Independent switches and
  memory controls are in the [MOSS guide](docs/moss-image.md).

## October 16, 2026 update

- Latest tested ComfyUI core is **v0.41.2**, with PyTorch **2.14.1** and
  Comfy Kitchen **0.2.37** on M5 Ultra. All 22 regression cases completed twice;
  captured outputs matched v0.40.0 and warm timing changes stayed between
  −2.6% and +1.2%. [Regression report](benchmarks/m5-ultra-comfy0412-smoke.md).
- Added automatic Flux.3 acceleration: phase-local prepared-weight reuse,
  exact RoPE and SwiGLU fusion, and Qwen3.5 prompt-expander weight reuse.
  Warm stock-to-optimized results are **12.12 → 6.17 s** distilled T2I,
  **69.15 → 35.47 s** base T2I, **21.94 → 13.35 s** distilled edit and
  **118.77 → 75.35 s** base edit. Edit comparisons use identical frozen
  expanded prompts; prompt generation separately improves from 116–117 s
  to about 10 s. Base uses 50 steps / CFG 3.5; distilled uses 16 steps / CFG 1.
  [Full benchmark and numerical scope](benchmarks/m5-ultra-flux3.md).
- New Flux.3 refinements preserve the preceding optimized implementation's
  tested latents and decoded bytes; this does not claim stock equivalence.
  Cache budgets, cleanup and independent switches are documented in the
  [Flux.3 guide](docs/flux3.md). The cast hook exists only during relevant phases.
- Added optional H3 INT8 QK / BF16 PV attention presets and their quality,
  timing and memory measurements. Existing exact defaults remain unchanged.
  [Attention report](benchmarks/m5-ultra-h3-floatpv.md).
- Published H3 QKV/FC1/SwiGLU experiments without enabling inconclusive or
  rejected candidates in production. [Fusion report](benchmarks/m5-ultra-h3-fusions.md).
- Updated third-party provenance, benchmark data and reproduction tools.

## October 13, 2026 update

- Corrected H3 Fast token reduction restores full spatial tokens earlier to
  address repeated outlines and horizontal banding observed in Turbo I2V.
  Saved Fast selections migrate to **Fast protected**. The independently
  selectable **Fast aggressive** recipe caps its deeper-reduction phase using
  the actual sampler schedule. Both remain optional approximations.
  [Correction](benchmarks/m5-ultra-h3-fast-token-fix.md) ·
  [Preset validation](benchmarks/m5-ultra-h3-token-dynamic.md).
- Added a narrowly gated exact attention schedule for long H3 sequences on
  the validated M5 Ultra / PyTorch 2.14.1 stack. A matched 10-second, 20-step
  workflow improves **412.03 → 399.21 s**, preserving video/audio latents and
  decoded output bytes. [Report](benchmarks/m5-ultra-h3-long-exact.md).
- Published attention pipeline, hardware-counter and matrix-execution research
  with reproduction tools. Candidates without a repeatable complete-workflow
  gain remain experiments; they are not enabled in production.
  [Benchmark index](benchmarks/README.md).
- Expanded token-policy, legacy-workflow and attention-routing regression
  coverage, and documented latent-upscaler usage after the ComfyUI upgrade.

## 1.1.0 release candidate — October 10, 2026

This candidate includes the automatic tested-runtime setup already on `main`,
the accepted October kernel refinements, and the release documentation/licensing
pass. No repository visibility change or push is part of this candidate commit.

### Inference changes

- Krea2 1MP: narrowly gated M5 Ultra INT8 output tiles reduce the matched
  optimized control from **6.24 to 6.13 s** end-to-end. Twenty native tests,
  actual GEMM checks and prompt/seed fixtures preserve the preceding optimized
  outputs. [Report](benchmarks/m5-ultra-krea2-oct2026.md).
- Ideogram 4 1MP: independent attention output-channel tiles and projection-specific
  INT8 schedules reduce the matched optimized control from **18.41 to 17.78 s**,
  preserving sampler and decoded bytes. Seventy-one native tests cover the accepted
  changes and related paths. Relaxed attention remains research-only.
  [Report](benchmarks/m5-ultra-ideogram-h3-oct2026.md).
- Further H3 exact candidates did not establish a repeatable complete-workflow
  win and were not promoted. Existing optional modes remain selectable.

### Distribution and documentation

- Root license now matches the combined GPL v3 source distribution. Original MIT,
  BSD and Apache notices remain, including the exact original FP8 MIT notice.
- Added missing full notices, corrected source provenance/package identity and
  preserved resolved dependency notices in optional super-resolution builds.
- README reduced from **2,235 to 291 lines**. Historical experiments moved into
  benchmark reports; kernel/settings/numerics/control details have dedicated
  references. Installation, support limits, current timings and troubleshooting
  remain on the front page.
- Added a [privacy audit](docs/privacy-audit.md), documenting machine-path and
  historical-email disclosures and the limits of metadata-only media screening.

### Validation

- **1,017 CPU tests passed**, with **515 native/GPU tests deliberately skipped**.
  The broad run passed 1,013 tests; four localhost HTTP tests initially hit the
  sandbox's socket-binding restriction and then passed in a permitted targeted
  rerun. MPS availability was disabled in the test process; no inference job ran.
- **23 frontend tests passed** with Node 26.10.0.
- Validated **23 notice hashes, 77 pinned vendor files and 71 runtime origins**.
- All local file links in 56 Markdown documents and their linked Markdown
  fragments resolve; package TOML parses and `git diff --check` passes.
- Privacy findings and scope are recorded in the linked report. Remote `main`
  matches the audited base revision; GitHub has no additional branches/tags,
  release assets, Actions runs, issues, PRs, discussions or forks at audit time.

GPU kernel and workflow measurements above are the earlier accepted October
tests; this documentation pass does not claim new GPU measurements or broader
Mac support.
