# Release notes

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
