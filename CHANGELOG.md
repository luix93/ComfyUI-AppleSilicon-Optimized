# Release notes

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
