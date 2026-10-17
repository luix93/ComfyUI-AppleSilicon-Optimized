# Benchmark reports


- [ComfyUI v0.41.2 regression check](m5-ultra-comfy0412-smoke.md): 22 cases, two runs each; matched v0.40.0 outputs and stable warm timings.

The [current performance table](../README.md#latest-results--m5-ultra-80-gpu-cores)
is the short release summary. Each report records its own hardware, stack,
workflow, reference and timing boundaries. An exact refinement can preserve a
preceding optimized output without matching stock ComfyUI.

## Current stack and model reports

| Model / area | Reports |
|---|---|
| ComfyUI v0.40.0 regression check | [23 cases, two runs each, prior-stack timing and tensor comparisons](m5-ultra-comfy040-smoke.md) |
| Full PyTorch 2.14.1 retest | [Models, compatibility and timings](m5-ultra-torch2141-retest.md) |
| SDXL / Z-Image | [2.14 production workflows](m5-ultra-torch214-production.md), [SDXL investigation](m5-ultra-sdxl-investigation.md) |
| Krea2 | [October exact tile and rejected attention experiments](m5-ultra-krea2-oct2026.md), [earlier exact work](m5-ultra-krea2-exact.md) |
| Ideogram 4 / H3 kernels | [QKV / FC1 exact fusion follow-up](m5-ultra-h3-fusions.md), [Exact matrix execution experiments](m5-ultra-h3-matrix-execution.md), [Hardware counters and occupancy](m5-ultra-h3-gpu-counters.md), [GPU profiling and exact K/V layout](m5-ultra-h3-attention-profile.md), [Exact attention pipeline screens](m5-ultra-h3-exact-attention-pipeline.md), [Long-clip exact attention follow-up](m5-ultra-h3-long-exact.md), [October kernel follow-up](m5-ultra-ideogram-h3-oct2026.md), [Ideogram baseline](m5-ultra-ideogram4.md) |
| Qwen 2.1 | [Generation](m5-ultra-qwen21.md), [image edit](m5-ultra-qwen21-edit.md) |
| Flux.3 Klein | [Base/distilled, T2I/edit, exact refinements and prompt expansion](m5-ultra-flux3.md) |
| Flux.2 Klein | [Text-to-image and edit](m5-ultra-flux2-klein.md) |
| LTX 2.5 | [Baseline](m5-ultra-ltx25.md), [VAE and exact follow-up](m5-ultra-ltx25-followup.md) |
| Wan 2.2 | [Ordinary and Turbo LoRA workflows](m5-ultra-wan22.md) |
| SeedVR2 | [2.14.1 exact VAE follow-up](m5-ultra-seedvr2-vae2141.md), [video memory diagnosis](m5-ultra-seedvr2-video-memory.md) |
| YuE2 | [Initial pass](m5-ultra-yue2.md), [RoPE/full-song follow-up](m5-ultra-yue2-followup.md) |
| H3 optional attention | [Reference/guide switch and exact prefix attention](m5-ultra-h3-reference-attention.md), [INT8 QK / floating PV at 1024×512](m5-ultra-h3-floatpv.md), [Kitchen/dense/hybrid attention and morphing](m5-ultra-h3-kitchen-attention.md), [SOL/top-k](m5-ultra-h3-sol-topk.md), [Veda](m5-ultra-h3-veda.md) |
| H3 cache/token controls | [Adaptive Fast presets](m5-ultra-h3-token-dynamic.md), [Fast artifact fix](m5-ultra-h3-fast-token-fix.md), [presets](m5-ultra-h3-presets.md), [bundled runtimes](m5-ultra-h3-owned-runtime.md), [combined aggressive tests](m5-ultra-h3-sol-aggressive.md) |

## Historical experiments

[Historical README results and field notes](readme-history-2026-10.md) preserve
the longer narrative previously shown on the front page, including earlier M5
Max/Ultra comparisons, VAE/sampling sprints, rejected experiments and thermal
measurement lessons. Follow its named stack and reference rather than treating
old results as the current configuration.

Additional reports include [relaxed sprint 3](m5-ultra-relaxed-sprint3.md),
[relaxed sprint 4](m5-ultra-relaxed-sprint4.md),
[exact sampling sprint](m5-ultra-sampling-sprint2.md),
[H3 long query-cache](m5-ultra-h3-long-qcache.md),
[Sage-style attention](m5-ultra-h3-sage.md),
[recent methods](m5-ultra-h3-recent-methods.md), and
[SeedVR2 acceleration](m5-ultra-seedvr2-acceleration.md).

Benchmark media are research artifacts. Some JSON/log/script records retain
machine paths; see the [privacy audit](../docs/privacy-audit.md). Large tensors,
downloaded checkpoints and generated optional runtimes are not part of this
source release.
