# Benchmark reports

- [MOSS CFG projection grouping](m5-ultra-moss-cfg-projections.md): byte-identical full-render comparisons, small/inconsistent gains, and a screened joined-state refinement; not promoted.

- [MOSS-specific dense attention tuning](m5-ultra-moss-attention-tuning.md): exact query tiles, INT8 preparation/layout changes, full-render comparisons and rejected candidates.

- [MOSS exact FC1/SiTU matrix-output fusion](m5-ultra-moss-ffn-epilogue.md): guarded opt-in kernel, full-render byte checks, small practical gains and rejected ConvRot fusion.

- [MOSS with H3 dense INT8 kernels](m5-ultra-moss-h3-dense.md): both dense variants, full-step 1024/1248 comparisons and numerical/visual tradeoffs; no SOL.

- [MOSS Kitchen-style INT8 Metal prototype](m5-ultra-moss-kitchen-int8.md): shared backend, independent FFN fusion, rejected end-to-end results and a standalone CUDA reference script.

- [MOSS SFT 1024² execution audit](m5-ultra-moss-execution.md): actual INT8 dispatch, redundant weight-cache cleanup and full-step results; no material new speed claim.

- [MOSS SFT 1.5 MP exact attention follow-up](m5-ultra-moss-15mp.md): 25/50-step timings, byte checks and profiling.

- [ComfyUI nightly c06bf5f5 regression](m5-ultra-comfy0413-nightly-smoke.md): 29 available cases run twice; YuE2/Flux.3 guard repairs, skipped checkpoints and upstream numerical changes documented.

- [ComfyUI v0.41.2 regression check](m5-ultra-comfy0412-smoke.md): 22 cases, two runs each; matched v0.40.0 outputs and stable warm timings.

The [current performance table](../README.md#latest-results--m5-ultra-80-gpu-cores)
is the short release summary. Each report records its own hardware, stack,
workflow, reference and timing boundaries. An exact refinement can preserve a
preceding optimized output without matching stock ComfyUI.

## Current stack and model reports

| Model / area | Reports |
|---|---|
| MOSS-Image SFT / Base | [Nightly stock baseline, exact fusions/caches, full-step byte comparisons](m5-ultra-moss-image.md) |
| ComfyUI v0.40.0 regression check | [23 cases, two runs each, prior-stack timing and tensor comparisons](m5-ultra-comfy040-smoke.md) |
| Full PyTorch 2.14.1 retest | [Models, compatibility and timings](m5-ultra-torch2141-retest.md) |
| SDXL / Z-Image | [2.14 production workflows](m5-ultra-torch214-production.md), [SDXL investigation](m5-ultra-sdxl-investigation.md) |
| Krea2 | [October exact tile and rejected attention experiments](m5-ultra-krea2-oct2026.md), [earlier exact work](m5-ultra-krea2-exact.md) |
| Ideogram 4 / H3 kernels | [QKV / FC1 exact fusion follow-up](m5-ultra-h3-fusions.md), [Exact matrix execution experiments](m5-ultra-h3-matrix-execution.md), [Hardware counters and occupancy](m5-ultra-h3-gpu-counters.md), [GPU profiling and exact K/V layout](m5-ultra-h3-attention-profile.md), [Exact attention pipeline screens](m5-ultra-h3-exact-attention-pipeline.md), [Long-clip exact attention follow-up](m5-ultra-h3-long-exact.md), [October kernel follow-up](m5-ultra-ideogram-h3-oct2026.md), [Ideogram baseline](m5-ultra-ideogram4.md) |
| Qwen3-VL 4B text generation | [Exact PyTorch decode acceleration, including synthetic multi-image input](m5-ultra-qwen3vl-decode.md), [large-image vision memory safeguard](qwen-vision-memory-safety.md) |
| BiRefNet / SDPose / Depth Anything 3 | [Warm preprocessing timings, exact SDPose refinement and DA3 multi-view compatibility](m5-ultra-vision-preprocessors.md) |
| SAM3.1 segmentation | [Repeated Metal compilation fix and exact refinement-reuse prototype](m5-ultra-sam3.md) |
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
