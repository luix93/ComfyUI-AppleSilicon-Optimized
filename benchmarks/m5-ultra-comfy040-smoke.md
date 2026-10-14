# ComfyUI v0.40.0: bounded optimized-workflow regression check

October 14, 2026 · M5 Ultra, 80 GPU cores / 256 GB · macOS 27 · Python 3.13.12 · PyTorch 2.14.1.

Two completed runs for each of **23 workflow/mode cases**, 46 renders total.
The batch completed in approximately **58 minutes**, including isolated-server startup, initial compilation/loading, and verification.
No production patch was changed during this check.

## Scope and method

- New core: **v0.40.0**, commit `3541246a9c997f4b8c32f9f88f3c985d02021c79`.
- Previous reference: `v0.39.0-33-g1d2ea294`, with the latest accepted October kernel refinements.
- Frontend 1.55.17, PyAV 18.1.0 and Comfy Kitchen 0.2.37. The older reference used PyAV 19.0.1; this compares complete installations, not an isolated core-only version change.
- Freshly installed pack: `14ad343`; reviewed Torch build `5c4886908584029761b579af026dcfb627c84070`.
- Installed default kernel selection, using isolated diagnostic servers and the previous authored/public fixtures. Private media and SATA files were not accessed.
- Each workflow runs twice. Run 2 is the warm observation; there are no tuning sweeps, new stock runs or extra resolution grids.
- Timings include generation, decode and export. Process startup is excluded from per-workflow times. Run 1 includes loading/compilation when needed; H3 attention modes share a process. Unchanged conditioning is cached, while YuE regenerates its tokens and conditioning.
- Sampling is forced to execute on both runs through diagnostic node run IDs. Synchronization and finite-output checks remain; tensor hashing/saving happens outside the timed workflow.
- Previous values are repeated-run means; current warm values are **one observation each**. Small differences do not establish a speedup or a statistically significant regression.
- Serial GPU work, bounded MPS watermarks, a 110 GiB process-RSS ceiling and a 48 GiB available-memory floor. SeedVR additionally enforces its 80 GiB RSS ceiling and 64 GiB available-memory floor. No memory watchdog tripped.
- A monitoring race while a compiler subprocess exited interrupted one Flux startup before rendering. The harness was corrected and resumed without repeating completed workflows.

## Results

All times are seconds. Positive change means more time than the previous optimized reference.

| Workflow | Previous warm | Run 1 | Run 2 (warm) | Time change |
|---|---:|---:|---:|---:|
| SDXL · 896×1152 · 12 steps | 2.40 | 6.37 | **2.49** | +3.6% |
| Z-Image Turbo · 1024² · 8 steps | 3.44 | 9.92 | **3.50** | +1.7% |
| Krea2 · 1448² · 8 steps · LoKr | 13.63 | 25.14 | **13.65** | +0.2% |
| Krea2 · 1448² · no LoKr | 13.96 | 16.62 | **13.60** | -2.6% |
| Flux.2 Klein 9B · T2I · 4 steps | 2.27 | 12.34 | **2.33** | +2.6% |
| Flux.2 Klein 9B · edit · 4 steps | 4.64 | 8.27 | **4.65** | +0.2% |
| Ideogram 4 · 1024² · 12 steps | 17.78 | 31.09 | **17.32** | -2.6% |
| Qwen-Image 2.1 · T2I · 25 steps | 12.66 | 21.68 | **12.70** | +0.3% |
| Qwen-Image 2.1 · edit · 40 steps | 23.00 | 29.11 | **23.21** | +0.9% |
| YuE2 · controlled 8 s | 5.78 | 7.75 | **5.93** | +2.7% |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 46.16 | 70.58 | **47.05** | +1.9% |
| LTX 2.5 · convolutional VAE · same clip | 32.05 | 43.27 | **32.40** | +1.1% |
| H3 exact · 864×480 · 124 frames · 20 steps | 141.65 | 166.48 | **144.28** | +1.9% |
| H3 SOL INT8 Fast · 480p | 114.26 | 123.71 | **114.34** | +0.1% |
| H3 SOL INT8 Hybrid Fast · 480p | 113.79 | 113.30 | **112.30** | -1.3% |
| H3 Veda Metal Protected · 480p | — | 121.34 | **120.87** | Compatibility only |
| Wan 2.2 I2V · 512² · 81 frames · 4-step LoRA / CFG 1 | 31.90 | 55.92 | **31.91** | +0.0% |
| Wan 2.2 I2V · same clip · 20 steps / CFG 3.5 | 304.33 | 310.32 | **304.34** | +0.0% |
| Krea2 · 1024² · 8 steps · no LoKr | 6.13 | 14.53 | **6.20** | +1.2% |
| YuE2 · saved full-song workflow | 34.12 | 36.46 | **34.90** | +2.3% |
| YuE2 · Mothersuperior · 20 s | 21.94 | 24.06 | **21.78** | -0.7% |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | 45.15 | 49.42 | **45.40** | +0.5% |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | 124.66 | 125.63 | **124.17** | -0.4% |

The three optional H3 modes use the same authored moving-camera study at 864×480, 124 frames and 20 steps. Their SOL Fast/Hybrid references are matched v0.39.0-stack measurements with that prompt. Veda lacks a matched full-clip v0.39.0 reference here and is reported as a compatibility check, not a timing-regression claim. These modes remain approximate relative to exact attention; this pass does not reassess their general visual/audio quality.

## Output and memory checks

Every case reproduced its own captured output bytes across the two runs. Every captured tensor compared with an available preceding-stack reference also matched, including sampled latents, decoded frames and audio where captured. Veda's full-clip check establishes repeatability and finite outputs, not cross-stack byte parity. No unexpected kernel-unavailable/latched-off fallback or model execution error was observed.

- SeedVR2 7B INT8 · 2400×1600 → 7200×4800: peak sampled Metal driver allocation **31.83 GiB**; finite output, repeated/reference encoded latents, sampled latents and frames match.
- SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames: peak sampled Metal driver allocation **33.58 GiB**; finite output, repeated/reference encoded latents, sampled latents and frames match.

This is a focused regression smoke test, not a new comprehensive benchmark or validation of all custom nodes, shapes, attention presets and hardware. Existing full benchmark figures remain unchanged.

[Compact measured evidence](data/m5-ultra-comfy040-smoke.json) · [Previous stack report](m5-ultra-torch2141-retest.md).

Local reproduction scripts, exact submitted graphs, logs, tensors and outputs are retained under `diagnostics/comfy040-smoke/` in the development workspace.
