# ComfyUI v0.41.2: bounded optimized-workflow regression check

October 16, 2026 · M5 Ultra, 80 GPU cores / 256 GB · macOS 27 · Python 3.13.12 · PyTorch 2.14.1.

Two completed runs for each of **22 workflow/mode cases**, 44 renders. Successful jobs took approximately **52 minutes** including isolated-server startup, model loading, compilation and verification. No production patch changed during this regression check.

## Scope and method

- New core: **v0.41.2**, commit `b53f77a74e21091cbdebfdf7b85fc80326932646`; previous core **v0.40.0**.
- Same PyTorch build `5c4886908584029761b579af026dcfb627c84070`, torchvision 0.29.1, frontend 1.55.17, PyAV 18.1.0 and Comfy Kitchen 0.2.37 as the previous run.
- Installed defaults and unchanged authored/public fixtures. Private media and SATA files were not accessed.
- Two renders per case. Run 2 is a single warm observation, including sampling, decoding and export. Initial loading is included in run 1; server startup is excluded. Tensor hashing/saving is outside workflow timing.
- Unchanged conditioning is cached. YuE regenerates tokens and conditioning. Diagnostic run IDs force sampling on each run.
- H3 scope: exact, SOL INT8 Fast and SOL INT8 Hybrid Fast at 864×480, 124 frames, 20 steps. No resolution or preset sweep. Optional SOL modes remain approximate relative to exact attention.
- Serial GPU jobs, 110 GiB process-RSS ceiling, 48 GiB available-memory floor and bounded MPS watermarks. SeedVR uses the stricter 80 GiB RSS / 64 GiB available-memory safeguards.
- The initial YuE attempt found a stale benchmark model symlink after models moved. The diagnostic path was corrected to the NVMe location and YuE rerun. No model or production patch repair was needed for that failure.

## Results

Seconds; positive change means slower. Small differences between single warm observations do not establish a meaningful speedup or regression.

| Workflow | v0.40.0 warm | v0.41.2 run 1 | v0.41.2 warm | Time change |
|---|---:|---:|---:|---:|
| SDXL · 896×1152 · 12 steps | 2.49 | 7.19 | **2.46** | -1.0% |
| Z-Image Turbo · 1024² · 8 steps | 3.50 | 9.95 | **3.45** | -1.4% |
| Krea2 · 1448² · 8 steps · LoKr | 13.65 | 27.70 | **13.48** | -1.2% |
| Krea2 · 1448² · no LoKr | 13.60 | 16.08 | **13.52** | -0.6% |
| Flux.2 Klein 9B · T2I · 4 steps | 2.33 | 12.17 | **2.30** | -1.2% |
| Flux.2 Klein 9B · edit · 4 steps | 4.65 | 8.06 | **4.59** | -1.4% |
| Ideogram 4 · 1024² · 12 steps | 17.32 | 32.82 | **17.32** | -0.0% |
| Qwen-Image 2.1 · T2I · 25 steps | 12.70 | 21.57 | **12.57** | -1.0% |
| Qwen-Image 2.1 · edit · 40 steps | 23.21 | 28.45 | **23.18** | -0.1% |
| YuE2 · controlled 8 s | 5.93 | 7.02 | **5.91** | -0.5% |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 47.05 | 69.87 | **46.52** | -1.1% |
| LTX 2.5 · convolutional VAE · same clip | 32.40 | 42.98 | **32.21** | -0.6% |
| H3 exact · 864×480 · 124 frames · 20 steps | 144.28 | 166.42 | **144.53** | +0.2% |
| H3 SOL INT8 Fast · 480p | 114.34 | 123.00 | **115.17** | +0.7% |
| H3 SOL INT8 Hybrid Fast · 480p | 112.30 | 113.23 | **112.50** | +0.2% |
| Wan 2.2 I2V · 512² · 81 frames · 4-step LoRA / CFG 1 | 31.91 | 58.57 | **31.94** | +0.1% |
| Wan 2.2 I2V · same clip · 20 steps / CFG 3.5 | 304.34 | 312.29 | **308.01** | +1.2% |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | 45.40 | 49.62 | **45.29** | -0.2% |
| Krea2 · 1024² · 8 steps · no LoKr | 6.20 | 16.62 | **6.17** | -0.5% |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | 124.17 | 124.83 | **124.34** | +0.1% |
| YuE2 · saved full-song workflow | 34.90 | 37.09 | **33.98** | -2.6% |
| YuE2 · Mothersuperior · 20 s | 21.78 | 24.11 | **21.93** | +0.7% |

## Validation

Every case reproduced its own captured tensor bytes across both runs and matched its captured v0.40.0 reference. This includes final latents, decoded frames and audio where captured. No memory watchdog tripped. Peak sampled Metal allocation was **31.73 GiB** for the large SeedVR image and **33.58 GiB** for its 120-frame video. This is a bounded regression check, not validation of every custom node, model shape, preset or Mac.

[Measured evidence](data/m5-ultra-comfy0412-smoke.json) · [Previous v0.40.0 check](m5-ultra-comfy040-smoke.md).

Local reproduction scripts, submitted graphs, logs, tensors and outputs are retained under `diagnostics/comfy0412-smoke/` in the development workspace.
