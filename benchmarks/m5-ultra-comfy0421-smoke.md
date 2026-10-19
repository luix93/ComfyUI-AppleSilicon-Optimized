# Updated ComfyUI 0.42.x: regression and compatibility repairs

October 19, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 / Kitchen 0.2.39.

**33 workflow/mode/preprocessor cases completed twice**, plus a separate two-run, 64-token prompt-enhancer check and two full-song YuE2 confirmation runs. Restored SDXL and both Wan configurations are included.

## Tested source and method

The instance was reported as updated to 0.42.1. Its actual core is `8f04fd6ecf4a548937971e840af2a985529fab76` (`v0.42.0-3-g8f04fd6e`), and its version file reports 0.42.0. Results are attached to that exact source revision rather than assuming a tagged 0.42.1 release.

Same established prompts, seeds, steps, dimensions and inputs were used, with the SeedVR input exception below. Run 2 is the warm observation. MOSS reruns text encoding; other unchanged conditioning follows the preceding regression method. These compare optimized stacks, not new stock/optimized baselines. SDXL/Wan references use v0.41.2; most others use nightly c06bf5f5. Larger MOSS uses the October 18 exact full-step control and its seed 856771198829162. An additional saved-seed larger render is retained separately, rather than comparing its different initial noise to the old control.

GPU jobs run serially with user-queue checks, 110 GiB process-RSS / 48 GiB available-memory safeguards, and stricter 80/64 GiB SeedVR guards. Tensor captures are hashed rather than dumped to disk. No private user images, clips or SATA files were accessed.

## Repairs

- **MOSS:** the changed core file and Kitchen 0.2.39 disabled the reviewed MOSS path. Adapted the new `rope(..., transformer_options)` interface, passed options through unchanged, and bypassed positional caching for custom RoPE controls. Default positions remain cached. Admit only the reviewed new core hash and Kitchen version; attention, SiTU and modulation definitions are unchanged.
- **Flux.3:** Kitchen 0.2.39 was outside the admitted versions. Review confirmed unchanged model blocks and eager RoPE arithmetic; the existing exact fusions and bounded caches are restored.
- **YuE2:** a new Llama source hash was rejected. Its attention/RoPE definitions are unchanged; the update adds generation cleanup. Admit that reviewed source hash, restoring the exact RoPE path. The shared Qwen3-VL source guard also benefits from this repair.

Only these three minimal compatibility repairs were installed into the live node, with original source backups. The repairs add no approximation. The two existing H3 SOL presets are enabled only in their named smoke cases; the exact default is retained. All 69 targeted tests passed against the actual updated core.

| Probe | Before repair | After repair | Output bytes |
|---|---:|---:|---|
| MOSS SFT 1024 | 34.39 s | **22.53 s** | Identical |
| Flux.3 distilled T2I | 7.41 s | **5.81 s** | Identical |
| YuE2 8 s | 7.59 s | **5.98 s** | Identical |

## Warm results

Seconds. Small differences between single warm observations should not be interpreted as repeatable performance changes.

| Workflow | Previous warm | Updated warm | Change |
|---|---:|---:|---:|
| SDXL · 896×1152 · 12 steps | 2.46 | **2.49** | +1.1% |
| Z-Image Turbo · 1024² · 8 steps | 3.48 | **3.46** | -0.6% |
| Krea2 · 1448² · 8 steps · LoKr | 13.63 | **13.73** | +0.8% |
| Krea2 · 1448² · no LoKr | 13.65 | **13.68** | +0.2% |
| Flux.2 Klein 9B · T2I · 4 steps | 2.29 | **2.30** | +0.3% |
| Flux.2 Klein 9B · edit · 4 steps | 4.66 | **4.66** | -0.0% |
| Ideogram 4 · 1024² · 12 steps | 17.43 | **17.44** | +0.0% |
| Qwen-Image 2.1 · T2I · 25 steps | 12.58 | **12.81** | +1.9% |
| Qwen-Image 2.1 · edit · 40 steps | 23.15 | **23.37** | +0.9% |
| YuE2 · controlled 8 s | 5.87 | **5.98** | +2.0% |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 45.32 | **45.48** | +0.3% |
| LTX 2.5 · convolutional VAE · same clip | 31.28 | **31.36** | +0.3% |
| H3 exact · 864×480 · 124 frames · 20 steps | 145.04 | **145.62** | +0.4% |
| H3 SOL INT8 Fast · 480p | 114.57 | **116.05** | +1.3% |
| H3 SOL INT8 Hybrid Fast · 480p | 112.67 | **115.04** | +2.1% |
| Wan 2.2 I2V · four-step LoRA / CFG 1 | 31.94 | **31.07** | -2.7% |
| Wan 2.2 I2V · 20 steps | 308.01 | **290.78** | -5.6% |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | New input | **45.76** | — |
| Krea2 · 1024² · 8 steps · no LoKr | 6.26 | **6.20** | -1.0% |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | New input | **125.37** | — |
| YuE2 · saved full-song workflow | 34.83 | **35.35** | +1.5% |
| YuE2 · Mothersuperior · 20 s | 22.41 | **22.26** | -0.7% |
| Flux.3 distilled · T2I · 16 steps | 5.79 | **5.81** | +0.3% |
| Flux.3 distilled · edit · 16 steps | 12.79 | **12.88** | +0.6% |
| Flux.3 Base · T2I · 50 steps / CFG 3.5 | 33.38 | **33.39** | +0.0% |
| Flux.3 Base · edit · 50 steps / CFG 3.5 | 70.59 | **70.88** | +0.4% |
| MOSS SFT · 1024² · 25 steps | 22.30 | **22.30** | -0.0% |
| MOSS Base · 1024² · 25 steps | 22.46 | **22.30** | -0.7% |
| MOSS SFT · 1248² · 25 steps | 34.12 | **34.31** | +0.6% |

## Full-song YuE2 follow-up

The initial sweep took 45.52 / 45.67 s. An isolated repeat took 36.95 / 35.35 s; the table shows this confirmation, and the original measurements remain in the JSON. The slow sweep spent about 30.5 s generating semantic tokens, versus roughly 22 s previously. Cache/fusion counters and generated token IDs were unchanged. All captured tensors in both sets match the previous stack. The initial timing anomaly did not reproduce; its cause is not established, so it is not attributed to a specific fix, thermal state or background process.

## Numerical and input scope

Every final main case repeats its own captured bytes. The repair probes preserve complete conditioning/noise/latent/decoded-output hashes. Cross-version mismatches, if any, are recorded explicitly in the JSON below; matching tensors do not certify untested shapes, LoRAs, models or third-party hooks.

**Wan numerical change:** conditioning and initial noise match v0.41.2, but divergence starts in high-noise sampling. A diagnostic replay of the older quantized-linear dispatch restores every previous captured hash on the full four-step LoRA workflow. This is the upstream module-owned input-processing change already isolated for Flux.3/LTX in the previous regression; Wan was unavailable in that pass. The rollback is not installed. The 20-step case is not separately replayed with the old dispatch. These are cross-semantics timing comparisons; no claim of equal visual quality is made from different output bytes.

The old prepared SeedVR image/video media was removed during the authorized storage cleanup. These two cases therefore use authored synthetic fixtures at the same sizes: 2400×1600 → 7200×4800, and 416×736 → 832×1472 / 120 frames. They validate execution, repeatability and memory behavior. Their timing/output comparison with the old input is not an apples-to-apples regression claim. The original input files were not reopened.

## Synthetic preprocessors

| Case | Previous warm | Updated warm | Same output |
|---|---:|---:|---|
| bg | 0.753 s | **0.752 s** | Yes |
| pose | 1.458 s | **1.432 s** | Yes |
| da3 | 0.417 s | **0.415 s** | Yes |
| base | 0.129 s | **0.115 s** | Yes |

The bounded Flux.3 Base prompt-enhancer check takes 9.90 s warm for the complete two-step diagnostic, including expansion, conditioning, sampling and decode; it is not a full 50-step timing. Repeat hash equality: True; preceding-stack hash equality: True.

[Raw timings, hashes, memory and source-review evidence](data/m5-ultra-comfy0421-smoke.json).
