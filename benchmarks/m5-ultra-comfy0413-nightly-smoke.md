# ComfyUI nightly c06bf5f5: bounded regression check

October 18, 2026 · M5 Ultra, 80 GPU cores / 256 GB · macOS 27 · PyTorch 2.14.1 · Kitchen 0.2.38.

**29 available workflow/mode/preprocessor cases completed twice.** SDXL and both Wan cases were skipped because their saved checkpoints were no longer available on the benchmark drive. Two compatibility guards caused real slowdowns and were updated after byte-level validation.

## Stack and method

- Core `c06bf5f569d268b80076e5f44126f87cdf835869` is a nightly after v0.41.3, not the tagged v0.41.3 release. Previous main references use v0.41.2 (`b53f77a7`), Kitchen 0.2.37 and the same Torch build. MOSS references were already on this nightly.
- Two completed runs per case; run 2 is the warm observation. Same saved prompts, seeds, inputs, resolutions, frame counts and samplers. No private images/clips or SATA files were accessed.
- Generation timing includes sampling, decode and export. Unchanged conditioning is cached, except YuE regenerates tokens/conditioning and MOSS reruns text encoding to match its accepted reference. Flux.3 editing uses the same frozen expanded prompts; the enhancer received a separate bounded 64-token compatibility check.
- H3 coverage is exact, SOL INT8 Fast and SOL INT8 Hybrid Fast at 864×480 / 124 frames / 20 steps. No resolution or preset sweep.
- Serial GPU execution; 110 GiB process-RSS ceiling and 48 GiB available-memory floor. SeedVR retains the stricter 80/64 GiB guards and bounded MPS allocation. No memory watchdog tripped.
- SeedVR sampled Metal peaks were 31.73 GiB for the large image and 33.55 GiB for the 120-frame video. These are allocator measurements, not total system RAM; overlapping unified-memory counters must not be added.
- The 46 focused cache-lifetime, fallback and GPU byte-equivalence tests passed. The main suite took about 48 minutes; targeted diagnosis, repairs and confirmation added about 20 minutes.
- Initial SeedVR image validation found a missing source photo. The identical prepared input retained with the old benchmark was reused; the production node did not need a fix. The initial uncapped prompt-enhancer diagnostic was stopped and replaced by the 64-token check.

## Compatibility repairs

- **YuE2:** the Llama source-file guard rejected the new nightly even though the reviewed attention and RoPE functions were unchanged. Added only the reviewed source hash. Warm times recovered from **7.73 → 5.87 s**, **49.10 → 34.83 s** (full workflow), and **31.47 → 22.41 s** (Mothersuperior). All captured tensors match both the pre-repair nightly and old references.
- **Flux.3:** the optimization accepted only Kitchen 0.2.37. The reviewed 0.2.38 build now also enables the existing exact fusions and phase-local caches. All four full sampling workflows preserve the pre-repair nightly tensors; cache cleanup is checked. No attention or precision approximation was added.

## Warm results

Seconds. Positive change means slower. Small differences between single warm observations are not evidence of a repeatable speed change. Flux.3 and LTX comparisons cross an upstream numerical change described below.

| Workflow | Previous warm | Current warm | Time change |
|---|---:|---:|---:|
| Z-Image Turbo · 1024² · 8 steps | 3.45 | **3.48** | +0.9% |
| Krea2 · 1448² · 8 steps · LoKr | 13.48 | **13.63** | +1.1% |
| Krea2 · 1448² · no LoKr | 13.52 | **13.65** | +0.9% |
| Flux.2 Klein 9B · T2I · 4 steps | 2.30 | **2.29** | -0.4% |
| Flux.2 Klein 9B · edit · 4 steps | 4.59 | **4.66** | +1.6% |
| Ideogram 4 · 1024² · 12 steps | 17.32 | **17.43** | +0.7% |
| Qwen-Image 2.1 · T2I · 25 steps | 12.57 | **12.58** | +0.0% |
| Qwen-Image 2.1 · edit · 40 steps | 23.18 | **23.15** | -0.1% |
| YuE2 · controlled 8 s | 5.91 | **5.87** | -0.7% |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 46.52 | **45.32** | -2.6% |
| LTX 2.5 · convolutional VAE · same clip | 32.21 | **31.28** | -2.9% |
| H3 exact · 864×480 · 124 frames · 20 steps | 144.53 | **145.04** | +0.4% |
| H3 SOL INT8 Fast · 480p | 115.17 | **114.57** | -0.5% |
| H3 SOL INT8 Hybrid Fast · 480p | 112.50 | **112.67** | +0.1% |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | 45.29 | **45.45** | +0.3% |
| Krea2 · 1024² · 8 steps · no LoKr | 6.17 | **6.26** | +1.4% |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | 124.34 | **124.70** | +0.3% |
| YuE2 · saved full-song workflow | 33.98 | **34.83** | +2.5% |
| YuE2 · Mothersuperior · 20 s | 21.93 | **22.41** | +2.2% |
| Flux.3 distilled · T2I · 16 steps | 6.17 | **5.79** | -6.1% |
| Flux.3 distilled · edit · 16 steps | 13.35 | **12.79** | -4.1% |
| Flux.3 Base · T2I · 50 steps / CFG 3.5 | 35.47 | **33.38** | -5.9% |
| Flux.3 Base · edit · 50 steps / CFG 3.5 | 75.35 | **70.59** | -6.3% |
| MOSS SFT · 1024² · 25 steps | 22.14 | **22.30** | +0.7% |
| MOSS Base · 1024² · 25 steps | 22.44 | **22.46** | +0.1% |

## Numerical regression findings

**Every final case repeats its own captured bytes.** All generation cases also match their previous references except LTX 2.5 and Flux.3. For those, conditioning and initial noise match; divergence begins during sampling.

A diagnostic replay of the old ComfyUI quantized-linear dispatch restores **all previous captured bytes** for the full LTX convolutional-VAE workflow and Flux.3 distilled T2I. This isolates an upstream dispatch change on those two probes. Both LTX VAE cases and all four Flux.3 cases show the same sampling-onward divergence pattern; the other variants were not separately replayed with the old dispatch.

The diagnostic rollback is **not installed**. Current ComfyUI retains its module-owned input-processing fallback. The compatibility fixes do not change the nightly outputs. Byte changes across the upstream upgrade are not a claim of equivalent visual quality; this regression pass is not a subjective quality study.

## Synthetic preprocessor checks

Two runs using the previous authored synthetic fixtures. These are whole-preprocessor timings, excluding model loading and file/video IO. The old reference is the mean of three warm runs; current is one warm run. All masks/depth/keypoints/rendered outputs match their old references.

| Case | Previous warm mean | Current warm |
|---|---:|---:|
| BiRefNet · 4 frames | 0.757 s | **0.753 s** |
| SDPose · 4 frames / 8 crops | 1.448 s | **1.458 s** |
| DA3 Mono Large · 8 frames | 0.416 s | **0.417 s** |
| DA3 Base multi-view · 8 frames | 0.122 s | **0.129 s** |

This is a bounded compatibility check of the listed cases, not certification of every model, shape, third-party node or Mac.

[Sanitized measurements](data/m5-ultra-comfy0413-nightly-smoke.json) · [Previous regression](m5-ultra-comfy0412-smoke.md).

Local scripts, captures, logs, source-review evidence and guard records are retained in the development diagnostics directory for this run.
