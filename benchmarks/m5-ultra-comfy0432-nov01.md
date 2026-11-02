# ComfyUI v0.43.2: regression on the dev instance

November 1, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 /
torchvision 0.29.1 / Comfy Kitchen 0.2.40 · frontend 1.57.4 · templates 0.11.100.

**35 workflow/mode/preprocessor cases completed twice, and every case repeats
its own bytes.** Four guard regressions were found and repaired: Fibo 2.0,
MOSS, Flux.3 and YuE2 had each lost their fast route. After repair, every one of
these cases is byte-identical to the same case run without the repair on this
core, and the timings are back at their October 23 level.

**One upstream numerical change.** The Flux.2 VAE now runs in FP16 instead of
BF16. As a result, decoded images and encoded reference latents differ from
October 23 for Flux.2 Klein, Ideogram 4, Flux.3, MOSS and Fibo 2.0. The
difference is small: MOSS frames are at 55 dB PSNR, with a maximum difference of
4/255. All other compared tensors are byte-identical to October 23.

## Tested source and method

- **Instance:** the `ComfyUI-AppleSilicon-dev` install, updated to the stable
  release **v0.43.2** (`423cbd20`, October 30). That release contains the
  October 23 nightly `7a283320` plus 47 later commits.
- **Pack:** repository `main` at `57bc1df`, plus the repairs below, synced into
  the instance's `custom_nodes`.
- **References:** each case is compared with its
  [October 23 run](m5-ultra-comfy042-nightly-oct23.md). The same fixtures,
  prompts, seeds, steps and inputs are used.
- **Runs:** run 2 is the warm observation. GPU jobs run serially, with the
  user-queue check and the 110 GiB RSS / 48 GiB available-memory guards.
  Captures are hashed, not dumped.

Scope, as on October 23:

- **MiniMax H3:** the exact default, *SOL Metal · fast* and
  *Dense Metal · balanced*. These now use their current preset names.
- **Wan 2.2:** only the four-step Turbo workflow. The 20-step value is
  **inferred**: the October 23 value is scaled by the measured Turbo change
  (30.80 → 30.45 s, −1.1%), giving 288.25 → **284.92 s**. It was not
  re-measured.

## Repairs

Each fast route checks a source hash of the upstream code it replaces, and
stands down when that code changes. Four routes stood down on v0.43.2. In each
case the diff was reviewed, the route was updated where the interface changed,
and the new hash was admitted.

| Route | What changed upstream | Repair | Before → after |
|---|---|---|---:|
| Fibo 2.0 | RoPE is built with `flux.math.rope`, with NTK axis scales (identical frequencies at scale 1) | Route mirrors the new construction | Dev 38.23 → **31.65 s** · Edit 7.07 → **6.32 s** |
| MOSS-Image | Edit references, a reference K/V cache and an attention bias. Kitchen 0.2.40 was outside the reviewed versions | Admit the model, `ops.py` and Kitchen 0.2.40. References, K/V and bias inputs keep the upstream path. The always-present `mask=None` no longer bypasses tile32 attention | SFT 34.17 → **18.92 s** · Base 35.10 → **19.53 s** · 1248² 50.97 → **29.83 s** |
| Flux.3 | Kitchen 0.2.40 only. Its eager RoPE/AdaLN files are unchanged, and its new fused INT8/VAE kernels are CUDA/HIP only | Admit Kitchen 0.2.40 | Distilled T2I 7.36 → **5.80 s** · Base T2I 43.31 → **33.38 s** · 64-token expansion 41.80 → **10.44 s** |
| YuE2 RoPE | `llama.py` speculative tree-decode caches; the Attention/RoPE definitions are unchanged | Admit the hash | 8 s 7.68 → **5.86 s** · full song 49.15 → **34.43 s** |

The 13 repaired cases were rerun after the unrepaired first pass. Every captured
tensor is identical between the two passes, so each repair changes speed only.

## Upstream numerical change: Flux.2 VAE in FP16

Core commit `d07dff45` ("Fuse and strip the Flux2 VAE through comfy-kitchen")
sets the Flux.2 VAE's working dtypes to FP16, BF16, FP32. The commit's comment
says activations peak near 300. On MPS the VAE therefore loads in FP16
(`VAE load … dtype: torch.float16`); on October 23 it was BF16. The commit's
fused strip-convolution path itself only runs on CUDA/ROCm.

Effects:

- **Text-to-image:** sampled latents still match October 23. Only the decoded
  frames differ.
- **Edit workflows:** the encoded reference latents differ, and so does
  everything sampled from them.
- **Size of the change:** compared with October 23, the MOSS outputs are at
  55.3 dB (SFT 1024²), 55.1 dB (Base) and 54.7 dB (SFT 1248²) PSNR, with a
  maximum per-channel difference of 4/255.
- **Speed:** decode time is unchanged; Flux.2 T2I decodes in 0.20 s on both
  cores.

FP16 carries more mantissa bits than BF16, so this is not a quality loss. The
pack does not override upstream's choice.

## Warm results

Seconds. A single warm observation per case; differences of about ±2% are
within run-to-run spread. "Same bytes" compares with October 23.

| Workflow | Oct 23 warm | v0.43.2 warm | Change | Same bytes |
|---|---:|---:|---:|---|
| SDXL · 896×1152 · 12 steps | 2.46 | **2.53** | +2.6% | 5/5 |
| Z-Image Turbo · 1024² · 8 steps | 3.44 | **3.53** | +2.7% | 5/5 |
| Krea2 · 1448² · 8 steps · LoKr | 13.52 | **13.39** | −1.0% | 4/4 |
| Krea2 · 1448² · no LoKr | 13.45 | **13.53** | +0.6% | 4/4 |
| Krea2 · 1024² · 8 steps · no LoKr | 6.03 | **5.98** | −0.8% | 4/4 |
| Flux.2 Klein 9B · T2I · 4 steps | 2.31 | **2.37** | +2.6% | 8/9 (VAE) |
| Flux.2 Klein 9B · edit · 4 steps | 4.65 | **4.64** | −0.2% | 6/11 (VAE) |
| Ideogram 4 · 1024² · 12 steps | 17.42 | **17.16** | −1.5% | 8/9 (VAE) |
| Qwen-Image 2.1 · T2I · 25 steps | 12.68 | **12.48** | −1.6% | 4/4 |
| Qwen-Image 2.1 · edit · 40 steps | 23.20 | **22.80** | −1.7% | 5/5 |
| YuE2 · controlled 8 s (repaired) | 5.83 | **5.86** | +0.6% | 3/3 |
| YuE2 · saved full-song workflow (repaired) | 35.24 | **34.43** | −2.3% | 3/3 |
| YuE2 · Mothersuperior · 20 s (repaired) | 22.15 | **22.23** | +0.4% | 3/3 |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 45.32 | **44.81** | −1.1% | 21/21 |
| LTX 2.5 · convolutional VAE · same clip | 31.19 | **30.92** | −0.9% | 21/21 |
| H3 exact · 864×480 · 124 frames · 20 steps | 146.24 | **144.88** | −0.9% | 9/9 |
| H3 SOL Metal · fast | 113.61 | **112.42** | −1.0% | 9/9 |
| H3 Dense Metal · balanced | 131.95 | **129.32** | −2.0% | 9/9 |
| Wan 2.2 I2V · four-step LoRA / CFG 1 | 30.80 | **30.45** | −1.1% | 19/19 |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | 44.97 | **44.76** | −0.5% | 3/3 |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | 124.28 | **123.02** | −1.0% | 3/3 |
| Flux.3 distilled · T2I · 16 steps (repaired) | 5.78 | **5.80** | +0.2% | 5/6 (VAE) |
| Flux.3 distilled · edit · 16 steps (repaired) | 12.70 | **12.80** | +0.8% | 3/8 (VAE) |
| Flux.3 Base · T2I · 50 steps / CFG 3.5 (repaired) | 33.52 | **33.38** | −0.4% | 5/6 (VAE) |
| Flux.3 Base · edit · 50 steps / CFG 3.5 (repaired) | 71.11 | **70.82** | −0.4% | 3/8 (VAE) |
| Flux.3 Base · 64-token prompt expansion · 2 steps (repaired) | 10.05 | **10.44** | +3.9%¹ | 3/8 (VAE) |
| MOSS SFT · 1024² · 25 steps (repaired) | 18.92 | **18.92** | 0.0% | 7/8 (VAE) |
| MOSS Base · 1024² · 25 steps (repaired) | 19.51 | **19.53** | +0.1% | 7/8 (VAE) |
| MOSS SFT · 1248² · 25 steps (repaired) | 29.75 | **29.83** | +0.3% | 7/8 (VAE) |
| Fibo 2.0 Dev · 1024² · 30 steps / CFG 5 (repaired) | 31.43 | **31.65** | +0.7% | 7/8 (VAE) |
| Fibo 2.0 Edit Turbo · 4 steps (repaired) | 6.28 | **6.32** | +0.7% | 6/10 (VAE) |
| BiRefNet · 4 synthetic frames | 0.75 | **0.77** | +3.0% | ✓ |
| SDPose + detector · 4 frames | 1.46 | **1.47** | +0.3% | ✓ |
| DA3 Mono Large · 8 frames | 0.41 | **0.41** | +0.6% | ✓ |
| DA3 Base multi-view · 8 frames | 0.11 | **0.12** | +2.9% | ✓ |

"(VAE)" means the only differing tensors are VAE outputs (decoded frames,
encoded references) and what is sampled from them; see above. The first-pass
H3 Dense Metal row now has an October 23 reference and matches it, which
confirms that the October 24 long-sequence change leaves 480p clips untouched.
The small sub-second cases (SDXL, Z-Image, preprocessors) vary by up to 3%,
which is within their usual spread; their bytes are unchanged.

¹ The difference is all in prompt expansion: 6.00 → 6.35 s warm and 6.29 → 6.58 s
first run, with the same generated text. The workflow runs with MTP off. Upstream
reworked the Qwen 3.x decode caches in this range (`f7c66437`, `503d785f`). This
is slightly above the 5.89–6.24 s seen on earlier cores. It was not bisected.

## Data

[m5-ultra-comfy0432-nov01.json](data/m5-ultra-comfy0432-nov01.json) · harness
`diagnostics/comfy0432-smoke/`, prepared from the October 23 suite by
`prepare.py`. The unrepaired first-pass results are in its `asis/` folder.
