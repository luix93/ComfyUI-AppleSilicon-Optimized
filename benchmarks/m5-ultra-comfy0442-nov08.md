# ComfyUI v0.44.2: regression on the dev instance

November 8, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27.0.1 · PyTorch 2.14.1 /
torchvision 0.29.1 / Comfy Kitchen 0.2.40 · frontend 1.57.6 · templates 0.11.103.

**34 jobs (35 workflow/mode/preprocessor cases) completed twice. Every case
repeats its own bytes, and every compared tensor is byte-identical to the
[November 1 regression](m5-ultra-comfy0432-nov01.md).** There is no upstream
numerical change in this release.

Two guard regressions were found and repaired: the MiniMax H3 VAE fast path and
the MOSS direct-FFN source check. Each repaired case is byte-identical to the
same case run without the repair on this core.

## Tested source and method

- **Instance:** the `ComfyUI-AppleSilicon-dev` install, updated to the stable
  release **v0.44.2** (`e6515b16`, November 6). That release contains v0.43.2
  (`423cbd20`) plus 51 later commits. Torch and Kitchen are unchanged.
- **Pack:** repository `main` at `6237366` (chip-capability gating), plus the
  repairs below, synced into the instance's `custom_nodes`.
- **References:** each case is compared with its November 1 run (for the
  routes repaired then, the repaired rerun). Fixtures, prompts, seeds, steps and
  inputs are the same.
- **Scope:** as on November 1. H3 covers the exact default, *SOL Metal · fast*
  and *Dense Metal · balanced*. Wan 2.2 covers only the four-step Turbo
  workflow.

## Repairs

| Route | What changed upstream | Repair | Effect |
|---|---|---|---|
| MiniMax H3 VAE | `Attention.forward` was split into `attend` / `rms_rope` / `attention`, and `TransformerBlock` now calls `attn.attend` followed by `post_attention` (added for the Sol-H3-VAE and p-h3-image decoders) | The fused QK-norm/RoPE/layout path wraps `attend`. `post_attention` only diverts the native INT8 out-projection. Older cores keep the previous wrapper | H3 exact video decode 11.85 → **10.43 s** (November 1: 10.22 s) |
| MOSS-Image | `ops.py`: the legacy `fp8_linear` reads the cast dtype under dynamic VRAM; BF16 paths are unchanged | Admit the new `ops.py` hash | Same bytes |

The new Sol-H3-VAE (`NonCausalViT3DDecoder`) calls `rms_rope` and `attention`
directly and is not affected by the wrapper. It runs upstream's code.

## Timings

Timings are in line with November 1, though they were not a clean measurement:

- A browser tab kept the GPU about 19% busy throughout. That load slowed
  long GPU-bound loops by 3–4%: H3, Wan, LTX, Krea2, Qwen-Image and MOSS.
- Short or decode-bound cases were within ±2%: Flux.3, YuE2, Fibo 2.0 Dev,
  SeedVR2, SDXL, Z-Image and the preprocessors.
- **The pack commit is not the cause.** An A/B on the same core and the same
  machine state gave the same timings with the November 1 pack (`4414edc`) and
  the current pack: Krea2 1024² 6.18 / 6.18 s, Wan Turbo 31.39 / 31.49 s.
- The first-pass Krea2 LoKr outlier (15.72 s) came from a local LLM that was
  running at the time. Its rerun took 13.79 s.

The README's latest-core column therefore keeps the November 1 v0.43.2
timings.

| Workflow | Nov 1 warm | v0.44.2 warm | Same bytes |
|---|---:|---:|---|
| SDXL · 896×1152 · 12 steps | 2.53 | 2.59 | 5/5 |
| Z-Image Turbo · 1024² · 8 steps | 3.53 | 3.50 | 5/5 |
| Krea2 · 1448² · LoKr (rerun) | 13.39 | 13.79 | 4/4 |
| Krea2 · 1448² · no LoKr (rerun) | 13.53 | 13.99 | 4/4 |
| Krea2 · 1024² (A/B) | 5.98 | 6.18 | 4/4 |
| Flux.2 Klein 9B · T2I / edit | 2.37 / 4.64 | 2.35 / 4.74 | 9/9 · 11/11 |
| Ideogram 4 · 1024² · 12 steps | 17.16 | 17.73 | 9/9 |
| Qwen-Image 2.1 · T2I / edit | 12.48 / 22.80 | 12.79 / 23.36 | 4/4 · 5/5 |
| YuE2 · 8 s / full song / Mothersuperior | 5.86 / 34.43 / 22.23 | 5.86 / 34.15 / 21.75 | 3/3 each |
| LTX 2.5 · diffusion / convolutional VAE | 44.81 / 30.92 | 46.25 / 32.00 | 21/21 each |
| H3 exact (repaired) | 144.88 | 150.80 | 9/9 |
| H3 SOL Metal · fast (repaired) | 112.42 | 118.49 | 9/9 |
| H3 Dense Metal · balanced (repaired) | 129.32 | 137.02 | 9/9 |
| Wan 2.2 Turbo (A/B) | 30.45 | 31.49 | 19/19 |
| SeedVR2 image / 120-frame video | 44.76 / 123.02 | 45.64 / 124.81 | 3/3 each |
| Flux.3 distilled · T2I / edit | 5.80 / 12.80 | 5.79 / 13.01 | 6/6 · 8/8 |
| Flux.3 Base · T2I / edit / 64-token expansion | 33.38 / 70.82 / 10.44 | 33.53 / 72.54 / 10.43 | 6/6 · 8/8 · 8/8 |
| MOSS SFT / Base 1024², SFT 1248² (repaired) | 18.92 / 19.53 / 29.83 | 19.43 / 19.92 / 31.72 | 8/8 each |
| Fibo 2.0 Dev / Edit Turbo | 31.65 / 6.32 | 31.95 / 6.53 | 8/8 · 10/10 |
| Preprocessors (BiRefNet, SDPose, DA3 mono / multi-view) | 0.77 / 1.47 / 0.41 / 0.12 | 0.77 / 1.45 / 0.41 / 0.13 | ✓ |

## Data

Harness: `diagnostics/comfy0442-smoke/`, prepared from the November 1 suite by
`prepare.py`. The unrepaired first-pass results of the rerun cases are in its
`asis/` folder, and the pack A/B is in `ab-pack/`.
