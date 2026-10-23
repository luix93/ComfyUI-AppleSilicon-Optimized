# ComfyUI nightly `7a283320`: regression on the dev instance

October 23, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 /
torchvision 0.29.1 / Comfy Kitchen 0.2.39 · frontend 1.57.1 · templates 0.11.93.

**35 workflow/mode/preprocessor cases completed twice.** Every case repeats its
own bytes. Every case with a previous record matches it byte for byte: 34
compared, 236 tensors in total. One guard regression was found and repaired:
YuE2 lost its exact single-token RoPE path.

## Tested source and method

- **Instance:** the new `ComfyUI-AppleSilicon-dev` install. Its core is
  `7a28332047b6af638dc6c7f72ef3481a718769fd` (`v0.42.0-19-g7a283320`, master
  as of October 20), and its version file reports 0.42.0.
- **Pack:** the repository working tree as of October 23, synced into the
  instance's `custom_nodes`. It includes the exact MOSS route, the Krea2 INT8
  FFN pair and the new Fibo 2.0 route.
- **References:** each case is compared with its October 19 run (core
  `8f04fd6e`, [previous regression](m5-ultra-comfy0421-smoke.md)), using the
  same fixtures, prompts, seeds, steps and inputs. Fibo 2.0 is compared with
  the October 23 [Fibo benchmark](m5-ultra-fibo2.md).
- **Runs:** run 2 is the warm observation. GPU jobs run serially, with the
  user-queue check and the 110 GiB RSS / 48 GiB available-memory guards.
  Captures are hashed, not dumped.

Scope changes requested for this pass:

- **MiniMax H3:** the exact default, *SOL INT8 · fast* and the new
  *Dense INT8 v3 · balanced* only. The other attention modes were skipped.
- **Wan 2.2:** only the four-step Turbo workflow. The 20-step value in the
  README is **inferred**: the October 19 time is scaled by the measured Turbo
  change (31.07 → 30.80 s, −0.9%), giving 290.78 → **288.25 s**. It was not
  re-measured.
- **Fibo 2.0:** Dev T2I and Edit Turbo added.

## Repair

| Probe | Before repair | After repair | Output bytes |
|---|---:|---:|---|
| YuE2 8 s | 7.65 s | **5.83 s** | Identical |

The core's `comfy/text_encoders/llama.py` changed in `7ef1f42a` ("correct
top-p sampling threshold"). The new source hash `e1018068…` was outside the
reviewed set, so the exact single-token RoPE path stood down (`rope_calls` 0)
and the run was 28% slower with identical bytes. The diff touches only the
top-p sampling mask; the Attention and RoPE definitions are unchanged. The
hash is now admitted in `_patches/yue2_rope_mps.py`. The Qwen3-VL decode paths
share this guard.

Two harness issues were fixed and the affected cases rerun; neither was a pack
regression:

- The copied frozen Flux.3 prompts gained a trailing newline, which changed the
  positive conditioning.
- The Fibo runner's workspace path was wrong.

## Warm results

Seconds. A single warm observation per case; differences of about ±2% are
within run-to-run spread.

| Workflow | Oct 19 warm | Updated warm | Change | Same bytes |
|---|---:|---:|---:|---|
| SDXL · 896×1152 · 12 steps | 2.49 | **2.46** | −0.9% | 5/5 |
| Z-Image Turbo · 1024² · 8 steps | 3.46 | **3.44** | −0.7% | 5/5 |
| Krea2 · 1448² · 8 steps · LoKr | 13.73 | **13.52** | −1.5% | 4/4 |
| Krea2 · 1448² · no LoKr | 13.68 | **13.45** | −1.7% | 4/4 |
| Krea2 · 1024² · 8 steps · no LoKr | 6.20 | **6.03** | −2.7% | 4/4 |
| Flux.2 Klein 9B · T2I · 4 steps | 2.30 | **2.31** | +0.7% | 9/9 |
| Flux.2 Klein 9B · edit · 4 steps | 4.66 | **4.65** | −0.2% | 11/11 |
| Ideogram 4 · 1024² · 12 steps | 17.44 | **17.42** | −0.1% | 9/9 |
| Qwen-Image 2.1 · T2I · 25 steps | 12.81 | **12.68** | −1.1% | 4/4 |
| Qwen-Image 2.1 · edit · 40 steps | 23.37 | **23.20** | −0.7% | 5/5 |
| YuE2 · controlled 8 s (repaired) | 5.98 | **5.83** | −2.6% | 3/3 |
| YuE2 · saved full-song workflow | 35.35 | **35.24** | −0.3% | 3/3 |
| YuE2 · Mothersuperior · 20 s | 22.26 | **22.15** | −0.5% | 3/3 |
| LTX 2.5 · diffusion VAE · 1280×704 · 121 frames | 45.48 | **45.32** | −0.3% | 21/21 |
| LTX 2.5 · convolutional VAE · same clip | 31.36 | **31.19** | −0.5% | 21/21 |
| H3 exact · 864×480 · 124 frames · 20 steps | 145.62 | **146.24** | +0.4% | 9/9 |
| H3 SOL INT8 · fast | 116.05 | **113.61** | −2.1% | 9/9 |
| H3 Dense INT8 v3 · balanced (new) | — | **131.95** | −9.8% vs exact | repeat only |
| Wan 2.2 I2V · four-step LoRA / CFG 1 | 31.07 | **30.80** | −0.9% | 19/19 |
| SeedVR2 7B INT8 · 2400×1600 → 7200×4800 | 45.76 | **44.97** | −1.7% | 3/3 |
| SeedVR2 7B INT8 · 416×736 → 832×1472 · 120 frames | 125.37 | **124.28** | −0.9% | 3/3 |
| Flux.3 distilled · T2I · 16 steps | 5.81 | **5.78** | −0.4% | 6/6 |
| Flux.3 distilled · edit · 16 steps | 12.88 | **12.70** | −1.3% | 8/8 |
| Flux.3 Base · T2I · 50 steps / CFG 3.5 | 33.39 | **33.52** | +0.4% | 6/6 |
| Flux.3 Base · edit · 50 steps / CFG 3.5 | 70.88 | **71.11** | +0.3% | 8/8 |
| Flux.3 Base · 64-token prompt expansion · 2 steps | 9.90 | **10.05** | +1.5% | 8/8 |
| MOSS SFT · 1024² · 25 steps (exact route) | 22.30 | **18.92** | −15.2% | 8/8 |
| MOSS Base · 1024² · 25 steps (exact route) | 22.30 | **19.51** | −12.5% | 8/8 |
| MOSS SFT · 1248² · 25 steps (exact route) | 34.35 | **29.75** | −13.4% | 8/8 |
| Fibo 2.0 Dev · 1024² · 30 steps / CFG 5 (new) | 31.77¹ | **31.43** | −1.1% | 8/8 |
| Fibo 2.0 Edit Turbo · 4 steps (new) | 6.53¹ | **6.28** | −3.9% | 10/10 |
| BiRefNet · 4 synthetic frames | 0.75 | **0.75** | −0.5% | ✓ |
| SDPose + detector · 4 frames | 1.43 | **1.46** | +2.1% | ✓ |
| DA3 Mono Large · 8 frames | 0.42 | **0.41** | −2.0% | ✓ |
| DA3 Base multi-view · 8 frames | 0.11 | **0.11** | −0.2% | ✓ |

¹ The October 23 Fibo benchmark on the same instance and pack, run through the
repository symlink rather than the installed copy.

The MOSS speedups are the October 23 exact route, byte-identical to the
previous pack. In the 64-token prompt check the time moves only in prompt
expansion: 5.89 s on Oct 19, 6.24 s on the first pass and 6.00 s on the rerun.
That is within the observed spread, and no cause is attributed.

## Latest release

The latest tagged release is **v0.42.5**, on the release branch. It does not
contain Fibo 2.0, the MOSS color-depth change or the top-p fix. Its guarded
MOSS model, Llama and `comfy/ops.py` sources all match hashes this pack has
already reviewed, so the same fast paths admit it. It was not separately
regression-timed.

Data: [m5-ultra-comfy042-nightly-oct23.json](data/m5-ultra-comfy042-nightly-oct23.json) ·
harness `diagnostics/comfy-dev-oct23-smoke/`, prepared from the October 19 suite
by `prepare.py`.
