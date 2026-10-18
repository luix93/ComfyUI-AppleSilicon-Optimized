# MOSS-Image SFT: 1.5 MP follow-up

October 18, 2026 · M5 Ultra 80 GPU / 256 GB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f5` · Comfy Kitchen 0.2.38.

## Reproduction

The user's latest output embeds **1248×1248 (1.5575 MP), 50 steps, CFG 2.5,
ER-SDE/simple**, SFT INT8 ConvRot, Qwen3.5-9B INT8 ConvRot encoding and the small
Flux.2 VAE. Prompt enhancement and SLG are both off. The live log already showed
MOSS acceleration enabled and a 67.32-second completed render.

This differs from the earlier **1024×1024 / 25-step** benchmark. The apparent
regression is predominantly the larger workload: doubling steps and increasing
pixels by 48.5%. A fresh isolated reproduction took 70.47 seconds including
loading, then 68.25 seconds warm. Sampling alone took 66.11 / 66.29 seconds.
The 35-second / 17–18-second Blackwell figures are user-reported; its complete
software, precision and timing boundaries have not been independently matched.

## Accepted exact change

The MOSS query-reuse gate previously stopped at 4,608 tokens. This workflow
uses **6,167 positive / 6,107 negative tokens**, so attention fell back to native
PyTorch while the other MOSS optimizations remained active. Extend the reviewed
range to 6,656 tokens; kernel arithmetic, tile sizes, reductions, precision,
CFG, sampler, step count and all keys/values remain unchanged.

## Complete workflow measurements

These compare the **preceding optimized pack with the new exact refinement**,
not stock ComfyUI. Same prompt/seed; fresh text encoding, sampling, decode and
PNG export every render. Models remain resident. Initial process/model warmup
is excluded. The full-step confirmation alternates current/candidate paths.

| Steps | Previous optimized warm runs | New exact warm runs | Warm means | Less time |
|---|---|---|---|---|
| 50 | 68.252 / 68.575 s | 67.411 / 66.058 s | **68.41 → 66.73 s** | 2.5% |
| 25 | 35.335 / 35.196 s | 34.125 / 34.020 s | **35.27 → 34.07 s** | 3.4% |

The 50-step control uses the initial warm reproduction and the warm alternating
control. Both candidate 50-step observations are retained, including the slower
one. These are small repeatable gains, not evidence of approaching a 2× speedup.
There is no measured stock baseline at this resolution/step count in this pass.

## Exactness and memory

- All full-step captures match their same-step reference: conditioning, initial
  latents/noise, complete sampler latents and decoded FP32 frames. The 25-step
  output is compared with its 25-step reference, not the 50-step output.
- Both 50-step candidates execute 4,200 accelerated attention calls; both
  25-step candidates execute 2,100. The previous path executes none at this size.
- Direct native-SDPA byte comparisons pass at 4,609, 6,107, 6,167 and 6,656
  tokens, including packed non-contiguous Q/K/V views and partial tiles.
- Sampled Metal allocation peaked at approximately 27.4 GiB in the alternating
  process. The 110 GiB RSS ceiling / 48 GiB available-memory floor did not trip.
  No private source images/videos or SATA files were read.

## Profiling and further candidates

A synchronized two-step profile attributes about **54%** of block time to the
matrix-projection paths including FFN activation, **32%** to attention including
Q/K normalization/RoPE, **11%** to modulation/normalization, and **3%** to residual
updates. Instrumentation changes execution timing; these are cost proportions,
not full-render estimates. Cast-weight timings are inclusive and must not be
added again to the projection totals.

Three alternative INT8 tile schedules preserve matrix outputs at both actual
branch lengths. The best tested schedule also preserves all eight-step workflow
captures, reducing warm sampling from 10.378 to 10.232–10.267 seconds and E2E
from 12.330 to 12.162–12.178 seconds. It remains experimental: the small screening
result has not received a full 25/50-step acceptance run. No matrix dispatch
change is installed.

The tested attention extension cannot close the roughly twofold Blackwell gap.
Further material gains need improvements to the dominant projection/attention
execution, rather than prompt enhancement, VAE decode or another cache toggle.

[Sanitized measurements](data/m5-ultra-moss-15mp.json) ·
[Initial MOSS benchmark](m5-ultra-moss-image.md) · [Controls](../docs/moss-image.md).
