# MOSS SFT: 1024² execution audit

October 18, 2026 · M5 Ultra 80 GPU / 256 GB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f5` · Comfy Kitchen 0.2.38.

## Finding

All four large block projections already execute through the pack's native
INT8 kernels: QKV, attention output, FFN input/gate and FFN output. The FFN
output is not falling back to floating matrix multiplication. An earlier
report's description of that path was incorrect and has been corrected.

There is nevertheless redundant work in **our compatibility layer**:
`ops_bias_fp8` dequantizes the FFN output weight to BF16. The core's fused
quantized helper rejects that plain tensor, discards it and calls the ordinary
INT8 module. The MOSS cache was retaining 1.97 GiB of these unused weights.

The exact cleanup bypasses only that failed preparation attempt. It invokes
the same SiTU activation and the same Linear module, preserving interrupt
checks and arithmetic. It requires the reviewed core source, expected weight
layout/shape and the known cast-wrapper chain. Foreign wrappers, active text
encoder cache scopes, hooks and weight functions retain the existing path.
The previous path remains selectable with `ASFP8_MOSS_FFN_DISPATCH=0`.

## Full-step results

SFT INT8 ConvRot, 1024×1024, CFG 2.5, ER-SDE/simple, Qwen3.5-9B INT8 ConvRot
encoder and Flux.2 small VAE. Prompt enhancement and SLG disabled. Identical
prompt/seed for every comparison; both prompts are freshly encoded and a PNG
is saved every run. These are **optimized-to-optimized**, not stock results.

| Steps | Previous optimized E2E | Cleanup E2E | Previous / cleanup sampling |
|---|---:|---:|---:|
| 25 | 22.372 s | 22.240 / 22.182 s | 20.539 / **20.344 s mean** |
| 50 | 42.854 / 42.714 s | 42.743 / 42.667 s | 40.932 / **40.861 s mean** |

Cleanup E2E means are **22.211 / 42.705 seconds**. The initial 25-step control
(24.401 s including process/model warmup) is excluded; consequently there is
only one warm 25-step control. The observed improvements are **0.16 / 0.08 s**,
well below 1% E2E. This is not evidence of a meaningful practical speed gain.
The benefit retained here is elimination of unnecessary weight preparation and
cache allocations, not a claim of reaching the hardware's speed limit.

A preceding three-step screen suggested about 4% lower sampling time. Full
renders did not reproduce that magnitude, so it is not a release speed claim.
Early attempts where the production guard did not activate were discarded as
candidate measurements; accepted runs explicitly check the activation counters.

Every direct run records **2,100 / 4,200** FFN dispatches at 25 / 50 steps and
zero cached weight bytes, versus 42 preparations and **2,113,929,216** cached
bytes in each control. This is a cache-accounting reduction, not a measured
2 GiB fall in process RSS: the allocator retains freed buffers during alternating
runs. Peak sampled Metal allocation in this process was approximately 26.45 GiB.

## Correctness and profiling

All conditioning, initial latents, noise, complete sampler latents and decoded
FP32 frames match byte-for-byte between modes and repeats of the same step
count. The 25-step output is not compared against the 50-step output.
The focused MOSS, YuE2 cache and Qwen3-VL tests pass: **72 tests**. Guard tests
cover foreign wrappers, inactive/active owned wrappers, cycles, hooks, explicit
disable and unreviewed core source. No other model workflow was benchmarked in
this focused follow-up.

A synchronized three-step dispatch audit counted 252 native executions of
**each** large projection. It measured 1.475 s in the prequantized matrix path
and 0.389 s in activation rotation/quantization. These are instrumented costs,
not uninstrumented render timings. The preceding 1.5 MP profile attributes
roughly 54% of main-block time to projection paths including FFN activation and
32% to attention including Q/K normalization and RoPE. These different profiles
must not be added together or treated as an exact 1 MP cost model.

The earlier alternative INT8 tile schedules did not establish a substantial
complete-render improvement. Reusing the H3 fused SwiGLU/rotation path is not
an exact drop-in: MOSS uses SiTU, and that path's reassociated Hadamard rotation
can change BF16 rounding. It was not enabled as an exact optimization.

## Interpreting the Spark comparison

The user reports 48 s at 50 steps with the official MOSS implementation and
30 s at 25 steps with a ComfyUI recipe on one DGX Spark. Their Ultra observations
of 40 / 20 s imply **1.2× / 1.5×**, not 2×, advantages. These external runs have
not been independently reproduced. Since they use different implementations,
we cannot derive Spark's per-step cost by fitting one line through the two.

The fresh Ultra measurements above include about 1.6 s of prompt encoding
plus decoding/export; sampling alone is consistent with roughly 40 / 20 s.
Previous 2× advantages on other models do not establish a 2× MOSS target as
achievable. Equally, these measurements do **not** establish a hardware ceiling.
Reaching 24 / 15 s end-to-end requires substantial improvements to matrix and
attention execution; cache cleanup or VAE work cannot supply that saving.

[Sanitized raw measurements](data/m5-ultra-moss-execution.json) ·
[Initial stock-versus-pack benchmark](m5-ultra-moss-image.md) ·
[1.5 MP follow-up](m5-ultra-moss-15mp.md).
