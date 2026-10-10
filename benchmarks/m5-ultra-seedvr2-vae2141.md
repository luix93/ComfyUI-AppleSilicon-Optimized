# SeedVR2 exact VAE follow-up on PyTorch 2.14.1

Completed 2026-10-09. M5 Ultra, 80 GPU cores, 256 GB, macOS 27.0. Runtime/core revisions match the [complete stack retest](m5-ultra-torch2141-retest.md). Only the **-Update** install receives the new defaults; the previous install is preserved. No SATA files, private clips or confidential images were read. The image is the same public Hallstatt fixture; video is the same generated H3 fixture.

## Accepted image change

A two-dispatch FP16 GroupNorm/SiLU/causal-channel-folding fusion preserves PyTorch's original per-thread sum and FMA sum-of-squares, both SIMD reduction levels, FP32 variance/rsqrt and affine order. GroupNorm rounds to FP16 before SiLU; SiLU rounds to FP16 before its temporal replicas enter the unchanged MPS Conv2D. Statistics require only two FP32 values per normalization group (256 bytes at batch one / 32 groups). Weight packing, convolution operands, attention, sampling and tile geometry are unchanged; memory fences remain enabled.

| 2400×1600 → 7200×4800, 7B INT8, one step | Before | Accepted native fusion |
|---|---:|---:|
| Encode | 10.10 s | 9.43 s |
| Sampling | 12.03 s | 11.92 s |
| Decode | 23.08 s | 21.57 s |
| Complete workflow, including PNG | **47.44 s** | **45.15 s** |
| Peak sampled Metal driver memory | 31.73 GiB | 31.95 GiB |
| Peak sampled RSS | 6.24 GiB | 6.21 GiB |

This is **4.8% less E2E time**, saving 2.30 seconds. Sampling is unchanged; its small timing difference is not a claimed sampler improvement. The Python prototype independently measured 44.99 s. The medium 960×640 → 2880×1920 workflow measured 6.98 → 6.73 s.

Each separate full workflow discards its first render and averages two measured renders. All measured encoded latents, sampled latents and final decoded RGBA tensor bytes match the preceding optimized implementation. This is exactness relative to the accepted optimized stack, not a new claim of equality with vanilla stock. The earlier stock plus scheduling safeguard measurement remains 66.29 s.

## Fixed-latent decoder confirmation

Six complete decodes use identical saved sampled latents. The first before/after pair warms both routes, followed by **before / after / after / before**. Decoding alone averages **22.86 → 21.66 s** (5.2% less time); all six frame tensor hashes match the reference. Allocation, normalization, folding, convolution, tiling and output layout are included. Sampling is absent.

## Convolution/layout and other fusion screens

The synchronized operation profile confirms existing folded Conv2D is the largest VAE cost. This profile is diagnostic, not an accepted timing: extra synchronization and captured-input copies perturb performance. Forty-nine captured convolution configurations and twenty-four captured normalization configurations were screened.

Input channels-last layouts changed output bytes in 32 of 49 convolution configurations. Channels-last weights preserved all 49, but did not provide a consistent operation win. The complete one-by-one-convolution layout candidate includes conversion costs and was slower. Existing convolution arithmetic and packing remain unchanged.

Comfy Kitchen 0.2.37 was checked first. Its optimized GroupNorm/SiLU/pad backends target CUDA/HIP; its eager route does not provide an optimized MPS kernel and emits channels-last-3D rather than this causal folded layout. It therefore cannot replace this fusion while retaining the reviewed exact convolution operands.

| Medium complete workflow | Warm E2E | Decision |
|---|---:|---|
| Previous exact implementation | 6.98 s | Reference |
| Legacy single-dispatch GN/SiLU/fold | 7.13 s | Neutral/slower; not enabled |
| Two-dispatch statistics + GN/SiLU/fold | 6.73 s | Accepted |
| Three-dispatch parallel statistics | 6.81 s | Exact, slower than accepted two-dispatch |
| Vectorized normalization application | 6.83 s | Exact, slower than accepted scalar application |
| Additional final normalization/conv wrapper | 6.98 s | No practical gain |
| SiLU + spatial-padding + channel-folding | 6.96 s | No repeatable practical gain |
| Channels-last one-by-one convolution | 7.25 s | Exact complete output, slower |
| Two spatial tiles per synchronization | 7.09 s | Exact, slower; retain tile waits |

Alternate reduction schedules, vectorization and padded SiLU preserve captured-operation bytes. Padded SiLU was checked against every finite FP16 input encoding. Decisions include allocation/layout work; isolated kernel timings do not qualify a change.

## Video fusion and scheduling screens

The generated 416×736 input is upscaled to 832×1472 with 7B INT8, Split Latent off, full context and unchanged temporal/spatial tiling. First render is excluded; two warm repeats are averaged. Outputs are finite; encoded, sampled and frame tensors match baseline.

| Nine-frame workflow | Warm E2E | Peak Metal driver |
|---|---:|---:|
| Existing per-block synchronization | 9.73 s | 18.50 GiB |
| At most two pending blocks | 9.77 s | 18.94 GiB |
| At most four pending blocks | 9.52 s | 19.54 GiB |
| In-place post-normalization SiLU | 9.92 s | 17.50 GiB |
| Per-frame GroupNorm/SiLU fusion | 9.87 s | 18.50 GiB |

The accepted scheduling change retains every spatial-tile fence, caps pending blocks at four, and also fences when driver allocation grows by 2 GiB or reaches 40 GiB. The 40 GiB threshold requests earlier completion; it is not a hard allocation limit. It is checked after each block, so one block can exceed it. Independent diagnostics stop above the configured driver/RSS budget, below 64 GiB system-available memory, or on timeout. Metal high/low watermarks remain active. No watchdog or OOM triggered. In-place SiLU preserves every finite FP16 value, but its lower allocation did not improve timing. The video normalization fusion is exact on this clip but also failed to improve E2E. Those two fusion candidates are not installed.

Five-second alternating whole-workflow confirmation: before **128.18 s**, bounded-four prototype **123.95 s**. The first before/after pair warms both routes; the measured order is before / after / after / before. Peak driver across the six-run process: **33.60 GiB**. All six encoded/sample/frame hashes match.

The final installed default independently averages **124.66 s**, after discarding its first of three full renders: **2.8% less E2E time** than the fresh 128.18 s baseline. Encode/decode average **31.13 / 76.36 s**. Sampling arithmetic is unchanged. Peak sampled driver memory is **33.58 GiB**, peak RSS **13.19 GiB**. Complete encode, sampling, decode, layout and MP4/audio export are included. All three installed-default latent/frame hashes match the baseline. The final wrapper resets its scoped counters at each tile fence and includes driver queries, which are included in this result.

## Validation and controls

Native SeedVR suites pass **64 tests**, including complete residual-block byte checks with identity and projected shortcuts, randomized normalization affine parameters, odd spatial dimensions and disabling the new switch. Scheduling checks cover early memory fences, the four-block cap, partial-tail flushes, tile fences, exception cleanup, FP32 fallback and unknown-build fallback. Installed-default image and five-second-video checks also pass. Source/build/hardware guards remain: the new image fusion and bounded video schedule activate only on the reviewed ComfyUI source, PyTorch 2.14.1 wheel and M5 Ultra. Unsupported image dtype/layout/shape, norm/convolution hooks, memory-state or normalization-memory-limit cases retain the earlier route.

**ASFP8_SEEDVR2_VAE_GN_SPLIT=0** restores the preceding exact normalization/SiLU/folding implementation. **ASFP8_SEEDVR2_VAE_BLOCK_BATCH=1** restores the original every-block video fence. Values 2–4 select smaller bounded batches; values above four are capped and invalid values use one. This schedule is limited to FP16 MPS tiled video. FP32, other devices, unknown builds and FAST-disabled execution retain every-block waits. State is scoped to one tiled call and holds counters, not persistent tensors.

**ASFP8_SEEDVR2_FAST=0** disables SeedVR performance additions while retaining memory protection. Legacy **ASFP8_SEEDVR2_VAE_GN** remains off and separate. Safety fences should remain enabled. Only the five-second full-context video fixture was revalidated with the new schedule; the earlier ten-second timing remains historical, not a new measurement. These fences do not make arbitrarily large inputs fit.

[Machine-readable measurements and hashes](data/m5-ultra-seedvr2-vae2141.json).
