# SeedVR2: PyTorch 2.14 retest and exact VAE cleanup

2026-09-27 · M5 Ultra, 80 GPU cores, 256 GB · macOS 27.0 · PyTorch 2.14.0
(`08187d9e0fba026dc8217405802ab5381dc88d90`) · ComfyUI
`b33e2b55cae074eca5aec96283cceac19aa249ba`.

## Complete upscale

The saved `seedvr2_image_2400x1600_3x.json` workflow uses the 7B INT8 ConvRot
model, FP16 VAE, 2400×1600 input, 3× Lanczos resize to **7200×4800**, 512-pixel
VAE tiles with 128 overlap, one Euler/simple step, CFG 1, seed 959948902156062,
and no color correction. The checkpoint's current subfolder is `SeedVR2/`.
The input PNG matches the shared workflow input (SHA-256
`9b3b87025e7c4fd45e9e7d2fb679fca792605f2e82e9ee4d61da6edfa80c44b1`).

| Phase | Stock 2.14 + tile safety wait | Existing optimized stack | Updated optimized stack |
|---|---:|---:|---:|
| VAE encode | 14.410 s | 10.541 s | 10.272 s |
| Sampling | 21.828 s | 11.817 s | 11.923 s |
| VAE decode | 35.549 s | 24.626 s | 24.555 s |
| Complete sequence, including PNG | **74.016 s** | **49.215 s** | **49.011 s** |
| Maximum sampled Metal allocation | 40.256 GiB | 32.756 GiB | 32.975 GiB |

Means of two warm repeats after discarding each process's first run. Final
warm totals were 48.992 and 49.029 seconds. Resize, preprocessing, encoding,
sampling, decoding, postprocessing and PNG writing execute every time; only
model loading is reused. These are synchronized timings of the actual node
sequence, without server queue/UI overhead. Verification and tensor saving
occur after the timed sequence. Per-tile memory observations are included in
all configurations.

The full optimized stack takes **33.8% less time** than stock + safety. The
new cleanup saves only **0.204 s (0.4%)** versus the previous optimized stack:
a small result near run-to-run variability, not a significant additional
speedup. Encoding improves consistently; sampling was not changed.

## Memory safeguards

The previous failure was accumulation of pending Metal temporary allocations
across VAE tiles. Both full-size configurations retain a synchronization after
each tile. Stock otherwise imports none of this custom node: its benchmark
adapter supplies only the equivalent scheduling wait, with native PyTorch
convolution, weight-only INT8 and attention arithmetic.

A deliberately bounded **320×213 → 960×638 unprotected probe** confirmed that
this issue still exists on 2.14. During decoding, driver allocation reached
**25.191 GiB** while live tensors were **8.241 GiB**. The diagnostic's 24 GiB
sampled threshold stopped the run immediately; it has no complete output or
end-to-end timing. This is a controlled diagnostic stop, not a system crash.

No full-size unprotected run was attempted. Every run used an external 96 GiB
RSS ceiling, 64 GiB minimum available system memory and 900-second timeout;
full runs also checked sampled driver/live allocations against 80 GiB after
each VAE call. Metal telemetry stays on the execution thread. No watchdog
stopped a completed benchmark. Metal allocation is sampled, not a continuous
peak or total system RAM; RSS is reported separately in the raw records and
must not be added to it.

The original one-tile scheduling policy, bounded INT8 dispatches, 64-bit row
gathers, and oversized-attention buffer safeguards remain unchanged. The
18720×12480 extreme case was **not rerun**; its previous successful test is
historical evidence, not a new 2.14 validation.

## Accepted change and numerical checks

The FP16 image VAE's residual blocks use output scale 1. Their existing
`(input + hidden) / 1` allocated and traversed another full tensor after the
addition. The optimized wrapper now returns the addition directly for unit
scale, retaining the original division for other scales. Convolutions,
normalization, activations, tile geometry, attention and quantization stay the
same. This runs under the existing `ASFP8_NATIVE=1` / SeedVR2 fast path.

The final full-size runs match the preceding optimized stack byte for byte:

| Tensor | SHA-256 |
|---|---|
| Encoded latent | `b4ddc3843b6354381408b871e45a64087b8da413fa284a52bb0b3ef50e734d1c` |
| Sampled latent | `0dd4a4ac8867da450bfb65397218d2b056d5d957f016e287c7dde83c84d51156` |
| Final frame tensor | `59a20c68d3cb6882459ce574a2c9c050141f0978f3fb80c141011ef0f2adc056` |

All frames are finite. Small workflows also compare the complete SeedVR2
fusion stack against the same native INT8/convolution stack with those
fusions disabled, using the original seed plus seeds 42 and 271828. A raw
FP16 checkpoint smoke test covers the conditioning-loader repair separately.

**35 applicable tests pass**, covering loader compatibility, memory scheduling,
packing-cache ownership, bounded INT8 projection, strided 64-bit row gathering,
attention, normalization/modulation, RoPE and residual scaling. The unit-scale
check includes every finite FP16 bit pattern; a non-unit-scale test retains
the division. Three tests for the optional PyTorch-2.13-only GroupNorm fusion
are deselected: its version guard keeps that implementation inactive on 2.14.
An initial test invocation omitted `ASFP8_INT8_EXT=1`; enabling that prerequisite
resolved its loader-related failure.

The complete stock and optimized stacks **are not bit-exact to each other**.
The existing native W8A8 activation quantization and convolution decomposition
already differ from stock. This retest verifies preservation relative to the
existing optimized implementation, not restoration of stock bytes.

## Experiments not enabled

- **Coalesced attention window copies, batch 16:** a bounded representative
  BF16 attention test improved from 3.629 to 2.904 ms with identical bytes.
  The full sampler did not improve: 11.833 versus 11.817 s. Its combined full
  run with the VAE cleanup averaged 48.732 s, but the apparent advantage was
  in unrelated VAE timings. Keep the original batch 8 and copy implementation.
- **Four-tile synchronization:** a combined medium screen with attention
  batching improved 7.257 to 7.100 s, but Metal allocation increased from
  16.393 to 19.129 GiB. This is not an isolated scheduling-speed measurement.
  Given the small combined benefit and the prior RAM failure, keep one-tile
  waits; no production flag or default was added for this experiment.
- **MLP chunks 16384 instead of 8192:** medium total 7.236 versus 7.257 s;
  sampled latents and frames identical, allocation 16.393 GiB. The result is
  effectively neutral, so retain the original chunk size.

The current remaining cost is principally the existing VAE convolution work.
This pass found no further substantial exact gain; it does not establish an
absolute hardware limit.

## Reproduction and artifacts

[Raw results, workflows, memory traces, and local benchmark harness](artifacts/seedvr2-torch214/)
retain the completed screens and the controlled unprotected-probe stop. Large latent tensors and output PNGs remain
in the local `diagnostics/seedvr2-torch214` directory. The runner contains this
machine's install/model/input paths; adjust those for another machine.

Use the 2.14 environment's Python with `guard.py LABEL python run.py --label
LABEL --mode stock|native --fix --input-size 2400 --repeats 3 --limit-gib 80`.
Normal stock comparisons retain the scheduling safeguard; the explicit diagnostic unprotected probe is restricted to a maximum 320-pixel input and 24 GiB sampled stop threshold. `SEED_RESIDUAL=0` restores the
preceding optimized residual implementation for reproduction;
`--mode native-base` disables the earlier SeedVR2 fusions as well. Diagnostic
candidate variables are confined to this harness and are not installed flags.

Earlier 2.13 results: [memory fix](m5-ultra-seedvr2-memory.md),
[exact kernels and large-image repair](m5-ultra-seedvr2-exact.md),
[FP16 compatibility](m5-ultra-seedvr2-fp16.md).
