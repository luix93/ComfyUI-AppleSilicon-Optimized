# SeedVR2 kernel coverage and exact optimization screening

**Historical screening report.** The candidates below were subsequently
implemented and tested at full resolution. See the
[exact optimization report](m5-ultra-seedvr2-exact.md) for current results
and installed defaults. The original screening measurements remain below.

2026-09-24, same M5 Ultra stack and workflow as the
[memory report](m5-ultra-seedvr2-memory.md). The installed memory safeguard
remains default-on. No additional production optimization was installed here.

## Which acceleration actually executes?

A traced 320×213 input, 3× resize, 960×638 final image ran the installed node
with `ASFP8_NATIVE=1`. Each one-step sampler executed:

- 288 native `i8_matmul2d_nt_fused_cs` calls: INT8 products with fused
  per-channel rescaling and BF16 output. Shapes include 3072→9216 QKV,
  3072→3072 output projections and 3072↔12288 MLP projections.
- 288 native rowwise INT8 quantization calls, with ConvRot activation rotation.
- 36 variable-length attention calls containing 324 individual attention
  windows. All chose the router's **default** attention path; no Flash41
  attention kernel ran in this sampler trace.
- VAE encoding used our Conv3D decomposition 162 times; decoding used it 228
  times. The convolution arithmetic runs in PyTorch/MPS Conv2D, not a custom
  Metal convolution kernel written by this project.

SeedVR2-specific transformer fusions have not been implemented. MiniMax, Wan,
Flux and Z-Image model-class fusions do not automatically apply. This 7B model's
MLP uses tanh-approximate GELU, so the MiniMax SwiGLU bridge is not applicable.
VAE attention also follows its existing ComfyUI implementation.

The trace's encoded latents, sampled latents and decoded FP32 frames matched the
previous installed-default reference hashes. Synchronized function timings
were collected only to identify costs: inserting waits distorts throughput,
so the uninstrumented measurements below determine performance.

## Small actual-workflow ablation

The memory safeguard and remaining patches stayed enabled. The disabled
comparison sets **only** `ASFP8_INT8_EXT=0` and `ASFP8_CONV3D=0`.
Each process includes a first run, excluded from the warm results below.

| Stage | INT8 + Conv3D acceleration disabled | Current acceleration |
|---|---:|---:|
| Encode | 1.510 s | 0.271 s |
| Sample | 0.864 s | 0.336 s |
| Decode | 3.452 s | 0.580 s |
| Complete sequence including PNG save | 5.864 s | 1.227 s |

These are medians of two disabled warm runs and three current warm runs.
They demonstrate an approximately 4.8× benefit at this **small** resolution;
they are not a measurement of that ratio at 7200×4800. Disabled-path output
bytes differ. The existing W8A8 activation-quantized path and convolution
decomposition are not an assertion of equivalence to unpatched ComfyUI.
New exact candidates must preserve the current accelerated reference.

## First exact candidate: convolution memory copies

A diagnostic-only implementation reuses packed convolution weights while the
inference weights are fixed, and returns the sole convolution chunk directly
instead of copying it through a one-element `cat`. It keeps the same Conv2D
operands, channel order, arithmetic and tile boundaries.

Eight full small-workflow runs alternated baseline and candidate. After the
initial baseline and candidate warmups, three runs of each produced:

| Stage | Current median | Candidate median | Reduction |
|---|---:|---:|---:|
| Encode | 0.2712 s | 0.2461 s | 9.3% |
| Decode | 0.5798 s | 0.5332 s | 8.0% |
| Complete sequence | 1.2273 s | 1.1607 s | 5.4% |

All eight runs have identical encoded latent, sampled latent and decoded frame
hashes. Raw tensors are retained by the diagnostic harness. This is a promising
screen, **not an accepted production change**: the cache needs model-lifetime
cleanup and invalidation for mutations, patches and movement, and the complete
large workflow still needs repeatability and byte-equivalence checks. The
prototype deliberately assumes immutable weights and remains uninstalled.

## Next candidates, preserving the accelerated reference

1. Finish the convolution packing/copy candidate above and measure both full
   VAE stages. In the prior large run they took 52.21 of 73.41 seconds (71%).
2. Reduce VAE normalization/activation/layout passes. Preserve the existing
   GroupNorm reduction algorithm and intermediate FP16 rounding. A custom
   reduction or different convolution algorithm is not presumed exact.
3. Reduce SeedVR2 sampling's window-dispatch overhead. The installed window
   geometry at the full 1×300×450 patch grid produces 324 regular and 361
   shifted windows, or 12,330 attention calls over 36 alternating layers.
   The largest contains just 483 tokens including the 58 conditioning tokens.
   Bucket equal-length windows and investigate dispatch batching while retaining
   the existing attention arithmetic. Batching can select different backend
   arithmetic, so attention tensors and final latents must be compared; it is
   an experiment, not an established exact optimization. See PyTorch's
   [numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
4. Add SeedVR2-specific RMS normalization/modulation, RoPE/layout and gated
   residual fusions. Cache identical RoPE sine/cosine results within a forward
   and remove redundant copies first. Preserve every BF16/FP32 cast and each
   multiplication/addition boundary; retain the INT8 accumulator and existing
   attention algorithm. GELU-to-quantization fusion is another candidate only
   if the existing tanh approximation and intermediate rounding are reproduced.

Changing tile sizes/overlap, reducing precision, replacing attention with Flash
or changing normalization reductions does not meet the exact requirement by
assumption. No full-resolution speedup is claimed for the new candidates.

Machine-readable evidence: [acceleration data](data/m5-ultra-seedvr2-acceleration.json).
Scripts and raw results: workspace `diagnostics/seedvr2-3x/trace.py`,
`screen_packing.py`, `kernel-census*`, `kernels-disabled*`, `packing-screen*`.
All diagnostic processes completed; no installed runtime files were changed.
