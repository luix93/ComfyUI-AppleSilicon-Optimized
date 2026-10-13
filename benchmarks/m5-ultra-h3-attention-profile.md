# H3 exact attention: GPU profiling and tiled K/V — October 13, 2026

Profiling identifies the QK/PV regions, including their loads, as the leading
costs. It does **not** establish register spilling as the bottleneck. A new
physical K/V layout preserves arithmetic and improves the captured attention
operation by about 3.5%, including packing. **The sampling screen did not
establish a repeatable practical gain, so the layout change remains
experimental and production is unchanged.**

**Follow-up:** [Hardware-counter profiling](m5-ultra-h3-gpu-counters.md) now
measures registers, occupancy, matrix utilization and memory pressure using
macOS 27’s bundled GPU debugger. Full Xcode was not required. The tooling
limitation described below applies only to this earlier timestamp pass.

## What was measured

M5 Ultra, 80 GPU cores, 256 GB, macOS 27, PyTorch 2.14.1. The reference is the
current exact optimized long-sequence query-cache schedule, **not stock**.
The public authored H3 864×480 / 362-frame capture contains contiguous BF16
Q/K/V `[1,56,44600,128]`. No private media or SATA files were accessed.

A native Objective-C++ runner compiles the same flattened shader using Metal
safe math, holds inputs/output in preallocated buffers, and records both GPU
command-buffer elapsed time and compute-pass timestamp-counter elapsed time.
The two clocks agree. Initial compile/warm-up runs are excluded; eight rounds
shuffle reference and probe order. The original native output, the all-enabled
probe output and the preceding PyTorch output have the same SHA-256:
`3dad4faf1ac08c28056e323b0f568bdecce315a281464b07cf59c998c769f4d3`.

### Controlled phase bypasses

One compiled diagnostic pipeline accepts a runtime mask for QK, softmax and PV.
This holds the compiled pipeline/register allocation constant between its
masks, though dynamic liveness and memory behavior can still differ. Disabled
regions use small data-dependent substitutes to retain downstream work.
Bypassed outputs are **not valid attention** and were never used in sampling.

| Native GPU measurement | Median | Difference from full probe |
|---|---:|---:|
| Original exact kernel | 537.11 ms | — |
| All probe phases enabled | 580.33 ms | — |
| QK region bypassed | 323.92 ms | −256.41 ms |
| PV region bypassed | 294.91 ms | −285.42 ms |
| Softmax/scale/mask/output-rescale region bypassed | 532.95 ms | −47.38 ms |

These are **sensitivity measurements, not additive stage percentages**. The
probe adds 8% overhead. QK and PV include their memory loads, matrix operations,
and local scheduling; the softmax bypass also removes output rescaling.
The result supports prioritizing the two matrix regions. It gives no evidence
that optimizing exponential evaluation alone could provide 1.6× attention
throughput.

The device exposes only the public timestamp counter set. Full Xcode and
Instruments are not installed. Pipeline reflection does not expose register
counts. A public Apple G13 disassembler could not reliably decode the M5
binary. Consequently no measured occupancy, register count, spill count or
memory-stall percentage is claimed. Calling this proven register pressure
would go beyond the evidence.

Two additional same-operand matrix-replay probes were implemented. Even their
one-repetition controls slowed from approximately 540 ms to 863–974 ms. This
perturbation is too large to interpret as original-kernel matrix-unit costs;
those probes are retained in the raw results but excluded from the diagnosis.

## Profiling-guided implementation

K is copied bit-for-bit into `[head,key_tile,dimension_tile,32,16]`; V into
`[head,key_tile,dimension_tile,32,32]`. Small vectorized Metal copy kernels
perform the packing. The attention shader changes only addresses/strides,
retaining the same matrix primitives, contraction/reduction order, FP32
probabilities, online softmax and output rounding. Partial final tiles are
zero padded, with the original attention bounds/masks retained.

This targets repeated strided fragment loads without creating the additional
live matrix tiles that slowed the preceding pipeline experiments. The
allocation cost and both copy dispatches are included in acceptance timings.
At 44,600 tokens the two temporary packed buffers require approximately
1.19 GiB together, released/reused after the operation; they are not a
cross-step attention cache.

| Initial screen, same process | Complete operation median |
|---|---:|
| Current exact reference | 537.44 ms |
| Pack K only | 547.27 ms |
| Pack V only | 533.43 ms |
| Pack both K and V | **521.40 ms** |
| Skip output multiplication when every lane's factor is exactly 1 | 545.62 ms |

All five outputs matched the captured reference bytes. The identity-rescale
branch and K-only packing are rejected for speed. V-only is not compelling.
K/V packing without the packing work timed at 512.82 ms in a separate batch;
that is not the practical number and is not used for the speedup claim.

A separate 24-pair randomized repeat measured **540.31 → 521.45 ms**. The median
paired throughput ratio is **1.0382×**, with a 95% percentile bootstrap interval
of **1.0332–1.0394×** (5,000 resamples). Ratio of overall medians corresponds to
3.49% less time. These intervals describe run variation, not all hardware or
thermal conditions. Further byte checks passed at 30,001, 32,768 and 44,600
tokens, including query magnitudes scaled by 0.01 and 4 at 44,600 tokens.

## Workflow validation

An isolated ComfyUI server ran the public 864×480, 362-frame workflow with
one warm-up evaluation, then two-step reference / packed / packed / reference
runs. All other optimizations, prompt, conditioning and initial noise stayed
fixed; approximate modes were disabled. Each measured run made 100 eligible
attention calls. Sampling times include layout copies, allocations and the
rest of the model.

| Run order | Sampling | Sample-only end-to-end |
|---|---:|---:|
| Reference | 75.115 s | 75.145 s |
| Packed | 75.975 s | 75.977 s |
| Packed | 75.063 s | 75.064 s |
| Reference | 77.050 s | 77.051 s |

Reference median is 76.082 s; packed median is 75.519 s. The apparent 0.74%
saving is smaller than the run variation and changes sign within the two
pairs. **This does not pass the repeatable practical-speed acceptance gate.**
All four records have identical conditioning, noise, video-latent and
audio-latent hashes, including both sampler output branches. These were
sample-only screens: no new decoded-frame comparison or full 20-step timing
is claimed. A full long clip was not run after this inconclusive speed gate.

To investigate the operator/workflow discrepancy, a separate live-model
probe timed original and packed attention on identical inputs inside one
sampling evaluation, alternating pair order and always returning the
original output. After excluding its first compilation pair, 24 pairs at
`[1,56,44691,128]` measured **538.735 → 520.180 ms** medians, with median paired
throughput **1.0357×**. The live workflow has 91 more tokens than the earlier
saved capture; every comparison itself uses matching shapes and inputs.
The operator gain therefore survives model residency. This intrusive probe
synchronizes calls and changes normal overlap/cache behavior, so its wall
time is not a clip benchmark. Why the gain weakens in normal sampling is
not yet established; allocator/cache interactions and normal timing variance
remain possibilities, not proven causes.

The guarded sampling screen retained at least 77.4 GiB available system
memory; MPS driver allocation reported 46.44 GiB. No memory guard fired.
The isolated servers terminated normally after collecting their results;
the main ComfyUI server and installed production kernel were not changed.

## Decision and next useful measurement

Keep the packing implementation and reproduction tools as research artifacts;
do not enable a new production path based on the isolated gain. The earlier
exact long-sequence schedule remains the shipping implementation. The
requested approximately 1.6× attention throughput has not been achieved.

The subsequent [hardware-counter investigation](m5-ultra-h3-gpu-counters.md)
uses macOS 27’s bundled `gpudebug`, without full Xcode, to measure registers,
occupancy and matrix/cache pressure on these isolated kernels. The original
full-Xcode prerequisite was incorrect. Capturing the kernels **inside loaded
H3** remains a separate step for investigating surrounding-work interactions;
it is not established by the isolated hardware profiles.
A fused producer that writes K/V directly into the packed layout could avoid
the extra copies, but has not been implemented or validated here; the small
observed operator gain is not evidence that this could deliver 25% overall.

## Reproduction and sources

[Offline profiling and candidate tools](experiments/h3-attention-profile/README.md)
and [machine-readable results](data/m5-ultra-h3-attention-profile.json).
Native timestamp units are calibrated against command-buffer GPU times:
resolved legacy-counter deltas on this device are nanoseconds. An initial
incorrect multiplication by the CPU mach timebase was corrected; published
measurements use the independently valid GPU command-buffer clock.

Apple documents [GPU counter availability and sampling](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
and [GPU analysis in Xcode](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app).
Its [Metal 4 presentation](https://developer.apple.com/videos/play/wwdc2025/262/)
also describes the relevance of tensor memory layout. The implementation here
uses ordinary packed buffers and the existing PyTorch/MPP arithmetic.
The unsuccessful disassembly check used [applegpu](https://github.com/dougallj/applegpu),
which targets the older Apple G13 ISA; none of that code is shipped in the node.
