# H3 exact-attention pipeline experiments — October 13, 2026

**Result: no additional production optimization accepted.** The best candidate
in a separate paired repeat reached only 1.005× median attention throughput;
its confidence interval includes no improvement. This does not establish a
path to the requested 1.6× attention throughput or approximately 200 seconds
off the 15-second workflow. The previously accepted
[long-sequence schedule](m5-ultra-h3-long-exact.md) remains unchanged.

## Reference and measurement boundaries

M5 Ultra, 80 GPU cores, 256 GB unified memory, macOS 27, validated PyTorch
2.14.1. The reference is the **current optimized exact** BF16 query-cache
kernel: BQ64/BK32/BD128, head-major dispatch, middle PV barrier at index 6.
It is not stock attention or the preceding, slower query-cache schedule.

The test reuses Q/K/V captured from the authored public 864×480, 362-frame H3
T2V fixture: three contiguous BF16 tensors `[1,56,44600,128]`. It does not open
private media, workflows, or SATA storage. Every successful candidate first
compares its complete output bytes against the reference. Then seven warmed,
shuffled timing rounds include output allocation, parameter lookup and Python
kernel dispatch, with GPU synchronization around each sample. Compile/warm-up
and CPU comparison are excluded from timing. Nonmatching candidates are not
timed. All runs are serial and guarded against user queue activity and memory
pressure.

There are 45 distinct candidate variants, including two deliberately incorrect
fragment layouts and two unsupported cooperative-tensor shapes. Initial API
integration failures are retained separately from their corrected tests.
Successful byte comparison on this captured operation **is not a claim of
complete-sampler identity or universal correctness**. No candidate passed the
speed gate, so no new full workflow was run and no clip-level saving is claimed.

## Results

Compare each row with its same-process control, not another screen's control.
Numbers are milliseconds per full attention call at 44,600 tokens.

| Family / representative candidate | Reference | Candidate | Decision |
|---|---:|---:|---|
| Persistent QK accumulator | 535.10 | 532.77 | Apparent sub-percent gain; not confirmed |
| Persistent PV accumulator within a key tile | 535.10 | 539.85 | Reject |
| Persistent output tensors across all key tiles | 536.26 | 603.52 | Reject |
| Direct tensor-input QK, static extents | 538.52 | 642.75 | Reject |
| Vectorized fragment loads | 537.40 | 536.00 | No meaningful gain on repeat |
| Earlier V loads, smallest variant | 538.52 | 670.16 | Reject; larger variants 959–1,081 ms |
| Two independent QK tiles computed ahead | 535.83 | 929.30 | Reject; four tiles 1,363.96 ms |
| Remove middle threadgroup synchronization | 535.83 | 1,064.67 | Reject |
| Native 32-query-row operations | 543.42 | 2,347.15 | Reject |
| 32-query rows, split output, no query cache | 540.38 | 992.28 | Reject despite reduced storage |
| Partial QK unrolling by four, query cache kept | 534.33 | 922.97 | Reject |
| Partial QK unrolling by four, query cache removed | 534.33 | 644.43 | Reject |
| Promote BF16 V to FP32 in registers | 537.40 | 1,259.51 | Exact on fixture, much slower |
| Rescale output fragments immediately before PV | 537.40 | 536.23 | No meaningful gain on repeat |

Other screens covered explicit contiguous strides, separate aligned/tail
loops, outer-loop unrolling, three/five/six SIMD groups, larger compiler thread
limits, explicit uniform scale, and synchronization every 2/4/8/16 key tiles.
None qualified. Both 64- and 128-wide register-only QK contractions are rejected
by the installed SDK: when both operands are cooperative tensors, K must be
16 or 32. The direct tensor-input route avoids that restriction and was
implemented successfully, but was slower here.

### Independent repeat of the closest candidates

Twenty randomized baseline/candidate pairs per candidate; all output bytes
matched again. Intervals are 95% percentile bootstrap intervals of the median
paired throughput ratio, using 5,000 resamples. These describe this run's
variation, not a guarantee across hardware, thermal states or workloads.

| Candidate | Reference median | Candidate median | Median paired throughput | Interval |
|---|---:|---:|---:|---:|
| Persistent QK | 537.48 ms | 536.43 ms | 1.0053× | 0.9996–1.0077× |
| Vector loads | 535.29 ms | 536.08 ms | 1.0005× | 0.9938–1.0020× |
| Later output rescaling | 536.11 ms | 535.74 ms | 1.0012× | 0.9963–1.0070× |
| Synchronization every eight tiles | 537.41 ms | 539.87 ms | 0.9938× | 0.9923–0.9976× |

A median of pairwise ratios is not necessarily the ratio of the two timing
medians. None supports a practical improvement. Replacing working production
code for these results would add risk without a demonstrated benefit.

## Research findings and implications

Apple recommends balancing tile size, register storage and synchronization;
larger tiles can lose when their live operands exceed fast local storage.
The measured regressions are consistent with that warning, but we did not
collect compiler register counts or GPU occupancy counters and therefore do
not assert a proven spill mechanism. We tested direct tensor operands,
cooperative intermediates and barrier schedules described in the
[MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
and [Apple's Metal tensor session](https://developer.apple.com/videos/play/wwdc2026/330/).

Current MLX attention uses a partially unrolled QK contraction and an explicit
uniform scale. We reproduced those scheduling ideas while retaining our
reference's arithmetic. They did not transfer as wins. Its
[NAX matrix helper](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/nax.h)
still uses MPP cooperative-tensor operations; its name alone does not identify
an untapped faster matrix API. See the
[upstream attention implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h).

[FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf) motivates
overlapping matrix operations and data movement, but its Hopper-specific
asynchronous mechanisms and numerical-error claims do not establish
byte-identity to this PyTorch/Metal kernel. Our limited lookahead experiment
preserved bytes but lost performance. That does not disprove every possible
Apple pipeline design; it rules out the tested implementation.

At the measured roughly 537 ms baseline, 1.6× throughput would require about
336 ms per call. None of these exact candidates is close. Larger key tiles,
altered softmax reductions, split-key accumulation or lower-precision
probabilities cannot be assumed byte-identical merely because they implement
the same mathematical dense attention. They were not substituted for the
user's exact requirement.

A further serious attempt should obtain per-kernel register/occupancy and
instruction-stall evidence before another pipeline rewrite, and isolate
QK versus FP32-probability PV versus softmax costs. Those are remaining research
questions, not implemented gains or evidence that the target is impossible.

## Artifacts and safeguards

- [Measurements, source hashes, failed attempts and raw paired repeats](data/m5-ultra-h3-exact-attention-pipeline.json)
- [Isolated experimental kernel builders and guarded reproduction harness](experiments/h3-exact-attention/README.md)
- Previous accepted full 10-second result remains **399.21 s** versus **412.03 s**
  for the preceding exact schedule, with complete decoded-output byte checks.
  That is a prior measurement, not a result of this pass.

No candidate was installed, no default changed, and no approximate mode was
enabled. Memory guards did not fire. Process RSS measurements are not GPU peak
memory; MPS allocation watermarks were also active. Large captured tensors are
not included in the repository.
