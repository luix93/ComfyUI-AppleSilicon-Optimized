# H3 exact attention: hardware counters — October 13, 2026

**The current exact kernel is primarily constrained by matrix execution, not
measured register-pressure throttling.** Its matrix accelerator is approximately
81% utilized. General compute occupancy is only 17%, but those metrics measure
different resources. A deliberately larger tile demonstrates the opposite
case: cache/memory pressure rises sharply and matrix utilization collapses.
No new production optimization passed acceptance in this investigation.

## Tools and scope

The earlier statement that full Xcode was required was too restrictive.
This macOS 27 installation already provides `gpucapture`, `gpudebug` and
`metalperftrace`. Native Metal capture and `gpudebug` hardware profiling work
with the installed Command Line Tools. No full Xcode download, license
acceptance, SDK selection change or PyTorch change was needed. Homebrew `mas`
7.0.0 was installed while investigating the App Store route; it is not a
profiling dependency.

Hardware: M5 Ultra, 80 GPU cores, 256 GB, macOS 27.0 (26A428), PyTorch 2.14.1.
The input is the same authored public BF16 Q/K/V `[1,56,44600,128]` fixture
used in the preceding experiments. This is the current exact optimized
long-sequence schedule, **not stock PyTorch**. No private clips, confidential
workflows or SATA files were accessed.

A native runner warms the original flattened shader three times, captures one
Metal dispatch, then writes its output for byte verification. Apple's profiler
replays that captured dispatch. These measurements isolate attention; they do
not include model residency, allocation, K/V packing or the rest of sampling.
They are not new end-to-end clip timings.

## Original kernel: register pressure, occupancy and stalls

| Counter/statistic | High-state profile | Medium-state repeat |
|---|---:|---:|
| Replay time | 530.206 ms | 780.646 ms |
| Matrix accelerator utilization | 80.88% | 81.87% |
| Matrix accelerator limiter | 82.44% | 83.34% |
| Compute occupancy | 17.09% | 17.04% |
| Occupancy target | 18.32% | 18.68% |
| Register-pressure influence on occupancy | 0% | 0% |
| L1-cache-pressure influence on occupancy | 0% | 0% |
| Memory-request-stall influence on occupancy | 0.82% | 1.82% |
| GPU bandwidth | 760.35 GiB/s | 425.59 GiB/s |
| Compiler temporary registers | 126 | 126 |
| Compiler spilled bytes | 352 | 352 |

Both profiles report **Consistent state: yes**. Counter averages describe the
whole dispatch; pressure influences are not percentages of execution time.
Register and L1 pressure influences remain zero throughout these baseline
traces. Memory-stall influence is episodic rather than the steady dominant
signal. Nonzero compiler spills therefore do not, by themselves, establish a
spill bottleneck. Zero pressure influence also does not mean register traffic
has zero cost.

The profiler reports **one sampled core out of 80** for these shader profiles.
This is not an exhaustive survey of die/core imbalance. The GPU peak-bandwidth
counter returned zero despite nonzero measured traffic; no percentage of peak
DRAM bandwidth is inferred from it. L1 residency categories are not added as
though they were disjoint.

Together with the earlier QK/PV phase-bypass measurements, the evidence points
to the matrix regions as the main limit. The counters do not separate QK from
PV within this shader or attribute every stall to a source line. We have not
established which individual QK/PV instruction is the remaining bottleneck.

## Controls and optimization experiments

### Removing persistent query fragments

The no-query-cache control retains attention arithmetic and matches the
reference bytes. It is slower: **605.848 ms** in a consistent high-state
profile, versus 530.206 ms. Ordinary warmed native runs were approximately
612 ms. Matrix utilization falls to **70.67%**. Register allocation stays at
**126**, while spilled bytes increase to **368**. Register/L1 pressure
influences remain zero.

Thus reducing apparent source-level lifetime did not reduce compiler register
allocation or improve throughput. The query cache is doing useful work.

### Packed K/V

Packed K/V keeps **126 registers and 352 spilled bytes**. A consistent medium
profile measures **760.015 ms**, versus 780.646 ms for the original, with
matrix utilization increasing to **84.10%**. That is a small kernel-only gain,
consistent with the preceding layout experiment. The high-state packed replay
reported **Consistent state: no** and is excluded from speed comparisons.

The earlier complete-operation measurement, including allocations and packing,
was 540.31 → 521.45 ms. Its complete-sampler screen was inconclusive, so packing
still remains experimental. See the [preceding report](m5-ultra-h3-attention-profile.md).

### Larger matrix contractions

The matrix-utilization finding motivated tests with larger cooperative QK and
PV contractions, retaining FP32 accumulation. All matched every output byte on
the 44,600-token fixture. Ten shuffled timing rounds included dispatch and
output allocation in each operation:

| Candidate | Median operation time | Decision |
|---|---:|---|
| Current exact reference | 533.662 ms | Keep |
| QK contraction 32 | 531.615 ms | Approximately 0.4%; no meaningful win |
| PV contraction 32 | 535.857 ms | Reject |
| Both | 537.430 ms | Reject |

These are fixture-level checks, not proof of exactness at every input shape.
None justified the next acceptance stage of complete-sampler validation.

### Why the larger query tile regressed

The byte-identical `m32row2` control measures approximately 2.32–2.36 seconds
in ordinary native execution. Its consistent medium-state replay takes
**3162.250 ms**, versus 780.646 ms for the reference. Timeline counters show:

| Counter | Original medium | 32-row control medium |
|---|---:|---:|
| Matrix accelerator utilization | 81.87% | 20.16% |
| L1-pressure influence on occupancy | 0% | 49.66% |
| Memory-stall influence on occupancy | 1.82% | 18.51% |
| GPU bandwidth | 425.59 GiB/s | 1292.52 GiB/s |

This control exposes substantial cache/memory pressure, unlike the original.
It supports avoiding this tile expansion. However, its profile sampled 80/80
cores and did not provide per-shader register/spill statistics. The exporter
correctly reported partial failure rather than treating missing statistics as
zero. Different sampling scope limits precise cross-profile ratios of counters;
its severe ordinary execution slowdown is independently observed. We cannot
assign the regression to a specific spill count.

An initial mistaken selection of the different `m32col2` variant failed the
byte check and was excluded before hardware profiling. Only the corrected,
byte-identical `m32row2` control informs the comparison above.

## Exactness, decision and next direction

Original, packed K/V, no-query-cache and corrected 32-row captured outputs all
have SHA-256:
`3dad4faf1ac08c28056e323b0f568bdecce315a281464b07cf59c998c769f4d3`.
The contraction experiments report zero mismatched bytes/elements and zero
maximum absolute error against that original. The production kernel remains
unchanged from the start of this profiling task.

**There is no measured basis for expecting 1.6× from occupancy tuning alone.**
If the reported 81% matrix utilization were the relevant attainable ceiling,
perfect utilization would imply only about 1.24× in that simplified model.
That is an inference, not a universal speed limit: a different exact matrix
implementation could change execution efficiency. It is also not a promise of
1.24× attention or whole-clip acceleration.

The next useful candidate must reduce the cost of the same exact matrix work
without reproducing the larger tile's cache traffic. Producer-side K/V packing
could remove copies from the existing small layout win; it has not been tested
and does not presently support a 25% clip-saving claim. Further source-level
lifetime tweaks should be judged by the resulting hardware counters, not by
register-count assumptions. In-model capture would be needed to establish why
the small layout gain weakens during normal sampling.

System-available-memory and main-ComfyUI-queue guards were used. No production
server was restarted and no new long clip was rendered. Profiler sessions were
closed after collection. The guard's process RSS excludes some debugger helper
processes; it must not be reported as total profiling memory usage.

[Machine-readable measurements](data/m5-ultra-h3-gpu-counters.json) ·
[Capture and profiling reproduction](experiments/h3-gpu-profile/README.md).

## Primary references

Apple's [Metal tools](https://developer.apple.com/metal/tools/) and
[interactive command-line debugging documentation](https://developer.apple.com/documentation/xcode/debugging-with-interactive-command-line-tools)
describe the capture/debugging tooling. Apple's
[M5 GPU architecture talk](https://developer.apple.com/videos/play/tech-talks/111431/)
explains occupancy targets, register/L1 pressure influences and matrix-unit
counters. [GPU occupancy documentation](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy)
provides the resource-usage context. Apple's
[profiling discussion](https://developer.apple.com/videos/play/wwdc2021/10157/?time=250)
explains consistent GPU performance-state checks. Numerical results above are
local measurements, not figures taken from those sources.
