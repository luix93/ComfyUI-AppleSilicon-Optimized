# H3 exact QK/PV execution experiments — October 13, 2026

**No new matrix path improved on the best preceding exact attention experiment.**
The existing packed K/V experiment again saves about 4% at the complete-operator
level. Adding a larger QK contraction ties that result, rather than adding to it.
The requested 1.6× attention throughput remains unachieved. Production is unchanged.

## Research and test design

The preceding [hardware profile](m5-ultra-h3-gpu-counters.md) showed approximately
81% matrix-accelerator utilization, without register/L1 occupancy-pressure
throttling. This motivated changing operand delivery and matrix-operation
shapes rather than chasing occupancy alone.

Apple's [TensorOps presentation](https://developer.apple.com/videos/play/wwdc2026/330/)
describes combining cooperative operands with tensor-backed operands and
passing intermediates between matrix operations. Its
[MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
discusses static tensor extents, tile size and cache-working-set tradeoffs.
The implementations below are local experiments, not copied sample kernels.
Existing PyTorch/Apple dependency notices apply; SDK headers are not bundled.

M5 Ultra / 80 GPU cores / 256 GB; macOS 27.0 (26A428); PyTorch 2.14.1.
Reference: the current exact optimized long-sequence query-cache kernel,
**not stock PyTorch**. Input: saved public-authored contiguous BF16 Q/K/V
`[1,56,44600,128]`. No private clips, confidential workflows or SATA files were
accessed. Approximate acceleration modes were not used.

Thirty distinct variants and controls were screened across seven batches.
Four unsupported register shapes and two invalid view prototypes were excluded
from timing; the remaining 24 matched the captured output bytes.

Each batch initializes/compiles and checks candidate output bytes before
randomizing 9–11 timing rounds. Timings include the Python wrapper, output
allocation and GPU synchronization. Packed variants additionally include every
packing allocation and copy dispatch. Compare a candidate with its own batch's
baseline; absolute baselines vary slightly between batches. Compilation and
initial warm-up are excluded.

## Results

### Mixed register/device operands

Q stays in cooperative registers while TensorOps loads K directly; separately,
P stays in registers while TensorOps loads V. Original tail handling remains.
All eight variants matched the reference output bytes.

| Candidate | Median | Same-batch reference |
|---|---:|---:|
| Mixed QK, contraction 16 | 575.557 ms | 533.989 ms |
| Mixed QK, contraction 32 | 575.112 ms | 533.989 ms |
| Mixed PV, N32/K16 | 603.715 ms | 533.989 ms |
| Mixed PV, N32/K32 | 586.374 ms | 533.989 ms |
| Mixed PV, N64/K16 | 641.668 ms | 533.989 ms |
| Mixed PV, N64/K32 | 564.637 ms | 533.989 ms |
| Mixed PV, N128/K16 | 600.263 ms | 533.989 ms |
| Mixed QK + PV | 658.583 ms | 533.989 ms |

Fewer explicit fragment loads or logical matrix calls did not produce faster
execution. All are rejected for speed.

### Register-only shapes and scheduling

The installed MPP compiler requires M, N and K to be 16 or 32 when **both**
operands are cooperative tensors, with at least one dimension equal to 32.
Four wider N64/N128 register-only variants fail those explicit SDK assertions;
they are unsupported, not numerical failures. Wider mixed-input variants above
are supported, but slower.

| Candidate | Median | Same-batch reference |
|---|---:|---:|
| Existing N32/K16 expressed directly as cooperative tensors | 549.779 ms | 550.385 ms |
| PV key-chunk loop outside output-column loop | 606.922 ms | 550.385 ms |
| Same ordering, additional synchronization | 611.495 ms | 550.385 ms |
| Register PV, N16/K32 | 564.838 ms | 537.730 ms |
| Register QK, N16/K32 | 565.336 ms | 537.730 ms |

All supported rows matched the reference bytes. The direct N32/K16 expression
is an equivalence control, with no meaningful gain. Independent-column
reordering keeps each accumulator's arithmetic order but performs worse.

### Lossless operand widening

BF16 values can be represented exactly in FP32. Testing FP32 cooperative Q
and/or K operands therefore explores another hardware path without reducing
input precision. Captured outputs matched, but performance was much worse:

| Candidate | Median | Same-batch reference |
|---|---:|---:|
| Widen Q | 1225.112 ms | 538.462 ms |
| Widen K | 919.206 ms | 538.462 ms |
| Widen both | 1624.533 ms | 538.462 ms |

### Packing combined with the new matrix shapes

| Candidate | Median | Same-batch reference |
|---|---:|---:|
| Previous packed K/V, original matrix path | **511.901 ms** | 534.901 ms |
| Packed K/V + mixed PV N32/K32 | 577.908 ms | 534.901 ms |
| Packed K/V + mixed PV N64/K32 | 621.163 ms | 534.901 ms |
| Packed K/V + register PV N16/K32 | 517.318 ms | 534.901 ms |

An independent combination batch gives:

| Candidate | Median |
|---|---:|
| Current production reference | 534.806 ms |
| Previous packed K/V experiment | **513.615 ms** |
| Packed K/V + register PV N16/K32 | 519.659 ms |
| Packed K/V + QK K32 + PV N16/K32 | 518.818 ms |
| Packed K/V + QK K32, original PV | **513.610 ms** |

For the last row versus previous packing, median paired throughput is 1.00084×;
a 5,000-resample bootstrap interval is **0.99511–1.00347×** over 11 rounds.
There is no established incremental gain. This repeats the *operator* packing
benefit; it does not overturn the earlier inconclusive complete-sampler result.
No packed path is promoted to production here.

### Transposed PV: invalid views versus valid physical layouts

Two non-unit-stride tensor-view prototypes failed the output-byte gate severely
(maximum absolute difference 18.0625; 319,568,160 mismatched elements out of
319,692,800). These are invalid implementation prototypes, **not** acceptable
numerical approximations or evidence that all transposed PV operations must
differ. A coordinate audit found identical left/destination cooperative slot
coordinates in the tested N32 case; the precise cause of the strided-view failure
is not established.

A follow-up physically transposes each V tile into contiguous key-major storage,
with zero-padded tail rows. Both N32 and N64 variants then match every output
byte. Including allocation and transpose dispatch, they take **607.405 ms** and
**572.346 ms**, respectively, versus **534.170 ms** for their same-batch reference.
The valid transposed implementation is therefore also rejected for speed.
An initial source-generation scope error was fixed before these measurements;
it is distinct from the SDK's genuine cooperative-shape restrictions.

## Hardware profiles guiding the decision

Native captures of the mixed PV N64/K32 and reordered PV kernels both match
the original output SHA-256. Profiles report consistent high/max GPU state.
The original column below is from the immediately preceding investigation on
the same stack; same-batch ordinary timings above independently establish the
regressions.

| Hardware measurement | Original | Mixed PV N64/K32 | Reordered PV |
|---|---:|---:|---:|
| Replay time | 530.206 ms | 561.132 ms | 586.303 ms |
| Matrix accelerator utilization | 80.88% | 76.39% | 73.12% |
| Compute occupancy | 17.09% | 21.17% | 18.55% |
| Temporary registers | 126 | 126 | 126 |
| Spilled bytes | 352 | 368 | 384 |
| Static instruction count | 1778 | 1857 | 1791 |
| Register-pressure influence | 0% | 0% | 0% |
| L1-pressure influence | 0% | 0% | 0% |
| Memory-stall influence | 0.82% | 0.95% | 0.71% |

Higher occupancy did not improve throughput. More spills exist, but the pressure
counters do not establish spill-induced throttling. The regressions are
consistent with added operand-handling/scheduling overhead reducing matrix-unit
utilization, rather than alleviating a memory-pressure bottleneck. Static
instruction counts are not dynamic operation counts and cannot assign all of
the slowdown to particular instructions.

As a rough consistency check, replay duration multiplied by matrix utilization
is approximately 429 ms in all three profiles. This is **not** an additive
QK/PV stage timing or a proven hardware lower bound. It indicates these routes
have not reduced the matrix-work component visible to the profiler.

The debugger sampled one core for the original/reordered profiles and two for
the mixed profile, out of 80. These are not exhaustive core-imbalance surveys.
No new in-model hardware capture was performed. The profiling export helper
was corrected to use relative output filenames: `gpudebug` v1.0 treated quotes
in its `fetch --out` command literally. Both new counter exports succeeded
after that correction.

## Exactness and acceptance

All candidates that enter timing must first match every final attention output
byte. The reference SHA-256 is:
`3dad4faf1ac08c28056e323b0f568bdecce315a281464b07cf59c998c769f4d3`.
This fixture-level gate is not proof for arbitrary inputs or every intermediate
FP32 value. A genuine speed winner would next need intermediate checks, shape
and input variation, complete-sampler latent equality and practical sampling
validation. No new candidate earned that next stage, so no new sampler or long
clip was run and no decoded-frame result is claimed.

The original FP32 probabilities, softmax recurrence, reference precision flags
and production code were retained. Changing the reference's established MPP
`relaxed_precision` descriptor flag was not part of these experiments.
That existing flag should not be confused with enabling a new approximate mode.

There is still **no demonstrated route to 1.6×** from these changes. The results
narrow the remaining work to reducing matrix execution itself without increasing
working-set traffic or changing reduction/rounding behavior. Producer-side
packing remains a possible way to remove the small packing overhead, not an
evidenced path to a 25% whole-clip saving. The existing shipping kernel remains
the best practically validated choice.

[All measurements](data/m5-ultra-h3-matrix-execution.json) ·
[Reproduction tools](experiments/h3-matrix-next/README.md).
