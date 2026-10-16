# H3 exact QKV / FC1 fusion follow-up — October 14, 2026

No new fusion is enabled by this experiment. The current exact default and the
optional INT8-QK / floating-PV attention recipes remain unchanged.

## Configuration

M5 Ultra (80 GPU cores, 256 GB), macOS 27, ComfyUI v0.40.0, PyTorch 2.14.1.
Same authored study-room camera-push scene and seed 812764035 as the
[preceding attention comparison](m5-ultra-h3-floatpv.md): 1024×512,
124 frames at 24 fps, CFG 1, `res_multistep` / `simple`, no LoRA,
cache or token reduction. Full runs use 20 steps; screens use two steps
at the same resolution and sequence length so matrix shapes remain representative.
H3 INT8 ConvRot weights, BF16 transformer activations, FP16 video VAE,
FP32 audio VAE. This tests exactness against the existing optimized INT8 route,
not against an unquantized checkpoint.

Actual captured projection shapes:

- QKV: `[19508,5376] × [21504,5376]ᵀ`.
- FC1: `[19508,5376] × [28672,5376]ᵀ`, then SwiGLU, ConvRot and
  row quantization to `[19508,14336]` for FC2.
- 50 eligible QKV and FC1 calls per sampler step; 1,000 of each over 20 steps.

## Existing fusions were already active

The baseline already fuses QKV projection, Q/K RMSNorm, partial RoPE and BHSD
output stores. It also uses the FC1/SwiGLU/ConvRot/row-quant bridge, reusing an
8,192-row scratch buffer. The FC1 bridge is multiple ordered kernels; it is
not a single matrix-plus-activation kernel.

A separate QKV control measured **22.92 ms fused versus 26.22 ms** with
normalization/RoPE in a second kernel, with identical bytes. This existing
saving is already included in the prior attention results; it is not a new gain.

## New implementations and operator screening

Diagnostic native extensions only; no production dispatch was changed. All
candidates preserve INT32 matrix accumulation, BF16 intermediate rounding,
SwiGLU exponential evaluation and ConvRot/row-quant arithmetic. No new
quantization, sparsity, token removal or attention approximation was added.

Each complete native operation includes output/scratch allocation and output
layout. The FC1 measurements include SwiGLU, ConvRot, scale reduction and
quantization, not just the matrix multiply. Inputs were captured from the
actual workflow. Seven interleaved timing rounds, three calls per round,
with MPS synchronization; medians below. Pipeline compilation and first-use
work are excluded. Each candidate first passed a byte comparison against the
saved operation outputs.

| QKV variant | Median ms | Compared with 22.80 ms current |
|---|---:|---:|
| Current 128-row tile, K block 256 | 22.80 | Reference |
| 32-row tile, smaller shared scratch | 25.61 | 12.3% slower |
| 64-row tile, smaller shared scratch | 22.84 | Essentially tied |
| 128-row tile, K block 512 | 24.20 | 6.1% slower |

| FC1 bridge variant | Median ms | Same-round current | Result |
|---|---:|---:|---|
| Single kernel through SwiGLU/ConvRot/group max, 16 rows | 55.47 | 33.85 | 63.9% slower |
| Same, 32 rows | 37.63 | 33.85 | 11.2% slower |
| Paired gate/up GEMMs, fused SwiGLU, separate compact rotation, 128 rows | 33.55 | 33.63 | Essentially tied |
| Shared A loads, paired register accumulators, fused SwiGLU, 128 rows | 37.14 | 33.63 | 10.4% slower |
| Same, 64 rows | 34.45 | 33.63 | 2.5% slower |
| Existing bridge, 4,096-row chunks | 34.27 | 33.63 | Slower |
| Existing bridge, 16,384-row chunks | 33.38 | 33.63 | 0.7% faster in isolation |
| Existing bridge, maximum aligned chunk (19,456 rows plus tail) | 33.14 | 33.63 | 1.5% faster in isolation |

The maximum chunk's roughly 0.49 ms saving predicts only **0.49 s across the
complete 20-step sampler**, before surrounding allocation/quantization/layout
interactions. Its wide scratch grows from 448 MiB to 1,064 MiB: **616 MiB
extra temporary storage**. This is not a plausible route to a large clip saving.

The single-kernel implementations remove global passes but constrain the
matrix tile or add accumulator/shared-memory state. The timing results show
that fewer kernels do not by themselves mean faster execution here. Register
occupancy was not profiled in this experiment, so the particular hardware
cause of each regression is not established by these timings.

## Workflow verification

The maximum-chunk bridge and the paired 128-row fusion were each checked at
all 100 affected calls of a two-step exact run: both quantized activation
bytes and FP32 row-scale bytes matched. Their sampler latents, decoded video
frames and decoded audio hashes also matched the exact baseline.

With the optional full INT8-QK / BF16-PV attention mode, the maximum chunk
also preserved all final tensor hashes relative to that mode's own baseline.
This does not make floating-PV attention exact relative to ordinary attention;
it means the FC1 change adds no further numerical change.

Uninstrumented two-step timings:

| Mode | Sampling seconds | End-to-end seconds |
|---|---:|---:|
| Exact baseline, warm | 19.044 | 30.455 |
| Exact + maximum chunk | 19.296 | 30.827 |
| Exact + paired 128-row fusion | 19.232 | 30.694 |
| Floating PV, warm repeat | 17.444 | 28.843 |
| Floating PV + maximum chunk, warm repeat | 17.405 | 28.808 |

The verification runs deliberately execute both implementations and are
excluded from performance comparisons. The first server run includes loading
and is also excluded. Final full-run results follow below.


## Full 20-step confirmation

Two warmups, then exact → maximum chunk → maximum chunk → exact.
The full workflow includes input quantization, allocations, layout conversion,
attention, video/audio decoding and video saving. Neither verification
instrumentation nor tensor export is inside the measured full runs.

| Run | Mode | Sampling s | s/step | End-to-end s |
|---|---|---:|---:|---:|
| 1 | Current exact | 191.659 | 9.583 | 203.044 |
| 2 | Maximum FC1 chunk | 190.934 | 9.547 | 202.237 |
| 3 | Maximum FC1 chunk | 191.537 | 9.577 | 202.881 |
| 4 | Current exact | 191.935 | 9.597 | 203.281 |

Mean results:

| Mode | Sampling s | s/step | End-to-end s |
|---|---:|---:|---:|
| baseline | 191.797 | 9.590 | 203.163 |
| maximum chunk | 191.235 | 9.562 | 202.559 |

The current exact baseline also matches the exact-output hashes from
the preceding attention study. All four full runs have identical conditioning, initial noise, sampler latents,
decoded frames and decoded audio hashes. Each candidate full run exercised
1,000 changed FC1 calls. This establishes byte agreement for this fixture, not
universal correctness for other shapes, LoRAs or hardware.

## Decision

The maximum chunk improves mean sampling by **0.562 s (0.29%)** and mean
end-to-end time by **0.604 s (0.30%)**. Its two end-to-end measurements differ
by 0.644 s. Both were slightly faster than the bracketing baselines, but two
repeats do not establish a robust sub-percent advantage across workloads.
That small measured gain and 616 MiB extra scratch do not justify a new
production setting. The tighter FC1 fusions and QKV variants were tied or
slower and are rejected for production.

**No default or production kernel changed.** The existing fused QKV and
8,192-row FC1 bridge remain active. Prototype implementations and a standalone
screen are retained under [dev/h3_fusion_experiments](../dev/h3_fusion_experiments/README.md),
separate from runtime imports. The archived benchmark was rerun successfully:
all candidate operation bytes still matched and the performance ranking
remained unchanged. The optional floating-PV attention mode is
unchanged. There is no new approximation or quality tradeoff in these trials.

Memory guards: 110 GiB process RSS ceiling, at least 48 GiB available system
RAM, bounded MPS watermarks, isolated server, and automatic stop if the user
queue becomes active. Driver memory includes cached allocations and is not
an isolated scratch-memory measurement.

[Machine-readable timings and tensor hashes](data/m5-ultra-h3-fusions.json).
