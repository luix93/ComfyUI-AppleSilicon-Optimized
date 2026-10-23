# Opt-in INT8 attention (M5)

An approximate dense attention backend for long sequences. **Every query still
attends to every key**: this is quantization, not sparse attention. It is never
a default and changes outputs. Exact routes are unchanged unless you select it.

## Where to select it

| Place | Choice | Scope |
|---|---|---|
| **Attention Backend (Apple Silicon)** node | `dense_metal` (v3, recommended; formerly `int8_dense_accurate`) or `dense_metal_v2` (formerly `int8_dense_approximate`) | Any model whose attention runs through ComfyUI's optimized attention: H3, Wan, LTX and others. Eligible calls only (below); everything else uses the Apple router. |
| **H3 Acceleration Presets → Attention** | `Dense Metal · protected` / `· balanced` / `· full experimental` (the v3 kernel) | MiniMax H3 with the same step/layer guards as the earlier dense presets. Text/audio prefix queries are recomputed exactly. |
| **H3 Apple Silicon Acceleration** (advanced) | sparse backend `dense_int8_v3` or `dense_int8_v2` | Custom H3 windows/layers. |

With the backend node on an H3 model that also has H3 sparse attention attached,
sparse remains outermost and every call it declines, including protected steps
and layers, uses the selected INT8 backend. Use the H3 presets when you want
exact protected steps.

**Eligible calls**: M5 GPU, BF16/FP16, head dimension 128, no mask, equal Q/K/V
heads, at least `ASFP8_INT8_ATTN_MIN_TOKENS` keys (default 8192; shorter calls
measured no faster) and at most 65,536 keys. Query and key lengths may differ.
Inputs and outputs keep ComfyUI's strided layouts without copies.

## How it works

**v3 (accurate, Kitchen numerics)**: Q and K get a randomized orthonormal
Hadamard rotation before INT8, and probabilities are quantized to 255 levels
against each 64-key tile's own maximum, then merged into FP32 accumulators with
deferred rescaling. Error stays flat with sequence length (about 0.010 on real
H3 activations) for roughly 8% more attention time than v2.

**v2 (fast)**:

- Q: INT8 with one scale per 16 rows. K: midrange-centered per channel (softmax
  invariant), INT8 with one scale per 64 keys. V: mean-centered, one INT8 scale
  per channel.
- Probabilities use 255 levels (stored as P−128) against the running row
  maximum. The P−128 offset is restored from prefix sums of V.
- PV accumulates in INT32 across all key tiles; accumulators are rescaled in
  place only for rows whose maximum rose. There is no per-tile float epilogue.
- Padded keys score exactly zero and carry zero V; their normalizer mass is
  removed once at the end, so the inner loop has no tail mask.

Hardware counters show INT8 running at 2× BF16 on M5 Ultra's neural
accelerators. The kernel currently reaches about 60% of that matrix-unit floor;
the remainder is issue/ALU work around 16-row matrix operations.
[Profiling data](../benchmarks/m5-ultra-int8-attention.md).

## Measured results (M5 Ultra 80-core, PyTorch 2.14.1)

See the [benchmark report](../benchmarks/m5-ultra-int8-attention.md) for
operator timings across model shapes, real-activation errors, complete H3
workflows and visual comparisons.

## Limits

- Approximate: video detail, composition, motion and audio can change, more so
  in full (every step/layer) modes. Compare against exact with a fixed seed.
  Protected (first 4 and last 2 of 20 steps exact) stays closest; balanced
  (first 2 and last 1) is in between. Which steps are exact matters more than
  which layers ([calibration](../benchmarks/m5-ultra-int8-attention.md#coverage-recipes-balanced-calibrated-layers-reference-rows)).
- Short sequences (image models around 4k tokens) gain nothing.
- Masked, GQA, non-128 head dimension and CPU calls are delegated unchanged.
- M1–M4 are not supported by this kernel.
