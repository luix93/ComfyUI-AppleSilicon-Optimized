# H3 long clips: exact attention scheduling

October 13, 2026. M5 Ultra, **80 GPU cores**, 256 GB unified memory, macOS 27,
PyTorch **2.14.1** (`5c4886908584029761b579af026dcfb627c84070`), ComfyUI `1d2ea294`.

**Result: a small, verified exact improvement; the requested 200-second / 25%
reduction on a 15-second clip was not achieved.** This compares against our
preceding optimized implementation, not stock ComfyUI. All measurements below
exclude optional caches, sparse attention, token reduction and relaxed VAE.

## Measured complete workflow

864×480, 243 frames at 24 fps (approximately 10 seconds), **20 steps**,
`res_multistep` / `simple`, BasicGuider (CFG 1), seed `812764035`, no LoRA.
INT8 ConvRot H3 diffusion/text checkpoints, original FP16 video VAE and FP32
audio VAE. The authored public study-room prompt includes a forward camera move,
clock, books, framed pictures and checkerboard floor; the complete prompt and
output hashes are in the [measurement data](data/m5-ultra-h3-long-exact.json).
No user media was read.

One-step warm-up loads the models, compiles kernels and warms the decoders.
Measured full runs include sampling, audio/video decode and export; unchanged
text conditioning and model loading are cached. Attention timings include
allocation and the existing projection, quantization and layout paths through
the complete sampler.

| Measurement | Previous exact | New exact | Less time |
|---|---:|---:|---:|
| 10-second clip, full 20-step sampling | 390.05 s | 377.35 s | 3.26% |
| 10-second clip, complete warm workflow | **412.03 s** | **399.21 s** | **3.11% / 12.81 s** |
| 15-second geometry, two-step sampling mean | 75.98 s | 73.98 s | 2.64% / 2.00 s |

The full-workflow row is one measured candidate and its same-process control;
an earlier independent full control was 413.63 seconds. All three complete
20-step runs have identical conditioning, initial noise, video/audio sampler
latents, decoded frames and decoded audio bytes. The 15-second screen uses two
measured runs per implementation, interleaved after warm-up: controls
75.275 / 76.684 s, candidates 74.021 / 73.932 s, with matching complete latents.
It uses 864×480 × 362 frames and 44,600 packed tokens.

Multiplying that two-step saving by ten suggests roughly **20 seconds saved at
20 steps**. This is an extrapolation, **not a newly measured full 15-second,
20-step end-to-end result**. The user's 795-second reference is not substituted
for a same-run control or relabeled as a new measurement.

## What changed

The guarded PyTorch MPP query-cache kernel swaps its head/query dispatch axes
and moves one middle-PV scheduling barrier from unrolled index 4 to 6. Query
and key tile sizes, matrix operations, softmax, precision and reduction order
remain unchanged. The final cooperative-tensor drain remains in place. No
activation history, attention prediction or cross-step result reuse is involved.

The refinement is automatic only for the already-supported H3-shaped BF16
route on **M5 Ultra / validated PyTorch 2.14.1**, with **30,001–65,536 tokens**.
The installed PyTorch headers remain hash-checked. Other shapes/builds retain
the preceding route. `ASFP8_H3_LONG_ATTN=0` restores the old query-cache schedule;
`ASFP8_MPP_QCACHE=0` disables the query-cache optimization altogether. Restart
ComfyUI after changing these environment variables. Short 5-second clips do
not enter this new scheduling branch.

Native tests cover ragged lengths, interleaved/offset storage, zero/custom
scales, opt-out, version/scope boundaries, and native PyTorch byte equality up
to 65,536 tokens. The focused production regression run passed **19 tests**;
the standalone draft also passed eight scale/length/opt-out comparisons.

## Why the long clip is expensive

At 44,600 tokens, every one of the 100 main attention calls in the two-step
control used the exact query cache. **There was no token-limit fallback.**
Q/K/V were already contiguous in head-major layout.

Synchronized profiling attributed approximately 54.06 seconds to dense
attention, 12.27 seconds to MLP, and 0.513 seconds to AdaLN over two steps.
The instrumented sampler loop took 76.44 seconds; the uninstrumented warm
control took 73.99 seconds. Profiling adds overhead, and enclosing module
measurements must not be added to their nested attention measurements.

Dense attention is roughly 70% of sampling here. Its pairwise work grows
approximately quadratically with token count; projections/MLP and much of
VAE decoding grow closer to linearly. This explains why three times the clip
length need not take three times as long. The available Spark totals alone
cannot establish its per-operator bottlenecks or kernel precision.

Saving 200 seconds from a roughly 795-second total would require about
**35–40% less attention time**, approximately **1.6× attention throughput**, if
the rest of the workflow were unchanged. The measured scheduling improvement
is far smaller. There is no evidence in this pass for a hidden switch or
missing fallback fix that would recover 25% while preserving all output bytes.

## Other screens and decisions

All attention screens use captured public-fixture Q/K/V from the real
15-second geometry. Numerical differences disqualify a candidate, even when
it still computes mathematically dense attention.

- **More query rows per SIMD group:** preserved bytes but took about
  2.50–3.44 seconds per attention call versus 0.51–0.54 seconds. Rejected.
- **Different query tiles, output splitting, grid swizzles and independent
  PV issue order:** none improved on the accepted schedule. PV reordering
  measured approximately 561–569 ms versus 513 ms for the accepted schedule.
- **Wider 32-value reduction instructions:** the correct QK packing matched
  bytes but measured about 511 ms versus 510 ms; widening PV measured 534 ms,
  and widening both 527 ms. Rejected as additional changes. Alternative
  packing failed byte comparison. A 64-column cooperative instruction is
  unsupported by the current MPP API and was not benchmarked.
- **Shared K/V staging across query SIMD groups:** tested unchanged arithmetic
  with padded/unpadded shared memory. It preserved bytes but increased runtime
  substantially; rejected. This also shows why reducing duplicated loads alone
  is not sufficient when the extra synchronization/storage costs dominate.
- **Larger MLP bridge chunks:** a synthetic shape-matched 44,600-row screen
  including FC1, activation, ConvRot, quantization and allocations measured
  77.49 ms at the current 8,192 rows, 76.51 ms at 16,384, and 76.41 ms with all
  rows. All outputs matched. The roughly 1 ms per-call saving would amount to
  about one second across 1,000 blocks and increases scratch memory. The
  existing bounded chunk size remains unchanged. Common FC2 work was excluded
  from this isolated screen; no whole-MLP speedup is claimed.
- **Schedule modulation precomputation:** the measured AdaLN fraction is only
  about 0.7% of sampling before accounting for precomputation cost, so it cannot
  deliver the requested gain by itself.

## Deferred TensorFold ideas

Reviewed the [H3 MLX pack](https://github.com/drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX),
its pinned engine revision `99fa80a32f7e61d331092064576ada72fdd2d1a4`, and the
newer [H3 engine branch](https://github.com/drowzeys/TensorFold/tree/b7e309e042e48b72e75e9fb87342df4e9bf22f37).
Mixed-precision attention and attention-output reuse change bytes; they do not
qualify for this target. Mixed FP16 probabilities reduced local error relative
to our existing INT8 attention but were slower than that existing optional
implementation. Neither route was installed or enabled by this exact change.
The upstream schedule-modulation memory benefit for FP32 weights must not be
reported as the same saving in our already quantized model.

The remaining high-value research direction is a materially faster **exact
dense attention pipeline** with the same reductions—better overlap of memory
loads and matrix work, or a better compiler/hardware schedule. This pass does
not establish a further gain or promise a 25% reduction from that work.

## Memory and reproducibility

GPU jobs ran serially in an isolated server. Guards stop on a busy main ComfyUI
queue, 110 GiB host RSS, or less than 48 GiB available system RAM; MPS allocator
watermarks were 0.4/0.3. The full exact batch peaked at 15.60 GiB host RSS and
retained at least 67.65 GiB available system RAM. Host RSS is **not** Metal
allocation, and end-of-run MPS snapshots are **not** peak GPU measurements.
Standalone kernel screens use a 96 GiB RSS cap, 64 GiB available-RAM floor and
900-second timeout. No OOM occurred. Large captured tensors and generated
videos stay outside the release repository; only sanitized measurements and
hashes are included.

## Further exact-attention investigation

The subsequent [pipeline and operand-lifetime screen](m5-ultra-h3-exact-attention-pipeline.md)
implemented 45 variants. None qualified as an additional production improvement;
the accepted schedule and the complete-workflow measurements above remain unchanged.
