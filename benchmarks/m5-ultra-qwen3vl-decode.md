# Qwen3-VL 4B INT8 ConvRot: native PyTorch prompt generation

October 17, 2026. M5 Ultra, 80 GPU cores / 256 GB; ComfyUI **v0.41.2**,
PyTorch **2.14.1**. Checkpoint: `qwen3vl_4b_int8_convrot.safetensors`.
This is autoregressive prompt generation, not diffusion sampling or text encoding.

## Follow-up: grouped projections and fused Q/K normalization

The earlier **42 tokens/s** figure was measured with ComfyUI's **basic attention**,
not PyTorch SDPA: the standalone harness set command-line arguments without
calling `comfy.options.enable_args_parsing()`. Removing per-stage synchronizations
alone still gave roughly 40–44 tokens/s. With argument parsing fixed, the preceding
implementation reached roughly 62–65 tokens/s with native SDPA. **Switching those
attention backends changes logits and sometimes generated tokens; it is not an
exact kernel speedup.** This patch does not change the user's selected backend.

For the following comparison both sides explicitly use the same native SDPA
backend. The baseline includes the preceding exact weight-cache/RoPE improvements;
the candidate adds only grouped projections and fused Q/K normalization/RoPE.
Three alternating before/after pairs run per task in one process. Discard the first
pair for warm-up; the table uses the median of the remaining two runs per mode.
Only generation boundaries synchronize. Timing includes prefill, rebuilding the
weight cache and preparing grouped projections.

| Task | Tokens | Previous exact pass | This pass | Speedup | Steady decode, before → after |
|---|---:|---:|---:|---:|---:|
| Sailboat prompt | 117 | 2.22 s | **2.05 s** | **1.08×** | 62.5 → **69.6 tok/s** |
| Clockmaker prompt | 111 | 2.14 s | **1.96 s** | **1.09×** | 62.6 → **69.9 tok/s** |
| Two synthetic images | 60 | 1.37 s | **1.34 s** | **1.02×** | 59.7 → **66.5 tok/s** |

The multimodal end-to-end gain is small, despite faster steady decoding. These
short synthetic requests do not represent a long-video BFS planner workload.
[All alternating runs, ablations and metadata](data/qwen3vl-decode-followup-oct17.json).

Accepted:

- Group Q/K/V and gate/up projections during single-token decoding, keeping native
  BF16 `F.linear` arithmetic. No activation quantization or vocabulary pruning.
  Replace separate owned cache allocations with slices of packed storage, instead
  of retaining a second complete weight copy. Hooks, custom forwards, weight
  functions, changed revisions and insufficient cache coverage retain the old path.
- Fuse Q/K RMSNorm with split-half RoPE, preserving the BF16 normalized intermediate,
  FP32 reduction order and separately rounded rotation products. Keep prefill and
  attention unchanged. Unsupported inputs/backends materialize the original norms
  before falling back; model wrappers are removed on success or cancellation.
- Avoid throwing a Python exception when checking the revision of an inference
  tensor; the generation-local cache still tracks ordinary tensor version changes.

A separate interleaved sailboat ablation measured warm medians of **2.192 s** for
baseline, **2.114 s** for grouping, **2.156 s** for norm/RoPE, and **2.032 s** combined.
An initially lighter prototype reached 73–74 tok/s, but the complete guarded
implementation is the result reported above.

Validation: 18 complete native-SDPA sequences match between variants, and a
separate six-run test compares **3,744 byte-identical tensors** across embeddings,
every block, final hidden states and logits. Another **3,744 byte comparisons** pass with the original basic-attention
backend. The normalization prototype also passed 100 synthetic seeds. Focused GPU regression suite: **93 passed, 1 skipped**.
These are tested parity results, not a universal proof over all possible inputs.

Retained converted weights remain **7.49 GiB**. Packing temporarily copies one
group at a time (largest approximately 95 MiB). MPS allocator/driver high-water
usage in these runs was roughly **20–21 GiB**, versus roughly **16.5 GiB** without
grouping; freed allocator storage can remain reserved. Existing memory-pressure
and cache-budget guards remain in effect. This pass is restricted to the validated
M5 Ultra / PyTorch build and checked core sources.

### Profiling and rejected experiments

CPU profiling identified substantial basic-attention work and a per-token `.item()`
wait for pending GPU execution. That wait is not evidence that scalar copying itself
costs 19 ms. Per-operation synchronized timings are diagnostic only: their dispatch/
synchronization floor makes them unsuitable for assigning whole-decoder percentages.

A Metal GPU capture of three repeated native BF16 GEMVs at **9728×2560**, profiled
at high GPU state, reported **915.88 GiB/s GPU read bandwidth**, **21.29% occupancy**,
**0.56% register-pressure influence**, and **61.71% memory-request-stall influence**.
These are one synthetic shape's counters, not sustained full-decoder DRAM bandwidth
or a full-model GPU-idle measurement. They indicate that this matrix kernel already
moves data quickly; register pressure is not its main indicated limiter.

A direct grouped BF16 GEMV based on the
[PyTorch 2.14.1 GEMV traversal](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/mps/kernels/Gemv.h)
passed one QKV shape but differed at gate/up and vocabulary shapes (maximum absolute
errors 0.0625 and 0.25 on random synthetic tensors). **Rejected from the exact path.**
No custom replacement matrix kernel is shipped from this experiment.

150 tok/s remains a weight-only ideal at 1,200 GB/s for roughly 8 GB/token. The
accepted implementation reaches about 70 tok/s on these text requests, not 150.

## Initial weight-cache pass (basic attention)

Second execution of each task, with identical greedy settings and a 192-token
maximum. Timing covers `clip.generate`: image encoding where present, prefill,
all generated tokens and rebuilding the temporary weight cache. Model file
loading, tokenization and final string decoding are outside timing; repetitions
reuse the loaded model. Both routes have matching per-stage MPS synchronization
for profiling. This is not the complete BFS planner or its media-loading time.

| Task | Output tokens (including stop) | Previous pack | Updated PyTorch | Speedup |
|---|---:|---:|---:|---:|
| Expand a sailboat scene | 119 | 35.72 s | **3.17 s** | **11.27×** |
| Expand a clockmaker scene | 111 | 33.51 s | **3.04 s** | **11.04×** |
| Describe two synthetic images and write a video prompt | 81 | 24.60 s | **2.35 s** | **10.46×** |

The baseline loads the **entire package**, with `ASFP8_QWEN3VL_DECODE=0`, and
uses ComfyUI basic attention implemented in PyTorch. The original standalone
harness did not enable CLI parsing, so its requested PyTorch SDPA flag was ignored.
Both sides used the same basic backend; the earlier wording was incorrect. **This is not a vanilla-ComfyUI comparison.** MLX generation
was explicitly disabled throughout. Both routes use the same local checkpoint;
there is no server, model substitution, precision reduction or token-limit change.

Steady decoding improves from approximately **3.3 to 42 tokens/s** on the warm repetitions. End-to-end
throughput is lower because each request includes prefill and cache preparation.
Both repetitions and complete generated text/token IDs are in the
[raw results](data/qwen3vl-decode-oct17.json).

## What was slow

The checkpoint stores INT8 ConvRot weights, but ComfyUI marks its text-encoder
projection layers for full-precision matrix execution and forced weight casting.
The native INT8 diffusion wrapper correctly declines those layers. Each token
therefore dequantizes and inverse-rotates the same weights again before floating
matrix execution. Instrumented warm transformer forwards took about **253 ms**.

The 4B model ties its vocabulary output to its embedding table. The package's
quantized-weight casting compatibility wrapper also converts this table to
floating point. That conversion repeats for embedding lookup and vocabulary
projection. An initial minimal harness omitted that wrapper and exposed a
different quantized head; full-package integration caught the difference, so
all initial results above were rerun with the complete stack active. The later
follow-up also verifies that command-line flags were actually parsed.

## Accepted changes

1. **Generation-scoped converted-weight cache.** Reuse ComfyUI's exact converted
   floating weights. Matrix operations, precision, attention and sampling stay
   unchanged. Cache keys include weight/data/scale revisions and cast arguments;
   patched weights, low-memory weight functions and requantization are excluded.
   In the initial isolated screen, transformer forwards fell to about **27 ms/token**
   before the RoPE fusion.
2. **Tied embedding/vocabulary table reuse.** Cache the existing floating table
   when the cast wrapper produces it, retaining the original `F.linear` and
   embedding lookup. Do not switch the head to INT8 arithmetic. A separate exact
   single-row INT8 kernel was also built and validated for casts that retain a
   quantized table; it is guarded and **not used in these full-package timings**.
   All vocabulary rows remain present.
3. **Single-token RoPE fusion and matrix reuse across layers.** Keep the two FP32
   products and addition separately rounded, followed by the original BF16 cast.
   Attention and normalization remain unchanged. Prefill keeps its original path.

Custom kernels are restricted to the validated M5 Ultra / pinned PyTorch builds
and supported shapes. RoPE also checks the core/Kitchen source and selected
backend. The weight cache is independent of those kernel guards and may provide
partial benefit on other MPS configurations; those devices are not benchmarked here.

## Correctness and memory

All **six complete generated token sequences** match the previous package.
Separate 16-token runs compare embeddings, every decoder block, final hidden tensor and
full vocabulary-logit tensor: **3,744 byte-identical tensor comparisons**, including
two synthetic-image runs. Dedicated GPU tests cover integer accumulation at
extreme INT8 values, ragged dimensions and RoPE parity with native Comfy Kitchen.
This establishes parity on these tests, not a universal proof for every prompt.

The cache holds **7.49 GiB** for this model, against an 8 GiB default maximum.
The cap is additionally limited to 10% of Metal's recommended working set;
new entries stop when driver allocations approach 65% of that recommendation.
Copies are discarded at generation exit, including cancellation/errors. The
PyTorch allocator may retain the freed storage for reuse. Budget-limited models
retain the ordinary operation on cache misses.

These tests used a 40 GiB process-memory guard, a 32 GiB available-RAM floor,
a bounded Metal allocator and an active-user-queue guard. Synthetic images were
two 224×224 tensors containing red/blue rectangles. No user clips or private
images were read. Large reference-image contexts and the complete BFS workflow
were not benchmarked; their vision encoding and prefill costs can differ.

## Controls

- `ASFP8_QWEN3VL_DECODE=0`: restore the previous PyTorch generation path.
- `ASFP8_QWEN3VL_CACHE_GIB=8`: converted-weight cache ceiling; `0` disables caching.
- `ASFP8_QWEN3VL_HEAD=0`: retain the previous vocabulary projection.
- `ASFP8_QWEN3VL_ROPE=0`: retain the previous RoPE operations and normalization.
- `ASFP8_QWEN3VL_GROUPED_PROJ=0`: disable projection grouping.
- `ASFP8_QWEN3VL_NORM_ROPE=0`: retain separate Q/K normalization.
- Set the last two flags to `0` to reproduce the preceding exact cache/RoPE pass.
- `ASFP8_DISABLE_MLX_TEXTGEN=1`: select native PyTorch instead of the pack's existing
  optional text-only MLX route. Image-containing requests already bypass that route.

No new workflow node or model download is required. Diffusion conditioning and
H3 sampling are outside the generation scope.

## Experiments not retained

Reusing the YuE2 cast cache without adapting the dequantization boundary produced
zero cache hits. The quantized wrapper survived the cast helper; conversion
happened immediately afterward. A minimal harness initially left out the general quantized cast compatibility
wrapper. Its INT8 vocabulary results are separate screening evidence, not the
full-stack headline. An initial vocabulary experiment also called
`Tensor.to` on the entire quantized table, inadvertently dequantizing it before
falling back. Its roughly 32 ms projection was rejected. The corrected INT8 head experiment uses the original cast/offload context and
quantized table. The headline full-package runs retain their floating head. No forced W8A8 transformer-body execution was accepted.
