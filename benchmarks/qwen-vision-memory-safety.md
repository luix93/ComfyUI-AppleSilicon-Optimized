# Qwen vision memory safeguard — 2026-10-18

Tested on M5 Ultra 80 GPU cores / 256 GB, ComfyUI v0.41.2, PyTorch 2.14.1.

## Failure and repair

`TextGenerate` failed before token generation in Qwen3-VL's FP32 vision tower.
Its `small_input=True` attention selector returned `attention_basic`, bypassing
the pack's global router. Image preprocessing permits approximately 12.8 MP:
one large image alone can cause a quadratic attention allocation over 100 GiB.

The patch wraps **only the Qwen3-VL / Qwen3.5 vision module's selector**. Small
basic-attention calls delegate unchanged. Oversized calls retain all keys and
values but slice queries, reusing ComfyUI's original attention implementation.
Neither precision nor resolution is reduced, and no sparse attention is used.
Already-selected non-basic backends, text generation, and other models retain
their existing routing.

A 512 MiB default scratch planning budget allows four score buffers per chunk;
device and system headroom can reduce it. The GPU is synchronized between
chunks so outstanding commands cannot accumulate the full original allocation.
Output storage and other model allocations are additional. Cancellation is
checked between chunks. Insufficient headroom or a changed unsupported layout
raises an explanatory error instead of attempting a giant allocation.

## Validation

Only synthetic inputs and randomly initialized vision weights were used. No
user media, saved workflows, or checkpoint files were read.

| Check | Result |
|---|---|
| Attention shape `[1,16,47776,64]`, FP32 | Completed, all outputs finite |
| Unsliced score tensor size (calculated, never allocated) | 136.05 GiB |
| Single bounded attention call | 2.38 s |
| Peak Metal driver allocation, sampled every 10 ms | 1.04 GiB |
| Peak active tensor allocation, sampled every 10 ms | 0.83 GiB |
| Selected query rows versus original full-key attention | Maximum absolute error 2.42e-8 |
| Full 24-block Qwen3-VL 4B vision architecture, random weights, 1024 patch tokens; forced chunks | Final merged features and three DeepStack outputs byte-identical in this test |

Memory figures are for a standalone attention test, **not full-model memory**.
The full-depth test used a smaller grid; the 47,776-token test covered one
attention operation, not a complete high-resolution encoder/generation run.

This is **not a general byte-exact guarantee**. Initial FP32 attention probes
showed maximum absolute differences up to 5.82e-7 from changing query matrix
shape. FP16/BF16 probes also differed. The operation remains full dense
attention, but reduction implementation/rounding can vary. Model-generated
text and real-image quality were not assessed by these synthetic tests.

Runtime tests cover selector scoping and idempotence, explicit backend
preservation, small-input byte parity, both output layouts, retention of every
key, cancellation, memory pressure, budget validation and unsupported layouts.
The combined vision-safety, Qwen decode and attention suites passed **77 tests**.
An installed-package synthetic smoke check with Desktop-style arguments passed.
Its first attempt hit a Metal command-buffer assertion during existing kernel
initialization, before this new patch was installed in that process; an
unchanged retry passed. That intermittent startup assertion remains undiagnosed
and is separate from the reported vision score-allocation failure.

Controls: `ASFP8_QWEN_VISION_SAFE=0` disables protection and restores the
original potentially unsafe allocations. `ASFP8_QWEN_VISION_MEMORY_MB` adjusts
the scratch budget within 64–1024 MiB. These settings are independent of the
exact Qwen text-decoding cache/fusions.

Raw measurements: [JSON](data/qwen-vision-memory-safety.json).
