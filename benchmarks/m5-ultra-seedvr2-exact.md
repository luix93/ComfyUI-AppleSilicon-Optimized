# SeedVR2 exact optimizations and large-image bounds

2026-09-24. M5 Ultra, 256 GB, macOS 27, installed PyTorch 2.13.0
(`cf30153c4c131c8164ee7798e5022d810682e2cb`). This supersedes the preliminary
[acceleration screening](m5-ultra-seedvr2-acceleration.md).

## Reference and measurement

SeedVR2 7B INT8 ConvRot with `ASFP8_NATIVE=1`, FP16 SeedVR2 VAE, one Euler
step, CFG 1, denoise 1, seed 959948902156062, 3× Lanczos resizing, 512/128 VAE
tile/overlap, color correction `none`. The reference is the previously
accelerated W8A8/Conv3D path with the VAE memory safeguard enabled, **not**
unpatched ComfyUI's weight-only INT8 path.

Times cover the actual node sequence from resize through PNG writing.
They include allocations, layout copies, quantization and GPU completion.
Interpreter/JIT startup and post-run hashes/tensor saves are outside the
measurement. First-run and warm timings are identified separately.

## Accepted implementation

- **VAE convolution packing:** reuse folded weights only within an active
  tiled encode/decode. Weak references, tensor versions and layout/device
  checks prevent stale reuse for normal tensors. Inference tensors without
  version counters are reused only during that immutable call. The context
  releases all packed tensors on normal return or exception. A sole convolution
  chunk is returned without an unnecessary one-element concatenation.
- **VAE SiLU/layout:** one Metal pass preserves the FP16 activation rounding
  and writes causal temporal replicas directly into the channel-folded Conv2D
  input. The original Conv2D operand geometry and arithmetic remain.
  Eligibility is restricted to the single-frame, no-memory-state FP16 path.
- **Sampling normalization/modulation/residual:** fuse memory passes while
  retaining BF16 intermediate stores. The unweighted RMSNorm still uses
  PyTorch's composite mean and reciprocal-square-root reduction. It does not
  substitute a different parallel reduction.
- **RoPE:** reuse the same FP32 sine/cosine values within the existing cache
  scope, process Q/K together, retain multiplication/addition order and BF16
  output rounding. Floating-point contraction is disabled in these kernels.
- **Window attention:** group equal-length windows in batches of eight,
  retaining the existing attention backend and arithmetic. Small safe images
  retain the original loop because dispatch batching was slower there.

All are default-on with the native preset, subject to shape/dtype guards.
`ASFP8_SEEDVR2_FAST=0` before startup disables the performance additions.
Individual flags are documented in the README. The memory safeguard and
attention buffer fix remain independent; native 64-bit layout guards remain
when FAST is disabled but the native extension is available.

## 7200×4800 acceptance measurements

Same-process baseline/baseline/candidate/candidate, excluding the first
baseline warmup:

| Stage | Previous exact reference | Candidate run 1 | Candidate run 2 |
|---|---:|---:|---:|
| VAE encode | 16.342 s | 10.551 s | 10.481 s |
| Sampling | 18.834 s | 14.099 s | 14.045 s |
| VAE decode | 36.037 s | 25.102 s | 25.079 s |
| Complete sequence with PNG | 73.516 s | 52.034 s | 51.831 s |

About **29% less total time**, or **1.42× throughput**. All four runs have the
same encoded latent, sampled latent and final FP32 RGBA frame hashes:

| Tensor | SHA-256 |
|---|---|
| Encoded | `c0d9ebdc04e62fc6eecfabdf9f5874182d5bc2f8817b1cf21dc37a70eabac50e` |
| Sampled | `b9dbc7e67f0b4a49e1fc9106e9e914bacd3f2043587e5c3d88b3888cf911fcc7` |
| Frames | `c5ab4984ce62faeb08e12cfbbc9978979d393f6eb665da8afe7614f3dde89151` |

These initial candidates included the optional GroupNorm fusion. An identical-
latent decoder ablation then measured 24.963/25.022 s without it versus 24.988 s
with it; all frame bytes matched. That gain is negligible, so
`ASFP8_SEEDVR2_VAE_GN=0` is the final default. The optional implementation is
restricted to the tested PyTorch 2.13 algorithm and remains available to test.

**Final installed defaults, fresh process:** 52.094 s first run and
**50.984 s warm**, both matching all three reference hashes above. The warm
encode/sample/decode stages were **10.462 / 13.036 / 25.178 s**. This final
version includes the row-gather and size guards below and leaves GroupNorm
fusion off. Relative to the 73.516 s warm reference, it reduces complete
workflow time by **30.6%** (about **1.44× throughput**).

## Why the 6240×4160 → 18720×12480 job failed

This output is **233,625,600 pixels**. Its patch grid has 912,600 tokens; a
3072-wide BF16 feature tensor has **2,803,507,200 elements**. A fused QKV
projection has **8,410,521,600 elements**. Those exceed signed 32-bit limits,
and QKV also exceeds unsigned 32-bit element indexing.

The reported exception was raised by attention on a small view backed by the
oversized feature buffer. A standalone 483-token reproduction triggered the
same `MPSGraph does not support tensor dims larger than INT_MAX` exception;
materializing that window produced the same bytes as an independent compact
reference.

Fixing only that exception was insufficient. Follow-up large tests exposed
INT8 fallbacks, invalid window indices, and a completed but visibly corrupted
render. That render was rejected. The large paths now:

- materialize bounded attention windows;
- bound INT8 projection rows before ConvRot/quantization/GEMM, preserving
  per-row quantization and the integer reduction;
- use explicit 64-bit row-gather offsets for large QKV/window layout buffers,
  including a virtual video/text concatenation;
- bound large Q/K normalization and native elementwise/RoPE work.

The native gather was separately exercised in a **4,608,000,000-element**
BF16 buffer with exact samples on both sides of the signed and unsigned
32-bit boundaries. A complete saved-latent sampler then ran all 36 blocks in
107.99 s without any INT8 fallback. This isolated sampler check is not an
end-to-end upscale timing.

The subsequent complete 6240×4160 → 18720×12480 run succeeded with finite
output and no INT8 fallbacks:

| Stage | Seconds |
|---|---:|
| VAE encode | 69.212 |
| Sampling | 114.916 |
| VAE decode | 170.285 |
| PNG writing | 13.390 |
| Complete node sequence, including resize/preprocessing/loading | **370.595** |

This is a **first complete run**, not a claimed warm median. The output image
and full-resolution crops were visually inspected; the corruption in the
rejected preliminary render is absent. Maximum observed Metal allocation was
125.51 GiB, observed after each transformer block, every 100 VAE tiles and at
stage boundaries. This sampling can miss short peaks and is not a measurement
of total system RAM. The saved tensors and final frame hashes are in the data
file. The huge Q/K normalization guard separately matched a compact reference
byte-for-byte at both ends of an 8.41-billion-element backing buffer.

## Numerical acceptance and rejected experiments

- Primitive tests cover BF16 modulation, gated residual, composite RMSNorm,
  RoPE, ragged FP16/BF16 window attention, bounded INT8 projections and strided
  row gathers. SiLU is checked against **every finite FP16 input bit pattern**.
- The focused GPU/CPU regression suite passed **32 tests**.
- Whole-workflow hashes compare encoded latents, sampled latents and final
  decoded frame tensors, not just PNGs.
- Final installed baseline/candidate pairs at 960×638 matched all three tensor
  hashes for seeds **959948902156062, 42 and 271828**. The two additional warm
  baseline/candidate pairs measured 1.218/0.865 s and 1.208/0.865 s.
- An initially faster direct RMS reduction diverged on captured real inputs
  and was removed. The accepted implementation retains the original reduction.
- RoPE with compiler FMA contraction differed; the accepted kernel disables it.
- A naive GroupNorm reduction differed. The exact PyTorch-derived version
  passed numerical checks but its full-decoder gain was neutral, so it stays off.

Exactness is measured against this installed accelerated reference and tested
software/hardware stack. It does not assert equivalence to a different
attention backend, PyTorch release, GPU, or unpatched W8A16 implementation.
The huge workflow cannot be compared byte-for-byte to the original crashing
path; operation tests and complete successful output checks have a different
scope from the reference comparisons at 7200×4800.

[Machine-readable results](data/m5-ultra-seedvr2-exact.json). Diagnostic scripts
and full local outputs are retained under workspace `diagnostics/seedvr2-fast`
and `diagnostics/seedvr2-3x`. The GroupNorm source attribution and license are in
[third-party notices](../THIRD_PARTY_NOTICES.md).
