# MOSS SFT: dedicated dense attention tuning

October 18, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f569d268b80076e5f44126f87cdf835869` after v0.41.3 ·
Comfy Kitchen 0.2.38.

## Scope and numerical contract

This tests attention specialized for MOSS's 24 heads, D128 and roughly 4–6K
joint image/text tokens. Every key, layer and denoising step is retained.
No SOL, sparse routing, token reduction or step skipping is used. Exact
attention remains the default; the new modes are independently opt-in.

- **`tile32`:** PyTorch MPP with 32 query rows per group and direct BNHD output.
  The key tile, QK/PV arithmetic and reduction order remain unchanged.
- **`dense_int8`:** existing H3-derived all-token INT8 QK/PV arithmetic, adapted
  to MOSS's input layout. Remove input copies, unused sparse summary work and
  allocations, route-bit traversal, output-layout conversion and statistics.
- **`dense_int8_floatpv`:** existing INT8 QK / BF16 PV arithmetic, with direct
  Q/K reads, direct BNHD output, no wrapper copy/count output, and 32-row query
  groups. Keep the V layout conversion: removing it did not consistently win.

The two INT8 modes are approximate versus exact attention. Matching their
previous dense algorithms' bytes does not make them exact versus PyTorch or
CUDA Comfy Kitchen. These changes introduce no additional approximation on top
of those previous dense algorithms in the validated comparisons.

## Measurement method

SFT INT8 ConvRot, Qwen3.5-9B INT8 ConvRot, Flux.2 small VAE, ER-SDE/simple,
25 steps, CFG 2.5, seed `856771198829162`. Same synthetic theatre-marquee prompt
as the earlier MOSS dense-attention report. Enhancement and SLG off.
The optional FFN fusion, SiTU/quantization prototype and Kitchen prototype are
all off, isolating the attention change.

Every job re-encodes text, samples, decodes and saves a PNG. Capture/hashing
happens outside the measured interval. Preparation, quantization, allocations,
layout operations and Python dispatch are included. Three-step warmups precede
paired full renders in forward/reverse mode order; exact controls bracket each
resolution. These are comparisons against the **current optimized exact path**,
not new stock baselines. The isolated server checks that the user's queue is
idle, caps RSS at 110 GiB and requires 48 GiB free system RAM.

## Paired warm complete renders

Means of two warm full runs per mode:

| Resolution | Attention | E2E | Sampling | E2E reduction vs exact |
|---|---|---:|---:|---:|
| 1024² | Current exact | 22.142 s | 20.373 s | — |
| 1024² | Tuned INT8 QK/PV | 21.642 s | 19.766 s | 0.500 s / 2.26% |
| 1024² | Tuned INT8 QK / BF16 PV | 21.597 s | 19.746 s | 0.545 s / 2.46% |
| 1248² | Current exact | 33.824 s | 31.956 s | — |
| 1248² | Tuned INT8 QK/PV | 32.385 s | 30.345 s | 1.439 s / 4.25% |
| 1248² | Tuned INT8 QK / BF16 PV | 32.409 s | 30.367 s | 1.415 s / 4.18% |

The separate exact-query-tile screen preserved final latents and decoded tensors:

| Resolution | Existing exact E2E | Tile32 E2E | Existing exact sampling | Tile32 sampling |
|---|---:|---:|---:|---:|
| 1024² | 22.358 s | 22.125 s | 20.510 s | 20.287 s |
| 1248² | 34.149 s | 33.505 s | 32.092 s | 31.458 s |

The 1024² tile32 E2E range overlaps the exact range; its small gain is not a
strong recommendation. The 1248² paired tile32 improvement is repeatable here.
Neither replaces the existing default.

## Output and quality checks

Conditioning, input latents and initial noise match across modes. Repeated runs
within each mode reproduce identical latents and decoded tensors. Both final
INT8 variants also match the corresponding preceding dense algorithm’s complete
latents and decoded tensors byte for byte at both resolutions.

| Resolution | Mode | Latent RMSE vs exact | Decoded-frame RMSE vs exact |
|---|---|---:|---:|
| 1024² | Tuned INT8 QK/PV | 0.435557 | 0.140707 |
| 1024² | Tuned INT8 QK / BF16 PV | 0.401582 | 0.139390 |
| 1248² | Tuned INT8 QK/PV | 0.320436 | 0.090634 |
| 1248² | Tuned INT8 QK / BF16 PV | 0.217754 | 0.060044 |

These errors are descriptive, not quality scores. Both approximate modes produce
coherent images in this fixture, but clothing, framing and background details
change versus exact. They do not preserve the exact composition. The retained
preparation/layout refinements add no further visual change versus each prior
dense mode on these tested fixtures.

![1024 exact / dense INT8 / floating-PV comparison](artifacts/moss-attention-tuning/comparison-1024.jpg)

![1248 exact / dense INT8 / floating-PV comparison](artifacts/moss-attention-tuning/comparison-1248.jpg)

Peak sampled Metal driver allocation was **29.89 GiB**. Process RSS was
1.14 GiB, which does not include all Metal allocations and is not total unified
memory consumption. The memory guard did not trigger.

[Per-run timings, input/output hashes and numerical checks](data/m5-ultra-moss-attention-tuning.json).

## Final operator measurements

Interleaved V stride 9216 / offset 6144, packed Q/K, including allocations,
quantization, layout conversion and the retained dispatch guards. Eleven
interleaved rounds of 16 calls; medians after warmup:

| Main token count | Existing exact | Exact tile32 | Tuned INT8 QK/PV | Tuned INT8 QK / BF16 PV |
|---|---:|---:|---:|---:|
| 4,179 | 2.145 ms | 2.004 ms | 1.848 ms | 1.835 ms |
| 6,167 | 4.490 ms | 4.203 ms | 3.746 ms | 3.720 ms |

This is roughly 14–17% less time for the complete attention operator, but
only 2–4% less time for the complete render. The model still spends substantial
time in projections/FFN and other work outside this attention operator.

## Screening and rejected candidates

- Floating-point query-group sizes 16, 32, 64 and 128 were screened. The 32-row
  group won the operator comparison while preserving bytes. Its whole-render
  gain is much smaller, with variability at 1024².
- Increasing the floating-point key tile from 32 to 64 or 128 changed bytes and
  substantially slowed the operator. Rejected.
- Converting BF16 Q/K/V to FP16, including conversion costs, was slower and
  changed bytes. Rejected.
- Removing query hoisting was slower. Rejected.
- INT8 query groups 16/32/64/128 and a plain dense K-block loop were screened.
  The plain loop retains the same traversal order and arithmetic. The retained
  full-INT8 mode uses 64-row groups; 32 provided no meaningful additional gain.
- Floating-PV groups of 32 were faster than 64; 16 and 128 were slower.
- Reading V directly inside floating-PV avoided a copy but lost at 1.5 MP.
  Retain the V copy; an isolated 1024 micro-win does not justify a second mode.

An initial full-INT8 direct-read prototype assumed packed V rows. Real MOSS
uses an interleaved QKV allocation: V has row stride 9216 and storage offset
6144, while normalized/rotated Q/K have row stride 3072. The prototype's image
check failed despite passing packed-input microchecks. Those full-render
results are **invalid and excluded**. The corrected implementation explicitly
supports both reviewed V layouts, tests their output bytes, and checks real
workflow inputs against the preceding dense operator during warmup.

## Controls, coverage and limitations

See [MOSS controls](../docs/moss-image.md#optional-moss-attention-tuning).
`ASFP8_MOSS_ATTN_MODE=exact` is unchanged by default. The optional modes are
limited to the reviewed M5 Ultra / PyTorch 2.14.1 configuration, BF16, batch 1,
24 heads, D128 and 4,096–6,656 tokens. They do not alter text-only embedder
attention or other models. Unsupported cases retain the exact path.

Focused tests cover shader/source guards, default/invalid-mode passthrough,
CPU/gradient/layout fallbacks, a latched compiler failure and output-byte
comparisons at 4096, 4179 and 6167 tokens, with packed and interleaved V.
Existing MOSS/FFN regression tests also passed. No CUDA equivalence or broad
quality certification is claimed. Visual validation uses one synthetic prompt
and seed at two resolutions; it cannot establish behavior across all prompts.

The derived kernels retain the existing PyTorch/MLX and CCV attribution and
licenses. See [third-party notices](../THIRD_PARTY_NOTICES.md).
