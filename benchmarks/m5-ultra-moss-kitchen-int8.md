# MOSS Kitchen-style INT8 Metal prototype — October 18, 2026

**Not promoted to the installed/default path.** The dense INT8 attention port
works on the tested MOSS inputs, but is slower than this pack's exact query-cache
attention. The first fused FFN prepass also loses end to end despite winning in
isolation. CUDA byte equivalence remains **unverified**; a standalone exporter
is supplied for a later CUDA run.

## Scope and setup

M5 Ultra, 80 GPU cores / 256 GB, macOS 27; PyTorch 2.14.1,
ComfyUI nightly `c06bf5f569d268b80076e5f44126f87cdf835869` after v0.41.3,
Comfy Kitchen 0.2.38. MOSS SFT INT8 ConvRot, 1024×1024, CFG 2.5,
ER-SDE/simple, 25 steps, seed `856771198829162`, prompt enhancer and SLG off.
This is the same synthetic text-to-image fixture as the
[execution audit](m5-ultra-moss-execution.md). The comparison is against the
current **optimized exact path**, not stock ComfyUI.

An isolated server ran with preview/assets disabled, an idle-user-server check,
MPS high/low watermarks 0.4/0.3, a 110 GiB child-RSS ceiling and 48 GiB available
physical-memory floor. No memory guard fired. Child RSS is not a measurement of
total unified GPU memory. The attention workspace is linear in token count;
there is no sequence-length-squared score allocation in the Metal implementation.

Each render refreshes text encoding, samples, decodes and saves PNG. Conditioning,
initial noise, initial latents, sampler latents and decoded floating tensors are
captured separately from the timed render. First model-loading runs are excluded.

## What was implemented

The shared `_patches/kitchen_attention_mps` API accepts raw head-major Q/K/V;
its reviewed integration is restricted to MOSS. It retains every key and ports:

- randomized H128 Q/K rotation and interleaved per-group INT8 scales;
- adaptive K-anchor selection from nine sampled keys;
- per-channel V quantization;
- unsigned 0–255 probability quantization, rounded-probability denominator,
  and per-key-tile delayed softmax rescaling;
- FP32 running softmax/numerator state and BF16 output.

The reference is [Kitchen v0.2.38, commit 42081bf9](https://github.com/Comfy-Org/comfy-kitchen/tree/42081bf9bb5fad3992c1256aab4f01b27cbb5816).
[Source hashes](data/kitchen-0238-reference.json) and third-party notices are
preserved. This differs from the existing CCV-derived H3 dense/SOL paths.

Metal MPP rejected mixed unsigned-P/signed-V integer multiplication. The port
uses `(P - 128) @ V + 128 * sum(V)` in INT32. This identity is exact within the
bounded tile range, including padded tails; it adds no probability approximation.
The CPU oracle checks the policy, but does not reproduce NVIDIA's exp2 intrinsic
or its complete floating reduction order. **Do not describe this as CUDA-byte-identical.**

Initial support: M5, BF16, D128, matching Q/K/V head counts, unmasked non-causal
attention, 1,025–65,536 keys. Masks, GQA, other head dimensions, FP16 and other
models are not validated. The MOSS adapter additionally retains its existing
reviewed shape/source/sampling-scope guards and respects explicit backend
replacements. It recognizes an explicit `comfy_kitchen_int8` checkpoint config;
INT8 weight precision alone does not select an attention backend.

The separate `_patches/moss_situ_quant_mps` prototype fuses exact BF16 SiTU LUT
lookup, regular Hadamard rotation and rowwise INT8 quantization. Its FHT
reassociates sums relative to the existing MPS matmul, so it is **non-bit-exact**.
The matrix multiplication and output epilogue remain the existing INT8 kernels.
It falls back for incompatible rotations, runtime LoRAs and precision overrides;
the surrounding MOSS adapter preserves hooks/weight-function guards.

## Measurements

Real captured attention tensors were `[1,24,4179,128]` and
`[1,24,4119,128]`. Synchronized repeated operator measurements include allocation
and quantization where indicated:

| Attention operation | 4,179 tokens | 4,119 tokens |
|---|---:|---:|
| Existing exact query-cache | 2.10 ms | 1.91 ms |
| Prototype quantization/preparation | 0.35 ms | 0.34 ms |
| Prototype attention kernel | 6.59 ms | 6.13 ms |
| Prototype complete operation | 6.87 ms | 6.64 ms |

The initial unrolled-loop-free port took 8.3–8.6 ms. Explicit cooperative-tensor
loop unrolling helped but did not close the gap. Removing Q caching, widening
the QK inner tile, and using one/two SIMD groups were screened with byte checks
against the prototype. Best isolated attention-only result was 6.09 ms, still
well behind exact. FP16 execution of exactly representable integer products
matched the prototype bytes on the captured tensor but took 9.06 ms in its
initial schedule; it was rejected.

| 25-step mode | Warm E2E | Sampling loop | Decision |
|---|---:|---:|---|
| Exact control | 22.19 s | 20.34 s | Retain |
| Kitchen-style attention | 32.42 s | 30.55 s | Reject as an acceleration |
| First fused FFN only | 22.60 s | 20.79 s | Reject |
| Both first prototypes | 32.88 s | 30.99 s | Reject; one full combined run |

Exact, attention and first FFN rows are means of two warm runs. The cold exact
run was 24.25 s. Three-step screens agreed with these directions. The first FFN
operator screen was 0.81 → 0.68 ms, illustrating why isolated kernel wins cannot
substitute for complete-render measurements.

A second FFN layout reduced per-thread live values from 64 to 16, using 512
threads per row. It preserved the first prototype's quantized bytes and improved
the isolated prepass from 0.66 to 0.53 ms in its paired screen. Yet full E2E was
22.96 / 22.74 s versus interleaved exact controls 22.25 / 22.01 s. It was also
rejected; the retained diagnostic implementation uses the first 128-thread
layout. These results point to integration/execution costs that the isolated
prepass timing does not capture; no specific hardware bottleneck is claimed
without a counter trace.

Neither a 50-step nor 1.5 MP sweep is justified for the rejected attention port.
No speedup is added to the public performance table from these experiments.

## Numerical and visual results

Q/K/V quantized bytes and all three scale arrays matched the local CPU policy
oracle on a captured 1,153-token/two-head slice. Against that oracle, the initial
attention output matched about 99.985% of BF16 elements; max absolute difference
was 0.0625 and RMSE 0.0001775. This is local-policy validation, **not** CUDA parity.
On full captured MOSS tensors, attention-vs-exact RMSE was 0.0646 / 0.0558,
against exact output RMS 5.254 / 4.876; max errors were 2.125 / 1.25.

The first fused FFN capture changed 12 quantized values out of 33,742,848;
rotation changed 268 BF16 values, and row scales were identical. That tiny
single-layer discrepancy did **not** imply negligible final-image differences.

On the complete 25-step fixture, decoded RGB `[0,1]` RMSE versus exact was
0.1365 for FFN, 0.1490 for attention and 0.1361 combined. These are differences
between generated compositions, not standalone perceptual quality scores.
Side-by-side inspection found coherent images, readable marquee text and the
same broad warm-light/cool-street palette, but changed coat length/opening,
walking pose and marquee framing. This single prompt does not establish general
quality preservation. Broader prompt/seed or temporal-quality certification was
not pursued for modes that already lost on speed.

Repeated attention renders matched each other byte-for-byte. Returning to exact
restored the conditioning, initial noise, sampler latents and decoded tensor
bytes. The installed exact implementation has not been replaced.

## Opt-in diagnostics and CUDA reference

These are **research switches, not recommended acceleration presets**:

- `ASFP8_MOSS_KITCHEN_INT8=1`: enable the shared prototype at eligible MOSS
  attention calls that explicitly request Kitchen INT8 in checkpoint metadata.
- `ASFP8_MOSS_SITU_QUANT=1`: independently enable the fused FFN experiment.

Both default to `0`. Neither changes the current H3 presets, global backend
availability, nor arbitrary INT8 models. Both require the existing MOSS patches
to be enabled. Clear the flags to restore the exact path.

The [standalone reference script](tools/kitchen_attention_reference.py) runs on
CUDA without importing this pack. Install the pinned Kitchen version in the
CUDA environment and use the same script/seed, or pass identical trusted Q/K/V
captures with `--input`:

```sh
python -m pip install comfy-kitchen==0.2.38
python kitchen_attention_reference.py --device cuda --output cuda.pt
```

On the Mac, from the repository with the tested ComfyUI Python environment:

```sh
python benchmarks/tools/kitchen_attention_reference.py --device mps --output metal.pt
python benchmarks/tools/kitchen_attention_reference.py --compare metal.pt cuda.pt
```

The comparison refuses different input hashes, canonicalizes CUDA V packing,
ignores unused CUDA Q-scale padding, and reports quantized-byte, scale and output
errors separately. Exports include framework/reference versions, device metadata
and timing including quantization. It does not upload or download any data.
The Metal export and comparison round-trip were exercised successfully.
The focused Metal/numerical/default-path test suite passed **39 tests**, including
short query tails, key tails, biased inputs, all-zero inputs, metadata guards
and preservation of explicit attention replacements.
No active CUDA machine was available for this pass; that run remains outstanding.

[Machine-readable results](data/m5-ultra-moss-kitchen-int8.json).
