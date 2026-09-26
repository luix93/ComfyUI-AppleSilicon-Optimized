# H3: Sage-style attention screen on M5 Ultra / PyTorch 2.14

Measured 2026-09-25 on the M5 Ultra, 80 GPU cores / 256 GB, macOS 27.0,
PyTorch 2.14.0 (`08187d9e0fba026dc8217405802ab5381dc88d90`).

## Result

The tested qlip-style Metal kernel is **slower** than both stock PyTorch
2.14 attention and the installed exact query-caching specialization.
It is not promoted to a production flag or installed in ComfyUI.

| Attention input | Stock 2.14 | Current exact query cache | Best Sage, normal compiler | Sage/current time |
|---|---:|---:|---:|---:|
| 2,968 tokens, cropped short geometry | 2.78 ms | 2.46 ms | 5.39 ms | 2.19× |
| **15,458 tokens, actual saved 5-second geometry** | **73.18 ms** | **64.01 ms** | **184.63 ms** | **2.88×** |

These are attention-call timings, **not video render timings**. The shorter
case crops real activations to the one-second geometry; it is not a fresh
one-second model capture. The long case uses the full saved H3 activation
set. Because every candidate loses at the actual long shape, no full video
or sampler quality-validation run is justified by this screen.

## Inputs and comparison

All routes receive identical saved BF16 Q/K/V tensors, shape
`[1, 56, 15458, 128]`, with contiguous batch/head/sequence/channel layout.
These were captured in the earlier H3 five-second profiling run, before the
2.14 migration. This experiment replays them through the current attention
implementations; it does not claim that they are newly captured 2.14
activations. The capture hash and current source hash are in the raw results.

The baseline imports the installed `torch_mpp_qcache.py`, checks the exact
Torch build/header hash, and uses the current H3 specialization. Its output
was byte-compared against unmodified PyTorch SDPA at both lengths. Every
comparison passed; no fallback was latched.

The candidate directly binds the Metal shaders from
[qlip-mlx](https://github.com/TheStageAI/qlip-mlx), commit
`142d28635aa65c57e6ee4168500f31033ee8a97a`. It retains the upstream INT8 Q/K,
FP32 softmax state, FP16 probability/value multiplication and FP16 output
rounding, followed by the BF16 conversion required by H3. It uses PyTorch
Metal buffers, with no MLX transfer overhead.

Unlike the earlier Qwen binding, the H3 adaptation reads and writes H3's
native head-first layout directly. It changes only source/destination
indexing and shape specialization, avoiding three input transpose copies
and an output transpose copy. A ragged 257-token / 56-head check compared
it byte for byte with the original sequence-first binding at all five
workgroup sizes and passed. H3 has equal query/key lengths, so this test
does not require the unequal-length extension used for Qwen.

Timings include new scratch/output allocations, Q/K quantization, V
conversion/padding, attention and final BF16 conversion. No persistent
activation cache, reused quantized input, or excluded preparation pass gives
the candidate an advantage. The baseline returns the same contiguous output
layout consumed by H3's existing INT8 output-projection fusion.

## Workgroup sweep and repeatability

Tested 1, 2, 4, 8 and 16 simdgroups per threadgroup. Each candidate has two
warmups and six rotating-order comparisons against both controls; each timed
measurement averages three attention calls and synchronizes MPS. Reported
numbers are medians. At 15,458 tokens, the five Sage configurations took
approximately 185.96 / 184.63 / 186.60 / 188.46 / 191.26 ms respectively.
The paired current controls stayed around 63.6–64.0 ms.

A separate `PYTORCH_MPS_FAST_MATH=1` process improves the best long Sage
case to **158.39 ms**, versus its paired **73.27 ms** stock PyTorch control
(**2.16× slower than stock**, and roughly 2.47× the normal exact path's time).
Its shorter case is still slower too: 6.64 ms versus 2.75 ms stock.

Fast math also changes the dynamically compiled query-cache kernel's output
bytes relative to stock, so this compiler control deliberately uses
**unmodified PyTorch** as the reference. It does not label a fast-math query
cache as our production exact baseline. The failed preliminary assertion is
retained in the diagnostic log; the normal configuration is unchanged.

## Numerical checks

For the complete long attention output, Sage versus the current BF16 path:

- Relative L2 error: **0.007027** (about 0.70%).
- RMSE: **0.005461**.
- Maximum absolute error: **0.21875**.
- Outputs finite at every tested workgroup size.

An independent FP32 comparison evaluates 32 evenly spaced query rows against
**all keys**, keeping the reference allocation bounded. On those rows the
relative error is 0.006913 for Sage and 0.001690 for the current BF16 output.
This measures numerical difference; it is not a claim about visual quality
or denoising stability. The candidate already fails the speed criterion, so
we did not render a full non-bit-exact clip to assess quality.

## Interpretation and records

The longer H3 sequence did not reverse the Qwen result. This particular
kernel is not a useful replacement for our 2.14 H3 attention, even after
removing avoidable layout copies. This does not establish a universal limit
for INT8 attention on Apple Silicon. A design that quantizes both matrix
multiplications, such as a future Kitchen-style Metal port, is a different
experiment with unmeasured performance and quality.

[Raw timings, source hashes and numerical measurements](data/m5-ultra-h3-sage.json)
are retained with this report. The standalone scripts and saved input tensors
remain under `diagnostics/h3-sage` and `diagnostics/ultra-round3` in the
workspace. The production kernels, defaults and existing H3 benchmark result
remain unchanged.
