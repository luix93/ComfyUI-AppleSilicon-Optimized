# MOSS SFT: matrix-output SiTU fusion

October 18, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f569d268b80076e5f44126f87cdf835869` after v0.41.3 ·
Comfy Kitchen 0.2.38.

## What changed

This follows the dense-attention experiment and targets the FFN matrix path.
MOSS already combines the hidden and gate projections in one INT8 matrix
multiply. The previous path writes `[M,16384]` BF16, then reads that tensor to
apply SiTU and write `[M,8192]`. The new kernel pairs hidden/gate columns within
its register accumulators, rounds both projections to BF16, applies the same
PyTorch-generated SiTU lookup tables, and writes the activated tensor directly.

The INT32 accumulation, scale multiplication order and BF16 intermediate
rounding are unchanged. Original weights are read directly with paired-column
addressing. ConvRot, rowwise INT8 quantization, FC2, attention, CFG and sampler
remain the preceding exact implementation. There is no approximation or SOL.
At 6,167 rows this eliminates a 192.7 MiB intermediate's write and read per
call; that is logical traffic avoided, not a measured DRAM-counter reduction.

The first prototype repacked weights once. Its direct-weight successor has
similar operator performance and avoids approximately 1.97 GiB of persistent
packed weights across 42 layers. The guarded integration does not change the
ordinary Linear output shape or pass specially marked tensors between methods.
It uses a reviewed MOSS block adapter and preserves unsupported paths.

## Measurements

SFT INT8 ConvRot, Qwen3.5-9B INT8 ConvRot, Flux.2 small VAE, ER-SDE/simple,
CFG 2.5, 25 steps, seed `856771198829162`. Same synthetic theatre-marquee
prompt as the dense-attention benchmark; prompt enhancement and SLG off.
Every run re-encodes the prompt, samples, decodes and saves a PNG. Tensor capture
and hashing are outside the timed interval. Sampling includes all input
rotation, quantization, allocations and layout costs. Models remain resident;
separate one-step warmups precede paired full renders, with exact controls
before and after candidates.

These are optimized-to-optimized comparisons, not stock baselines. Means of two
warm full renders per mode:

| Resolution | Exact E2E | Fused E2E | E2E reduction | Exact sampling | Fused sampling |
|---|---:|---:|---:|---:|---:|
| 1024×1024 | 21.996 s | 21.627 s | 0.370 s / 1.68% | 20.300 s | 19.785 s |
| 1248×1248 | 33.848 s | 33.440 s | 0.409 s / 1.21% | 31.987 s | 31.440 s |

Sampling reductions are 2.54% and 1.71% respectively. The 1248 candidate runs
span 33.159–33.720 s against exact controls of 33.718–33.978 s: the E2E ranges
slightly overlap. This is a small, variable benefit, not a strong new speed
claim. Both fused sampling runs are below both exact sampling controls. The
option stays disabled by default pending broader evidence. No guard tripped.
Peak sampled Metal allocation across these production suites was 29.89 GiB.

The direct-weight prototype at 1248² measured 33.450 / 33.780 s exact and
33.038 / 33.034 s fused, with sampling means 31.769 / 31.077 s. The production
adapter above is measured separately; do not average these suites together.

Operator timings include FC1 plus SiTU. On captured 6,167-token inputs the
existing path takes about 3.48 ms and the fused kernel 3.07 ms (about 12% less
time). The complete-render improvement is much smaller because this is only
one part of every model block.

## Screened and rejected alternatives

- Six layouts were screened: packed-weight 128×128, 64×128, 64×64 and 128×256,
  and direct-weight 128×128 / 64×128. All matched captured activated outputs
  byte-for-byte. The 128×128 direct-weight version avoids weight duplication
  and was selected. The 128×256 version took about 13.4–13.7 ms and was rejected.
- A second prototype fused FC1, SiTU **and ConvRot** using a shared-memory
  regular-Hadamard transform. It removed another pass but changed the floating
  summation order: 351 / 400 rotated BF16 values and 4 / 14 quantized bytes
  differed on two real captures. It took about 14.9 / 14.6 ms versus
  3.78 / 3.59 ms for the existing sequence. Rejected for both speed and exactness;
  it was not run through full denoising or added to the production dispatcher.
- These measurements do not identify register pressure or memory stalls from
  hardware counters. Larger fused kernels are not assumed to be faster.

## Correctness and deployment

Intermediate FC1/SiTU outputs match byte-for-byte on first/last positive-branch
and last negative-branch captures. The three-step verification checks all 252
fused calls; a direct-weight verification checks another 84 calls. Complete
sampler latents and decoded float frames match their exact reference, as do
conditioning and initial noise. Production counters confirm 2,100 fused calls
per 25-step render. Restoring the previous path restores the same output bytes.

The option is **disabled by default**: `ASFP8_MOSS_FFN_EPILOGUE=1` enables it on
the reviewed MOSS stack. It requires the existing MOSS/SiTU/FFN-dispatch switches,
BF16 batch one, 4,096–6,656 tokens and the reviewed ConvRot INT8 weight shapes.
Hooks, LoRA weight functions/runtime adapters, forced precision, incompatible
weights, unsupported core versions and compilation failures retain the existing
path. The earlier approximate `ASFP8_MOSS_SITU_QUANT` experiment takes precedence
when explicitly enabled. No installed ComfyUI files or user workflows were changed.

The focused MOSS regression/fusion suite passes **46 tests**. A final one-step
smoke run confirms 84 fused calls and byte-identical tensors after adding the
block/Linear source-identity guards. Final checks cover default/explicit disable, inactive scope, compile-failure latching,
custom-layer behavior and a GPU matrix-output comparison with an unaligned row
tail and different column scales. Small gains do not justify a claim that the
previous Blackwell/Spark performance target has been reached.

[Raw timings, tensor hashes and source hashes](data/m5-ultra-moss-ffn-epilogue.json).
