# MOSS-Image: exact incremental acceleration

October 18, 2026 · M5 Ultra 80 GPU / 256 GB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f5` · Comfy Kitchen 0.2.38.

## Workflow and timing boundaries

Both downloaded MOSS-Image SFT and Base INT8 ConvRot models were tested with
Qwen3.5-9B INT8 ConvRot text encoding and the BF16 Flux.2 small VAE. The W4A8
encoder was not used for these comparisons. Both graphs use 1024×1024,
25 steps, CFG 2.5, ER-SDE/simple and their saved fixed seeds. Base retains
SLG Simple, single layer 22, start 0 / end 0.5; SFT has it disabled.

Prompt enhancement was disabled in diagnostic copies as authorized. Every
measured render re-encodes both prompts, samples, decodes and saves a PNG.
Warm runs retain loaded models, but do not reuse encoded conditioning or output
images. Tensor capture/hashing happens outside the timed execution. Original
saved workflows were not modified.

## Complete workflow results

| Model | Stock warm | Previous pack warm | New first / warm | Stock → new |
|---|---:|---:|---:|---:|
| SFT | 68.472 s | 33.876 s | **24.143 / 22.144 s** | **3.09×** |
| Base | 69.163 s | 34.311 s | **24.568 / 22.442 s** | **3.08×** |

Fresh-process and warm timings above use the final installed files and prepared
workflow copies. Full sampling-loop timings:

| Model | Previous sampling | New sampling | Time reduction |
|---|---:|---:|---:|
| SFT | 32.163 s | **20.378 s** | **36.6%** |
| Base | 32.536 s | **20.664 s** | **36.5%** |

The new changes preserve the previous pack's complete outputs byte for byte.
The full pack does not preserve stock bytes: existing CPU-to-MPS encoder
placement and W8A8 linear execution already alter that reference. The stock
column measures actual completed runs, not an extrapolation.

## What changed

1. **SiTU activation:** PyTorch evaluates the soft-capped hidden/gate functions
   for all BF16 input codes. One Metal pass looks up those exact intermediate
   values and multiplies them with the same BF16 rounding. No polynomial or
   approximate activation was introduced.
2. **Q/K RoPE:** keep native RMS reductions and norm weights. Fuse split-half
   rotary products, addition and output allocation with separate FP32 operations
   and explicit BF16 rounding; do not contract them into FMA.
3. **Modulation:** keep native RMS normalization and norm-weight folding; fuse
   only the following elementwise sequence, retaining each intermediate round.
4. **Prepared FFN output weights:** reuse owned copies instead of repeating
   preparation every model call. The floating matrix multiplication is unchanged.
5. **Text/position reuse:** cache timestep-independent features only when input
   tensors compare equal. Recompute timestep embeddings and all diffusion blocks.
6. **Exact dense attention:** reuse the already validated PyTorch shader adapter,
   hoisting query loads out of the key loop. QK, PV, softmax and reduction order
   are retained. No sparse, quantized or approximate attention was added.

## Profiling and screening

A synchronized two-step SFT profile measured 3.329 s inside main blocks,
including 1.731 s attributed to weight preparation, 1.107 s in the activated
FFN output path, 0.811 s in attention (0.495 s in SDPA), and 0.415 s in
normalization/modulation. These are **overlapping inclusive measurements**;
they must not be added or treated as ordinary render times.

Three-step screens, complete sampling loop, seconds:

| Candidate | Time | Byte comparison |
|---|---:|---|
| Previous pack, repeated baseline | 3.914–3.967 | Reference |
| SiTU alone | 3.394 | Identical |
| RoPE alone | 3.643 | Identical |
| Modulation alone | 3.699 | Identical |
| Prepared-weight cache alone | 3.575 | Identical |
| All three elementwise fusions + text/position reuse | 2.916–2.923 | Identical |
| Above + prepared-weight cache | 2.595–2.606 | Identical |
| Above + exact query reuse | 2.500–2.526 | Identical |

Single-candidate numbers are screening observations, not independent full-workflow
claims. Combined improvements were subsequently confirmed at the full 25 steps
for both models, alternating old/new/old/new.

Alternative exact INT8 matrix tiles were screened at token counts 4,119,
4,179 and 4,311 for QKV, FFN input and output projections. All tested GEMM
outputs matched. A 128-row layout with different scheduling showed small
isolated gains, but the complete eight-step screen measured **6.565–6.600 s**
existing versus **6.492–6.505 s** candidate sampling (about 1–1.5%). Warm
end-to-end results were 8.118 s existing and 8.227–8.235 s candidate, with
encoding/decode variation larger than the small sampling gain. There was no
repeatable complete-workflow win. **Not enabled in production**; the existing
matrix kernels remain unchanged. No additional approximation was introduced.

VAE decode is already approximately 0.15 s in the full native workflow. It was
measured but not rewritten: it cannot explain or materially remove the tens of
seconds spent in sampling.

## Strict stock-byte check

Enable only `moss_fast` through the existing diagnostic selector
`ASFP8_ENABLE_ONLY=moss_fast`. This retains stock CPU text encoding and stock
linear execution. Both models passed two full renders with **every captured
stock tensor byte-identical**, including final decoded frames:

| Model | Stock e2e | MOSS-only first / warm e2e | Stock / MOSS-only warm sampling |
|---|---:|---:|---:|
| SFT | 68.472 s | 60.588 / **58.969 s** | 47.101 / **38.172 s** |
| Base | 69.163 s | 61.276 / **60.334 s** | 47.625 / **38.567 s** |

These are different configurations from the full-pack table. Stock prompt
encoding alone costs about 21 s versus about 1.5–1.6 s on the pack's existing
MPS route. That placement change is not part of the new exact work. The
prepared floating-weight cache is unused on the stock quantized return path;
the other exact MOSS paths still run. The diagnostic selector disables other
patch modules globally and is not a per-model production preset.

## Validation and memory

- Full SFT and Base comparisons capture conditioning, input latents, initial
  noise, complete sampler latents and decoded FP32 frame tensors. Comparisons
  with the preceding pack match SHA-256 byte hashes, including repeats.
- Operation tests cover every finite BF16 activation code with four hidden/gate
  pairings; random modulation including batch two; strided split-half RoPE and
  tails. Lifecycle, source/shape fallback and weight ownership/invalidation tests
  cover cache and dispatch behavior.
- The SiTU/RoPE/modulation/query counters confirm the paths actually execute.
  The SFT full run records 2,100 SiTU calls, 2,100 accelerated attention calls,
  2,058 weight-cache hits and 42 preparations. Attention stays dense.
- Sampling-local prepared weights consume 2,113,929,216 bytes (1.97 GiB).
  Positional/text caches remain below 32 MiB and are released with the weight
  cache on success, interruption or failure.
- Final generated images were visually inspected for basic coherence. Since
  pixels match the previous optimized outputs exactly, this is a functional
  sanity check, not a new stock-versus-native quality assessment.
- Serial isolated test servers used a 110 GiB RSS ceiling, 48 GiB available-RAM
  floor and bounded Metal watermarks. The interactive server was not restarted
  or interrupted. No private reference media or SATA files were accessed.

The final installed workflows each ran twice and matched the original
previous-pack references. All active-path counters were checked, and all
**35 operation/lifecycle/shared-cache tests passed**. Final native runs observed
up to **30.26 GiB** Metal driver allocation; no memory safeguard fired.
The new SiTU/RoPE/modulation kernels and query path were active, not silently
falling back. Reproduce the operation checks with the tested ComfyUI Python:
`python -m pytest tests/test_moss_fast.py tests/test_flux3_fast.py -q` from this
pack's directory (pytest must be available in that test environment).

These are two prompt/seed fixtures at 1 MP, not proof for every workflow,
precision, device or third-party hook. New routes fail closed on unreviewed
source/builds. This is a focused MOSS nightly test, not a complete regression
certification of every model on the newer core/Kitchen stack.

[Controls and numerical contract](../docs/moss-image.md) ·
[Machine-readable results](data/m5-ultra-moss-image.json).
