# YuE2 exact RoPE follow-up on M5 Ultra

2026-10-08 · M5 Ultra, 80 GPU cores, 256 GB unified memory · macOS 27.0 ·
PyTorch 2.14.0 · same ComfyUI/checkpoint/workflows as the
[October 7 first pass](m5-ultra-yue2.md).

The full saved workflow improves **47.95 → 33.48 s**, **30.2% less total time**,
against the immediately preceding exact GPU implementation. Two paired warm
runs regenerate the same 504 ABC tokens and 2080 music tokens, producing the same
83.2-second audio. Original 8192-token score allowance, 120-second maximum,
32 acoustic steps, prompt, seed, sampler and attention remain unchanged. The
120-second maximum does not force a 120-second song.

| Complete workflow | Previous exact GPU | New exact GPU | Less time |
|---|---:|---:|---:|
| 128 ABC tokens / 8 s screen, three warm pairs | 7.300 s | **5.597 s** | 23.3% |
| Full saved workflow, warm pair 1 | 47.955 s | **33.535 s** | 30.1% |
| Full saved workflow, warm pair 2 | 47.942 s | **33.422 s** | 30.3% |
| **Full saved workflow mean** | **47.949 s** | **33.479 s** | **30.2%** |
| Mothersuperior AR + v4 NAR LoRAs, 20 s cap / 32 steps | 30.947 s | **21.181 s** | 31.6% |

Score plus music token generation improves **41.11 → 26.67 s**, about **35%**.
Acoustic sampling remains about 6.07 s. This pass adds no acoustic or VAE change.
Complete timings include allocation, layout work, sampling, decoding and export;
models are resident after warmup. Short comparisons discard an old/new warmup,
and full comparisons run alternating old/new in the same isolated process.

## What changed

YuE2's single-token decode repeatedly launches small eager RoPE operations for
Q and K at each of 28 transformer layers. A dedicated BF16 Metal kernel performs
the same rotation in one launch per layer. It keeps the native **separately
rounded FP32 products and addition**, followed by the original BF16 conversion.
An FMA prototype changed output bytes and was rejected. The unchanged frequency
matrix is built once per token and shared across the 28 layers.

The original floating matrix multiplies, PyTorch attention, CPU multinomial
sampling, full 184704-entry probability/RNG vector, progress and EOS logic all
remain. Converted weights retain their previous phase-local cache. The INT8
checkpoint still computes AR matrix operations through the same BF16 floating
path; this does not introduce a new quantization mode.

The route runs only inside YuE2 AR generation, on BF16 single-token inputs with
16 Q / 8 KV heads and D=128. It is guarded by the validated M5 Ultra, exact Torch
build and ComfyUI Llama / Kitchen eager source hashes. Noncontiguous inputs,
other shapes/dtypes, batched guidance and a different Kitchen backend retain
the preceding implementation. Prefill and acoustic conditioning retain their
original paths. Compilation failure latches the fusion off and falls back.

The per-phase frequency/matrix reference is released with the generation scope,
including interruption. No persistent activation cache or CPU thread setting is
introduced. Peak MPS driver reservation was **13.67 GiB**, with **8.67 GiB** active
tensors in the final base-model validation; retained cast-weight copies remain
**4.03 GiB per phase**.

## Exactness checks

- All three short and two full final-patch pairs preserve score/music IDs,
  conditioning, latents, decoded waveform and exported FLAC bytes.
- Full outputs also match the frozen October 7 reference. FLAC SHA-256 remains
  `01ecd6a3e544e4aa08483905791af6a3d1c2c7d520262a3e1b05ff3d0d093e71`.
- Both Mothersuperior LoRAs retain all loaded patches, and two warm paired
  20-second runs preserve tokens, tensors and FLAC bytes. They also match the
  frozen previous LoRA reference; full 120-second LoRA generation was not timed.
- Three additional short public styles/seeds, autoregressive CFG 1.01 and
  temperature-zero argmax preserve tokens and all captured tensors.
- 3,136 real-model RoPE calls compare Q/K output bytes against native MPS.
- Synthetic checks cover storage offsets, position 0–24575 and every BF16 bit
  encoding, including nonfinite values: zero byte differences.
- 27 focused CPU/MPS tests cover parity, disabled paths, changed runtime,
  unsupported layout/backend, failure fallback and scope cleanup.

These checks compare with our previous **GPU BF16** stack. Stock CPU FP32 and
GPU BF16 were already numerically different before this follow-up. The original
stock short baseline remains 124.049 s; the new short result is 5.597 s, 95.5%
less time. The full stock CPU workflow was not timed.

## Candidates screened and rejected

The isolated prototypes used the same 128-token / 8-second workflow and three
warm repeats. Kernel timing alone did not determine acceptance.

| Prototype | Reference E2E | Candidate E2E | Decision |
|---|---:|---:|---|
| Read sampled token on CPU before uploading | 7.398 s | 7.414 s | No practical gain |
| Copy CPU token directly to existing MPS decode buffer | 7.398 s | 7.462 s | No practical gain |
| BF16 SiLU lookup + SwiGLU fusion | 7.398 s | 7.416 s | No practical gain |
| Reuse recent-token IDs on GPU | 7.461 s | 7.472 s | No practical gain |
| Reuse matrix, retain eager RoPE | 7.461 s | 7.154 s | Included in fused route |
| Reuse matrix + fused RoPE | 7.461 s | 5.606 s | Qualified in production above |

Reducing global CPU threads preserved sampled IDs/RNG state and gave a small
screening gain. Added to fused RoPE, the full prototype improved only
33.408 → 33.116 s, under 1%. It is not shipped: globally changing thread settings
adds scope and concurrency complexity for little practical benefit. CPU threads
and RNG stay unchanged. The CPU profile's large `torch.tensor` time includes
waiting for queued GPU work; it is not purely Python tensor-construction cost.

## Controls and reproduction

`ASFP8_YUE2_ROPE=0` disables only the new rotation/matrix-reuse route.
`ASFP8_YUE2_FAST=0` disables all YuE2 additions. Both default to enabled; changing
startup flags requires a ComfyUI restart. Unsupported runtimes fall back rather
than applying an unqualified exactness claim.

No workflow node or model conversion is needed. Only public text prompts and
synthetic tensors were used; no user recordings or private media were opened.
The isolated server used an idle main-server guard, 100 GiB process/driver cutoff
and 48 GiB available-memory floor, and was stopped after each test group.

[Machine-readable paired runs, hashes, guards and screening results](data/m5-ultra-yue2-followup.json)
include the precise source manifest. The original controls and first-pass
baseline remain in the [earlier report](m5-ultra-yue2.md).
