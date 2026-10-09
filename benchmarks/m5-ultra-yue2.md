# YuE2 on M5 Ultra: autoregressive weight reuse and exact acoustic attention

2026-10-07 · M5 Ultra, 80 GPU cores, 256 GB unified memory · macOS 27.0 ·
PyTorch 2.14.0 · ComfyUI `b33e2b55cae074eca5aec96283cceac19aa249ba`.

**October 8 follow-up:** the latest exact implementation averages **33.48 s**.
See the [paired RoPE follow-up benchmarks](m5-ultra-yue2-followup.md) for the new
measurements, controls and rejected candidates. The measurements below are the
October 7 first pass, retained as history.

The saved `audio_yue2_text2music.json` averaged **48.46 s**, compared with
**354.18 s** on the previous optimized GPU stack: **7.31× faster, 86.3% less
total time**. Both warm repeats produce identical score tokens, music tokens,
conditioning, sampler latents, decoded waveform and saved FLAC bytes. The new
routes are enabled by default and can be disabled independently.

## What was slow

YuE2 first generates an ABC score, then music semantic tokens, then acoustic
latents and finally audio. This is substantially different from timing only a
diffusion sampler. Autoregressive score/music generation dominated this workflow.

The original application log already had the encoder on MPS, at approximately
7.5 tokens/s. Its interrupted score progress was 362/8192 tokens; the long ETA
assumed the entire score-token allowance would be consumed. It was not a measured
end-to-end time for a completed 120-second song.

ComfyUI forces weight casts for this encoder. Its quantized weights are logically
FP16, while the existing MPS forward uses BF16. That path repeatedly dequantizes,
applies ConvRot and converts weights during every generated token. Our generic
INT8 linear kernels do not run through this forced-cast autoregressive seam.
Caching the exact floating results of ComfyUI's own cast operation for each
generation phase removes that repeated work without changing its matrix math.

A separate stock-only test revealed another cost: unmodified ComfyUI selected
CPU FP32 for the encoder. Our existing text-encoder device patch already resolves
that placement. It was not the cause of the slow MPS run reported by the user.

## Matched saved workflow

Checkpoint: `yue2_3b_int8_convrot.safetensors` (3,960,938,800 bytes), loaded from
the shared local model directory. Seed 0, blank lyrics, saved cinematic soundtrack
style. No reference images, audio or video were used. The API graph and source
hashes are included in the [machine-readable results](data/m5-ultra-yue2.json).

- ABC: full mode, original 8192-token allowance, temperature 0.7, top-p 0.9,
  top-k 30, repetition penalty 1.005, window 100.
- Semantic music: full mode, maximum duration 120 s, temperature 1.0, top-p 0.95,
  top-k 100, repetition penalty 1.2, CFG 1.
- Acoustic sampler: 32 steps, `dpm_2`, `sgm_uniform`, CFG 1, denoise 1;
  63 model evaluations. Same PyTorch attention backend in both GPU stacks.
- Original audio VAE and FLAC export retained.

**120 seconds is a maximum, not a fixed generated duration.** For this prompt and
seed, generation naturally finishes at 504 ABC tokens and 2080 semantic tokens,
producing **83.2 seconds of audio** in both configurations. We did not shorten
the original token allowance or duration setting to obtain the full-workflow gain.
An exactly 120-second generated output has not been separately benchmarked.

| Full saved workflow | Previous GPU stack | New exact YuE2 routes |
|---|---:|---:|
| Warm repeat 1, complete execution | 353.302 s | 47.930 s |
| Warm repeat 2, complete execution | 355.063 s | 48.991 s |
| **Mean complete execution** | **354.183 s** | **48.461 s** |
| ABC generation, mean | 67.400 s | 8.199 s |
| Semantic token generation, mean | 279.459 s | 33.354 s |
| Conditioning construction, mean | 0.281 s | 0.312 s |
| Acoustic sampling, mean | 6.349 s | 6.112 s |
| Audio VAE, mean | 0.575 s | 0.355 s |
| FLAC export, mean | 0.107 s | 0.118 s |

Phase timings synchronize MPS and include their allocations/casts. Complete
execution includes graph overhead. Models are resident after warmup, but score,
semantic tokens, conditioning, sampling, decode and export are recomputed on
every measured run. The previous decoder's first long run includes shape-specific
compilation; there is no new VAE optimization and its timing difference is not
claimed as a decoder speedup.

## Stock PyTorch 2.14 screen

The stock baseline has this pack absent. All three configurations rerun the
entire graph with an identical controlled **128-token ABC / 8-second music**
screen, original 32 acoustic steps and seed 0. Each has a discarded warmup plus
two warm repeats. The intentionally truncated score is a speed screen, not a
full-score quality example or a substitute for the full saved workflow above.

| Controlled short screen | Warm repeats | Mean |
|---|---|---:|
| Stock 2.14, CPU FP32 encoder | 123.571 / 124.526 s | **124.049 s** |
| Previous optimized stack, MPS BF16 encoder | 44.871 / 45.034 s | **44.953 s** |
| New exact routes, MPS BF16 encoder | 7.451 / 7.319 s | **7.385 s** |

Stock → new is **16.80× faster**, 94.0% less time. Previous GPU → new is
**6.09× faster** in this short screen. The CPU FP32 stock and GPU BF16 stacks
are **not bit-exact to each other**. The new patches are exact against the
previous GPU stack; the full stock CPU workflow was not timed.

## Accepted changes

1. **Phase-local cast-weight cache.** Retains owned copies of ordinary floating
   weights returned by the original cast/offload function. Never caches a streamed
   buffer lease or quantized wrapper. Keys include module/parameter identity,
   tensor revisions, device and requested dtypes. Weight hooks, low-VRAM hooks
   and training bypass it. Copies are released in `finally`, including on error
   or interruption. The checkpoint remains stored as INT8; autoregressive
   matrix computation remains the same dequantized BF16 path as before.
2. **Vocabulary projection slices.** Computes only the contiguous permitted
   score/music vocabulary rows with the existing floating `linear`. Preserves
   the full 184704-entry probability vector and CPU multinomial RNG behavior.
   Permitted logits match exactly; discarded rows are zero before the original
   mask makes them negative infinity. Unmasked unused logits deliberately differ.
   This route is restricted to the validated M5 Ultra / exact PyTorch 2.14 build,
   BF16 single-row inputs and the known vocabulary layout.
3. **Cheap inference-tensor revision checks.** Uses `torch.is_inference` instead
   of repeatedly raising exceptions to read nonexistent inference version counters;
   avoids signature-binding objects in the common cast call. No tensor arithmetic
   changes.
4. **Scoped acoustic query reuse and output layout.** Extends our existing
   source-hash-checked PyTorch MPP query-hoisting kernel to YuE2 GQA
   (16 query / 8 KV heads, D=128; 2048–4096 query tokens, up to 8192 KV tokens).
   Its reductions, arithmetic and tile geometry stay unchanged. Direct BSHD
   output removes an adjacent layout conversion. Unsupported shapes fall back
   to the existing attention implementation; autoregressive attention is untouched.

Frozen full-workflow conditioning isolates the acoustic test from token
generation. Three warm sampler repeats average **6.3656 → 6.1712 s**, a
**3.1%** improvement including allocations, quantization and layout work.
Complete sampler latents and decoded audio match bytes on every repeat. This
small gain is included in the final complete-workflow results, not added to them.

## Validation and memory

- Two complete original-workflow repeats before and after; all captured token
  lists, score text, conditioning, latents and decoded waveforms match bytes.
- All four corresponding FLAC exports match SHA-256:
  see the complete hash in the JSON results.
- Three additional short styles/seeds reproduce the previous GPU stack's token
  sequences, conditioning, latents and decoded audio exactly.
- 37 GPU tests passed, including scoped strided/offset YuE2 GQA cases and existing
  H3, Krea2, Qwen, LTX and Wan attention regression cases. Separate CPU/cache
  tests cover cleanup, mutation invalidation, limits and disabled paths.
- Actual retained cast copies: **4.03 GiB per phase**. The final guarded process
  peaked at **13.20 GiB MPS driver memory** (8.67 GiB active tensors). Cache scope
  ends between generation phases; allocator reservation can remain after release.
- Cache default limit: min(8 GiB, 20% of MPS recommended memory), with an additional
  driver headroom guard. A missed cache entry retains the original cast path.
- Isolated server, idle main-server check, 100 GiB process/driver cutoffs and
  48 GiB available system-memory floor. All isolated servers were shut down.

There is no visual/audio quality tradeoff in the accepted routes: final audio
bytes match the previous GPU implementation. This does not claim equivalence
between stock CPU FP32 and existing GPU BF16 arithmetic.

## Controls

| Environment variable | Default | Effect |
|---|---|---|
| `ASFP8_YUE2_FAST` | `1` | Master switch; `0` restores the preceding YuE2 paths |
| `ASFP8_YUE2_WEIGHT_CACHE` | `1` | Phase-local cast-weight reuse |
| `ASFP8_YUE2_CACHE_GIB` | `8` | Maximum cache size, also capped by memory guards; `0` disables storage |
| `ASFP8_YUE2_VOCAB_HEAD` | `1` | Guarded permitted-vocabulary projection |
| `ASFP8_YUE2_ATTN` | `1` | Guarded acoustic query-cache/layout route |
| `ASFP8_YUE2_VERBOSE` | `0` | Optional phase cache counters |

Acoustic acceleration also requires the existing `ASFP8_ATTN_BACKEND=pytorch`
and `ASFP8_MPP_QCACHE=1`. Restart after changing startup settings. No additional
workflow node, model conversion or third-party node package is required.

## Investigated but not shipped

A prefix/conditioning reuse experiment preserved bytes but improved acoustic
sampling only about 2.2%, required an extra buffer and a copied model forward,
and overlapped the smaller accepted attention gain. Its maintenance/memory cost
did not justify installation. A first vocabulary-head prototype switched to
native W8A8 and failed permitted-logit byte parity; it was rejected. Earlier
attention hook prototypes with zero active kernel calls are excluded from claims.

Further AR W8A8 kernels, bucketed attention or an MLX backend could improve
throughput, but would need separate numerical and complete-audio qualification.
They are not active in these results.

Primary research references: the [YuE repository](https://github.com/multimodal-art-projection/YuE)
and [paper](https://arxiv.org/abs/2609.33757), plus
[YuE-Studio's Apple Silicon implementation](https://github.com/tonywestonuk/YuE-Studio)
and [vocabulary projection discussion/code](https://github.com/tonywestonuk/YuE-Studio/blob/main/src/yue2/batched.py).
No code from those repositories was vendored into this patch.
