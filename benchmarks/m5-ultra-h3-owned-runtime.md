# H3 pack-owned runtime parity — M5 Ultra

October 5, 2026. The combined H3 node now calls locally maintained MotionCache,
Spectrum and Veda runtimes through direct internal APIs. They do not import
the pinned vendor packages or instantiate/register their public nodes.

## Matched complete renders

M5 Ultra 80 GPU cores, 256 GB; PyTorch 2.14.0; ComfyUI
`b33e2b55cae074eca5aec96283cceac19aa249ba`. Existing exact model/attention/VAE kernels enabled,
relaxed kernel flag off. Native H3 FL2VA INT8 convrot DiT/text encoder,
FP16 video VAE and FP32 audio VAE. **864×480, 22 frames (~0.92 s), 20 steps,
res_multistep/simple, seed 812764035**. The generated red-paper-boat prompt and
workflow are recorded in the linked data. No personal clips or images were used.

The two-step initial warmup is excluded. Main modes alternate old/new for two
complete repetitions each, with identical noise and conditioning. Combined
modes are one complete screening render per implementation. End-to-end includes
sampling, both decoders and file export; model/text load is warmed.

| Mode | Previous runtime | Our runtime | Identical output bytes | Runs each |
|---|---:|---:|---|---:|
| MotionCache | 13.06 s | 13.03 s | Yes | 2 |
| Spectrum protected | 13.26 s | 13.24 s | Yes | 2 |
| Spectrum fast | 11.54 s | 11.55 s | Yes | 2 |
| Veda Metal (forced on short clip) | 27.00 s | 26.92 s | Yes | 2 |
| Veda + MotionCache | 23.32 s | 23.09 s | Yes | 1 |
| Veda + Spectrum protected | 18.89 s | 18.86 s | Yes | 1 |

Every pair matches raw-byte SHA-256 for conditioning, initial noise, all captured
sampler latent streams, decoded video tensors and decoded audio tensors. Repeats
within each implementation also match. A subsequent Exact render recovers the
previous complete Exact render's bytes after all cache/sparse runs.

These measurements establish preserved behavior and similar runtime speed.
They do **not** establish a material additional end-to-end speedup, equivalence
to dense quality, or new full-clip benchmark numbers. Veda is deliberately forced
on this short clip to exercise its predictor/Metal route; the normal minimum of
8192 tokens remains unchanged because sparse processing is slower here.

## What changed

- MotionCache: direct model API instead of the upstream V3 node wrapper;
  lazy ComfyUI imports, keyword model calls, and sampler option restoration even
  when preparation fails. Two reuse metrics and six update metrics use one host
  readback each. Individual FP32 reductions, Python scalar rate/clamp arithmetic
  and reuse decisions are retained.
- Spectrum: direct measured-preset API; forecasting, replay and compatibility
  hook order retained. Public node registrations and three unused objective-media
  modules excluded. The isolated worker uses a pack-specific namespace.
- Veda: locally maintained checkpoint/plan/scoring/selection code, explicit Apple
  dispatch and a dedicated bounded execution class. Generic backend discovery
  and the generic CUDA allocator/attention loop are excluded. Resident projection
  memory is counted once when cached, then released at cleanup. Existing Metal
  kernels, predictor weights, budgets, self-test and dense fallback are retained.

Existing workflow IDs, UI settings/order, presets, exact/SOL routes and optional
composition remain unchanged. This is not predictor retraining. Original licenses
and copyright notices remain; Spectrum continues to be GPL-3.0-or-later.

## Verification and memory guards

The final GPU parity suite covers detector decisions, forecasts, trained tile
selection, partial/reference tiles, native kernels, FP8 checkpoint dequantization,
projection/cache cleanup, dense fallback, cancellation and option restoration.
The broader CPU regression suite and frontend controls are also checked; exact
counts: **46 GPU/runtime checks passed**, **171 CPU regression checks passed**
(40 GPU-dependent cases skipped in that sandboxed run), and **14 frontend checks
passed**. These suites overlap; their counts are not additive.

The isolated renderer uses MPS high/low watermark ratios 0.4/0.3, a 110 GiB child
RSS stop, 48 GiB minimum available system memory and one-hour timeout. Peak child
RSS: **11.01 GiB**. Live-server allocator flags are unchanged.

[Measurements and tensor hashes](data/m5-ultra-h3-owned-runtime.json).
[Runtime ownership and controls](../docs/h3-acceleration.md#internal-runtime-ownership).
