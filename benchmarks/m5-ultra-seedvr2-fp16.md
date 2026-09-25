# SeedVR2 7B FP16 loader compatibility

2026-09-24, M5 Ultra, macOS 27, PyTorch 2.13.0, ComfyUI 0.37.0.

## Failure and repair

`UNETLoader` rejected `seedvr2_ema_7b_fp16.safetensors` with **Could not detect
model type**. The 16,479,334,424-byte file has 1,128 FP16 tensors and the
recognized 36-layer, separate-stream 7B architecture, but lacks the two
conditioning keys required by ComfyUI's `SeedVR2.matches()`.

The macOS `seedvr2_loader` patch supplies the original fixed conditioning
before model-class matching, only for that recognized architecture and only
when both conditioning keys are absent. Existing weights, embedded
conditioning, other architectures and checkpoint files are unchanged.
`ASFP8_SEEDVR2_LOADER=0` disables the repair independently of kernel flags.

The bundled 1.25 MB asset contains ByteDance's original BF16 `pos_emb.pt` and
`neg_emb.pt`, with only container/key-name conversion. Both also match the
conditioning in the installed normal and sharp INT8 ConvRot checkpoints.
[Provenance, hashes and license](../_patches/seedvr2_assets/README.md).

## Validation

- 12 focused tests pass: official tensor-byte hashes, prefixed/unprefixed
  loading, weight-object preservation, existing and partial conditioning,
  other architectures, config-only probes, upstream exception propagation,
  damaged assets and the disable switch.
- The small complete 3x workflow produces identical encoded latents, sampled
  latents and final frame bytes on first/warm runs.
- An independent reference disables the compatibility hook and embeds the
  original upstream PT tensors at the checkpoint-read boundary, emulating a
  Comfy-ready checkpoint. Its complete small-workflow hashes match the repair.
- Two 2400×1600 → 7200×4800 runs have identical encoded, sampled and final
  frame hashes. Outputs are finite; the rendered preview has no obvious
  corruption. This is a loading-compatibility check, not a quality comparison
  between FP16 and INT8.

Actual loaded storage dtype is **FP16** and ComfyUI's chosen compute dtype is
**BF16**. Existing BF16 SeedVR2 fusions apply; dense projections use MPS rather
than the INT8 kernels. No FP16 → INT8 conversion is performed.

## Full workflow timing

Same established photograph, seed `959948902156062`, 3x Lanczos resize,
one Euler step, CFG 1, tiled FP16 VAE (512/128), no color correction.
All accepted native optimizations enabled. Timing includes resizing,
preprocessing, VAE encode, sampling, VAE decode and PNG saving. The first
run also includes model loading; process startup/JIT and post-run hash
verification are excluded. Second run reuses loaded models.

| Stage | First FP16 run | Warm FP16 run |
|---|---:|---:|
| Encode | 10.617 s | 10.513 s |
| Sample, including synchronization | 17.332 s | 16.276 s |
| Decode | 25.148 s | 25.142 s |
| Complete workflow, including PNG | **57.449 s** | **54.223 s** |

The prior warm INT8 result was 50.984 s total / 13.036 s sampling. FP16 takes
about **3.24 s more overall** in this comparison (6.4%). These are recorded
runs, not a new alternating FP16/INT8 performance study.

Maximum observed Metal allocation is **40.98 GiB**, sampled at stage
boundaries; it is not a continuous peak or total system RAM. The external
watchdog observed up to 32.62 GiB process RSS; the two memory measures must
not be added as independent allocations on unified memory. No watchdog
limit was reached. The 18,720×12,480 FP16 case was not tested here.

Raw timing and hashes: [JSON](data/m5-ultra-seedvr2-fp16.json).
