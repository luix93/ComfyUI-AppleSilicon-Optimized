# M5 Ultra: further relaxed-kernel experiments and an optional H3 preset

Date: 2026-09-23. M5 Ultra, 80 GPU cores, 256 GB; macOS 27,
PyTorch 2.13.0, ComfyUI `b33e2b55` / 0.37.0, Comfy Kitchen 0.2.35.
Exact sprints one and two remain the baseline (`723573f`). The previous relaxed
round is preserved at local commit `38a387b` and in the sprint-three diagnostic
directory. Default operation stays exact. New routes are startup options;
`ASFP8_NATIVE=1` alone does not select numerical relaxation.

**Outcome:** the optional preset reduces the matched full workflow from
**161.13 to 153.35 seconds E2E**,
saving **7.78 s / 4.83%**.
It saves a further **2.58 seconds**
over the previous tail-plus-dense combination. Sampling is **7.11 s/step**;
the 7.0-second target is still unmet. The sampling changes remain experimental
because same-seed composition, motion and joint audio latents change.

## Selection and scope

`ASFP8_MINIMAX_RELAXED=1` selects tuned long-H3 attention plus the three VAE
features below. It is an experimental preset. The individual
settings below override it, including explicit `0` or `exact`:

| Setting | Choices / effect |
|---|---|
| `ASFP8_MINIMAX_ATTN_MODE` | `exact`, original `tail`, or `tuned` |
| `ASFP8_MINIMAX_VAE_DENSE_ATTENTION` | `1` enables dense FP16 VAE attention |
| `ASFP8_MINIMAX_VAE_ATTN_TUNING` | `1` tunes tiles within dense VAE attention |
| `ASFP8_MINIMAX_VAE_QKV` | `1` enables the fused FP16 QKV projection |

Native prerequisites are unchanged. Restart ComfyUI after changing settings.
Removing the preset and all individual relaxed overrides restores the exact
stack. Dense-only and the original tail experiment remain individually usable.
The preset excludes the slower software-exponential, delayed-rescaling and
earlier GEMM experiments. It does not enable cache/sparsity/token-reduction
nodes or alter steps, VAE tiling, temporal chunking or overlap.

Sampling selection patches only MiniMax's attention binding. It accepts
unmasked, inference-only BF16 calls with 56 heads, dimension 128 and at least
8,192 tokens. Explicit workflow attention backends and overrides take
precedence. Unsupported calls retain their original route. Optional failures
are logged and latch back to the existing implementation for that process.
Owned attention-input containers retain their fallback contract.

The VAE projection retains FP16 inputs and FP32 accumulation. It fuses the
projection's bias/FP16 rounding, Q/K normalization, partial RoPE and output
layout. The existing pre-normalization, attention, output projection and
feed-forward path remain separately controlled. Lazy/custom layers, hooks and
runtime weight/bias transforms retain the existing projection path.

## Experiments beyond sprint three

All GPU jobs execute serially. Kernel timings include padding, output
allocation and layout costs. Whole decoder timings repeatedly decode identical
saved latents; sampler comparisons capture and byte-check initial noise.
Cold loading/compilation runs are excluded and timing order alternates.

### Attention scheduling and tail combinations

The first screen covers 96 shape/configuration combinations: 16/32 query rows,
16/8/4 SIMD groups, 32/64/96/128 key columns, one/two score blocks, and sequence
lengths 8,613/15,458. Doubling query rows is substantially slower. A ten-round
follow-up finds only a 1.3–1.7% full-shape gain from the best schedules alone;
the existing tail variant remains faster than either schedule alone.

Combining the tail with eight SIMD groups and one 96-column score block gives
a further **2.35% captured full-shape attention improvement over the original
tail**, across eight alternating rounds. This changes floating-point reduction
order. The source is derived with guarded transformations into a separate
extension; the exact extension's source and defaults are unchanged.

The 22-frame sampler does not dispatch this kernel: its sequence is below the
router's 4,096-token threshold. Those screening runs are explicitly excluded
from attention-gain and attention-quality claims. The tests therefore advance
to 124 frames at 640×352 before full-resolution confirmation. Each completed
20-step medium run records 1,000 calls to the intended attention entry point.

| Medium scene / seed | Exact sampling | Tuned sampling | Video latent relative L2 |
|---|---:|---:|---:|
| Boat / 812764035 | 67.2014 s | 65.9564 s | 25.74% |
| Scarf / 20260924 | 66.5460 s | 64.5670 s | 43.37% |
| Butterfly / 20260925 | 66.5840 s | 64.5010 s | 34.39% |

These one-pair screens show gains in all three scenes; the full-resolution
alternating repeats provide the repeatability check. Noise is identical within
each comparison. Video and audio latents change.

An additional FP16-operand attention screen includes converting BF16 Q/K/V to
FP16 and converting the output back to BF16. At full shape, the base version
is 1.98% slower than exact, and tail/tuned versions save only 1.42%/3.15%
against exact. They do not beat the stronger BF16 candidate. Outputs remain
finite on the captured inputs, with relative L2 around 0.168%; these variants
are not advanced or included in the preset.

### Normalization/modulation into rowwise INT8 quantization

The sprint-two prototype is retested without its reference-feeding verification
mode. Three prompts/seeds, 22 frames at 864×480, 20 steps each, use warmed
exact/candidate/candidate/exact order.

| Scene | Exact mean sampling (s) | Candidate mean (s) | Video latent relative L2 |
|---|---:|---:|---:|
| Boat | 19.13853 | 19.11322 | 7.41% |
| Scarf | 19.60307 | 19.58293 | 8.08% |
| Butterfly | 19.58739 | 19.57566 | 7.03% |

The nominal 0.06–0.13% differences do not establish a useful practical gain.
Candidate repeats reproduce their own outputs, but differ from exact. This
route stays in the diagnostic harness and is excluded from the preset.

### FP16 VAE attention tuning and fused QKV projection

The attention sweep tests 24 tile/group configurations at each of batches one
and four on real captured Q/K/V. The chosen configurations are four SIMD
groups with two 32-column blocks for batch one, and four groups with one
64-column block for larger batches. Whole-decoder results, rather than the
much larger isolated batch-one gain, determine their practical value.

Eight FP16 QKV GEMM schedules are tested at each batch size. The selected
64×64 output tile, K block 256, four-SIMD-group kernel reduces the complete
projection/normalization/RoPE/layout chain from **0.9831 to 0.5305 ms** at batch
one and **2.2197 to 1.6972 ms** at batch four. All captured Q/K/V output bytes
match the reference. A different GEMM implementation is still treated as
optional: these matching examples are not a universal bit-exactness proof.

| Complete decoder | Exact | Prior dense VAE | Tuned dense VAE | Prior dense + QKV |
|---|---:|---:|---:|---:|
| 22 frames, six alternating rounds | 1.87548 s | 1.52357 s | 1.50532 s | 1.45932 s |
| 124 frames, four alternating rounds | 13.14113 s | 10.66436 s | 10.53487 s | 10.24595 s |

At full length, tuning separately saves **0.12949 s / 1.21%** over the prior
dense decoder; QKV separately saves **0.41841 s / 3.92%**. The live combined
measurement below captures their joint return. Projection changes
remain independently switchable from the attention changes.

The production wrappers are then tested on the original saved short reference
and three new 22-frame boat/scarf/butterfly latent tensors. Each uses two
warmups per mode followed by four alternating rounds. Exact mode reproduces
the saved frame bytes. **QKV alone reproduces exact frames in all four cases;
QKV plus dense attention reproduces dense-only frames in all four cases.**
The combined VAE mode takes 1.440–1.462 s, versus 1.865–1.875 s exact.
The kernel cache is checked to establish that the projection actually ran.

For the full saved clip, production dense+tuning+QKV has a 10.0495 s warm
median over three measured decodes after two warmups. Its complete frame tensor
matches the tuned-attention reference byte for byte; the full QKV+dense frame
tensor also matches dense-only. Thus the new projection adds no observed frame
error in these comparisons. Fresh-process tests separately verify exact defaults,
the preset, and explicit exact overrides against saved attention/frame tensors.

## Numerical and visual review

The complete VAE preset on identical full-clip latents has pixel RMSE
**0.00009626**, PSNR **80.33 dB**, sampled luminance SSIM **0.99999724**,
maximum absolute pixel error **0.004594**, and mean RGB biases below 0.00000025.
Temporal-difference error RMS is **0.00012753**; reference/candidate motion RMS
is 0.06485255/0.06485258. About 1.0964% of RGB8 channel values change, by at most
two levels. These pixel metrics describe differences, not a quality guarantee.

Inspected lossless detail crops preserve paper folds, water ripples/reflections,
scarf weave, hair strands, wing markings and flower detail at the displayed
scale. The enlarged crop at the worst full-clip pixel and twelve consecutive
frames around the worst temporal error also look matched, without an obvious
new flicker in that inspected sequence. The three additional VAE scenes have
PSNR 79.39–80.54 dB and sampled SSIM at least 0.9999967.

Sampling has a materially different tradeoff. Medium boat/scarf/butterfly frame
PSNR is **21.94/16.05/17.10 dB**, sampled SSIM **0.814/0.643/0.680** and
temporal-difference error RMS **0.0500/0.0139/0.0709**. Side-by-side frame
sequences show changed boat drift and reflection positions, portrait/scarf
framing and window/background light, and butterfly wing timing and flower
placement. Red, teal and orange/purple scene colors and fine subject detail
remain plausible in the inspected examples. Consecutive butterfly frames show
different wing motion, not matched trajectories. These observations do not
establish equivalent temporal or aesthetic quality across prompts/seeds.

The sampling preset therefore remains explicitly **experimental and default
off**, for user review. The VAE and sampler can be switched independently.
Raw FP32 frames, lossless crops and synchronized side-by-side MP4s are retained;
MP4s are compressed review copies and are not numerical references.

The full combined preset changes video/audio latents by **38.44%/11.35%
relative L2**. It has frame PSNR **17.55 dB**, sampled SSIM **0.7092**
and temporal-difference error RMS **0.07172** against exact. Inspection of six
matched frames, enlarged detail crops and twelve consecutive frames confirms
changed boat orientation/drift and ripple/reflection positions. Paper edges,
folds and water detail remain plausible in those views; both sequences have
continuous motion at the inspected points. This is a visible alternate result,
not a byte-only change. Audio latents also differ; these silent visual-review
copies do not establish equivalent audio quality.

## Full workflow confirmation

The same 864×480, 124-frame, 24-fps, 20-step workflow uses seed 812764035,
res_multistep/simple, denoise 1, the INT8 convrot H3 model, Qwen text encoder,
FP16 video VAE and FP32 audio VAE. Every mode receives a full-shape two-step
warmup before these measured runs. Order is exact/previous/new/new/previous/exact.

| Run | Sampling (s) | Video VAE (s) | E2E (s) |
|---|---:|---:|---:|
| Exact 1 | 147.096889 | 13.184528 | 161.361 |
| Previous combination 1 | 144.752693 | 10.729042 | 156.537 |
| New preset 1 | 142.285457 | 10.149746 | 153.472 |
| New preset 2 | 142.044136 | 10.144264 | 153.223 |
| Previous combination 2 | 143.572320 | 10.706361 | 155.327 |
| Exact 2 | 146.583424 | 13.254498 | 160.890 |
| **Exact mean** | **146.840156** | **13.219513** | **161.1255** |
| **Previous combination mean** | **144.162507** | **10.717701** | **155.9320** |
| **New preset mean** | **142.164796** | **10.147005** | **153.3475** |

The new preset saves **4.675 seconds
sampling (3.18%)** and
**3.073 seconds video decode
(23.24%)** against the exact control.
Compared with the previous combination, it saves another
1.998 seconds sampling and
0.571 seconds video decode.
All new-preset timing ranges are below both control ranges in these repetitions.

Timers include the complete synchronized sampling/decoder paths, allocations,
quantization, padding and layout operations. E2E includes audio decode and video
export. Numerical comparisons, raw-frame saving and review encoding occur
after the execution timestamp. These are paired comparisons with common
instrumentation, not a claim against an old historical minimum.

All six captured initial-noise tensors match. Both exact controls reproduce
all four prior exact sampler-latent tensors and the prior full-frame SHA-256
`538f33de59395e714e0fb87ade8011dfe61f4131d11f34ba5c142d72dd71d5be`.
Both previous-combination runs reproduce the earlier tail experiment's four
sampler-latent tensors. Each mode repeats its own complete latent/frame bytes.
No optional-kernel fallback occurs. The existing exact matrix-multiply and
attention source files are unchanged.

## Local evidence and preservation

`diagnostics/relaxed-sprint4/`, above the repository, contains the source
snapshot, hash manifest, scripts, mode selectors, all saved noise/latents/frame
tensors, timings and review media. `baseline/` and `baseline-hashes.json`
preserve the starting sources. Sprint-three sources, candidate generators,
compiled-kernel recipes and review media remain in `diagnostics/relaxed-sprint3/`.

New experiments are selected through `modes.json` and `run_sampling.py`,
`schedule_probe.py`, `repeat_schedule.py`, `tail_schedule_probe.py`,
`vae_probe.py --kind qkv|attention`, and `vae_decode.py`. Production-wrapper
checks use `vae_validate.py` and `verify_runtime.py`. Rejected experiments
remain reproducible without changing the installed exact kernels.

Small numerical evidence is included in
[`data/m5-ultra-relaxed-sprint4.json`](data/m5-ultra-relaxed-sprint4.json).
The local `review/index.html` collects the full workflow and VAE-only comparisons.

## Validation and local installation

**55 GPU/routing/native regression tests pass.** Fresh processes using both
the workspace and installed custom node verify default exact behavior, the
combined preset, and explicit per-feature exact overrides against saved
attention and frame outputs. Projection failure latching, container ownership,
unknown attention arguments, masks, workflow overrides, custom projection
hooks/weight functions, ragged rows and partial/full RoPE have coverage.

The installed-source audit changes only `__init__.py` and
`_patches/minimax_vae_fast.py`, and adds the relaxed selector, separate attention
loader and optional VAE projection module. The other **77 existing sources**
match the frozen starting state. Replaced sources are backed up under
`diagnostics/relaxed-sprint4/installed-backup/`. Launch settings are unchanged,
so the preset is **not active until explicitly enabled and ComfyUI restarted**.
The isolated benchmark server is stopped after testing.
