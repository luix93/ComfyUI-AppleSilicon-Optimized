# ComfyUI-AppleSilicon-Optimized

An Apple Silicon acceleration and compatibility pack for ComfyUI, with maximum
acceleration validated on **M5 chips: M5 Max and M5 Ultra have been tested**.
Support for earlier chips is experimental. Use the **latest macOS (currently
macOS 27)** for maximum acceleration: our full native kernels rely on **Metal
4.1**. The node configures the tested Python dependencies and runtime settings
automatically; standard workflows need no acceleration node or main flag.
It makes FP8/INT8 models work on MPS and accelerates sampling, attention and
video/image VAEs using custom Metal kernels and runtime patches.

**Our tested exact optimizations are bit-exact:** they preserve identical output
bytes against their documented reference. This guarantee does not extend to
every quantized compatibility path or optional approximate mode; see the
[numerics policy](#numerics-policy).

The pack also includes optional H3 acceleration/LoRA controls, Prompt Studio,
and Apple Silicon Super Resolution. Cache, sparse attention, token reduction,
opt-in INT8 attention and relaxed H3 modes are explicit workflow choices.
[Latest changes](#latest-results--m5-ultra-80-gpu-cores).

The complete source bundle is **[GPL-3.0](LICENSE)** because it includes
Spectrum and ComfyUI-derived code. Original MIT/BSD/Apache notices are retained; model weights have their
own terms. See [third-party provenance and notices](THIRD_PARTY_NOTICES.md).

[Installation](#installation) · [Performance](#latest-results--m5-ultra-80-gpu-cores)
· [Numerics](#numerics-policy) · [Settings](#environment-variables)
· [Credits and licenses](#credits)
## Installation

**Install the node pack, then launch ComfyUI.** On Apple Silicon, the first
prestartup runs setup automatically using that instance's Python, including
ComfyUI Desktop. No separate installer command is needed for a plain clone.
The installer selects our tested
**PyTorch 2.14.1 + torchvision 0.29.1** profile when the existing build is not
reviewed. This includes replacing newer untested builds. Reviewed 2.14.0/0.29.0
and 2.14.1/0.29.1 installations are kept when their source revision, imports and
native-extension headers match. Torchaudio is preserved; there is no 2.14.1
Torchaudio release in this profile.

Requirements: an Apple Silicon Mac, ComfyUI in a virtual environment, and Python
3.10–3.14 supported by the selected wheels. Python 3.14.1 is excluded by
our torchvision wheel's requirements. The full native kernels need a Metal 4.1
compiler/toolchain; our current measurements use **macOS 27 / M5 Ultra**.
Older Apple Silicon attempts the enabled routes with existing fallbacks, but
M1–M4 performance and parity are not yet validated.

**Latest regression: ComfyUI nightly core
[`7a283320`](https://github.com/Comfy-Org/ComfyUI/commit/7a28332047b6af638dc6c7f72ef3481a718769fd)**
(`v0.42.0-19-g7a283320`, with Fibo 2.0 support), with **PyTorch 2.14.1 / Kitchen
0.2.39 / frontend 1.57.1**. On October 23, 35 workflow/mode/preprocessor cases
completed twice. All are byte-identical to the October 19 regression on
`8f04fd6e`. One reviewed guard repair restores YuE2's exact RoPE path after an
upstream top-p fix. H3 covered exact, SOL INT8 fast and Dense INT8 v3; Wan
covered the four-step workflow.
[Timings, repair and numerical scope](benchmarks/m5-ultra-comfy042-nightly-oct23.md).
The [October 19 regression](benchmarks/m5-ultra-comfy0421-smoke.md) (core
`8f04fd6e`, including Wan's upstream numerical change and SeedVR's synthetic inputs),
the [preceding nightly check](benchmarks/m5-ultra-comfy0413-nightly-smoke.md)
and the [v0.41.2 check](benchmarks/m5-ultra-comfy0412-smoke.md) retain prior evidence.

**Stock/optimized benchmark core:** `v0.39.0-33-g1d2ea294` — package version **0.39.0**,
33 commits after the v0.39.0 tag, at
[`1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8`](https://github.com/Comfy-Org/ComfyUI/commit/1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8).
This is the checkout used for established stock/optimized pairs, not the tagged
v0.39.2 release. The complete table also includes Flux.3 on **v0.41.2**, MOSS on
**core `7a283320`**, and latest regression timings. The installer manages the
tested Torch profile; it does not pin or update ComfyUI core.

### ComfyUI-Manager

1. If your Manager permits **Install via Git URL**, install
   `https://github.com/luix93/ComfyUI-AppleSilicon-Optimized`.
2. Manager installs the dependencies and invokes this pack's `install.py`.
3. Restart ComfyUI. First-launch setup or queued package changes run at
   prestartup, before Torch loads. Wait for
   **Setup verified** or **Already configured** in the console.

Unchanged subsequent launches use a local completion record: no downloads,
package installation or Torch probe. A changed tested profile, pack requirements
or relevant installed runtime triggers setup again. If another extension has
already imported Torch before setup runs, stop ComfyUI and use the manual
installer below; the running process is left unchanged.

### Clone / ComfyUI Desktop

Stop ComfyUI. In Terminal, replace the first path with your actual ComfyUI folder:

```bash
cd "/path/to/ComfyUI"
git clone https://github.com/luix93/ComfyUI-AppleSilicon-Optimized custom_nodes/ComfyUI-AppleSilicon-Optimized
```

For Desktop, clone into the selected instance's actual `custom_nodes` folder,
not the application bundle. Launch that instance; the hook uses the Python
Desktop launched, so no virtual-environment activation is needed. Pip-less
environments use a disposable pip helper targeting that exact Python.

Start ComfyUI normally. Native acceleration is automatic on macOS/MPS.
**M5, M5 Pro, M5 Max and M5 Ultra with stable PyTorch 2.14.x use native PyTorch
attention by default**, avoiding the older attention router on eligible calls.
Custom query-cache kernels retain their separate validated Ultra/build guards;
selecting native PyTorch does not enable those kernels on a Max. Explicit backend
settings remain authoritative. First launch/use can take longer while native extensions compile.
If Apple developer tools are missing, `xcode-select --install` opens their
installer; ensure the selected tools include the required Metal compiler.
Unavailable extension paths retain their existing fallbacks.

### Check, opt out and troubleshoot

From your ComfyUI directory:

```bash
# Optional: finish setup manually while ComfyUI is stopped.
./.venv/bin/python custom_nodes/ComfyUI-AppleSilicon-Optimized/install.py

# Read-only: show the interpreter, package versions and reviewed-build status.
./.venv/bin/python custom_nodes/ComfyUI-AppleSilicon-Optimized/install.py --check

# Install this pack's dependencies while retaining your Torch/vision versions.
./.venv/bin/python custom_nodes/ComfyUI-AppleSilicon-Optimized/install.py --keep-torch
```

For Manager installs, `ASFP8_SETUP_KEEP_TORCH=1` supplies the same opt-out.
Set it before first launch to retain Desktop's existing Torch/vision. An explicit
`--keep-torch` choice is recorded and retained across automatic setup checks;
run the installer without that flag/environment variable to select the tested
profile again. If your environment is named `venv` or lives elsewhere, replace
`.venv/bin/python` with that environment's Python. The installer rejects a
global/system interpreter and leaves non-Apple-Silicon environments unchanged.
`ASFP8_NATIVE=0` disables automatic native feature activation at startup;
individual feature overrides still take priority. Approximate H3 modes remain off
unless selected explicitly. See [all settings](#environment-variables).

**M5 Max attention regression:** earlier package versions incorrectly required
Ultra-specific custom-kernel validation before selecting native PyTorch 2.14.
Installing the pack could therefore select older attention paths instead. The
startup selection is now independent; this fix has routing-unit-test coverage,
but the tester's reported 140 → 200 s/step slowdown has not been reproduced or
re-benchmarked. After updating, restart and look for
`native PyTorch attention defaults` and `PyTorch 2.14 SDPA selected` in the log.
Existing explicit backend overrides must be checked separately.

Setup downloads and resolves wheels before changing packages, verifies the
reviewed Torch/vision checksums and Torch source revision, and then installs
from that local wheel set. Before/after package-version records are saved under
`_runtime/setup/`; they are diagnostic records, not full environment backups.
PyTorch is shared with other custom nodes, so dependency conflicts are reported
by `pip check` after setup. No models or private media are read by setup.

The pack's startup prerequisite helper can install a missing `ninja` and restore
stripped Torch headers from a matching local cache. `ASFP8_AUTOFIX=0` disables
that helper. Neither optional Super Resolution weights/runtime nor LLM models
are downloaded by the main installer; their separate setup is documented in
[Super Resolution](docs/super-resolution.md) and [Prompt Studio](docs/prompt-studio.md).
## Supported configurations

**Latest ComfyUI releases are supported**, with source checks controlling which
native fast paths can run.
**Timing-confirmed regression:** October 23, 2026, nightly core
[`7a283320`](https://github.com/Comfy-Org/ComfyUI/commit/7a28332047b6af638dc6c7f72ef3481a718769fd)
(`v0.42.0-19-g7a283320`, master of October 20, the first core with Fibo 2.0),
with frontend 1.57.1 and Kitchen 0.2.39. 35 cases were run twice, and all are
byte-identical to the previous regression. This is the checkout behind the
latest timings below. The latest tagged release,
[v0.42.5](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.42.5), is built
from a release branch without Fibo 2.0. Its guarded MOSS, Llama and ops sources
match reviewed hashes, but it was not separately timed. Unreviewed source
revisions use compatibility fallbacks rather than a claim of validated maximum
acceleration. [Regression results and scope](benchmarks/m5-ultra-comfy042-nightly-oct23.md).

| Configuration | Validation status |
|---|---|
| M5 Ultra, 80 GPU cores / 256 GB, macOS 27, PyTorch 2.14.1 + torchvision 0.29.1 | Stock/optimized benchmarks use `v0.39.0-33-g1d2ea294`, Flux.3/preprocessing on v0.41.2 and MOSS on core `7a283320`. Latest [two-run regression](benchmarks/m5-ultra-comfy042-nightly-oct23.md): nightly core `7a283320`, Kitchen 0.2.39; see its input and numerical scope. |
| PyTorch 2.14.0 + torchvision 0.29.0 | Reviewed build retained by setup; earlier workflow evidence is linked in the reports. |
| M5 Max | Tested in earlier optimization work. The latest regression and current performance table use M5 Ultra; the current release stack was not retested on a Max. |
| Latest macOS / Metal 4.1 toolchain | Required for maximum native acceleration. Current benchmarks use macOS 27; install the matching Xcode/Metal tools as described above. |
| M1–M4 / older macOS / other framework builds | Experimental. Available routes are attempted with existing fallbacks; full native acceleration, performance and parity are not established. |
| Intel Mac / Windows / Linux | Automatic Apple runtime setup leaves these environments unchanged; Apple acceleration is not supported. |

Kernel availability depends on the selected compiler, headers, device and input
geometry. A model loading successfully does not prove that its fast path ran.

## Latest results — M5 Ultra (80 GPU cores)

**Table updated 2026-10-23 · 256 GB unified memory · macOS 27.0 · PyTorch 2.14.1**

**Latest changes.** The Fibo and MOSS rows in the table below are measured with these changes; the other figures are incremental comparisons:

- **Fibo 2.0 (new):** Dev and Edit Turbo get an exact route: joined CFG rows,
  fused norm/modulation/residual passes and tile32 attention. Dev 1024², 30
  steps: **88.49 s stock → 31.87 s (2.78×)**, of which 38.47 → 31.87 s comes
  from this route. It is byte-identical to the previous pack.
  [Report](benchmarks/m5-ultra-fibo2.md) · [Controls](docs/fibo2.md)
- **MOSS-Image, exact:** the exact route now joins the two CFG branches through
  its INT8 projections and uses a MOSS INT8 tile schedule and 32-row attention
  tiles. It also gains contributed RMS/modulation and Q/K-RoPE fusions. SFT at
  25 steps, warm with the prompt already encoded, now takes **20.43 → 17.48 s**
  at 1024² and **32.39 → 28.43 s** at 1248², byte-identical to the previous
  route.
  [Report and stock pairs](benchmarks/m5-ultra-moss-exact-combo.md#readme-table-pairs-october-23)
- **Opt-in INT8 attention (M5):** a new `dense_metal` backend (Dense Metal) with
  Kitchen-style numerics, available for H3 and other long-sequence models. The
  MiniMax H3 864×480 15-second clip takes **799.65 → 588.57 s (1.36×)** with
  every call quantized, or **662.04 s** with steps and layers protected. It
  changes outputs and is never a default.
  [Report](benchmarks/m5-ultra-int8-attention.md) · [Controls](docs/int8-attention.md)
- **Krea2 INT8:** an exact paired gate/up FFN, about 1% faster.

### Complete performance table

**Stock and optimized columns are matched measurements from the linked benchmark
reports.** Earlier optimized implementations are never used as stock baselines. The last column shows the latest tested core's warm second run. Speedups use the paired benchmark columns,
not a comparison across different ComfyUI versions.

- Established model pairs: `v0.39.0-33-g1d2ea294`, including later kernel follow-ups.
- Flux.3 and preprocessing pairs: **v0.41.2 / Kitchen 0.2.37**.
- MOSS and Fibo 2.0 pairs: measured 2026-10-23 on **core `7a283320` (`v0.42.0-19`) / Kitchen 0.2.39**.
- Latest regression (last column): **nightly core `7a283320` (`v0.42.0-19`) / Kitchen 0.2.39**, 2026-10-23. All use **PyTorch 2.14.1**.

<!-- Canonical current benchmark table: update the date, measurements and linked report together after validated runs. -->

| Workflow | Before (stock) | After (optimized) | Speedup | Latest core check |
|---|---:|---:|---:|---:|
| SDXL, 896×1152 / 12 steps | 2.61 s | **2.40 s** | **1.09×** | 2.46 s |
| Z-Image Turbo, 1024×1024 / 8 steps | 7.86 s | **3.44 s** | **2.28×** | 3.44 s |
| Flux.2 Klein 9B, text-to-image / 1024×1024 / 4 steps | 5.61 s | **2.27 s** | **2.47×** | 2.31 s |
| Flux.2 Klein 9B, image edit / 1024×1024 / 4 steps | 9.25 s | **4.64 s** | **1.99×** | 4.65 s |
| Ideogram 4, 1024×1024 / 12 steps / dual-model CFG | 43.54 s | **17.78 s** | **2.45×** | 17.42 s |
| Krea2 Turbo, 1448×1448 / 8 steps + LoKr | 27.45 s | **13.63 s** | **2.01×** | 13.52 s |
| Krea2 Turbo, same workflow without LoKr | 27.54 s | **13.96 s** | **1.97×** | 13.45 s |
| Krea2 Turbo, 1024×1024 / 8 steps / no LoKr | 15.63 s | **6.13 s** | **2.55×** | 6.03 s |
| Qwen-Image 2.1, 1024×1024 / 25 steps | 28.94 s | **12.66 s** | **2.29×** | 12.68 s |
| Qwen-Image 2.1 edit, generated reference / 1024×1024 / 40 steps | 50.47 s | **23.00 s** | **2.19×** | 23.20 s |
| MiniMax H3, 864×480 / 124 frames / 20 steps | 236.97 s | **141.65 s** | **1.67×** | 146.24 s |
| LTX 2.5, diffusion VAE / 1280×704 / 121 frames / 8+3 steps | 90.82 s | **46.16 s** | **1.97×** | 45.32 s |
| LTX 2.5, convolutional VAE / same clip and original tiling | 50.98 s | **32.05 s** | **1.59×** | 31.19 s |
| Wan 2.2 I2V 14B×2, 512×512 / 81 frames / 20 steps / CFG 3.5 | 421.16 s | **304.33 s** | **1.38×** | ≈288.25 s² |
| Wan 2.2 I2V 14B×2, same clip / 4-step LoRAs / CFG 1 | 46.14 s | **31.90 s** | **1.45×** | 30.80 s² |
| SeedVR2 7B INT8, 2400×1600 → 7200×4800 / 1 step | 66.29 s | **45.15 s** | **1.47×** | 44.97 s³ |
| YuE2 3B INT8, controlled 8 s / 128 ABC tokens / 32 steps | 124.19 s | **5.78 s** | **21.49×** | 5.83 s |
| Flux.3 distilled, T2I / 1024×1024 / 16 steps | 12.12 s | **6.17 s** | **1.96×** | 5.78 s |
| Flux.3 Base, T2I / 1024×1024 / 50 steps / CFG 3.5 | 69.15 s | **35.47 s** | **1.95×** | 33.52 s |
| Flux.3 distilled, edit / 1376×752 / 16 steps | 21.94 s | **13.35 s** | **1.64×** | 12.70 s |
| Flux.3 Base, edit / 896×1152 / 50 steps / CFG 3.5 | 118.77 s | **75.35 s** | **1.58×** | 71.11 s |
| MOSS-Image SFT, 1024×1024 / 25 steps / CFG 2.5 | 69.34 s | **19.09 s** | **3.63×** | 18.92 s |
| MOSS-Image Base, 1024×1024 / 25 steps / CFG 2.5 | 70.38 s | **19.76 s** | **3.56×** | 19.51 s |
| MOSS-Image SFT, 1248×1248 / 25 steps / CFG 2.5 | 88.41 s | **30.25 s** | **2.92×** | 29.75 s |
| MOSS-Image SFT, 1248×1248 / 50 steps / CFG 2.5 | 155.49 s | **58.67 s** | **2.65×** | —⁴ |
| Fibo 2.0 Dev INT8, 1024×1024 / 30 steps / CFG 5 | 88.49 s | **31.87 s** | **2.78×** | 31.43 s |
| Fibo 2.0 Edit Turbo INT8, 1024×1024 reference / 4 steps | 16.19 s | **6.61 s** | **2.45×** | 6.28 s |
| SeedVR2 7B INT8 video, 416×736 → 832×1472 / 120 frames | Not measured | **124.66 s** | — | 124.28 s³ |
| SeedVR2 7B INT8 video, 25-frame processing screen / same dimensions | 34.18 s | **27.81 s** | **1.23×** | — |
| YuE2, saved full-song workflow / 83.2 s song | Not measured | **34.12 s** | — | 35.24 s |
| YuE2 Mothersuperior, 20 s | Not measured | **21.94 s** | — | 22.15 s |
| BiRefNet, 4 synthetic frames | 21.786 s (CPU) | 0.757 s (MPS) | 28.78× | 0.748 s |
| Multi-person SDPose + detector, 4 frames / 8 crops | 2.943 s | 1.448 s | 2.03× | 1.463 s |
| DA3 Mono Large, 8 frames | 0.456 s | 0.416 s | 1.10× | 0.407 s |
| DA3 Base multi-view, 8 frames | — | — | — | 0.114 s |

Speedup = stock time ÷ optimized time. **2×** means twice the processing speed;
**1.2×** means 20% faster. **—** means no measurement is published for that cell.

¹ On October 19 the full-song YuE2 initial sweep was slower; an isolated two-run
confirmation returned to the preceding timing range. The latest regression matched
that confirmation (35.24 s).

² Wan retains current upstream quantized-linear semantics; output bytes differ
from v0.41.2. The timing comparison crosses that core change. The latest regression
ran only the four-step workflow (31.07 → 30.80 s, byte-identical). The 20-step
value applies the same −0.9% to its October 19 time and was not re-measured.

³ SeedVR uses new synthetic media at the same dimensions. Its latest timings
validate execution and bounded memory, rather than input-matched regression.
The SeedVR 25-frame screen was not rerun.

⁴ MOSS SFT 1248² / 50 steps was not part of the latest regression. Stock MOSS
1248² values are one run each; the other stock/optimized pairs are two-run means.

The full-song YuE2, Mothersuperior and 120-frame SeedVR2 rows retain their
optimized timings, but have **no measured full-workflow stock comparison**.
YuE2's original 83.2-second song took **354.18 s** on the earlier GPU stack;
that already used our MPS placement patch, so it is not stock. Later 50.46 s /
31.35 s YuE2 controls and the 128.18 s SeedVR2 control already included further
optimizations and must not be used to calculate stock speedup.
[Original YuE2 measurements](benchmarks/m5-ultra-yue2.md) ·
[Later YuE2 GPU comparisons](benchmarks/m5-ultra-torch2141-retest.md#yue-full-song-and-adapter-gpu-checks) ·
[SeedVR2 incremental comparison](benchmarks/m5-ultra-seedvr2-vae2141.md#video-fusion-and-scheduling-screens).

The SeedVR2 25-frame row is a genuine stock-arithmetic comparison with memory
safety guards retained. It includes processing and first-frame PNG, but excludes
video export; it is not a measured five-second stock run.
[25-frame stock comparison](benchmarks/m5-ultra-seedvr2-video-memory.md#scheduling-experiments).

Generation timings include sampling, decode and export; model loading and unchanged
conditioning are cached. YuE regenerates tokens/conditioning. MOSS and Fibo also include
fresh text encoding (Fibo Edit also re-encodes its reference), with prompt enhancement off. Flux.3 edit uses identical frozen
expanded prompts, excluding prompt expansion. Preprocessing rows exclude model
loading and video/file IO. Stock baselines omit this pack; Qwen retains its minimal encoder
compatibility shim and SeedVR retains scheduling-only memory protection.
Optional H3 acceleration modes are off in the main table.

The latest regression is a single warm observation per case, not a replacement
for repeated benchmark averages. **LTX, Flux.3 and Wan differ from
v0.41.2 after the upstream linear-dispatch change**; these compatibility repairs
preserve updated-core outputs. All final cases repeat their own captured bytes,
and the October 23 pass matches the October 19 bytes in every compared case.
H3's SOL INT8 fast preset (113.61 s) and the new Dense INT8 v3 balanced preset
(131.95 s) were also checked; they are opt-in and not in the main table.
The [latest regression report](benchmarks/m5-ultra-comfy042-nightly-oct23.md)
includes previous-version timings, the YuE2 guard repair and numerical scope.
[October 19 regression](benchmarks/m5-ultra-comfy0421-smoke.md) ·
[Established baseline](benchmarks/m5-ultra-torch2141-retest.md) ·
[Krea2 follow-up](benchmarks/m5-ultra-krea2-oct2026.md) ·
[Ideogram/H3 follow-up](benchmarks/m5-ultra-ideogram-h3-oct2026.md).

### Model notes and additional measurements

- **MOSS:** the rows include the October 23 exact route, which is byte-identical
  to the preceding pack (see Latest changes). The complete pack is not
  stock-byte-identical; a separate MOSS-only configuration passed stock-byte
  comparisons.

  INT8 attention modes, including the Kitchen policy, quantize every call. At
  1.5 MP they saved only 2–4% at 25–28 dB PSNR, so they stay opt-in.
  [Initial benchmarks](benchmarks/m5-ultra-moss-image.md) ·
  [1.5 MP follow-up](benchmarks/m5-ultra-moss-15mp.md) ·
  [Controls](docs/moss-image.md).
- **Flux.3:** distilled uses CFG 1, Base CFG 3.5; both use Euler/simple and INT8
  ConvRot models. Prepared Qwen3.5 weights reduce prompt expansion from 116–117 s
  to about 10 s. With fresh expansion/reference encoding, the original v0.41.2
  edit tests measured 24.01 s distilled / 85.95 s Base.
  [Benchmarks and numerical scope](benchmarks/m5-ultra-flux3.md) · [Controls](docs/flux3.md).
- **Preprocessing:** BiRefNet's large gain includes moving CPU execution to MPS,
  which changes mask rounding. SDPose's feature-only execution preserves tested
  pose outputs. DA3 Base multi-view avoids a Metal camera-head dtype crash;
  experimental batching stays off by default.
  [Measurements and limitations](benchmarks/m5-ultra-vision-preprocessors.md).
- **SAM3.1:** correcting repeated failed Metal compilation reduced two-frame
  warm node time from 4.22 to 0.63 s (6.7×), with identical masks/boxes. This
  compares the previous pack with the corrected pack, not stock ComfyUI, and
  excludes video loading/export. [Report](benchmarks/m5-ultra-sam3.md).

### Qwen3-VL 4B prompt generation in PyTorch

On ComfyUI **v0.41.2**, native PyTorch decoding reaches approximately **70 tok/s**
with the existing Qwen3-VL 4B INT8 ConvRot checkpoint. Latest exact improvements,
compared with our preceding cache/RoPE pass using the same PyTorch SDPA backend:

| Task | Previous exact pass | Updated | Speedup |
|---|---:|---:|---:|
| Text prompt expansion, 117 tokens | 2.22 s | **2.05 s** | **1.08×** |
| Text prompt expansion, 111 tokens | 2.14 s | **1.96 s** | **1.09×** |
| Two synthetic images → video prompt, 60 tokens | 1.37 s | **1.34 s** | **1.02×** |

Warm alternating runs include prefill and cache preparation. Tested outputs match
byte-for-byte; temporary converted weights use **7.49 GiB**. The earlier 42 tok/s
benchmark used basic attention because its standalone harness did not parse CLI
flags; changing attention backend is separate from these exact gains. Historical
10–11× cache results, memory limits, tests and switches are in the
[benchmark report](benchmarks/m5-ultra-qwen3vl-decode.md).

**Large-image safety:** Qwen’s vision encoder can bypass the global attention
router and request a full quadratic score tensor. Oversized basic-attention
calls now use bounded query chunks, preserving image resolution and all keys.
Small calls are unchanged; chunked matrix execution can change floating-point
rounding. This safeguard is separate from the byte-exact decoding optimizations
above. See the [memory-safety validation](benchmarks/qwen-vision-memory-safety.md).

October 13 long-H3 exact follow-up: **412.03 → 399.21 s** for a 10-second,
864×480 / 20-step clip (**3.1% less time**, unchanged latent, frame and audio
bytes). The 15-second two-step screen suggests about 20 seconds saved over
20 steps, not the requested 200 seconds; that projection is not a full-clip
measurement. [Long-clip report](benchmarks/m5-ultra-h3-long-exact.md).

Optional H3 SOL INT8 Hybrid Fast measures **113.79 s at 864×480** and
**371.75 s at 1344×768**, against matched exact-optimized controls of
146.40 / 606.72 s. It changes numerical results and remains opt-in.
Top-k Wide has reported background morphing; the legacy relaxed H3 preset is
currently slower than the exact-optimized default. See the
[current-stack report](benchmarks/m5-ultra-torch2141-retest.md).

For reference/guide-heavy H3 workflows, SOL and dense quantized presets have a
separate **default-off visual reference approximation switch**. SOL also has an
exact protected-prefix optimization.
A matched five-second guide + image-reference test measured **218.98 → 159.07 s
(1.38×)** with SOL Fast and reference approximation; this is an additional
approximation, not a change to the exact default.
[Matched guided tests and quality notes](benchmarks/m5-ultra-h3-reference-attention.md).

The table combines named workflow reports; it is not a promise for other inputs,
hardware or cold model loads. The INT8 routes are not a blanket guarantee of
parity with vanilla ComfyUI. [All benchmark reports](benchmarks/README.md) contain
settings, comparison boundaries, memory measurements and rejected candidates.

## Numerics policy

**Exact means byte-identical to the stated tested reference**, on its validated
stack and inputs. Later exact fusions preserve the selected optimized path;
they do not undo numerical changes already introduced by that path.

| Route | Output behavior |
|---|---|
| Exact kernel/scheduling refinements | Preserve their documented reference bytes, including intermediate rounding where required. |
| Other default native routes | Some use altered quantization or rounding, such as H3 W8A8 SwiGLU and reassociated ConvRot; the entire pack is not bit-exact to stock. |
| H3 cache, sparse/dense INT8 attention and token reduction | Explicit workflow approximations. They can alter detail, composition, motion and temporal stability. |
| Relaxed H3 kernel/VAE flags | Experimental, independently selectable and off by default. |

For a quality comparison, keep prompt, seed, model, dimensions, sampler and step
count fixed; inspect the complete output, especially background stability in
video. A small tensor error alone does not establish equivalent quality.
[Detailed numerical contracts and switches](docs/numerics.md).

## Workflow nodes

Standard model workflows receive startup patches automatically. Additional nodes
are for controls or new functions:

| Node | Purpose |
|---|---|
| H3 Acceleration · Simple | Combine one cache preset, one attention preset and optional token reduction. SOL/dense quantized reference approximation is a separate default-off switch. Placement defaults to MotionCache balanced; set all three selectors to Off for pass-through. |
| H3 Apple Silicon Acceleration | Advanced settings for the same runtime. MotionCache and Spectrum implementations are bundled. |
| H3 LoRA (Apple Silicon) | Ordered adapter list, including Turbo, with enable/weight/validation per entry. |
| Attention Backend (Apple Silicon) | Select attention on the connected model, including the opt-in approximate `dense_metal` and `dense_metal_v2` for long video sequences. |
| H3 Latent Upscale (Apple Silicon) | MPS-aware latent upscaling with device/precision and memory controls. |
| Apple Silicon Super Resolution | Optional image/video VSR and neural-rendering stages; separate runtime/model installation. |

[H3 setup and quality guidance](docs/h3-acceleration.md) ·
[LoRA/control details and migration](docs/h3-workflow-reference.md) ·
[Latent upscaler](docs/h3-latent-upscaler.md) ·
[Super Resolution](docs/super-resolution.md).

## Prompt Studio

**Apple Silicon Prompt Studio** generates editable prompts for image, video and
audio workflows through **oMLX, LM Studio, Ollama or an OpenAI-compatible server**.
It outputs `prompt` and `negative_prompt` as `STRING` values. The compact node
opens a workspace with creative controls, editable target-model profiles,
image/video-frame/audio references, a searchable prompt library with favorites,
and JSON backup/restore. Classic nodes and Nodes 2.0 are supported.

Start with **Open Studio → Connections**, save your endpoint, discover models,
then select your LLM. The canvas node now has a **model picker**, using the
saved per-connection discovery list, and one **Generate on queue** toggle.
With it off, workflows reuse the editable saved text; with it on, each execution
requests a fresh prompt. The Generate button sends the brief together with
connected image/video/audio inputs automatically, running only the required
upstream nodes. References require a compatible LLM/server. Endpoint setup stays
in Open Studio; you can switch models directly on the node.
API keys stay in a separate private server file, outside workflows and backups.

[Setup and reference support](docs/prompt-studio.md) ·
[Example workflow](workflows/apple_silicon_prompt_studio.json)
## Troubleshooting

- **Setup/download failure:** read the first setup error and run `install.py --check`
  with this instance's Python. Stop ComfyUI before rerunning the installer. Use
  `--keep-torch` if you intend to retain the existing framework. Setup journals
  record package versions; they are not full environment backups.
- **Missing Metal compiler or failed native build:** verify the selected developer
  tools include Metal 4.1. Check the first compiler error; unavailable extensions
  use their existing fallback routes. M1–M4/older-macOS behavior is experimental.
- **A regression after an update:** restart after setting `ASFP8_NATIVE=0` to
  disable automatically enabled native groups. Explicit per-feature settings
  still win; compatibility/memory safeguards and connected approximation nodes
  are separate. Disable H3 approximation selectors/relaxed flags for an A/B test.
- **Unexpected speed or output:** compare warm runs with identical settings.
  `ASFP8_ATTN_DEBUG=1` logs attention routing and `ASFP8_EXEC_TRACE=1` provides
  node timing. Keep these off for normal use.
- **High memory use:** keep SeedVR2 scheduling safeguards enabled; test small
  clips first. SOL's temporary-buffer budget is configurable in the advanced
  node; Auto is the default. Raising it can increase peak memory.
- **Removing the pack:** stop ComfyUI and remove its folder from `custom_nodes`.
  Restart to remove patches. Packages installed into the shared environment
  remain installed; restore previous versions using the setup journal if needed.

Before sharing logs/workflows, remove API keys, personal paths and confidential
prompts/media. Prompt Studio references are sent to the configured LLM endpoint;
choose an endpoint you trust. [All environment variables](docs/settings.md).

## Environment variables

Most workflows need none. The main controls are `ASFP8_NATIVE=0` (automatic native
groups), `ASFP8_SETUP_KEEP_TORCH=1` (installer opt-out) and `ASFP8_AUTOFIX=0`
(prerequisite helper). Per-model switches and experimental modes are documented
in the [complete settings reference](docs/settings.md).
For the new Flux.3 paths, `ASFP8_FLUX3_FAST=0` restores the preceding pack;
[Flux.3 controls](docs/flux3.md) also separate prompt caching, sampling caching
and each fusion.
For MOSS-Image, `ASFP8_MOSS_FAST=0` restores its preceding execution;
[MOSS controls](docs/moss-image.md) separate each fusion, exact attention and
bounded sampling caches.

## Documentation and release checks

- [Benchmark index and historical experiments](benchmarks/README.md)
- [Kernel/compatibility implementation](docs/kernel-reference.md)
- [YuE2 Mothersuperior workflow](docs/yue2-mothersuperior.md)
- [Separate YuE2 hum-to-song workflow](docs/yue2-hum-to-song.md) — requires FS Audio Suite.
- [RefMod compatibility](docs/h3-refmod-compatibility.md)
- [Attribution audit](docs/release-attribution.md) and [privacy audit](docs/privacy-audit.md)
- [Release-candidate changes and validation](CHANGELOG.md)

## Credits

Built from Paweł Mazurkiewicz's **ComfyUI-AppleSilicon-FP8** and **mtlflashattn**,
with adapted/reference work from **Cider/Mininglamp**, **CCV/Liu Liu**, **PyTorch**,
**MLX/Apple**, **ComfyUI/Comfy Org**, **Comfy Kitchen/Kijai**, **SeedVR/ByteDance**,
**MotionCache/starsFriday**, **Spectrum/xmarre**, **h3.c/Salvatore Sanfilippo**,
**Sol-Attn/NVIDIA**, **LBH-123-AI's H3 upscaler**, **MLX-DLSS/MLX Swift** and
**DLSSMac/Mappsnet7**. Additional sparse/cache research references are credited too.

[Full author/project credits](docs/credits.md) ·
[Source inventory and required notices](THIRD_PARTY_NOTICES.md).

## Special thanks

**Neat**, for collaboration on this project and a generous financial donation.

## License

The combined distribution, including bundled Spectrum, uses
[GPL-3.0](LICENSE). Spectrum retains its GPL-3.0-or-later terms; original project code retains its
[MIT notices](third_party/original-FP8-MIT.txt). See [third-party attribution](THIRD_PARTY_NOTICES.md)
for the licenses and provenance of bundled components.
