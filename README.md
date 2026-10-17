# ComfyUI-AppleSilicon-Optimized

An Apple Silicon acceleration and compatibility pack for ComfyUI. It makes
FP8/INT8 models work on MPS and speeds up sampling, attention and video/image
VAEs using custom Metal kernels and runtime patches. Standard workflows work
without adding an acceleration node or setting a main flag.

The pack also includes optional H3 acceleration/LoRA controls, Prompt Studio,
and Apple Silicon Super Resolution. Cache, sparse attention, token reduction
and relaxed H3 modes are explicit workflow choices.

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

**Latest tested ComfyUI core: v0.41.2**, commit
[`b53f77a7`](https://github.com/Comfy-Org/ComfyUI/commit/b53f77a74e21091cbdebfdf7b85fc80326932646).
All 22 workflow/mode cases passed a two-run regression check on October 16;
captured outputs matched v0.40.0 and warm times remained within −2.6% to +1.2%.
[Results and validation scope](benchmarks/m5-ultra-comfy0412-smoke.md).

**Optional H3 attention:** new INT8 QK / BF16 PV presets measured **8.51 s/step**
full or **8.76 s/step** protected, versus **9.29 s/step exact**, at 1024×512,
124 frames, 20 steps. These are approximate, opt-in modes; exact defaults are
unchanged. [Speed, quality and comparison with SOL](benchmarks/m5-ultra-h3-floatpv.md).

**Stock/optimized benchmark core:** `v0.39.0-33-g1d2ea294` — package version **0.39.0**,
33 commits after the v0.39.0 tag, at
[`1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8`](https://github.com/Comfy-Org/ComfyUI/commit/1d2ea2948d33dfda4d7cfe58c6d234968aa62cf8).
This is the checkout used for the established model table, not the tagged v0.39.2
release. New Flux.3 measurements use **v0.41.2** and are listed separately. The installer manages the tested Torch profile; it does not pin or
update ComfyUI core.

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

| Configuration | Validation status |
|---|---|
| M5 Ultra, 80 GPU cores / 256 GB, macOS 27, PyTorch 2.14.1 + torchvision 0.29.1 | Benchmarked on ComfyUI `v0.39.0-33-g1d2ea294`; v0.41.2 passed the [two-run workflow regression check](benchmarks/m5-ultra-comfy0412-smoke.md), with new Flux.3 benchmarks on v0.41.2. |
| PyTorch 2.14.0 + torchvision 0.29.0 | Reviewed build retained by setup; earlier workflow evidence is linked in the reports. |
| M5 Max | Historical measurements exist; the current release stack was not retested on a Max. |
| M1–M4 / older macOS / other framework builds | Experimental. Available routes are attempted with existing fallbacks; full native acceleration, performance and parity are not established. |
| Intel Mac / Windows / Linux | Automatic Apple runtime setup leaves these environments unchanged; Apple acceleration is not supported. |

Kernel availability depends on the selected compiler, headers, device and input
geometry. A model loading successfully does not prove that its fast path ran.

## Latest results — M5 Ultra (80 GPU cores)

**Updated 2026-10-16 · 256 GB unified memory · macOS 27.0 · PyTorch 2.14.1**

### New: Flux.3 on ComfyUI v0.41.2

Warm diffusion workflow time, including VAE decode and PNG export. Edit rows use
the same frozen expanded prompt in both configurations; prompt generation is
measured separately below. Base uses **50 steps / CFG 3.5**, distilled uses
**16 steps / CFG 1**, both Euler/simple and INT8 ConvRot checkpoints.

| Workflow | Stock | Optimized | Speedup |
|---|---:|---:|---:|
| Flux.3 distilled, text-to-image, 1024×1024 | 12.12 s | **6.17 s** | **1.96×** |
| Flux.3 base, text-to-image, 1024×1024 | 69.15 s | **35.47 s** | **1.95×** |
| Flux.3 distilled, image edit, 1376×752 | 21.94 s | **13.35 s** | **1.64×** |
| Flux.3 base, image edit, 896×1152 | 118.77 s | **75.35 s** | **1.58×** |

The new refinements also cut **14–22%** from the preceding optimized pack,
with byte-identical tested latents and decoded tensors. Prompt expansion drops
from **116–117 s to about 10 s** by reusing prepared Qwen3.5 weights within
each generation. These exact incremental changes do not make the existing
W8A8 path identical to stock. With fresh prompt expansion and reference encoding,
the complete edit workflows measured **24.01 s distilled / 85.95 s base** warm.
[Measurements and rejected experiments](benchmarks/m5-ultra-flux3.md)
· [Controls and temporary memory budgets](docs/flux3.md).

### Established model benchmarks

**ComfyUI core used for this table:** `v0.39.0-33-g1d2ea294` (package version
0.39.0). [Recorded test environment](benchmarks/m5-ultra-torch2141-retest.md).

The [v0.41.2 regression check](benchmarks/m5-ultra-comfy0412-smoke.md) passed
all 22 cases twice, with byte-identical captured outputs against v0.40.0 and
warm timing changes of −2.6% to +1.2%. These single warm observations do not
replace the benchmark averages below.

<!-- Canonical current benchmark table: update the date, measurements and linked report together after validated runs. -->

| Workflow | Before (stock) | After (optimized) | Less total time | Speedup |
|---|---:|---:|---:|---:|
| SDXL, 896×1152 / 12 steps | 2.61 s | **2.40 s** | 8.2% | **1.09×** |
| Z-Image Turbo, 1024×1024 / 8 steps | 7.86 s | **3.44 s** | 56.3% | **2.28×** |
| Flux.2 Klein 9B, text-to-image / 1024×1024 / 4 steps | 5.61 s | **2.27 s** | 59.6% | **2.47×** |
| Flux.2 Klein 9B, image edit / 1024×1024 / 4 steps | 9.25 s | **4.64 s** | 49.8% | **1.99×** |
| Ideogram 4, 1024×1024 / 12 steps / dual-model CFG | 43.54 s | **17.78 s** | 59.2% | **2.45×** |
| Krea2 Turbo, 1448×1448 / 8 steps + LoKr | 27.45 s | **13.63 s** | 50.4% | **2.01×** |
| Krea2 Turbo, same workflow without LoKr | 27.54 s | **13.96 s** | 49.3% | **1.97×** |
| Krea2 Turbo, 1024×1024 / 8 steps / no LoKr | 15.63 s | **6.13 s** | 60.8% | **2.55×** |
| Qwen-Image 2.1, 1024×1024 / 25 steps | 28.94 s | **12.66 s** | 56.3% | **2.29×** |
| Qwen-Image 2.1 edit, generated reference / 1024×1024 / 40 steps | 50.47 s | **23.00 s** | 54.4% | **2.19×** |
| MiniMax H3, 864×480 / 124 frames / 20 steps | 236.97 s | **141.65 s** | 40.2% | **1.67×** |
| LTX 2.5, diffusion VAE / 1280×704 / 121 frames / 8+3 steps | 90.82 s | **46.16 s** | 49.2% | **1.97×** |
| LTX 2.5, convolutional VAE / same clip and original tiling | 50.98 s | **32.05 s** | 37.1% | **1.59×** |
| Wan 2.2 I2V 14B×2, 512×512 / 81 frames / 20 steps / CFG 3.5 | 421.16 s | **304.33 s** | 27.7% | **1.38×** |
| Wan 2.2 I2V 14B×2, same clip / 4-step LoRAs / CFG 1 | 46.14 s | **31.90 s** | 30.9% | **1.45×** |
| SeedVR2 7B INT8, 2400×1600 → 7200×4800 / 1 step | 66.29 s | **45.15 s** | 31.9% | **1.47×** |
| YuE2 3B INT8, controlled 8 s / 128 ABC tokens / 32 steps | 124.19 s | **5.78 s** | 95.3% | **21.49×** |

Speedup = stock time ÷ optimized time. **2×** means twice the processing speed; **1.2×** means 20% faster.


Warm end-to-end averages, including sampling, decode and export; loading and unchanged conditioning are cached. YuE regenerates tokens and conditioning each time. Stock omits this pack; Qwen uses its minimal INT8 encoder compatibility shim, and SeedVR stock retains scheduling-only memory protection. The optional H3 cache, sparse attention, token reduction and relaxed kernels are **off**. H3 sampling averages **6.52 s/step**.
Latest exact follow-ups: Krea2 1MP **6.24 → 6.13 s** and Ideogram 4
**18.41 → 17.78 s** against their preceding optimized controls, preserving
sampler/decoded bytes. These are separate comparisons from the stock column.
[Krea2 report](benchmarks/m5-ultra-krea2-oct2026.md) ·
[Ideogram/H3 report](benchmarks/m5-ultra-ideogram-h3-oct2026.md).

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
| H3 Apple Silicon Acceleration | Advanced settings for the same runtime. MotionCache, Spectrum and Veda implementations are bundled; Veda predictor weights require separate setup. |
| H3 LoRA (Apple Silicon) | Ordered adapter list, including Turbo, with enable/weight/validation per entry. |
| Attention Backend (Apple Silicon) | Select attention on the connected model. |
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

## Documentation and release checks

- [Benchmark index and historical experiments](benchmarks/README.md)
- [Kernel/compatibility implementation](docs/kernel-reference.md)
- [YuE2 Mothersuperior workflow](docs/yue2-mothersuperior.md)
- [RefMod compatibility](docs/h3-refmod-compatibility.md)
- [Attribution audit](docs/release-attribution.md) and [privacy audit](docs/privacy-audit.md)
- [Release-candidate changes and validation](CHANGELOG.md)

## Credits

Built from Paweł Mazurkiewicz's **ComfyUI-AppleSilicon-FP8** and **mtlflashattn**,
with adapted/reference work from **Cider/Mininglamp**, **CCV/Liu Liu**, **PyTorch**,
**MLX/Apple**, **ComfyUI/Comfy Org**, **Comfy Kitchen/Kijai**, **SeedVR/ByteDance**,
**MotionCache/starsFriday**, **Spectrum/xmarre**, **Veda**, **h3.c/Salvatore Sanfilippo**,
**Sol-Attn/NVIDIA**, **LBH-123-AI's H3 upscaler**, **MLX-DLSS/MLX Swift** and
**DLSSMac/Mappsnet7**. Additional sparse/cache research references are credited too.

[Full author/project credits](docs/credits.md) ·
[Source inventory and required notices](THIRD_PARTY_NOTICES.md).

## License

The combined distribution, including bundled Spectrum, uses
[GPL-3.0](LICENSE). Spectrum retains its GPL-3.0-or-later terms; original project code retains its
[MIT notices](third_party/original-FP8-MIT.txt). See [third-party attribution](THIRD_PARTY_NOTICES.md)
for the licenses and provenance of bundled components.
