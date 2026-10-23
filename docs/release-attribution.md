# Release attribution audit — October 10, 2026

The [README credits](../README.md#credits) identify authors and projects used by
this pack. [Third-party notices](../THIRD_PARTY_NOTICES.md) distinguish copied or
adapted source, installed dependencies, optional runtimes and research references.
The [provenance index](../third_party/provenance.json) records audited revisions
and checksums; a license-audit revision is not necessarily the revision from which
code was adapted.

## Changes made for release

- Added full Cider, h3.c, Sana, Comfy Kitchen and MLX-DLSS license/notice texts.
- Preserved the original FP8 project's MIT license verbatim in
  `third_party/original-FP8-MIT.txt`. Root `LICENSE` and `LICENSE-BUNDLED` now contain
  the full GPL v3 text for the combined distribution. This selects version 3 from
  Spectrum's GPL-3.0-or-later grant and retains ComfyUI's GPL-3.0 terms; individual
  permissive notices and Spectrum's original terms remain intact.
- Corrected the SOL shader's misleading FlashVSR attribution. The maintainer
  confirmed it was their local prototype; mtlflashattn ancestry remains credited.
- Corrected package/repository identity and removed the previous author's
  registry PublisherId. Registry publication requires this maintainer's own
  registered ID; cloning from GitHub does not require one.
- Made the optional super-resolution builder preserve the resolved Swift/MLX
  dependencies' license and notice files beside the generated binary, with a
  checksum inventory. Missing dependency licenses stop the build.

## Release contents and limits

Release the tracked source and its notices together. Generated `_runtime` files
and downloaded model weights are excluded from the Git source distribution.
Model and LoRA licenses remain separate, including NVIDIA-derived DLSSMac
weights. The bundled SeedVR conditioning asset has its own
explicit Apache provenance. A binary release needs its actual resolved dependency
notices and any separately applicable binary/model terms.

mtlflashattn declares MIT in its package metadata but has no standalone upstream
license file at the audited revision. Its source copyright and the original
author's full MIT notice are retained; this limitation is recorded in the index.

The inventory checker detects changed notices, changed pinned vendor files and
missing or inconsistent runtime origins. It does not establish the provenance of
unrecorded code or grant redistribution rights to external assets.

## Validation

```sh
python dev/check_attribution.py
python -m pytest -q tests/test_runtime_notices.py tests/test_super_resolution.py tests/test_runtime_install.py
```

The audit validated **23 notice files, 77 pinned vendor files and 71 runtime
origins**. The targeted tests passed **68/68**. Attribution/header changes do not
alter inference arithmetic; no GPU benchmark was rerun for this audit. Historical
benchmark source hashes retain the versions used for their original measurements.
