# SeedVR2 video memory evidence

See [the report](../../m5-ultra-seedvr2-video-memory.md) for scope, timings,
exactness claims and limitations.

The JSON records preserve configurations, measured stages, SHA256 tensor
hashes, sampled memory peaks, completion/controlled-stop status, and external
watchdog results. To keep the archive compact, per-operation memory events
are reduced to phase peaks and a probe count; full traces and large tensors
remain in the local `diagnostics/seedvr2-video` directory. MP4/PNG/PT outputs
are deliberately excluded from this Git archive.

`workflow-original.json` is the unmodified saved API workflow. Despite its
3B filename, it loads the 7B INT8 convrot model and leaves Split Latent off.
`install-audit.json` identifies the installed runtime file.

The harness records the test machine's concrete model/ComfyUI/input paths.
`screens.py` and `long-runs.py` launch tests **sequentially**, each through
`guard.py`; do not run these alongside another inference job. Long runs use
the installed node, native ComfyUI video decoding/export, the original full
sampler context, and two repetitions. Compare tensor hashes only between
matching reader modes: the early PyAV-only screens and native ComfyUI reader
perform different input conversions.

The stock screen uses stock arithmetic with scheduling-only safety waits.
The `*-before` records use the previous optimized stack with only its
spatial-tile waits. A controlled stop is not a completed benchmark and is
not counted as a system OOM. `full120-block` is a first-pass pilot without
video export; `full120-installed`/`full240-installed` are the final exported
video measurements. Startup/extension compilation and verification hashes
are excluded from processing time.

`tests.log` covers scheduling scope, default/disabled behavior, nesting,
exception cleanup and argument/result preservation. The adjacent image
benchmark archive holds the original complete-image reference; the final
verification record compares it against the new installed safeguard.
