# BiRefNet, SDPose and Depth Anything 3 — October 18, 2026

M5 Ultra (80 GPU cores), 256 GB unified memory, macOS 27, ComfyUI v0.41.2
(`b53f77a7`), PyTorch 2.14.1. Marigold was excluded.

## Measurements and scope

These are short synthetic-frame workloads using the downloaded real models,
not timings of the user's private clips. Four runs per confirmation case;
the first is excluded and the other three averaged. Warm times exclude model
loading, video decoding/encoding, disk output and workflow scheduling.

| Workload | Stock 2.14.1 | Previous pack | Current default | Stock/current |
|---|---:|---:|---:|---:|
| BiRefNet, four 864×480 input frames, native 1024 model preprocessing | 21.786 s (CPU) | Same existing MPS route | 0.757 s (MPS) | 28.78× |
| RT-DETR + SDPose + drawing, four frames / eight person crops | 2.943 s | 1.554 s | 1.448 s | 2.03× |
| DA3 Mono Large, eight 864×480 frames, 504 shorter side | 0.456 s | 0.416 s | 0.416 s | 1.10× |

BiRefNet's large improvement is the **existing CPU-to-MPS placement fix**, not a
new kernel developed in this pass. CPU and MPS masks are **not byte-identical**:
maximum absolute mask difference 0.03279, mean 0.0000202 on this synthetic case.
No new BiRefNet approximation was installed.

SDPose includes detector execution, but its pose stage deliberately receives two
known synthetic crops per frame so every run exercises multi-person processing.
The synthetic drawings are not a detector-accuracy or real-person-quality test.
The same pose keypoints, confidence scores and rendered bytes were produced by
stock, the previous pack and the new feature-only route. The saved workflow's
1024-long-side input resize and video IO are outside these timings.

DA3 uses the saved Mono Large model and lower-bound resize policy: model input is
896×504 for this aspect ratio. Depth rendering is included. Renderer RNG is reset
before each timed render because stock v2-style normalization samples pixels.

Observed MPS driver allocation high-water marks were 8.75 GiB for BiRefNet,
7.25 GiB for SDPose, 1.73 GiB for default DA3 Mono and 2.29 GiB for DA3 Base.
These include allocator caching and are not total system-memory peaks. CPU RSS
and driver memory overlap in unified memory and must not be added together.
Guards stopped tests for an active user queue, low available memory, excessive
process RSS or a 20-minute per-process timeout. No guard was triggered.

## Accepted default: SDPose feature-only execution

The node takes its heatmap features from the last 640-channel U-Net feature,
before output block 9, but previously completed the remaining U-Net output path
and discarded the sampler result. A node-local model clone now stops after the
original capture hook. Normal sampler cleanup still runs; the unused sampler
output is a zero tensor. Ordinary sampling is not modified.

Alternating on/off confirmation retained identical keypoints and rendered outputs
on every call. See raw paired timings in the data file. CPU, unexpected U-Net
layouts, existing model-function wrappers and changed upstream source files fall
back to the original node. This route does not batch person crops or lower their
resolution. `ASFP8_SDPOSE_FEATURE_ONLY=0` disables it.

## DA3 Base multi-view compatibility fix

Both stock and the previous pack aborted in Metal on the two-frame Base test:
`Destination NDArray and Accumulator NDArray cannot have different datatype`.
Operator tracing identified the camera decoder's FP32 feature input multiplied
by FP16 output-head weights. The three camera heads now use ComfyUI's weight
casting while called on MPS, restoring their flags afterward. The intended FP32
head computation is retained; the image backbone remains FP16.

The fixed Base multi-view path completed eight frames in **0.122 s warm**, with
finite depth/rendered outputs. It used **504 on the longer side**, or 504×280,
with `middle` reference view and `cam_dec`. This is a different model and lower
resolution than the Mono Large timing; it is not a direct speed/quality comparison.
The host logs missing unused auxiliary ray-head weights for this checkpoint;
these tests used the camera-decoder route, not `ray_pose`.

A separate local eight-frame Base workflow is provided for user assessment. It
retains source FPS and omits audio because it exports only that short frame window.
No improvement in real-video temporal stability is claimed from the synthetic
compatibility test. Stock per-frame normalization can still cause visible flicker.

## Optional experiment: DA3 mono batches of two

`ASFP8_DA3_BATCH=2` enables bounded two-frame FP16 mono inference. **Default is 1.**
Landscape confirmation improved 0.416 → 0.362 s for eight frames and happened to
preserve depth/rendered bytes. A noisy portrait clip with three frames did not:
raw depth max/mean absolute error 0.0008574 / 0.0000164; rendered depth max/mean
error 0.3778 / 0.000511. Thresholded sky handling and normalization can amplify
small model differences. Do not infer visual equivalence from small raw error.

Batching remains experimental, not an accepted exact default. It falls back for
non-MPS, non-FP16, camera-enabled models, processing grids over 512K pixels, low
available device memory, or single frames. Batches are capped at two, preserve
frame order and check cancellation. This does not make loading an entire long
video into CPU image tensors memory-bounded. It does not window multi-view attention.

## Rejected screens

- BiRefNet shifted-window mask caching: exact, but no repeatable whole-node gain
  (two-frame screen 0.398 → 0.405 s).
- BiRefNet batch-of-two inference: output differences; not installed.
- BiRefNet Swin SDPA: about 0.352 s versus 0.398 s for two frames, but changed mask
  bytes. Prototype only; no visual quality claim or default replacement.
- SDPose person-crop batching: confidence/keypoint differences; not installed.
- DA3 CPU-first Lanczos preprocessing: exact on the screened input, but improvement
  was within timing noise; not installed.

Source guards match the tested ComfyUI files and retain stock behavior after
unrecognized host changes. `ASFP8_PREPROCESSORS=0` disables this module, including
the Base compatibility fix. Existing pack attention/device patches are separate.

Validation: 60 focused tests passed, three skipped (unavailable kernel-specific
paths). This includes attention-routing regressions, SDPose capture preservation,
exception propagation, camera flag restoration, odd frame batches, frame order,
low-memory fallback and source guards.

[Raw timings, errors and memory observations](data/vision-preprocessors-oct18.json).
Upstream adaptations: [ComfyUI](https://github.com/Comfy-Org/ComfyUI), GPL-3.0;
see the package third-party notices.
