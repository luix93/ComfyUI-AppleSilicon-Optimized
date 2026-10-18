# SAM3.1 segmentation — M5 Ultra, 2026-10-18

ComfyUI v0.41.2, PyTorch 2.14.1, M5 Ultra 80 GPU cores / 256 GB.
Checkpoint: `sam3.1_multiplex_fp16.safetensors`, FP16 execution.

## Workflow and test boundary

The inspected utility workflow passes all video frames to **SAM3_Detect**, with
text conditioning, threshold 0.5, two refinement iterations and union masks.
It does not use SAM3_VideoTrack. The detector resizes each frame to 1008×1008
internally and separately refines detected-object crops at 1008×1008.

Testing used two authored 864×480 geometric frames and the prompt “a red circle.”
Threshold was 0 in the benchmark only, ensuring both frames exercised refinement;
the saved workflow was not modified. Actual pretrained weights were used, but
no user input video or images were opened. A memory/queue guard protected the
test. Timings cover the complete Detect node, including resizing, model
management, detection, refinement, mask transfer and aggregation. Checkpoint
loading, text encoding, video loading and preview/export are excluded.

The before/after baseline uses the existing pack's applicable attention/RoPE
patches. Unrelated model patches were excluded from benchmark startup. This
is **not a stock-ComfyUI comparison**.

## Accepted router fix

SAM's refinement cross-attention uses head dimension 16, with query/key lengths
8 and 5184 (and the reverse). The router attempted the cooperative Flash41
kernel, but its shared Metal library also compiled FP8 entry points whose
static tensor extent must be divisible by 32. Compilation therefore failed
even though this was an FP16 call. The router caught the exception and used
its bmm fallback, then repeated the failed compilation on the next call.

Twenty failed attempts across four refinement calls cost approximately 3.54 s
in the synchronized stage profile. The fix prevents unsupported, non-32-aligned
head dimensions from entering this library. It directly selects the **same bmm
fallback that already supplied the result**. Supported 32/64/96/128-wide heads
retain their existing route. No precision, attention arithmetic, resolution,
threshold or refinement-count change is introduced by this fix.

## Warm results

Three alternating rounds per mode; discard round one and take the median of
the remaining two. Masks (all bytes) and bounding-box/score outputs matched
in every measured run.

| Mode | Two-frame Detect time | Relative to previous route | Status |
|---|---:|---:|---|
| Previous pack route | 4.22 s | 1.00× | Baseline |
| Corrected router | 0.630 s | 6.69× | Implemented |
| Corrected router + refinement feature reuse / unused feature-branch removal | 0.427 s | 9.88× | Diagnostic prototype only |

These are synthetic tests of one detected object per frame. They do not establish
a full-video speedup or broad real-video quality validation. Frames with no
detection do not incur the same refinement/failed-compilation costs.

## Additional candidates

- `_refine_mask` passes the identical crop through the full image backbone on
  both iterations. A cache scoped to that single refinement operation removes
  the duplicate encoding. The prototype retains only the current crop's feature
  tensors and clears them immediately afterward; it does not accumulate video
  features. Combined with skipping discarded feature branches, this saves a
  further approximately 32% after the routing fix. Production integration still
  needs lifecycle, input-mutation, extension-hook and source-compatibility guards.
- Detection computes tracker-neck features it discards; segmentation computes
  detector-neck features it discards. The prototype avoids both without changing
  the used features. Included in the final prototype row.
- Selecting the winning detection mask before resizing all predicted masks
  preserved tested outputs but provided only a small marginal difference in
  this 480p test. It was not integrated; higher-resolution output may warrant a
  separate check.
- Video tracking and fewer refinement iterations change behavior and require
  quality assessment. Neither is part of the accepted fix or reported speedup.

## Profile interpretation

Before the fix, a synchronized two-frame profile took 4.26 s. Refinement heads
accounted for 3.61 s, predominantly failed attention compilation. The six image
backbone executions totaled 0.51 s. Mask upscaling and prompt encoding were
small contributors. Profile timings overlap and must not be summed as
independent phases.

Router regression tests cover unsupported head sizes in both cross-attention
directions and retention of supported Flash41 routes. The full-node benchmark
verifies final masks and boxes; it does not claim byte comparisons at every
internal tensor. The unchanged fallback arithmetic explains why the routing
repair does not add approximation.

The combined MPS routing, attention-backend and Qwen regression suites passed
**126 tests**, with three opt-in Flash41 integration tests skipped. The SAM3
full-node GPU checks ran separately as described above.

Raw results and profile: [JSON](data/sam3-oct18.json).
