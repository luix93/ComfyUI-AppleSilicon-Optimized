# H3 Apple Silicon Acceleration

Connect this node between the H3 model loader/LoRA loader and the guider. Each mechanism is optional. Select **Exact · no approximation**, or set **Temporal cache** to Off and disable **Token reduction** and **Sparse attention** to retain the exact optimized model path. These workflow controls are separate from the older `ASFP8_MINIMAX_RELAXED` environment preset; the measurements below keep that preset off.

## Simple node

Use **H3 Acceleration · Simple** for three standard dropdowns, a reference/guide approximation switch and a
logging switch. It is compact on first placement in Nodes 2.0 and the classic canvas;
there are no hidden controls or expandable panels. It shares the advanced
node's implementation and keeps each selected mechanism independent.

| Selector | Recipes |
|---|---|
| Temporal cache | Off; MotionCache protected / balanced; FirstBlockCache protected / fast; Spectrum protected / fast draft |
| Attention acceleration | Off; SOL Metal conservative / balanced / fast / hybrid fast / top-k wide / top-k all-step draft; Dense Metal protected / balanced / full experimental; VSA Metal · FastH3 Omni (128) |
| Token reduction | Off; Protected / Balanced / Light / Fast protected / Fast aggressive (experimental) |

Defaults: **MotionCache balanced**, **sparse Off**, **tokens Off**, **Detailed logs Off**, **Approximate reference / guide attention Off**.
Use **one** Simple or advanced acceleration node between model/LoRA loader and
guider. All three Off returns the incoming model unchanged. Other nodes or
environment approximations still apply. Choose the advanced node when you want
custom settings or a manual temporary-memory ceiling. The Simple node uses
**Auto** memory.

**SOL Metal** is the INT8 SOL sparse attention and **Dense Metal** the dense
INT8 v3 attention (Kitchen-style numerics), both native Metal kernels on M5.
Earlier names (`SOL INT8 · …`, `Dense INT8 v3 · …`) still load and run the same
recipes. Removed attention recipes (October 23, 2026) also still load and run
the nearest remaining recipe, with a log line: Dense INT8 v1/v2 and Dense QK
INT8 / PV BF16 map to the matching Dense Metal recipe, SOL hybrid balanced to
SOL Metal balanced, Veda protected / fast draft to SOL Metal fast / top-k wide,
and VSA ComfyUI (64) to VSA FastH3 Omni (128). The advanced node keeps the Dense INT8 v1,
v2 and floating-PV backends selectable.

### Logging

New Simple and advanced nodes default to summary-only logging. Each enabled
mechanism prints one compact final summary. MotionCache reports model calls
skipped and estimated DiT speedup. FirstBlockCache accounts for the first block
it still computes; its estimate concerns transformer block compute. Spectrum
counts real DiT work, including discarded/rollback calls; archived-output replay
is not added to the native baseline. Cache/predictor/routing overhead is excluded,
so these are work estimates, not measured end-to-end speedups. Do not multiply
estimates from combined modes.

SOL reports attention work omitted; token reduction reports evaluations
reduced and token counts. These do not have a reliable wall-time speedup estimate
without a matched dense run. Inactive routes report zero acceleration. Routine
startup/configuration, protected-window fallback and per-step diagnostics are
hidden; real errors, unavailable kernels and incompatible inputs still warn.
Turn **Detailed logs** on in Simple, or the individual logging switches in the
advanced node, for diagnostics. Existing workflows retain their saved logging
values: turn those switches off to use summaries only. All selectors Off adds no
acceleration log output. ComfyUI's own progress/model-loading logs remain.

| Recipe | Fixed settings |
|---|---|
| MotionCache protected | Threshold 0.10, window 0.25–0.85, max one skip, four warmup evaluations |
| MotionCache balanced | Threshold 0.15, window 0.15–0.95, max two skips, four warmup evaluations |
| FirstBlockCache protected | Cumulative threshold 0.10, window 0.25–0.85, max one skip, audio veto 1 |
| FirstBlockCache fast | Cumulative threshold 0.20, window 0.15–0.95, max two skips, audio veto 1 |
| Spectrum protected / fast draft | Existing first 4/final 2 or first/final 1 actual-evaluation recipes, blend 0.5 |
| SOL Metal conservative | INT8 corrected, tau 0.25, sigma window 0.50–0.85 |
| SOL Metal balanced / fast | INT8 corrected, tau 0.5 / 1.3, step fraction 0.20–0.90 |
| SOL Metal top-k wide | Fixed 15% video-block budget, step fraction 0.10–0.95; protected layers retained |
| SOL Metal top-k all-step draft | Fixed 10% video-block budget, every step/layer eligible |
| Dense Metal protected | Kitchen-style INT8 attention (Q/K rotation, per-tile P), step fraction 0.20–0.90; protected layers stay exact |
| Dense Metal balanced | Same kernel, step fraction 0.10–0.95 |
| Dense Metal full experimental | Same kernel in every step/layer; can change composition |
| SOL Metal hybrid fast | SOL Fast in ordinary layers, dense INT8 in protected layers inside steps 20–90%; protected steps stay exact |
| VSA Metal · FastH3 Omni (128) | FastH3 trained gates, 128-token tiles, 10% video keep; cache and tokens Off |
| Token Protected | Early/later restore at 12/8, step fraction 0.20–0.90 |
| Token Balanced | Early/later restore at 30/20, step fraction 0.10–0.95 |
| Token Light / Fast protected | Horizontal pairs after block 3; early/later restore at 20/12 or 33/30, scale 1, first ten evaluations considered early |
| Token Fast aggressive | Horizontal pairs after block 3; restore at 40 for the first half of sampler steps, capped at ten steps, then at 30; scale 1 |

SOL conservative/balanced/fast/top-k wide, hybrid fast and Dense Metal protected/balanced keep layers
**0,1,33,34,35** dense, with two leading blocks dense; SOL top-k all-step draft and Dense Metal full
experimental have neither protection. Every sparse recipe keeps
attention dense below **8192 packed tokens**, including during token reduction.
Thus a short clip can show no sparse gain, and a combined recipe can use sparse
attention only outside the reduced span. These thresholds can be changed in the
advanced node. Sparse selection has overhead; enabling more mechanisms is not
a guarantee of more speed.

Cache/attention recipes target **20 steps without Turbo**. The Fast token recipes
also have limited Turbo I2V screening described below. Protected means more original work
is retained, not a quality guarantee. Fast draft and token recipes can alter
identity, composition, detail, motion and audio. MotionCache protected and Token
Light are newly screened on short authored clips; the other cache/sparse
recipes retain the previous full-clip measurements linked below. Token reduction
stays off by default. [Current validation and comparisons](../benchmarks/m5-ultra-h3-presets.md).

### Reference and guide attention

**Approximate reference / guide attention** is available directly on the Simple
node and under **Sparse attention → Settings** on the advanced node. It is
**off by default** and applies to **SOL INT8** presets (including hybrid
and top-k), **Dense INT8**, and **Dense QK INT8 / PV BF16**, both protected and
full variants. It has no effect with attention Off or BF16 SOL.

- **Off:** all reference/guide/text/audio queries use exact attention. In SOL,
  prefix keys remain dense and only generated-video interactions are sparsified.
- **On, SOL:** visual guide and reference-image/video queries and keys may use
  existing quantization, sparse routing and pooled correction. Text/audio keys
  remain dense.
- **On, dense:** visual reference queries use the selected dense kernel: INT8
  QK/PV or INT8 QK with BF16 PV. Every key connection is retained; no sparse
  routing is introduced. Text/audio query rows are recomputed from the original
  BF16 Q/K/V inputs. Dense quantization already applies to all keys used by
  approximate queries; this is not a promise that those keys use exact arithmetic.

In both paths, text/audio query rows remain exact for the same Q/K/V inputs.
Boundary tiles round outward to protect complete 64-token blocks. This selective
visual switch does **not** reproduce ordinary Kitchen's whole-sequence
quantization policy, which also approximates text/audio query rows.

The selected preset's sampling-step and layer protections still apply. Hybrid's
protected-layer dense INT8 path retains the original reference protection.
Unknown or changed packed layouts fall back to full prefix protection, including
when token reduction changes the sequence length. Pure text-to-video has no
visual reference to relax. The completion log reports how many calls actually
used the switch.

This is an additional approximation: it can change reference fidelity, motion,
background geometry and final audio through later layers, despite directly
protecting audio attention. It is not a quality guarantee. Full-video guides can
occupy about half the sequence; protecting their queries and keys then protects
roughly three quarters of the attention matrix. More total tokens alone therefore
does not imply a larger SOL speedup.

SOL's exact protected-prefix attention also uses the validated rectangular query-cache
on M5 / PyTorch 2.14.1, within its tested geometry limits. This changes scheduling,
not arithmetic. `ASFP8_H3_PREFIX_QCACHE=0` restores native prefix SDPA for debugging;
unsupported builds/shapes fall back automatically. This optimization is independent
of the reference-approximation switch.
[Matched guide/reference timing and quality tests](../benchmarks/m5-ultra-h3-reference-attention.md).

### Additional SOL and token controls

**Top-k video fraction** in the advanced node selects a fixed fraction of generated
video key blocks per query block. **0** retains the existing tau route; a positive
value overrides tau. Global conditioning/audio blocks, outward-rounded boundary
queries and local neighbors are retained in addition to that budget, so measured
attention density exceeds the nominal fraction. Rejected blocks still contribute
through the existing pooled tail. With reference approximation off, no reference/audio prefix is discarded. With it
on, the budget covers visual guides/references too; text/audio remain protected.

Top-k uses a deterministic Metal sort with index tie-breaks. Its maximum is
**262,144 packed tokens** because the router's workgroup sort uses up to 32 KiB
of shared memory. This is specific to the optional top-k router; existing tau
routing and its memory controls are unchanged. Temporary memory uses the existing
Auto/manual policy. Shapes beyond the top-k capacity report a kernel fallback;
choose fraction 0 to use tau routing instead.

At 20 steps, **top-k wide** keeps the first **2** and last **1** steps dense,
while **top-k all-step draft** makes every step and layer eligible. Dense protected
queries/global keys still apply inside the Draft kernel, subject to the separate
visual-reference switch described above. Both change inference
bytes. More coverage can change composition, color, motion and background stability;
Draft is a speed/quality tradeoff, not a quality-preserving replacement for Exact.

**Quality feedback (October 7):** Top-k Wide caused background morphing in a
user test where **SOL Metal · fast** did not. Prefer Fast for that workflow; Wide
is experimental, not a recommended stability upgrade. Fast uses adaptive tau
1.3 routing with the first four and final two of 20 steps dense. Wide changes
both routing and the step window, so the cause is not isolated. A controlled
comparison would hold Fast's 0.20–0.90 window/layers fixed and change only the
routing before testing a wider window. This comparison has not been run.

Token **Protected** reduces only eight early / four later blocks, restoring before
blocks 12/8; the first four and final two of 20 sampler steps keep full tokens.
**Balanced** reduces 26 early / 16 later blocks, restoring before blocks 30/20;
the first two and final one steps keep full tokens. Both use the existing horizontal
pairing and original per-pair detail restoration. They do not use a new ranking or
content-adaptive merging algorithm. "Protected" describes retained compute, not
identical output or guaranteed quality.

The advanced token **Start/End step fraction** controls use actual sampler sigmas,
keeping repeated CFG calls at the same timestep together. Protection rounds outward
on short schedules. Light and both Fast recipes use start 0 / end 1.
**Fast protected** retains the corrected Fast recipe: restore before block
33 in its early phase, then before 30 after ten model evaluations.
The old Fast recipe reproduced repeated outlines and banded edges in a 10-step
Turbo I2V test. Endpoint protection alone did not remove it; earlier restoration did.
Saved Simple nodes selecting the old Fast label map to **Fast protected**, including
API workflows. Reload the browser for the renamed dropdown.
Advanced nodes retain their explicit values: change **Early Restore Block** from
40 to **33** there. Historical Fast benchmarks measured the old 40/30 recipe.
See the [reproduction and fix report](../benchmarks/m5-ultra-h3-fast-token-fix.md).

**Fast aggressive** opts into deeper reduction before block 40, but only for the
first `min(10, floor(steps / 2))` sampler steps; remaining steps restore before
block 30. It uses the actual supplied sigma schedule, so repeated CFG calls and
cache-skipped evaluations do not extend the early phase. The block indices refer
to layers inside each model call, not sampling steps. At one sampling step it uses
only the gentler block-30 phase. With partial denoising, the supplied schedule
determines the step count.

| Sampling steps | Restore at 40 | Then restore at 30 |
|---|---|---|
| 4 | First 2 | Last 2 |
| 8 | First 4 | Last 4 |
| 10 | First 5 | Last 5 |
| 20 | First 10 | Last 10 |
| 40 | First 10 | Last 30 |

The advanced node exposes **Early phase policy** (`evaluation_count` or
`capped_steps`) and **Early step fraction**. In capped mode, **Early evaluations**
acts as the maximum number of early sampler steps. Existing advanced workflows
retain the original evaluation-count policy.

The tested alternative, restoring at 40 while sigma remained at least 0.7,
selected nine of ten steps on H3's shifted schedule and reproduced duplicated
outlines. A noise threshold is not a reliable proxy for the percentage of steps.
The capped policy limits this exposure; it does not guarantee identical quality.
Both Fast modes remain optional and approximate. In the five-second test,
Aggressive saved only 2.3% sampling time and added visible repeated ridges on
objects; prefer Protected when detail matters. [Adaptive Fast tests and timings](../benchmarks/m5-ultra-h3-token-dynamic.md).

Defaults remain
MotionCache balanced alone, sparse Off and tokens Off. All three selectors remain
independently combinable.

[New SOL/token measurements and limitations](../benchmarks/m5-ultra-h3-sol-topk.md).

### Token compatibility

The original reducer tried converting per-token modulation tensors to Python
integers and assumed a single scalar final-video segment. Masked H3 inputs can
instead carry a vector of modulation indices. Pair-compatible masks now retain
those indices at reduced width. If two horizontal partners have different
modulation indices, that evaluation keeps its original tokens and logs the
reason, avoiding a mixed conditioning timestep.

SOL counts now follow the actual reduced sequence
and return to full geometry when tokens expand. Conditioning's shared layout is
never modified. Cleanup restores metadata even if a block is interrupted.
A summary logs evaluations, actual reductions, full/reduced row counts and mask
bypasses. Token reduction alone and the existing MotionCache/Spectrum/SOL
combinations retain their earlier unmasked output bytes in the matched screening
checks.

[Download the single-node template](../workflows/h3_acceleration_simple.json)
and connect its model sockets to your loader and guider.

## Presets and defaults

The node uses a compact panel in both Nodes 2.0 and the classic canvas. Three
cards expose **Temporal cache**, **Sparse attention**, and **Token reduction**.
Each active card has its own collapsible **Settings**. There are no empty rows
for hidden controls. **Starting point** applies a tested recipe; you can then
combine the mechanisms and tune their values independently.

The default **MotionCache · recommended** uses the measured 0.15 recipe below.
It gave the closest content preservation in our recent tests, but remains
approximate. Token reduction and sparse attention default off. MotionCache and
Spectrum are included: no extra installation or workflow nodes are needed.

| Preset | Applied configuration |
|---|---|
| MotionCache · recommended | MotionCache 0.15 alone; four warmup evaluations, maximum two skips |
| Exact · no approximation | All three mechanisms off; exact startup kernels remain active |
| Spectrum · protected | Spectrum alone; first four and last two evaluations actual |
| Spectrum · fast draft | Spectrum alone; first and last actual, bootstrap enabled |
| FirstBlockCache · 0.20 | Cumulative policy, threshold 0.20, window 0.15–0.95, maximum two skips, audio veto 1 |
| SOL Metal · fast | INT8 corrected, tau 1.3, step-fraction 0.20–0.90, layers 0,1,33,34,35 dense, minimum 8192 tokens |
| SOL Metal + FirstBlockCache | Same SOL protection at tau 0.5; cache 0.10, window 0.25–0.85, maximum one skip, audio veto 1 |
| Custom combination | Indicates manually combined or tuned values; leaves those values intact |

All approximate presets were measured at **20 steps without a Turbo LoRA**.
They are not quality guarantees or recommendations for 4/6/8-step Turbo runs.
Choose **one temporal cache** (MotionCache, Spectrum or FirstBlockCache), then
optionally add SOL or dense INT8 attention, token reduction, or both. The temporal
predictor is applied first; actual model evaluations use the selected inner
mechanisms. MotionCache and Spectrum are alternative predictors, so this node
does not stack them with each other. Combinations compound approximation and
need their own visual/audio comparison. Select Exact for this node's pass-through
path; other workflow nodes and environment flags still apply.

Saved workflows preserve their values, including older sparse backend/window
semantics. The starting point is inferred from active settings; unmatched values
show Custom combination. Selecting a recipe explicitly resets its numerical
settings. On a new node, first enabling sparse attention initializes the tested
SOL INT8 recipe. Loaded workflows keep their saved sparse values. Each Settings
panel remembers whether it was open. Presentation controls are excluded from
backend inputs. Connected widget inputs remain in control; disconnect them
before applying a recipe.

[Nodes 2.0 checks, bundled-source parity and short combination renders](../benchmarks/m5-ultra-h3-node-v2.md).

## Attention acceleration

### FastH3 Omni VSA on Metal

**VSA Metal · FastH3 Omni (128)** uses the checkpoint's trained compression
gates and 128-token video cubes. It is a separate backend from SOL;
SOL is not a substitute for this trained VSA branch. Current Metal execution
requires M5 Ultra and the reviewed PyTorch 2.14.0/2.14.1 builds.

For an existing FastH3 Omni reference workflow, replace **Model Sparse
Attention** with **H3 Acceleration · Simple**. Keep the model input after
the loader/sigma-shift/dense-backend nodes and connect the output to both
the guider and scheduler. Set:

- Temporal cache: **Off**.
- Attention acceleration: **VSA Metal · FastH3 Omni (128)**.
- Token reduction: **Off**.

Use eight Euler/simple steps, guidance 1, video/audio shifts 12/3 and a
FastH3 checkpoint carrying `to_gate_compress` weights. Select **pytorch
attention** in Model Attention Backend on Mac. No additional predictor is
downloaded. [Example API workflow](../workflows/video_fasth3_omniref_metal_vsa.api.json).

The recipe retains 10% of target-video tiles and, independently, 10% of each
reference video's tiles. Image references, text and audio remain dense in the
fine branch. The learned pooled branch is added as trained, including on
prefix rows. Final output and audio are not identical to dense inference.
The reference/guide approximation switch used by SOL does not control VSA.

The advanced node's **VSA tile size 64** keeps ComfyUI's 64-token geometry with
all reference conditioning dense, as a comparison. It uses floating-point Metal
fine attention, so matching geometry does not establish byte parity with
Kitchen's CUDA INT8 VSA implementation.

[Measured record](../benchmarks/data/m5-ultra-fasth3-omniref-vsa.json).

### Dense Metal and the SOL Metal hybrid

**Dense Metal** retains every key token and uses the general M5 INT8 kernel
with Kitchen-style numerics (Q/K rotation, per-tile probabilities). **Protected**
keeps the first four/final two steps of a 20-step run and layers 0,1,33,34,35
exact; **balanced** keeps the first two/final one; **full experimental** applies
the kernel in every step/layer and can change scene composition even when
single-operation errors are small. **SOL Metal · hybrid fast** uses SOL Fast
(tau 1.3) in ordinary layers and the Dense Metal (v3) kernel in the protected
layers, only inside the active 20–90% window; its first four/final two steps
remain exact. `ASFP8_H3_HYBRID_DENSE=v1` restores the original v1 kernel there.
[v1 vs v3 measurement](../benchmarks/m5-ultra-h3-hybrid-v3.md).

All of them retain exact packed-prefix query attention, rounded outward to a
64-row boundary. This protects the direct prefix calculation, not final audio
bytes: changed video information can propagate through later layers. Every
recipe is non-bit-exact, uses Auto memory, and keeps the 8192-token eligibility
gate. Exact remains available by choosing Off; defaults have not changed.
[Design, limits and measurements](int8-attention.md).

The head-major SOL layout removes copies without changing the prior SOL's
output bytes. `ASFP8_SOL_LAYOUT=legacy` provides a diagnostic rollback. This is
an exact implementation improvement **within SOL**, whose arithmetic remains
approximate relative to native attention. No token-rescue or cumulative-mass
router is enabled by these presets; both remain screened research experiments.
[Matched clip timings, numerical errors and per-preset temporal assessment](../benchmarks/m5-ultra-h3-kitchen-attention.md).

### Other dense backends (advanced node)

The first Dense INT8 kernel, **Dense · INT8 v2 (M5)** and **Dense · INT8 QK /
BF16 PV (M5)** remain selectable as the advanced node's attention method. Their
Simple presets were removed on October 23, 2026: v3 matches or beats them on
error at equal or better speed, and floating PV was only 1.05–1.08× faster than
exact. Historical measurements:
[Kitchen/dense/hybrid](../benchmarks/m5-ultra-h3-kitchen-attention.md) ·
[floating PV at 1024×512](../benchmarks/m5-ultra-h3-floatpv.md) ·
[INT8 v2/v3](../benchmarks/m5-ultra-int8-attention.md).

### Existing sparse implementations

**Attention method = SOL · INT8** selects the M5 INT8 SOL implementation. This quantizes attention Q/K/V and approximates dropped video-token blocks with summaries; it is non-bit-exact even when the diffusion checkpoint already uses INT8 weights. **SOL · BF16** retains the older floating-point sparse implementation. Higher **Routing Tau** generally computes fewer exact token blocks, with a larger approximation budget.

### Temporary memory

Open **Sparse attention → Settings → Temporary memory (GiB)** when using the
`int8` kernel. This replaces the former hardcoded 4 GiB scratch limit.

- **0 = Auto (default):** at most 12.5% of physical RAM, capped at half the safe
  available headroom. On a 256 GiB Ultra, the upper ceiling is **32 GiB**.
- **Positive value:** a manual ceiling, for example **16** for up to 16 GiB.
  It still respects current available memory and Metal's recommended working
  set, retaining at least 2 GiB or 5% of physical RAM, whichever is larger.

This is a ceiling for estimated temporary attention buffers, including layout
copies, routing/reduction workspaces and outputs. It is **not preallocated** and
does not include the model's resident weights or the attention inputs. The
estimate is conservative; actual use can be lower when layout copies are views.
Memory pressure elsewhere in the workflow can still affect the run.

The value is saved with the workflow and survives changing acceleration recipes.
Older workflows without this field use Auto. Raising the ceiling changes no
kernel arithmetic, but it can enable approximate SOL where the old limit forced
exact dense attention, so the resulting clip can change.

If a call does not fit, the log gives the estimated requirement, effective budget,
token/head count and safe headroom. That call uses dense attention; a memory-budget
rejection **does not latch SOL off** for later calls that fit. Genuine unavailable
kernel errors still latch off for the current run. To reproduce the old allowance,
set **4**; the updated estimate includes additional temporary output buffers.

API workflows use `sparse_memory_budget_gib` on
`AppleSiliconH3UnifiedAcceleration`. Restart ComfyUI and refresh the browser after
updating the node to expose the new control.

Validated on M5 Ultra with a **65,536-token, 48-head** direct attention call:
the conservative estimate was **6.15 GiB**; a 4 GiB ceiling rejected it before
allocating scratch, while 8 GiB and Auto both completed with byte-identical
outputs and routing counts. Protected-prefix/ragged tests also matched across
budgets. This validates the memory control, not a new complete-clip speed claim.
[Validation data](../benchmarks/data/h3-sol-memory-budget.json).

**Protected Dense Layers = 0,1,33,34,35** keeps those zero-based transformer layers on the current exact attention path. Prefix queries (including text and audio) use exact attention in the INT8 route, although video-query attention and subsequent cross-stream interactions can still change audio and video.

**Window Schedule** has two choices:

- `sigma_percent` preserves the behavior of existing workflows. Start/end are converted through the model's diffusion-time schedule. These percentages need not correspond to the same numbered sampler steps across schedulers, and float32 endpoints can move a boundary by one step.
- `step_fraction` derives the window from the actual sampler sigma schedule. **Start = 0.20, End = 0.90, 20 steps** selects steps **5–18**: first four and final two remain dense. Fractions that land between steps round toward more protection. Repeated CFG evaluations use the same sigma window rather than advancing a model-call counter. A finite, strictly decreasing schedule is required.

The 14-step recipe uses `corrected`, `int8`, `step_fraction`, start/end `.20/.90`, leading dense blocks `2`, and protected layers `0,1,33,34,35`. Select tau and any cache combination using the measured tradeoffs in the [wider-window benchmark report](../benchmarks/m5-ultra-h3-sol-aggressive.md). The earlier seven-step recipe remains available. Keep **Minimum Sequence Tokens = 8192** to avoid spending sparse-kernel overhead on very short clips; diagnostic short-clip quality tests explicitly override this to zero.

## Cache method

- `first_block_cache`: block 0 runs on every evaluation. Changes in its video and audio residuals decide whether blocks 1–49 may reuse a cached aggregate residual. Change accumulates across skipped evaluations; the threshold, window and maximum consecutive skips constrain reuse.
- `first_block_cache_per_step`: tests the latest change rather than accumulated change, using a full detector instead of the subsampled detector. It can reuse more often and is a more aggressive policy at the same threshold.
- `oracle` and `block_oracle`: measurement modes. They run the full computation and record prediction errors; **they do not accelerate sampling**. With the other mechanisms disabled, both produced byte-identical sampler latents, frames and audio in the compatibility test.

Cache thresholds are not output-error guarantees. Both actual cache modes are non-bit-exact. The audio-change veto helps constrain reuse; it does not prove perceptual audio quality. Reuse counts are reported in the console. State is reset between sampler runs.

## Token Reduction

This averages adjacent horizontal target-video tokens inside the transformer, preserving the packed prefix and restoring full video resolution before the final blocks. **Early/Later Restore Block** controls the reduced span. **Update Scale** changes how the pooled update is applied; set the toggle off to disable the mechanism. It can combine with any of the temporal cache choices and sparse attention, but the quality changes accumulate. It is a separate approximation, not a bit-exact memory optimization.

## Compatibility fix

Current ComfyUI's block-replacement dispatcher passes `attention=None`. Our fused block previously omitted that keyword, so enabling cache or token reduction could fail with `fast_forward() got an unexpected keyword argument 'attention'` while ordinary sampling succeeded. The wrapper now accepts the argument and honors actual custom attention callables through upstream. Its ordinary arithmetic is unchanged. Restart ComfyUI and refresh the browser after updating.

The public cache modes, token reduction, and combined cache/token/sparse settings have been exercised on complete short workflows. Exact output was recovered after the approximate runs, checking that state did not leak into subsequent sampling. Timing and visual differences are documented separately from compatibility success.

## Spectrum (included)

The pinned [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
v0.2.28 algorithm is maintained in our own internal runtime. Choose a Spectrum mode in **Temporal cache**;
no additional node is needed. Avoid chaining a separate forecasting/cache node.

- **Protected**: first four and last two evaluations fully computed; no bootstrap.
- **Fast**: first and last fully computed, bootstrap enabled. Larger appearance
  changes were observed in the measured full-clip tests.

Both keep upstream replay enabled and history/archive on the model device (MPS).
**Video blend weight** is editable under Settings and defaults to 0.5. Other
Spectrum parameters retain the measured recipe. Exact native kernels still run
on actual evaluations. The separate upstream node remains useful if you want
its full research control set, but is not needed by this node.

These are 20-step, non-Turbo measurements; first-four/last-two protection leaves
little or no room for forecasting at 4/6/8 steps.
[Full-clip timings and visual comparisons](../benchmarks/m5-ultra-h3-recent-methods.md).

## MotionCache (included)

The pinned [MiniMax-H3-MotionCache](https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache)
algorithm is maintained in our own internal runtime. Defaults: threshold 0.15, four warmup evaluations,
maximum two consecutive skips, window 0.15–0.95, motion strength 1, subsample
factor 8. Settings exposes the threshold, motion strength, warmup, skip limit
and window. This reuses full model residuals; FirstBlockCache instead computes
block 0 as its detector.

Keep ComfyUI's native **VAE Decode** node. The upstream MotionCache pack's
**Fast VAE Decode** is not included; its older tiling implementation was not
accepted in our current-core tests. None of these cache choices changes the
default exact decoder.

## Bundled source and licenses

All three runtimes load only when selected and require no runtime download. Existing
external provider installations can remain for older workflows. Source commits,
hashes, attribution and full license texts are included. MotionCache is MIT;
Spectrum is GPL-3.0-or-later, so the combined distribution uses GPL-3.0
while retaining existing MIT notices. See [third-party notices](../THIRD_PARTY_NOTICES.md)
and [bundled license](../LICENSE-BUNDLED).


## Internal runtime ownership

MotionCache and Spectrum now execute from `_patches/h3_runtime` through
our combined node's direct APIs. No upstream custom-node class is instantiated
and no extra nodes are registered. Existing workflow IDs, field order, settings,
cache/attention composition and measured presets are unchanged.

MotionCache retains its detector/reuse arithmetic while batching scalar GPU
readbacks. It supports keyword inputs and guarantees restoration of sampler options if preparation raises. Spectrum
retains forecasting, offline replay and the compatibility hook order; its unused
objective-media node schemas/tools are omitted.

`_patches/_vendor` is retained only as a pinned comparison reference. Production
imports do not use it. `_patches/h3_runtime/ORIGINS.json` maps derived files to
upstream inputs, and full original licenses remain alongside each runtime.
Upstream updates are reviewed/ported explicitly, then checked against pinned
sources and representative complete H3 renders. This is a runtime rebuild, not
predictor retraining or a change to the approximation/quality tradeoff.

[Rebuild parity checks and complete-render timings](../benchmarks/m5-ultra-h3-owned-runtime.md).
