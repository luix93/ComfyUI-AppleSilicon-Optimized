# MOSS SFT: grouped CFG projections

October 18, 2026 · M5 Ultra 80 GPU / 256 GiB · macOS 27 · PyTorch 2.14.1 ·
ComfyUI nightly `c06bf5f569d268b80076e5f44126f87cdf835869` after v0.41.3 ·
Comfy Kitchen 0.2.38.

## Decision

**Not promoted to a runtime option.** Grouping positive/negative CFG matrix
operations preserves tested output bytes, but the gain is small at 1024² and
inconsistent at 1248². No production patch, default or installed instance was
changed. This follows the optional FFN and dense-attention experiments; those
options were disabled throughout these comparisons.

## Implementation and boundaries

MOSS uses each prompt's true token count for positional encoding. The fixture
has 83 positive and 23 negative text tokens. Ordinary ComfyUI therefore runs
the two branches separately. Simply padding and attending over both padded
sequences would change the model calculation.

The diagnostic adapter carries both original lengths through CFG batching,
trims transport padding before text encoding, and prepares each branch's
positions, text embeddings and timestep modulation separately. It joins token
rows for QKV, output projection and FFN matrix operations, then splits views
for branch-local attention. The refined version retains the joined FC1 output
through SiTU and FC2, avoiding a wide split/rejoin copy. Neither attention nor
CFG arithmetic is approximated. All tokens, layers and steps remain.

The adapter is limited to the reviewed synthetic batch-one, global-conditioning
fixture. It is not a released generic CFG batching implementation. Model/block
operation ordering follows ComfyUI's GPL-3.0 MOSS implementation; existing
kernel attribution remains in THIRD_PARTY_NOTICES.md.

## Warm complete-render results

SFT INT8 ConvRot, Qwen3.5-9B INT8 ConvRot, Flux.2 small VAE, ER-SDE/simple,
25 steps, CFG 2.5, seed `856771198829162`. Same synthetic theatre-marquee prompt
as the earlier MOSS experiments; prompt enhancer and SLG off. These compare
against the **previous optimized exact path**, not stock ComfyUI.

Each resolution has one-step warmup/verification runs, then exact, grouped,
grouped, exact full runs in one resident server. Means of two full runs:

| Resolution | Existing exact E2E | Grouped E2E | E2E reduction | Existing sampling | Grouped sampling |
|---|---:|---:|---:|---:|---:|
| 1024×1024 | 22.180 s | 21.613 s | 0.567 s / 2.55% | 20.463 s | 19.812 s |
| 1248×1248 | 33.979 s | 33.914 s | 0.065 s / 0.19% | 32.120 s | 31.876 s |

At 1024², E2E controls were 22.046 / 22.313 s; grouped runs were
21.659 / 21.567 s. At 1248², controls were 33.841 / 34.117 s and grouped
runs 34.170 / 33.658 s: overlapping ranges, with one grouped run slower.
The larger-resolution result is not a reliable practical speedup.

Sampling measurements include concatenation, allocations, ConvRot, rowwise
quantization, matrix execution, attention layouts and CFG handling. E2E also
includes prompt encoding, VAE decode and PNG saving. Tensor capture/hash work
is outside the measured interval. Instrumented screening-job E2E times are
not generation benchmarks because they include repeated operator experiments.

## Screens and final refinement

Actual positive/negative inputs to the first block were captured from a
one-step run at each resolution. Each candidate was warmed and checked, then
timed in 12 shuffled groups of three executions, including input clones,
required joins, allocation and complete block operations.

- Naive grouping of all four projections was neutral/slower: 19.384 versus
  19.391 ms at 1024²; 30.901 versus 30.741 ms at 1248². Grouping QKV, output
  projection or FFN separately did not provide a substantial gain.
- Retaining the joined FC1 intermediate through FC2 reduced copy overhead.
  This was the adapter tested through full denoising above.
- A final screen kept the hidden state joined between blocks and used the
  identical shared timestep modulation for both row segments. Attention still
  received separate views and original positions. First-block outputs matched
  byte-for-byte. At 1024² it measured **18.770 versus 19.493 ms (3.71%)**;
  at 1248², **29.812 versus 30.444 ms (2.08%)**. These are operator measurements,
  **not complete-render timings**. This final variant was not run through full
  denoising or accepted: it still screens in the small-gain range the user
  declined to pursue. Its model adapter remains an unvalidated local prototype.

Grouping does not remove the arithmetic of either CFG branch. Each existing
matrix already processes more than 4,000 or 6,000 token rows. Combining them
reduces launch and tail overhead, but does not halve matrix work. The results
do not establish a hardware-counter diagnosis of register pressure or stalls.

## Correctness and memory

For the full-render adapter, all 42 paired blocks match their separate-branch
reference during one-step verification at both sizes. All full-run
conditioning, initial noise, input latents, final sampler latents and decoded
float frames match their exact controls byte-for-byte. Each grouped 25-step
run executes 1,050 paired blocks and retains all **2,100 attention calls**.
This is evidence for these fixtures, not universal equivalence across models,
hooks, LoRAs or arbitrary CFG arrangements.

The isolated server checks that the user server stays idle and enforces
110 GiB process RSS / 48 GiB available-memory safeguards with MPS watermarks
0.4 / 0.3. No guard tripped. Peak sampled Metal driver allocation during the
full suite was **29.90 GiB**. Process RSS peaked at 1.14 GiB; RSS alone omits
much Metal memory and must not be interpreted as total model memory. Allocator
reuse makes per-run driver readings unsuitable for claiming a memory reduction.

[Raw repetitions, full timings, counters, tensor hashes and guards](data/m5-ultra-moss-cfg-projections.json).
