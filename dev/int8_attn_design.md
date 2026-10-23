# int8 attention (opt-in ASFP8_INT8_ATTN) — implementation spec

Status: design finalized 2026-07-22 from liuliu/ccv NAInt8AttentionKernel
(BSD-3; refs downloaded to scratchpad ccv_ref/, re-fetch from
github.com/liuliu/ccv unstable lib/nnc/mfa/kernels/ if needed).
User contract: COMPLETELY OPTIONAL switch, default off, NOT under
ASFP8_NATIVE. Targets Wan 111→~95-100s, LTX 86→~78s. Krea gains ~nothing.
Ideogram: bmm route today; D=256 na spills regardless of operand width
(fp32 accumulator unchanged) — probe before promising anything.

## DrawThings recipe facts (verified in source)

- Separate GPU pre-pass kernels quantize on the fly per attention call:
  `quantize_q` (per-16-row tiles), `quantize_k` (per-64-row tiles),
  `compute_v_mean` + `quantize_v` (V minus per-(head,dim) mean, then
  per-64-row tiles). Symmetric int8, scale = absmax/127 (absmax==0 →
  1/127), clamp ±127, extent-clamped at the sequence tail.
- Forward kernel: blockDimensions {16,64,32}, executionSIMDGroups=4 for
  D≤192 (we keep OUR na geometry: 16 SGs × 16 rows, NABC=64, NABLK=2 —
  A/B 4 vs 16 later).
- Softmax: P = fast::exp2(float(cS_int32) * (q_scale*k_scale*DOT) - M);
  scale tiles align with processing tiles → one scalar per block.
- PV has two paths; their FLOAT path is
  get_left_input_cooperative_tensor<float, int8_t, accum> — fp left ×
  int8 right. We ship that one first (P unquantized).

## Our variant (flash41.mm: flash_attn_fwd_na_i8, clone of na)

- QK: Q8 int8 × K8 int8 → int32 destination tensor (the 2× half).
  Descriptor (NASR, NABC, DD, false, true, @NARP@, multiply_accumulate).
- Tail masking: int32 cannot hold -inf. Init cS=0 always; in tail blocks
  AFTER qk.run set OOB-column elements to INT_MIN before the max reduce,
  and keep the existing P=0 forcing for OOB columns. K8/V8 pad rows may
  hold garbage (extent-clamped quant) — int math is finite and P=0 zeroes
  their PV contribution exactly; no NaN class exists in int8.
- Per-block scales: q_scale = QScl[bh][q0/16] (q0 is a multiple of 16 →
  one scalar per SG); k_scale{0,1} = KScl[bhkv][c/64 (+1)]. Block factor
  f = q_scale * k_scale * dot_scale (dot_scale already folds log2e).
  Row max M compared in REAL units (int_max * f per block; f > 0 keeps
  ordering within a block).
- P stays @DT@; fold the block's V scale in: cP[k] = @DT@(P * v_scale_blk).
  PV: cP (@DT@ left) × V8 (int8 right) → float accumulate into ctO with
  the existing correction sweeps UNCHANGED. cL accumulates UNSCALED P
  sums (reduce before the v_scale fold).
- Epilogue: O = ctO/L + vmean[d]  (Σ_c P_c = L ⇒ mean addback after
  normalization is literally +mean; exact, not approximate).
- vmean is per (b,hkv,d) fp32, computed over TRUE Lk rows only.

## Pre-pass kernels (same @DT@ template file)

- `na_i8_v_mean`: grid (D/4, BHkv); 256 threads; float4 partial sums over
  rows stride-256, simd_shuffle_xor reduce, scratch[NSG], write
  mean4[(bhkv*D + vec4)/4] = sum / Lk_true. (Copy DrawThings' vectorized
  structure.)
- `na_i8_quant`: grid (ntiles, BH_or_BHkv); 256 threads; params
  [Lpad_stride, Ltrue, D, tile_size, use_mean]; extent =
  min(tile, Ltrue - tile*TS); pass 1 absmax over extent*D (minus mean if
  use_mean), simd+tg reduce; scale = absmax/127 or 1/127; pass 2 quantize
  rint(x*inv_scale) clamp ±127 as char4 vector stores. Scales layout
  [BH, ntiles] fp32.

## Host flash_fwd_na_i8(q,k,v,scale,nsg=16,rp=1,bc=64,blk=2)

Same padding as flash_fwd_na (LqP to 16*nsg, LkP to bc*blk); allocates
q8/k8/v8 int8 (padded shapes, torch.empty), qscl [BH, LqP/16],
kscl/vscl [BHkv, LkP/64], vmean [BHkv, D] fp32; encodes 4 pre-pass
dispatches + main kernel on the torch MPS stream (no COMMIT_AND_WAIT).
SH stays 8 ints incl. padded strides. D ≤ 128, D%32==0, fp16/bf16 io.
pso_for: new fn names need a new branch in the key enum (currently
na==3 else 4 — EXTEND THE KEY, don't collide) and tokens are already
substituted globally.

## Routing / gating

- Env ASFP8_INT8_ATTN=1 (default OFF, independent of ASFP8_NATIVE).
- attn_router + wan's attention path: where na is currently chosen and
  the env is set, call flash_fwd_na_i8 instead (same eligibility gates:
  unmasked, D≤128, D%32==0). Announce once on stderr. Latch off on error.

## Verification plan

- Parity vs fp16 na at Wan/LTX/krea shapes incl. GQA + ragged tails,
  NaN-poisoned pool. Expect rel err ~1e-2 class (int8 scores + int8 V);
  set threshold from measurement, assert no NaN/Inf and err stable.
- Micro-bench: Wan shape (B1 H40 L21504? use real trace shapes) vs na —
  need ≥1.2x to be worth routing; below that, kill.
- E2e: Wan 512²×81f and LTX 1024² with 2-min cooldowns; same-seed grids
  vs na for quality (typography-drift class expected, no systematic
  bias); cross-seed batch for blind-compare sanity.

## VERDICT (2026-07-23): KILLED — int8 cannot beat fp16 attention on M5 Max

Rested interleaved A/B (dev/bench_i8_attn.py): best config nsg=4 blk=1 =
0.89x na at Wan shape, 0.78x at LTX shape; accuracy stable at rel 1.1-1.3%.

Root cause, measured on our own 105 TF/s int8 GEMM kernel (register-fragment
operands — the fastest int8 path this hardware has):

    K=6144: 103.9   K=1024: 101.2   K=512: 93.2
    K=256:   32.7   K=128:   25.2   K=64:  12.6   TF/s
    attention aggregate (M=N=21504, K=128): 15.8 TF/s

The int8 2x advantage exists only at deep reductions. Attention's QK reduces
over K = head_dim = 128, where the int8 fragment path runs 25 TF/s — BELOW
the fp16 na kernel's 36-40. Identical structure to the fp16 manual-fragment
finding: the win comes from a regime that does not exist inside attention.

The kernel stays in-tree, dark and unrouted: functional, accurate
(rel ~1.2%), fully documented — re-evaluate only on new silicon or if a
future MPP release adds a shallow-K int8 path. Do NOT re-attempt on M5.

## Superseded (2026-10-22)

The verdict above was drawn from a standalone K=128 GEMM that wrote a large
INT32 output, not from a fused kernel. Hardware counters on M5 Ultra later
showed INT8 running at exactly 2x BF16 on the neural accelerators inside fused
attention (`diagnostics/int8-attention-profile-oct21`). The general opt-in
kernel `_patches/int8_attention_mps.metal` measures 1.26-1.53x against exact
attention for long video sequences; see `benchmarks/m5-ultra-int8-attention.md`.
