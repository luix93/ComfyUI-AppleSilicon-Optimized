r"""int8 GEMM kernel-variant tournament (dev R&D, GEMM throughput sprint).

Every variant is a source-level edit of _patches/int8_ext/int8_gemm.mm that may
only change the GEMM core's *schedule* — int32 accumulation is associative, so
any candidate that passes `torch.equal` against the shipped kernel is bit-exact
by construction. Candidates that fail to compile or miss parity are skipped
with a printed reason; the sweep never aborts.

Edits are anchored string replacements and every anchor MUST match exactly once
(silent str.replace drift shipped a broken kernel once — never again).

Run (GPU should be otherwise idle):
  <venv>/bin/python3 dev/tournament_int8.py [variant ...]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "_patches", "int8_ext", "int8_gemm.mm")
BUILD_ROOT = "/tmp/asfp8_build/int8_tournament"
COOLDOWN_S = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "3.0"))

# (label, M, K, N) — the production shapes that matter most.
BENCH_SHAPES = [
    ("krea wq 6144^2", 4300, 6144, 6144),
    ("krea wk/wv N=1536", 4300, 6144, 1536),
    ("krea down K=16384", 4300, 16384, 6144),
    ("ideo qkv N=13824", 5037, 4608, 13824),
]

# Parity shapes: aligned, ragged-M/N, ragged-K (remainder path), tiny.
PARITY_SHAPES = [(512, 1024, 512), (517, 704, 385), (4300, 1536, 384), (33, 96, 63)]


def apply_edits(src_text, edits):
    for anchor, replacement in edits:
        n = src_text.count(anchor)
        assert n == 1, f"anchor matched {n} times (must be 1):\n{anchor[:120]}"
        src_text = src_text.replace(anchor, replacement)
    return src_text


# ── Variant definitions ─────────────────────────────────────────────

# V1: vectorized fragment loads — packed_char4 (1-byte aligned by definition,
# safe at any address) turns 8 byte-loads per fragment into 2 vector loads.
_V1 = [(
    """template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}""",
    """template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}

// int8 specialization: one packed_char4 vector load per fragment row.
template <>
inline void nax_frag_load<int8_t>(thread int8_t *dst, const device int8_t *src,
                                  int ld, short2 sc, short off_m, short off_n) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    *(thread packed_char4 *)(dst + i * kElemCols) =
        *(const device packed_char4 *)(src + (i * kElemRowsJump) * ld);
  }
}""",
)]

# V4: drop the per-BK-block scheduling barrier (mem_none — it only herds
# simdgroups into lockstep for cache locality; correctness is unaffected).
_V4 = [(
    "    threadgroup_barrier(mem_flags::mem_none);\n    for (int kk1 = 0; kk1 < BK; kk1 += SK) {",
    "    for (int kk1 = 0; kk1 < BK; kk1 += SK) {",
)]

# V5: drop the volatile compiler barrier (forces load/compute grouping; the
# compiler may schedule better without it — or blow registers; measure).
_V5 = [
    ("      volatile int compiler_barrier;\n", ""),
    ("      (void)compiler_barrier;\n", ""),
]

# V6: vectorized interior stores — when the simdgroup's whole 32x32 tile is
# in-bounds, write the dequant epilogue as packed 4-wide vector stores and
# skip the per-element bounds checks. Values and rounding are computed
# identically (bf16_rte per element first, then one packed store).
_V6 = [(
    """template <typename OutT>
inline void nax_frag_store_dequant(const thread int32_t *src, device OutT *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *row_scale,
                                   const device OutT *bias, bool has_bias) {
  for (short i = 0; i < 2; i++) {""",
    """template <typename OutT>
inline void nax_frag_store_dequant(const thread int32_t *src, device OutT *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *row_scale,
                                   const device OutT *bias, bool has_bias) {
  // Interior fast path: whole fragment in-bounds -> packed 4-wide stores.
  if (m_base + sc.y + off_m + kElemRowsJump < M &&
      n_base + sc.x + off_n + kElemCols <= N) {
    for (short i = 0; i < 2; i++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      float rs = row_scale[mi];
      OutT vals[kElemCols];
      for (short j = 0; j < kElemCols; j++) {
        OutT r = bf16_rte(float(src[i * kElemCols + j]) * rs);
        if (has_bias) {
          r = bf16_rte(float(r) + float(bias[n_base + sc.x + off_n + j]));
        }
        vals[j] = r;
      }
      *(device packed_bfloat4 *)(dst + (sc.y + off_m + i * kElemRowsJump) * ld +
                                 (sc.x + off_n)) =
          *(thread packed_bfloat4 *)vals;
    }
    return;
  }
  for (short i = 0; i < 2; i++) {""",
)]

# V7: keep the int32 accumulator resident in ct_c across the TK loop —
# the shipped loop copies c_frags <-> ct_c around EVERY gemm_op.run; hoisting
# the copies out of the kk loop halves the register traffic per fragment pair.
_V7 = [(
    """      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn + 1][kk][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }""",
    """      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          short c0 = mm * TN + nn, c1 = c0 + 1;
          for (short i = 0; i < kElemsPerFrag; i++) {
            ct_c[i] = c_frags[c0][i];
            ct_c[kElemsPerFrag + i] = c_frags[c1][i];
          }
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn + 1][kk][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
          }
          for (short i = 0; i < kElemsPerFrag; i++) {
            c_frags[c0][i] = ct_c[i];
            c_frags[c1][i] = ct_c[kElemsPerFrag + i];
          }
        }
      }""",
)]

VARIANTS = {
    "v7_cresident": _V7,
    "v1_vec4": _V1,
    "v4_nobarrier": _V4,
    "v5_novolatile": _V5,
    "v14": _V1 + _V4,
    "v145": _V1 + _V4 + _V5,
    "v15": _V1 + _V5,
    "v16": _V1 + _V6,
    "v156": _V1 + _V5 + _V6,
}


def build_variant(name, edits):
    import torch.utils.cpp_extension as cpp
    from torch.utils.cpp_extension import load as cpp_load

    src_text = open(SRC).read()
    src_text = apply_edits(src_text, edits)
    build_dir = os.path.join(BUILD_ROOT, name)
    os.makedirs(build_dir, exist_ok=True)
    variant_src = os.path.join(build_dir, f"int8_gemm_{name}.mm")
    with open(variant_src, "w") as f:
        f.write(src_text)
    return cpp_load(
        name=f"asfp8_int8_{name}",
        sources=[variant_src],
        extra_cflags=["-std=c++17", "-ObjC++"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        build_directory=build_dir,
        verbose=False,
    )


def parity(cand, base):
    g = torch.Generator().manual_seed(3)
    for M, K, N in PARITY_SHAPES:
        a = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8).to("mps")
        b = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to("mps")
        if not torch.equal(cand.i8_matmul2d_nt(a, b), base.i8_matmul2d_nt(a, b)):
            return f"int32 mismatch at {(M, K, N)}"
        rs = (torch.rand(M, generator=g) * 0.01).to("mps")
        cs = (torch.rand(N, generator=g) * 0.01).to("mps")
        if not torch.equal(cand.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None),
                           base.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None)):
            return f"fused_cs mismatch at {(M, K, N)}"
    return None


def bench_one(mod, M, K, N, reps=15):
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="mps")
    b = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="mps")
    rs = torch.rand(M, dtype=torch.float32, device="mps") * 0.01
    cs = torch.rand(N, dtype=torch.float32, device="mps") * 0.01
    f = lambda: mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None)
    f(); torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / reps


def bench_interleaved(cand, base, M, K, N, rounds=8, reps=8):
    """Alternate candidate/baseline within one session; median of per-round
    ratios. Robust to thermal drift — both sides see the same GPU state."""
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="mps")
    b = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="mps")
    rs = torch.rand(M, dtype=torch.float32, device="mps") * 0.01
    cs = torch.rand(N, dtype=torch.float32, device="mps") * 0.01

    def run(mod):
        mod.i8_matmul2d_nt_fused_cs(a, b, rs, cs, None)

    def t_of(mod):
        run(mod); torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            run(mod)
        torch.mps.synchronize()
        return (time.perf_counter() - t) / reps

    ratios, base_ts = [], []
    for _ in range(rounds):
        tb = t_of(base)
        tc = t_of(cand)
        ratios.append(tb / tc)
        base_ts.append(tb)
    ratios.sort(); base_ts.sort()
    med_ratio = ratios[len(ratios) // 2]
    med_base = base_ts[len(base_ts) // 2]
    return med_ratio, 2 * M * K * N / med_base / 1e12


def main_ab(name):
    from _patches.int8_ext import loader

    os.environ.setdefault("ASFP8_INT8_EXT", "1")
    base = loader.module()
    assert base is not None
    cand = build_variant(name, VARIANTS[name])
    err = parity(cand, base)
    if err:
        print(f"{name}: PARITY FAILED — {err}")
        return
    print(f"{name}: parity OK (torch.equal); interleaved A/B, median of 8 rounds")
    for label, M, K, N in BENCH_SHAPES:
        r, base_tf = bench_interleaved(cand, base, M, K, N)
        print(f"  {label:>20}: base {base_tf:6.1f} TF/s, cand {base_tf*r:6.1f} TF/s ({r:5.3f}x)")
        time.sleep(COOLDOWN_S)


def main():
    from _patches.int8_ext import loader

    os.environ.setdefault("ASFP8_INT8_EXT", "1")
    base = loader.module()
    assert base is not None, "baseline kernel failed to build"

    picks = sys.argv[1:] or list(VARIANTS)
    print(f"benching baseline + {picks}")

    # Baseline row (measured in the same session so ratios are thermal-robust).
    base_t = {}
    for label, M, K, N in BENCH_SHAPES:
        base_t[label] = bench_one(base, M, K, N)
        print(f"  baseline {label:>20}: {2*M*K*N/base_t[label]/1e12:6.1f} TF/s")
        time.sleep(COOLDOWN_S)

    for name in picks:
        edits = VARIANTS[name]
        try:
            cand = build_variant(name, edits)
        except Exception as e:
            print(f"{name}: BUILD FAILED — {str(e)[-200:]}")
            continue
        err = parity(cand, base)
        if err:
            print(f"{name}: PARITY FAILED — {err}")
            continue
        rows = []
        for label, M, K, N in BENCH_SHAPES:
            t = bench_one(cand, M, K, N)
            rows.append(f"{label}: {2*M*K*N/t/1e12:6.1f} TF/s ({base_t[label]/t:5.3f}x)")
            time.sleep(COOLDOWN_S)
        print(f"{name}: parity OK (torch.equal)\n    " + "\n    ".join(rows))


def main_rate_desc64():
    """Rate-only probes: bigger descriptors with garbage packing (parity is
    EXPECTED to fail — this measures issue rate only). API constraints:
    N in {16,32}, at least one of M/N/K == 32. If wall time stays flat while
    each op does 2x the MACs, the descriptor is a real lever; if wall time
    doubles, the unit is MAC-rate-limited and there is nothing to win."""
    for dm, dn, dk in ((16, 32, 32), (32, 32, 16), (32, 32, 32)):
        probe_desc(dm, dn, dk)


def probe_desc(dm, dn, dk):
    scale = (dm * dn * dk) / (16 * 32 * 16)
    edits = [(
        "  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(\n"
        "      16, 32, 16, false, true, true,",
        "  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(\n"
        f"      {dm}, {dn}, {dk}, false, true, true,",
    )]
    from _patches.int8_ext import loader
    base = loader.module()  # also fixes ninja PATH for cpp_load
    try:
        cand = build_variant(f"v8_desc_{dm}x{dn}x{dk}", edits)
    except Exception as e:
        print(f"desc {dm}x{dn}x{dk}: BUILD FAILED — {str(e)[-300:]}")
        return
    M, K, N = 4300, 6144, 6144
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="mps")
    b = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="mps")
    for mod, tag in ((None, "skip"),):
        pass
    for tag, mod in (("base 16x32x16", base), (f"desc {dm}x{dn}x{dk} (x{scale:.0f} MACs/op)", cand)):
        f = lambda: mod.i8_matmul2d_nt(a, b)
        f(); torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            f()
        torch.mps.synchronize()
        dt = (time.perf_counter() - t) / 20
        print(f"  {tag}: {2.0*M*K*N/dt/1e12:6.1f} nominal TF/s, {dt*1000:6.2f} ms/call (rate only)")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--rate-desc64":
        main_rate_desc64()
    elif len(sys.argv) == 3 and sys.argv[1] == "--ab":
        main_ab(sys.argv[2])
    else:
        main()
