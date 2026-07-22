r"""int8 attention verdict harness (dev R&D).

Interleaved A/B: alternate flash_fwd_na (fp16/bf16 reference) and
flash_fwd_na_i8 within one session, median of per-round ratios — immune to
thermal drift and clock ramp (the GEMM sprint's phantom-win lesson).

Run rested (GPU otherwise idle):
  <venv>/bin/python3 dev/bench_i8_attn.py [--configs]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# Production shapes: Wan self-attn, LTX-class, krea GQA (sanity only).
SHAPES = [
    ("wan 40h L21504", 1, 40, 40, 21504, 21504, 128, torch.float16),
    ("ltx 32h L9216", 1, 32, 32, 9216, 9216, 128, torch.bfloat16),
]


def bench_pair(mod, cfg, B, Hq, Hkv, Lq, Lk, D, dt, rounds=6, reps=4):
    torch.manual_seed(0)
    q = (torch.randn(B, Hq, Lq, D) * 0.5).to("mps").to(dt)
    k = (torch.randn(B, Hkv, Lk, D) * 0.5).to("mps").to(dt)
    v = (torch.randn(B, Hkv, Lk, D) * 0.5).to("mps").to(dt)
    sc = D ** -0.5
    nsg, blk = cfg

    def run_na():
        mod.flash_fwd_na(q, k, v, sc)

    def run_i8():
        mod.flash_fwd_na_i8(q, k, v, sc, nsg, 1, 64, blk)

    def t_of(f):
        f(); torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            f()
        torch.mps.synchronize()
        return (time.perf_counter() - t) / reps

    ratios, na_ts = [], []
    for _ in range(rounds):
        tn = t_of(run_na)
        ti = t_of(run_i8)
        ratios.append(tn / ti)
        na_ts.append(tn)
    ratios.sort(); na_ts.sort()
    ref = mod.flash_fwd_na(q, k, v, sc)
    out = mod.flash_fwd_na_i8(q, k, v, sc, nsg, 1, 64, blk)
    torch.mps.synchronize()
    rel = ((out.float() - ref.float()).abs().max()
           / (ref.float().abs().max() + 1e-9)).item()
    fl = 4.0 * B * Hq * Lq * Lk * D
    med_na = na_ts[len(na_ts) // 2]
    med_r = ratios[len(ratios) // 2]
    return med_na, med_r, rel, fl


def main():
    from _patches.mfa4_ext import loader
    os.environ.setdefault("ASFP8_ATTN_ROUTER", "1")
    mod = loader.module()
    assert mod is not None and hasattr(mod, "flash_fwd_na_i8")

    configs = [(16, 1), (16, 2), (8, 1), (4, 1)]
    for label, B, Hq, Hkv, Lq, Lk, D, dt in SHAPES:
        print(f"== {label} ({str(dt).split('.')[-1]}) ==")
        first = True
        for cfg in configs:
            med_na, r, rel, fl = bench_pair(mod, cfg, B, Hq, Hkv, Lq, Lk, D, dt)
            if first:
                print(f"  na ref: {med_na*1000:7.1f} ms  {fl/med_na/1e12:5.1f} TF/s")
                first = False
            print(f"  i8 nsg={cfg[0]:2} blk={cfg[1]}: {r:5.3f}x vs na   rel={rel:.4f}")
            time.sleep(3)


if __name__ == "__main__":
    main()
