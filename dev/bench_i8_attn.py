r"""Native attention schedule and low-precision verdict harness (dev R&D).

Interleaved A/B: alternate flash_fwd_na (fp16/bf16 reference) and
flash_fwd_na_i8 within one session, median of per-round ratios — immune to
thermal drift and clock ramp (the GEMM sprint's phantom-win lesson).

Run rested (GPU otherwise idle).  For example:
  <venv>/bin/python3 dev/bench_i8_attn.py --na-configs --shape "minimax h3"
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from _patches.metal_autotune import (  # noqa: E402
    JsonTuningCache,
    KernelVariant,
    paired_tournament,
)

# Production shapes: Wan self-attn, LTX-class, krea GQA (sanity only).
SHAPES = [
    ("minimax h3 56h L5378", 1, 56, 56, 5378, 5378, 128, torch.bfloat16),
    ("minimax h3 56h L8192", 1, 56, 56, 8192, 8192, 128, torch.bfloat16),
    ("minimax h3 56h L17000", 1, 56, 56, 17000, 17000, 128, torch.bfloat16),
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", action="append", default=[],
        help="case-insensitive label substring; repeat to select several",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--na-configs", action="store_true",
        help="tournament bf16 NA schedules instead of INT8 attention",
    )
    modes.add_argument(
        "--fp8-qk", action="store_true",
        help="benchmark experimental block-scaled FP8 QK/BF16 PV vs BF16 NA",
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--json", default=None,
                        help="write machine-readable tournament results")
    parser.add_argument(
        "--cache", default=None,
        help="atomically merge each tournament case into this review cache",
    )
    args = parser.parse_args()

    from _patches.mfa4_ext import loader
    os.environ.setdefault("ASFP8_ATTN_ROUTER", "1")
    mod = loader.module()
    assert mod is not None and hasattr(mod, "flash_fwd_na_i8")
    if args.fp8_qk:
        assert hasattr(mod, "flash_fwd_na_fp8qk")

    configs = [(16, 1), (16, 2), (8, 1), (4, 1)]
    selected = SHAPES
    if args.shape:
        selected = [s for s in SHAPES if any(
            needle.lower() in s[0].lower() for needle in args.shape
        )]
        if not selected:
            parser.error("--shape did not match any benchmark label")
    json_cases = []
    for label, B, Hq, Hkv, Lq, Lk, D, dt in selected:
        if args.na_configs or args.fp8_qk:
            torch.manual_seed(0)
            q = (torch.randn(B, Hq, Lq, D) * 0.5).to("mps", dtype=dt)
            k = (torch.randn(B, Hkv, Lk, D) * 0.5).to("mps", dtype=dt)
            v = (torch.randn(B, Hkv, Lk, D) * 0.5).to("mps", dtype=dt)
            scale = D ** -0.5
            shipped = (16, 1, 1, 64, 2)
            if args.fp8_qk:
                fp8_schedules = [
                    shipped,
                    (12, 1, 1, 64, 2),
                    (8, 1, 1, 64, 2),
                    (16, 0, 1, 64, 2),
                    (16, 1, 1, 32, 2),
                    (16, 1, 1, 64, 1),
                    (16, 1, 1, 128, 1),
                ]
                baseline = KernelVariant("bf16-na", shipped)
                variants = [baseline] + [
                    KernelVariant(f"fp8qk-{cfg}", cfg)
                    for cfg in fp8_schedules
                ]
            else:
                schedules = (
                    [(nsg, 1, 1, 64, 2)
                     for nsg in (16, 15, 14, 12, 10, 8, 6, 4)]
                    + [
                        (16, 0, 1, 64, 2),
                        (16, 1, 0, 64, 2),  # threadgroup P
                        (16, 1, 1, 32, 1),
                        (16, 1, 1, 32, 2),
                        (16, 1, 1, 64, 1),
                        (16, 1, 1, 96, 1),
                        (16, 1, 1, 128, 1),
                        (16, 1, 1, 128, 2),
                    ]
                )
                variants = [KernelVariant(str(cfg), cfg) for cfg in schedules]
                baseline = variants[0]

            def make_runner(variant):
                op = (mod.flash_fwd_na if variant.name == "bf16-na"
                      else (mod.flash_fwd_na_fp8qk
                            if variant.name.startswith("fp8qk-")
                            else mod.flash_fwd_na))
                return lambda: op(q, k, v, scale, *variant.config)

            def compare(ref, out):
                delta = (out.float() - ref.float()).abs()
                max_abs = delta.max().item()
                max_rel = (max_abs / (ref.float().abs().max().item() + 1e-9))
                return torch.equal(out, ref), max_abs, max_rel

            mode_label = ("FP8 QK / BF16 PV" if args.fp8_qk
                          else "NA schedule tournament")
            print(f"== {label} {mode_label} ({dt}) ==")
            results = paired_tournament(
                variants, make_runner, baseline=baseline,
                synchronize=torch.mps.synchronize,
                rounds=args.rounds, reps=args.reps, compare=compare,
            )
            for result in results:
                if result.error:
                    print(f"  nsg/rp/byp/bc/blk={result.config}: ERROR "
                          f"{result.error}")
                    continue
                print(
                    f"  {result.name} nsg/rp/byp/bc/blk={result.config}: "
                    f"{result.speedup:5.3f}x, "
                    f"{result.candidate_seconds * 1000:6.2f} ms "
                    f"(paired ref {result.baseline_seconds * 1000:6.2f}), "
                    f"exact={result.exact} maxdiff={result.max_abs:.3e}"
                )
            case_record = {
                "label": label,
                "shape": [B, Hq, Hkv, Lq, Lk, D],
                "dtype": str(dt),
                "baseline": list(baseline.config),
                "results": [result.to_json() for result in results],
            }
            json_cases.append(case_record)
            if args.cache:
                mode_key = "fp8-qk" if args.fp8_qk else "na-schedules"
                JsonTuningCache(args.cache).update(
                    f"{mode_key}:{label}:{dt}", case_record)
            continue
        print(f"== {label} ({str(dt).split('.')[-1]}) ==")
        first = True
        for cfg in configs:
            med_na, r, rel, fl = bench_pair(mod, cfg, B, Hq, Hkv, Lq, Lk, D, dt)
            if first:
                print(f"  na ref: {med_na*1000:7.1f} ms  {fl/med_na/1e12:5.1f} TF/s")
                first = False
            print(f"  i8 nsg={cfg[0]:2} blk={cfg[1]}: {r:5.3f}x vs na   rel={rel:.4f}")
            time.sleep(3)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"cases": json_cases}, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
