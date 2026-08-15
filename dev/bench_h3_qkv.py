"""Paired M5 benchmark for the MiniMax H3 INT8 QKV norm/RoPE epilogue."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _patches.metal_autotune import KernelVariant, paired_tournament  # noqa: E402


SHAPES = (5378, 8192, 17000)
HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
ROTARY_HALF = 48
EPSILON = 1.0e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=int, default=[])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-strength", type=float, default=0.75)
    parser.add_argument("--json")
    args = parser.parse_args()
    shapes = args.shape or list(SHAPES)
    if any(value not in SHAPES for value in shapes):
        parser.error(f"--shape must be one of {SHAPES}")
    if args.lora_rank < 0:
        parser.error("--lora-rank must be non-negative")

    os.environ.setdefault("ASFP8_INT8_EXT", "1")
    os.environ.setdefault("ASFP8_DIT_EXT", "1")
    from _patches.dit_ext import loader as dit_loader
    from _patches.int8_ext import loader as int8_loader

    int8 = int8_loader.module()
    dit = dit_loader.module()
    assert int8 is not None and hasattr(int8, "i8_matmul2d_nt_fused_cs_qkv")
    assert dit is not None and hasattr(dit, "rms_rope_attention_partial")

    generator = torch.Generator().manual_seed(61)
    width = 3 * HEADS * HEAD_DIM
    weight = torch.randint(
        -128, 128, (width, HIDDEN), generator=generator,
        dtype=torch.int8).to("mps")
    col_scale = (
        torch.rand(width, generator=generator) * 0.002 + 1.0e-5).to("mps")
    bias = torch.randn(
        width, generator=generator, dtype=torch.bfloat16).to("mps")
    q_weight = torch.randn(
        HEAD_DIM, generator=generator, dtype=torch.bfloat16
    ).float().to("mps").contiguous()
    k_weight = torch.randn(
        HEAD_DIM, generator=generator, dtype=torch.bfloat16
    ).float().to("mps").contiguous()
    if args.lora_rank:
        lora_input = torch.randn(
            max(shapes), HIDDEN, generator=generator,
            dtype=torch.bfloat16).to("mps")
        lora_down = torch.randn(
            args.lora_rank, HIDDEN, generator=generator,
            dtype=torch.bfloat16).to("mps")
        lora_up = torch.randn(
            width, args.lora_rank, generator=generator,
            dtype=torch.bfloat16).to("mps")

    cases = []
    for rows in shapes:
        activation = torch.randint(
            -128, 128, (rows, HIDDEN), generator=generator,
            dtype=torch.int8).to("mps")
        row_scale = (
            torch.rand(rows, generator=generator) * 0.002 + 1.0e-5).to("mps")
        angles = torch.randn(
            rows, ROTARY_HALF, generator=generator).to("mps")
        cosine, sine = torch.cos(angles), torch.sin(angles)
        freqs = torch.stack(
            (cosine, -sine, sine, cosine), dim=-1
        ).reshape(1, rows, 1, ROTARY_HALF, 2, 2).to(
            torch.bfloat16
        ).contiguous()
        long_schedule = rows >= 8192

        def materialized():
            q, k, v = int8.i8_matmul2d_nt_fused_cs_qkv(
                activation, weight, row_scale, col_scale, bias,
                HEADS, HEAD_DIM, long_schedule)
            if args.lora_rank:
                low_rank = F.linear(lora_input[:rows], lora_down)
                for base, up_part in zip(
                        (q, k, v), lora_up.split(HEADS * HEAD_DIM, dim=0)):
                    delta = F.linear(low_rank, up_part).reshape(
                        rows, HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
                    base.add_(delta, alpha=args.lora_strength)
            dit.rms_rope_attention_partial(
                q, k, freqs, q_weight, k_weight, EPSILON)
            return q, k, v

        def fused_lora():
            low_rank = F.linear(lora_input[:rows], lora_down)
            output = []
            part_width = HEADS * HEAD_DIM
            for kind in range(3):
                start, end = kind * part_width, (kind + 1) * part_width
                delta = F.linear(low_rank, lora_up[start:end]).contiguous()
                output.append(int8.i8_matmul2d_nt_fused_cs_qkv_lora_part(
                    activation, weight[start:end].contiguous(), row_scale,
                    col_scale[start:end].contiguous(),
                    bias[start:end].contiguous(), delta, freqs,
                    k_weight if kind == 1 else q_weight,
                    HEADS, HEAD_DIM, kind, args.lora_strength,
                    EPSILON, long_schedule))
            return tuple(output)

        variants = [KernelVariant("two-kernel", (False,))]
        fused_available = (
            hasattr(int8, "i8_matmul2d_nt_fused_cs_qkv_lora_part")
            if args.lora_rank else
            hasattr(int8, "i8_matmul2d_nt_fused_cs_qkv_rms_rope")
        )
        if fused_available:
            variants.append(KernelVariant("fused", (True,)))

        def make_runner(variant):
            if not variant.config[0]:
                return materialized
            if args.lora_rank:
                return fused_lora
            return lambda: int8.i8_matmul2d_nt_fused_cs_qkv_rms_rope(
                activation, weight, row_scale, col_scale, bias,
                freqs, q_weight, k_weight, HEADS, HEAD_DIM,
                EPSILON, long_schedule)

        def compare(reference, output):
            exact = all(
                torch.equal(actual, expected)
                for actual, expected in zip(output, reference)
            )
            maximum = max(
                (actual.float() - expected.float()).abs().max().item()
                for actual, expected in zip(output, reference)
            )
            denominator = max(
                expected.float().abs().max().item() for expected in reference
            ) + 1.0e-9
            return exact, maximum, maximum / denominator

        results = paired_tournament(
            variants, make_runner, baseline=variants[0],
            synchronize=torch.mps.synchronize,
            rounds=args.rounds, reps=args.reps, compare=compare)
        if args.diagnose and len(variants) > 1:
            reference = materialized()
            actual = make_runner(variants[1])()
            torch.mps.synchronize()
            for label, expected, observed in zip("qkv", reference, actual):
                delta = (observed.float() - expected.float()).abs()
                mismatches = torch.count_nonzero(observed != expected).item()
                per_token = torch.count_nonzero(
                    observed != expected, dim=(0, 1, 3))
                bad_tokens = torch.nonzero(per_token).flatten()
                token_summary = (
                    "none" if bad_tokens.numel() == 0 else
                    f"{bad_tokens[:12].tolist()}..{bad_tokens[-1].item()}"
                )
                print(
                    f"  diagnose {label}: mismatches={mismatches} "
                    f"max_abs={delta.max().item():.8g} tokens={token_summary}")
        lora = f" LoRA-rank={args.lora_rank}" if args.lora_rank else ""
        print(f"== H3 QKV M={rows} K={HIDDEN} H={HEADS} D={HEAD_DIM}{lora} ==")
        for result in results:
            print(
                f"  {result.name:12s} {result.candidate_seconds * 1000:7.2f} ms "
                f"paired-ref={result.baseline_seconds * 1000:7.2f} ms "
                f"speedup={result.speedup:5.3f}x exact={result.exact}")
        avoided = 8 * rows * HEADS * HEAD_DIM
        print(f"  fused epilogue avoids {avoided / 2**20:.1f} MiB of Q/K traffic")
        cases.append({
            "rows": rows,
            "hidden": HIDDEN,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "lora_rank": args.lora_rank,
            "lora_strength": args.lora_strength if args.lora_rank else None,
            "avoided_qk_bytes": avoided,
            "results": [result.to_json() for result in results],
        })

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"cases": cases}, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
