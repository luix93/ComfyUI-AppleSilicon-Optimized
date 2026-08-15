"""Paired M5 benchmark for the MiniMax H3 FC1/SwiGLU/FC2 bridge."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from _patches.metal_autotune import KernelVariant, paired_tournament  # noqa: E402


SHAPES = (5378, 8192, 17000)
HIDDEN = 5376
FFN = 14336


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=int, default=[])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--tile-rows", action="append", type=int, default=[])
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
    from _patches.int8_ext import loader

    kernel = loader.module()
    assert kernel is not None and hasattr(kernel, "i8_mlp_bridge_prequant")
    generator = torch.Generator().manual_seed(53)
    fc1 = torch.randint(
        -128, 128, (2 * FFN, HIDDEN), generator=generator,
        dtype=torch.int8).to("mps")
    fc2 = torch.randint(
        -128, 128, (HIDDEN, FFN), generator=generator,
        dtype=torch.int8).to("mps")
    fc1_scale = (
        torch.rand(2 * FFN, generator=generator) * 0.002 + 1.0e-5).to("mps")
    fc2_scale = (
        torch.rand(HIDDEN, generator=generator) * 0.002 + 1.0e-5).to("mps")
    if args.lora_rank:
        from comfy.quant_ops import TensorWiseINT8Layout
        from comfy_kitchen.tensor import QuantizedTensor
        from _patches import int8_linear_kernel_mps as int8_patch
        from _patches.minimax_h3_turbo import H3RuntimeLoRA

        def layer(weight, scale):
            params = TensorWiseINT8Layout.Params(
                scale=scale, orig_dtype=torch.bfloat16,
                orig_shape=tuple(weight.shape), convrot=True,
                convrot_groupsize=256)
            quantized = QuantizedTensor(
                weight, "TensorWiseINT8Layout", params)
            return type("Linear", (), {
                "weight": quantized,
                "bias": None,
                "weight_function": [],
                "bias_function": [],
                "_full_precision_mm": False,
                "comfy_force_cast_weights": False,
            })()

        fc1_layer = layer(fc1, fc1_scale)
        fc2_layer = layer(fc2, fc2_scale)
        lora1 = H3RuntimeLoRA((
            torch.randn(
                2 * FFN, args.lora_rank, generator=generator,
                dtype=torch.bfloat16).to("mps"),
            torch.randn(
                args.lora_rank, HIDDEN, generator=generator,
                dtype=torch.bfloat16).to("mps"),
            float(args.lora_rank), None, None, None,
        ), fast=True)
        lora2 = H3RuntimeLoRA((
            torch.randn(
                HIDDEN, args.lora_rank, generator=generator,
                dtype=torch.bfloat16).to("mps"),
            torch.randn(
                args.lora_rank, FFN, generator=generator,
                dtype=torch.bfloat16).to("mps"),
            float(args.lora_rank), None, None, None,
        ), fast=True)
        lora1.multiplier = args.lora_strength
        lora2.multiplier = args.lora_strength
        setattr(fc1_layer, int8_patch._RUNTIME_LORA_ATTR, lora1)
        setattr(fc2_layer, int8_patch._RUNTIME_LORA_ATTR, lora2)
        int8_patch._kernel = kernel

    cases = []
    for rows in shapes:
        activation = torch.randint(
            -128, 128, (rows, HIDDEN), generator=generator,
            dtype=torch.int8).to("mps")
        row_scale = (
            torch.rand(rows, generator=generator) * 0.002 + 1.0e-5).to("mps")
        long_schedule = rows >= 8192
        if args.lora_rank:
            lora_input = torch.randn(
                rows, HIDDEN, generator=generator,
                dtype=torch.bfloat16).to("mps").contiguous()

        def baseline():
            if args.lora_rank:
                int8_patch._act_cache = None
                wide = int8_patch._int8_linear_kernel(
                    lora_input, fc1, fc1_scale, None,
                    torch.bfloat16, True, 256)
                lora1.add_to(wide, lora_input)
                act = int8_patch._swiglu_activation(wide)
                result = int8_patch._int8_linear_kernel(
                    act, fc2, fc2_scale, None,
                    torch.bfloat16, True, 256)
                return lora2.add_to(result, act)
            wide = kernel.i8_matmul2d_nt_fused_cs(
                activation, fc1, row_scale, fc1_scale,
                None, False, long_schedule)
            rotated = kernel.i8_swiglu_rot(wide)
            del wide
            quantized, scale = kernel.i8_quant_rowwise(rotated)
            del rotated
            return kernel.i8_matmul2d_nt_fused_cs(
                quantized, fc2, scale.reshape(-1), fc2_scale,
                None, True, long_schedule)

        def bridge(tile_rows):
            if args.lora_rank:
                int8_patch._act_cache = None
                os.environ["ASFP8_INT8_MLP_LORA_TILE_ROWS"] = str(
                    tile_rows or rows)
                return int8_patch.int8_mlp_bridge(
                    fc1_layer, fc2_layer, lora_input)
            quantized, scale = kernel.i8_mlp_bridge_prequant(
                activation, fc1, row_scale, fc1_scale,
                None, long_schedule, tile_rows)
            return kernel.i8_matmul2d_nt_fused_cs(
                quantized, fc2, scale.reshape(-1), fc2_scale,
                None, True, long_schedule)

        default_tile_rows = 2048 if args.lora_rank else 1024
        tile_rows = args.tile_rows or [default_tile_rows]
        if any(value != 0 and value < 128 for value in tile_rows):
            parser.error("--tile-rows must be zero or at least 128")
        variants = [KernelVariant("materialized", (0,))]
        variants.extend(
            KernelVariant(f"bridge-{value or 'full'}", (value,))
            for value in tile_rows)

        def make_runner(variant):
            if variant.name == "materialized":
                return baseline
            return lambda: bridge(variant.config[0])

        def compare(reference, output):
            delta = (output.float() - reference.float()).abs()
            maximum = delta.max().item()
            relative = maximum / (reference.float().abs().max().item() + 1.0e-9)
            return torch.equal(output, reference), maximum, relative

        results = paired_tournament(
            variants, make_runner, baseline=variants[0],
            synchronize=torch.mps.synchronize,
            rounds=args.rounds, reps=args.reps, compare=compare)
        lora = f" LoRA-rank={args.lora_rank}" if args.lora_rank else ""
        print(f"== H3 MLP M={rows} K={HIDDEN} F={FFN}{lora} ==")
        for result in results:
            print(
                f"  {result.name:12s} {result.candidate_seconds * 1000:7.2f} ms "
                f"paired-ref={result.baseline_seconds * 1000:7.2f} ms "
                f"speedup={result.speedup:5.3f}x exact={result.exact}")
        effective_tile = min(rows, (default_tile_rows // 128) * 128)
        saved = 6 * rows * FFN - 4 * effective_tile * FFN
        print(
            "  default bridge eliminated peak BF16 storage: "
            f"{saved / 2**20:.1f} MiB")
        cases.append({
            "rows": rows,
            "hidden": HIDDEN,
            "ffn": FFN,
            "lora_rank": args.lora_rank,
            "lora_strength": args.lora_strength if args.lora_rank else None,
            "eliminated_peak_bytes": saved,
            "results": [result.to_json() for result in results],
        })

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"cases": cases}, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
