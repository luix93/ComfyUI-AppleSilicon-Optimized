#!/usr/bin/env python3
"""Bounded operator validation; never loads models or starts/stops ComfyUI.

Optional --upstream-backend and --upstream-helpers compare AST-extracted pure
functions from SHA256-pinned FastVideo sources. No CUDA imports are executed.
--metal additionally runs parity and warm operator-only timings on M5 Ultra.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _patches import h3_vsa_mps as vsa

SOURCE_REVISION = "cdda1b7535fe32699335e39db82c1d4db21ceb88"
SOURCE_HASHES = {
    "backend": "1583fac16c6ec03542fbc015cff8a0c6235400073266a3817b21ee3834e29b59",
    "helpers": "6f05e429129bb131532f42e24aa868bd7752f6600bc958ea66934b6390157fc2",
}


def upstream_functions(backend, helpers):
    ns = {"torch": torch, "math": math, "functools": functools,
          "VSA_TILE_SIZE": (4, 4, 8), "VSA_H3_TILE_SIZE": (4, 4, 8), "_TILE_ELEMS": 128}
    groups = [("helpers", helpers, {"get_tile_partition_indices", "construct_variable_block_sizes",
        "get_non_pad_index", "compute_topk"}),
        ("backend", backend, {"_validate_h3_segment_geometry", "_h3_segment_tile_geometry",
         "_pool_tiles", "_build_block_mask"})]
    for name, path, wanted in groups:
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError(f"{name} source differs from the validated upstream snapshot")
        tree = ast.parse(data)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        if {node.name for node in nodes} != wanted:
            raise ValueError(f"{name} source lacks oracle functions")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
        exec(compile(module, str(path), "exec"), ns)
    return ns


def oracle(q, k, v, gate, segments, ns):
    _, sizes, inverse, prefix, _, spans = ns["_h3_segment_tile_geometry"](
        segments, torch.device("cpu"), (4, 4, 8))
    n = sizes.numel()*128
    tiled = []
    for source in (q, k, v, gate):
        buf = torch.zeros((1, n, *source.shape[1:]), dtype=source.dtype)
        buf[0, inverse] = source
        tiled.append(buf)
    qt, kt, vt, gt = tiled
    pq, pk, pv = [ns["_pool_tiles"](t, sizes, 128) for t in (qt, kt, vt)]
    scores = torch.matmul(pq, pk.transpose(-2, -1))/math.sqrt(q.shape[-1])
    mask = ns["_build_block_mask"](scores, prefix, .9, True, spans, (.9,)*len(spans))
    live = torch.arange(128)[None, :] < sizes[:, None]
    allowed = mask.repeat_interleave(128, 2).repeat_interleave(128, 3) & live.flatten()[None, None, None]
    logits = torch.einsum("bqhd,bkhd->bhqk", qt.float(), kt.float())/math.sqrt(q.shape[-1])
    fine = torch.einsum("bhqk,bkhd->bqhd", logits.masked_fill(~allowed, -torch.inf).softmax(-1), vt.float()).to(q.dtype)
    coarse = torch.matmul(scores.softmax(-1), pv).permute(0, 2, 1, 3).to(q.dtype)
    out = torch.addcmul(fine.view(1, -1, 128, *q.shape[1:]), coarse[:, :, None],
                       gt.view(1, -1, 128, *q.shape[1:])).view_as(fine)[0, inverse]
    return out, sizes, inverse, mask[0], tiled


def compare_upstream(ns):
    reports = []
    cases = [(17, 5, (5, 5, 9)), (3, (1, 5, 9), 7, (5, 3, 9)),
             (1, (1, 1, 1)), (129, (3, 3, 3), 5, (4, 4, 9)),
             (13, (4, 4, 88), 7, (4, 4, 16))]
    for segments in cases:
        plan = vsa.build_plan(segments)
        rng = torch.Generator().manual_seed(725)
        args = list(torch.randn(4, plan.seq_len, 2, 128, generator=rng))
        args[-1] *= .2
        expected, sizes, inverse, mask, old_tiled = oracle(*args, segments, ns)
        got = vsa.attention(*args, plan, backend="reference")
        torch.testing.assert_close(got, expected, atol=8e-7, rtol=2e-5)
        # Verify geometry and selection separately, not only merged output.
        first_slots = torch.arange(plan.n_tiles)*128
        old_tile_ids = inverse[plan.perm[first_slots]]//128
        assert torch.equal(plan.valid_count.long(), sizes[old_tile_ids])
        ours = [vsa.gather(t, plan) for t in args]
        for new, old in zip(ours, old_tiled):
            reordered = old[0].view(-1, 128, 2, 128)[old_tile_ids].reshape_as(new)
            assert torch.equal(new, reordered)
        pq, pk = (vsa.pool(t, plan) for t in ours[:2])
        index, keep = vsa.select(torch.matmul(pq, pk.transpose(-2, -1))/math.sqrt(128), plan)
        own_mask = torch.ones((2, plan.n_tiles, plan.n_tiles), dtype=torch.bool)
        own_mask[:, :plan.n_video_tiles, :plan.n_video_tiles] = False
        own_mask[:, :plan.n_video_tiles, :plan.n_video_tiles].scatter_(-1, index, keep)
        assert torch.equal(own_mask, mask[:, old_tile_ids][:, :, old_tile_ids])
        reports.append({"segments": segments, "tokens": plan.seq_len,
                        "geometry_and_routes_match": True,
                        "max_abs_error": float((got-expected).abs().max())})
    return reports


def timed_pair(sparse, dense, repeats=7, batch=3):
    """Alternate timing order, amortizing Python/GPU dispatch over a small batch."""
    functions = (sparse, dense)
    for _ in range(3):
        for fn in functions:
            fn()
    torch.mps.synchronize()
    values = [[], []]
    for i in range(repeats):
        for which in ((0, 1) if i % 2 == 0 else (1, 0)):
            started = time.perf_counter()
            for _ in range(batch):
                functions[which]()
            torch.mps.synchronize()
            values[which].append((time.perf_counter()-started)*1000/batch)
    return [{"median_ms": statistics.median(v), "rounds_ms": v,
             "calls_per_round": batch} for v in values]


def check_metal():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    # Check the actual CPU oracle, not an MPS reference with shared bugs.
    segments = (33, (1, 5, 9), 7, (5, 5, 9))
    cpu = vsa.build_plan(segments)
    gpu = vsa.build_plan(segments, "mps")
    rng = torch.Generator().manual_seed(825)
    args = list(torch.randn(4, cpu.seq_len, 4, 128, generator=rng).bfloat16())
    args[-1] *= .2
    expected = vsa.attention(*args, cpu, backend="reference")
    got = vsa.attention(*(t.to("mps") for t in args), gpu).cpu()
    torch.testing.assert_close(got.float(), expected.float(), atol=.008, rtol=.025)
    # Warm synthetic attention timings only: includes routing/gather/coarse/merge,
    # excludes projections, RoPE, model loading and all video generation/decode.
    timings = []
    for segments, heads in [((256, 124, (7, 24, 48)), 4), ((256, 124, (7, 32, 64)), 4),
                     ((256, (7, 24, 48), 124, (7, 24, 48)), 4),
                     ((256, 124, (13, 32, 64)), 4), ((256, 124, (13, 32, 64)), 56)]:
        plan = vsa.build_plan(segments, "mps")
        rng.manual_seed(925)
        q, k, v, gate = torch.randn(4, plan.seq_len, heads, 128, generator=rng).to("mps", torch.bfloat16)
        gate *= .2
        sparse, dense = timed_pair(lambda: vsa.attention(q, k, v, gate, plan),
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)))
        timings.append({"segments": segments, "tokens": plan.seq_len, "heads": heads,
                        "vsa_complete_operator": sparse, "dense_sdpa_only": dense,
                        "operator_speedup_vs_dense": dense["median_ms"]/sparse["median_ms"]})
    return {"parity_tokens": cpu.seq_len, "max_abs_error": float((got.float()-expected.float()).abs().max()),
            "timings": timings,
            "boundary": "Synthetic attention only. No full-model or quality claim; dense has no coarse branch."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-backend", type=Path)
    parser.add_argument("--upstream-helpers", type=Path)
    parser.add_argument("--metal", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"torch": torch.__version__, "upstream_revision": SOURCE_REVISION,
              "upstream_hashes": SOURCE_HASHES}
    with torch.inference_mode():
        if args.upstream_backend or args.upstream_helpers:
            if not (args.upstream_backend and args.upstream_helpers):
                parser.error("supply both upstream source paths")
            report["upstream_parity"] = compare_upstream(upstream_functions(
                args.upstream_backend, args.upstream_helpers))
        if args.metal:
            report["metal"] = check_metal()
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
