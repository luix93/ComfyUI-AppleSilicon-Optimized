#!/usr/bin/env python3
"""A/B bench driver: submit exported API-format workflows to a running ComfyUI
server and time each execution via the /history timestamps.

Usage:
  bench_workflows.py --base http://127.0.0.1:8299 --runs 4 --out results.jsonl \
      /path/to/wf1.json /path/to/wf2.json

Each run bumps every sampler seed by the run index so ComfyUI's node cache
doesn't short-circuit re-execution (identical prompts are cached and would
report ~0s). Run 1 is warmup (model load + kernel JIT) — report runs 2+.
"""
import argparse
import copy
import json
import sys
import time
import urllib.request
import uuid


def submit(base, prompt, client_id):
    req = urllib.request.Request(
        f"{base}/prompt",
        data=json.dumps({"prompt": prompt, "client_id": client_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:800]}") from None
    if resp.get("node_errors"):
        raise RuntimeError(f"node_errors: {json.dumps(resp['node_errors'])[:500]}")
    return resp["prompt_id"]


def wait(base, prompt_id, timeout=7200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{base}/history/{prompt_id}") as r:
            hist = json.load(r)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                return entry
        time.sleep(1.0)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")


def exec_seconds(entry):
    ts = {}
    for name, data in entry.get("status", {}).get("messages", []):
        if name == "execution_start":
            ts["start"] = data["timestamp"]
        elif name in ("execution_success", "execution_error"):
            ts["end"] = data["timestamp"]
    if "start" in ts and "end" in ts:
        return (ts["end"] - ts["start"]) / 1000.0
    return None


def output_artifacts(entry):
    """Return file descriptors from a Comfy history entry's node outputs."""
    artifacts = []

    def visit(value, node_id):
        if isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str):
                artifact = {
                    "node_id": str(node_id),
                    "filename": filename,
                    "subfolder": str(value.get("subfolder", "")),
                    "type": str(value.get("type", "output")),
                }
                if "format" in value:
                    artifact["format"] = str(value["format"])
                artifacts.append(artifact)
                return
            for child in value.values():
                visit(child, node_id)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, node_id)

    for node_id, output in entry.get("outputs", {}).items():
        visit(output, node_id)
    return artifacts


def bump_seeds(prompt, offset):
    p = copy.deepcopy(prompt)
    for node in p.values():
        ins = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in ins and isinstance(ins[key], int):
                ins[key] = ins[key] + offset
    return p


def set_steps(prompt, steps):
    p = copy.deepcopy(prompt)
    if steps is None:
        return p
    changed = 0
    for node in p.values():
        if node.get("class_type") == "BasicScheduler" and "steps" in node.get("inputs", {}):
            node["inputs"]["steps"] = steps
            changed += 1
    if not changed:
        raise RuntimeError("no BasicScheduler node found for --steps")
    return p


def set_duration(prompt, duration):
    """Override the exported MiniMax workflow's user-facing duration input."""
    p = copy.deepcopy(prompt)
    if duration is None:
        return p
    changed = 0
    for node in p.values():
        meta = node.get("_meta", {})
        if (node.get("class_type") == "PrimitiveFloat"
                and "duration" in str(meta.get("title", "")).lower()
                and "value" in node.get("inputs", {})):
            node["inputs"]["value"] = duration
            changed += 1
    if changed != 1:
        raise RuntimeError(
            f"expected one duration PrimitiveFloat, found {changed}")
    return p


def set_megapixels(prompt, megapixels):
    """Override ResolutionSelector nodes without changing their aspect ratio."""
    p = copy.deepcopy(prompt)
    if megapixels is None:
        return p
    changed = 0
    for node in p.values():
        if (node.get("class_type") == "ResolutionSelector"
                and "megapixels" in node.get("inputs", {})):
            node["inputs"]["megapixels"] = megapixels
            changed += 1
    if changed != 1:
        raise RuntimeError(
            f"expected one ResolutionSelector, found {changed}")
    return p


def set_easycache(prompt, values):
    """Insert ComfyUI's native EasyCache node in front of every model consumer."""
    p = copy.deepcopy(prompt)
    if values is None:
        return p
    threshold, start, end = values
    loaders = [node_id for node_id, node in p.items()
               if node.get("class_type") == "UNETLoader"]
    if len(loaders) != 1:
        raise RuntimeError(f"expected one UNETLoader, found {len(loaders)}")
    loader_id = loaders[0]
    cache_id = "asfp8:easycache"
    if cache_id in p:
        raise RuntimeError(f"workflow already contains reserved node {cache_id}")
    for node in p.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if value == [loader_id, 0]:
                inputs[key] = [cache_id, 0]
    p[cache_id] = {
        "inputs": {
            "model": [loader_id, 0],
            "reuse_threshold": threshold,
            "start_percent": start,
            "end_percent": end,
            "verbose": True,
        },
        "class_type": "EasyCache",
        "_meta": {"title": "EasyCache (benchmark injection)"},
    }
    return p


def set_h3cache(
    prompt, values, audio_weight=1.0, prediction_mode="velocity_hold",
    cached_suffix_blocks=0,
):
    """Insert this package's H3Cache node in front of every model consumer."""
    p = copy.deepcopy(prompt)
    if values is None:
        return p
    threshold, start, end, max_skips = values
    loaders = [node_id for node_id, node in p.items()
               if node.get("class_type") == "UNETLoader"]
    if len(loaders) != 1:
        raise RuntimeError(f"expected one UNETLoader, found {len(loaders)}")
    loader_id = loaders[0]
    cache_id = "asfp8:h3cache"
    if cache_id in p:
        raise RuntimeError(f"workflow already contains reserved node {cache_id}")
    for node in p.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if value == [loader_id, 0]:
                inputs[key] = [cache_id, 0]
    p[cache_id] = {
        "inputs": {
            "model": [loader_id, 0],
            "reuse_threshold": threshold,
            "start_percent": start,
            "end_percent": end,
            "max_consecutive_skips": int(max_skips),
            "audio_weight": float(audio_weight),
            "cached_suffix_blocks": int(cached_suffix_blocks),
            "prediction_mode": prediction_mode,
            "verbose": True,
        },
        "class_type": "AppleSiliconH3Cache",
        "_meta": {"title": "H3Cache (benchmark injection)"},
    }
    return p


def _rewire_through(prompt, loader_id, node_id, node):
    """Insert `node` between the UNETLoader and every consumer of its MODEL."""
    if node_id in prompt:
        raise RuntimeError(f"workflow already contains reserved node {node_id}")
    for existing in prompt.values():
        inputs = existing.get("inputs", {})
        for key, value in inputs.items():
            if value == [loader_id, 0]:
                inputs[key] = [node_id, 0]
    prompt[node_id] = node


def _single_loader(prompt):
    loaders = [node_id for node_id, node in prompt.items()
               if node.get("class_type") == "UNETLoader"]
    if len(loaders) != 1:
        raise RuntimeError(f"expected one UNETLoader, found {len(loaders)}")
    return loaders[0]


def set_h3_sparse(prompt, values):
    """Insert the H3 sparse attention node after any cache node already added.

    Applied last so the MODEL chain is loader -> cache -> sparse, matching the
    order the combined preset node uses.
    """
    p = copy.deepcopy(prompt)
    if values is None:
        return p
    mode, tau, start, dense_blocks, min_seq = values
    # The cache injections above rewired consumers to their own node; chain from
    # whichever node currently feeds the samplers.
    loader_id = _single_loader(p)
    source_id = loader_id
    for candidate in ("asfp8:h3cache", "asfp8:easycache"):
        if candidate in p:
            source_id = candidate
    node_id = "asfp8:h3sparse"
    if node_id in p:
        raise RuntimeError(f"workflow already contains reserved node {node_id}")
    for existing in p.values():
        inputs = existing.get("inputs", {})
        for key, value in inputs.items():
            if value == [source_id, 0]:
                inputs[key] = [node_id, 0]
    p[node_id] = {
        "inputs": {
            "model": [source_id, 0],
            "mode": str(mode),
            "tau": float(tau),
            "start_percent": float(start),
            "dense_leading_blocks": int(dense_blocks),
            "min_sequence_tokens": int(min_seq),
            "verbose": True,
        },
        "class_type": "AppleSiliconH3SparseAttention",
        "_meta": {"title": "H3 Sparse Attention (benchmark injection)"},
    }
    return p


def set_h3_preset(prompt, preset):
    """Insert the combined H3 accelerator node with one preset."""
    p = copy.deepcopy(prompt)
    if preset is None:
        return p
    loader_id = _single_loader(p)
    _rewire_through(p, loader_id, "asfp8:h3preset", {
        "inputs": {
            "model": [loader_id, 0],
            "preset": str(preset),
            "verbose": True,
        },
        "class_type": "AppleSiliconH3Acceleration",
        "_meta": {"title": "H3 Acceleration preset (benchmark injection)"},
    })
    return p


def set_spectrum_h3(prompt, mode=None):
    """Inject the external Spectrum MiniMax H3 node for reference A/B runs."""
    p = copy.deepcopy(prompt)
    if mode is None:
        return p
    loaders = [node_id for node_id, node in p.items()
               if node.get("class_type") == "UNETLoader"]
    if len(loaders) != 1:
        raise RuntimeError(f"expected one UNETLoader, found {len(loaders)}")
    step_counts = {
        int(node["inputs"]["steps"])
        for node in p.values()
        if (node.get("class_type") == "BasicScheduler"
            and "steps" in node.get("inputs", {}))
    }
    if len(step_counts) != 1:
        raise RuntimeError(
            f"expected one BasicScheduler step count, found {len(step_counts)}")
    steps = next(iter(step_counts))
    loader_id = loaders[0]
    cache_id = "asfp8:spectrum_h3"
    if cache_id in p:
        raise RuntimeError(f"workflow already contains reserved node {cache_id}")
    for node in p.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if value == [loader_id, 0]:
                inputs[key] = [cache_id, 0]

    if mode == "exact":
        warmup_steps = 64
        tail_actual_steps = 1
        flex_window = 0.0
    elif mode == "single":
        warmup_steps = 5
        tail_actual_steps = max(0, steps - 6)
        flex_window = 0.0
    elif mode == "single-late":
        forecast_step = max(5, steps - 5)
        warmup_steps = forecast_step
        tail_actual_steps = max(0, steps - forecast_step - 1)
        flex_window = 0.0
    else:
        warmup_steps = 5
        tail_actual_steps = 1
        flex_window = 0.75
    p[cache_id] = {
        "inputs": {
            "model": [loader_id, 0],
            "enabled": True,
            "blend_weight": 0.50,
            "degree": 4,
            "ridge_lambda": 0.10,
            "window_size": 2.0,
            "flex_window": flex_window,
            "warmup_steps": warmup_steps,
            "tail_actual_steps": tail_actual_steps,
            "max_history": 8,
            "history_storage": "vram",
            "debug": True,
        },
        "class_type": "SpectrumApplyMiniMaxH3",
        "_meta": {"title": "Spectrum H3 (reference benchmark injection)"},
    }
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflows", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8299")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None,
                    help="override every BasicScheduler step count (profiling aid)")
    ap.add_argument("--duration", type=float, default=None,
                    help="override the exported MiniMax duration input in seconds")
    ap.add_argument("--megapixels", type=float, default=None,
                    help="override ResolutionSelector megapixels, preserving aspect ratio")
    ap.add_argument("--easycache", type=float, nargs=3, metavar=("THRESH", "START", "END"),
                    help="inject native EasyCache with threshold/start/end")
    ap.add_argument("--h3cache", type=float, nargs=4,
                    metavar=("THRESH", "START", "END", "MAX_SKIPS"),
                    help="inject H3Cache; THRESH=0 records calibration without skipping")
    ap.add_argument("--h3cache-audio-weight", type=float, default=1.0,
                    help="weight of audio change in H3Cache's reuse veto")
    ap.add_argument("--h3cache-suffix-blocks", type=int, default=0,
                    help="cache only the final N H3 blocks; 0 caches the full model")
    ap.add_argument("--h3cache-predictor",
                    choices=("secant_velocity", "velocity_hold",
                             "output_hold", "residual_carry",
                             "denoised_hold", "denoised_linear",
                             "first_block_cache",
                             "first_block_cache_per_step", "oracle",
                             "block_oracle", "block_taylor"),
                    default="velocity_hold",
                    help="prediction used on an H3Cache hit")
    ap.add_argument("--spectrum-h3",
                    choices=("exact", "single", "single-late", "conservative"),
                    help="inject the external Spectrum H3 reference node")
    ap.add_argument("--h3-sparse", nargs=5,
                    metavar=("MODE", "TAU", "START", "DENSE_BLOCKS", "MIN_SEQ"),
                    help="inject H3 sparse attention; combines with --easycache "
                         "or --h3cache (applied after the cache node)")
    ap.add_argument("--h3-preset", choices=("exact", "balanced", "aggressive"),
                    help="inject the combined H3 accelerator node with one preset")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to the per-run seed bump; use distinct offsets "
                         "across single-run invocations so ComfyUI's execution "
                         "cache can't return a prior result in 0s")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.steps is not None and args.steps < 1:
        ap.error("--steps must be >= 1")
    if args.duration is not None and args.duration <= 0:
        ap.error("--duration must be > 0")
    if args.megapixels is not None and args.megapixels <= 0:
        ap.error("--megapixels must be > 0")
    if args.easycache is not None:
        threshold, start, end = args.easycache
        if threshold < 0 or not 0 <= start <= end <= 1:
            ap.error("--easycache requires THRESH>=0 and 0<=START<=END<=1")
    if args.h3cache is not None:
        threshold, start, end, max_skips = args.h3cache
        if (threshold < 0 or not 0 <= start <= end <= 1
                or max_skips < 1 or not max_skips.is_integer()):
            ap.error("--h3cache requires THRESH>=0, 0<=START<=END<=1, and integer MAX_SKIPS>=1")
        if args.h3cache_audio_weight <= 0:
            ap.error("--h3cache-audio-weight must be positive")
        if not 0 <= args.h3cache_suffix_blocks <= 49:
            ap.error("--h3cache-suffix-blocks must be between 0 and 49")
    selected_caches = sum(value is not None for value in (
        args.easycache, args.h3cache, args.spectrum_h3))
    if selected_caches > 1:
        ap.error("--easycache, --h3cache, and --spectrum-h3 are mutually exclusive")
    if args.h3_sparse is not None:
        mode, tau, start, dense_blocks, min_seq = args.h3_sparse
        if mode not in ("corrected", "hard_drop", "dense_reference"):
            ap.error("--h3-sparse MODE must be corrected, hard_drop, or dense_reference")
        try:
            tau_v, start_v = float(tau), float(start)
            dense_v, seq_v = int(dense_blocks), int(min_seq)
        except ValueError:
            ap.error("--h3-sparse expects MODE TAU START DENSE_BLOCKS MIN_SEQ")
        if tau_v < 0 or not 0 <= start_v <= 1 or not 0 <= dense_v <= 49 or seq_v < 0:
            ap.error("--h3-sparse requires TAU>=0, 0<=START<=1, 0<=DENSE_BLOCKS<=49, MIN_SEQ>=0")
        args.h3_sparse = (mode, tau_v, start_v, dense_v, seq_v)
    if args.h3_preset is not None and (args.h3_sparse is not None or selected_caches):
        ap.error("--h3-preset already selects a cache and sparse configuration; "
                 "do not combine it with --easycache/--h3cache/--spectrum-h3/--h3-sparse")

    with urllib.request.urlopen(f"{args.base}/system_stats") as r:
        stats = json.load(r)
    torch_ver = stats.get("system", {}).get("pytorch_version", "?")
    print(f"server up: torch {torch_ver}", flush=True)
    if args.h3_sparse is not None or args.h3_preset is not None:
        # The sparse route needs torch 2.13's MPS stack; refusing here beats
        # discovering it from a node error after a model load.
        parts = str(torch_ver).split(".")
        try:
            version = (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            version = (0, 0)
        if version < (2, 13):
            raise SystemExit(
                f"server reports torch {torch_ver}; H3 sparse attention needs 2.13+")

    out = open(args.out, "a") if args.out else None
    client_id = uuid.uuid4().hex
    for wf_path in args.workflows:
        with open(wf_path) as f:
            base_prompt = json.load(f)
        name = wf_path.rsplit("/", 1)[-1]
        base_prompt = set_steps(base_prompt, args.steps)
        base_prompt = set_duration(base_prompt, args.duration)
        base_prompt = set_megapixels(base_prompt, args.megapixels)
        base_prompt = set_easycache(base_prompt, args.easycache)
        base_prompt = set_h3cache(
            base_prompt, args.h3cache, args.h3cache_audio_weight,
            args.h3cache_predictor, args.h3cache_suffix_blocks)
        base_prompt = set_spectrum_h3(base_prompt, args.spectrum_h3)
        base_prompt = set_h3_sparse(base_prompt, args.h3_sparse)
        base_prompt = set_h3_preset(base_prompt, args.h3_preset)
        times = []
        prompt_ids = []
        artifacts = []
        for i in range(args.runs):
            prompt = bump_seeds(base_prompt, args.seed_offset + i)
            t_wall = time.time()
            pid = submit(args.base, prompt, client_id)
            prompt_ids.append(pid)
            entry = wait(args.base, pid)
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                errs = [m for n, m in status.get("messages", []) if n == "execution_error"]
                msg = errs[0].get("exception_message", "?")[:300] if errs else "?"
                print(f"  {name} run{i}: ERROR {msg}", flush=True)
                times.append(None)
                artifacts.append([])
                continue
            secs = exec_seconds(entry) or (time.time() - t_wall)
            times.append(secs)
            artifacts.append(output_artifacts(entry))
            tag = " (warmup)" if i == 0 else ""
            print(f"  {name} run{i}: {secs:.1f}s{tag}", flush=True)
        rec = {
            "label": args.label,
            "torch": torch_ver,
            "workflow": name,
            "prompt_ids": prompt_ids,
            "times_s": times,
            "artifacts": artifacts,
            "warmup_s": times[0],
            "steady_s": sorted(t for t in times[1:] if t is not None)[
                len([t for t in times[1:] if t is not None]) // 2
            ] if any(t is not None for t in times[1:]) else None,
        }
        print(f"{name}: warmup={rec['warmup_s']}, steady_median={rec['steady_s']}", flush=True)
        if out:
            out.write(json.dumps(rec) + "\n")
            out.flush()
    if out:
        out.close()


if __name__ == "__main__":
    main()
