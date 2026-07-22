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


def bump_seeds(prompt, offset):
    p = copy.deepcopy(prompt)
    for node in p.values():
        ins = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in ins and isinstance(ins[key], int):
                ins[key] = ins[key] + offset
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflows", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:8299")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to the per-run seed bump; use distinct offsets "
                         "across single-run invocations so ComfyUI's execution "
                         "cache can't return a prior result in 0s")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with urllib.request.urlopen(f"{args.base}/system_stats") as r:
        stats = json.load(r)
    torch_ver = stats.get("system", {}).get("pytorch_version", "?")
    print(f"server up: torch {torch_ver}", flush=True)

    out = open(args.out, "a") if args.out else None
    client_id = uuid.uuid4().hex
    for wf_path in args.workflows:
        with open(wf_path) as f:
            base_prompt = json.load(f)
        name = wf_path.rsplit("/", 1)[-1]
        times = []
        for i in range(args.runs):
            prompt = bump_seeds(base_prompt, args.seed_offset + i)
            t_wall = time.time()
            pid = submit(args.base, prompt, client_id)
            entry = wait(args.base, pid)
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                errs = [m for n, m in status.get("messages", []) if n == "execution_error"]
                msg = errs[0].get("exception_message", "?")[:300] if errs else "?"
                print(f"  {name} run{i}: ERROR {msg}", flush=True)
                times.append(None)
                continue
            secs = exec_seconds(entry) or (time.time() - t_wall)
            times.append(secs)
            tag = " (warmup)" if i == 0 else ""
            print(f"  {name} run{i}: {secs:.1f}s{tag}", flush=True)
        rec = {
            "label": args.label,
            "torch": torch_ver,
            "workflow": name,
            "times_s": times,
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
