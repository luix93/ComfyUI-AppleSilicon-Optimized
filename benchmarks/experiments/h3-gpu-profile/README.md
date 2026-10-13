# Native hardware profiling of exact H3 attention

On the tested macOS 27 machine, `/usr/bin/gpucapture` and `/usr/bin/gpudebug`
are already available. Full Xcode is not required for this procedure. Earlier
claims that only timestamp profiling was possible were too restrictive:
the public Metal counter API exposes timestamps here, but Apple's developer
profiler exposes additional hardware data.

Use the validated M5 Ultra / PyTorch 2.14.1 environment. Generate native inputs
and `baseline.metal` using [the preceding preparation script](../h3-attention-profile/README.md)
and an authored public BF16 `[1,56,N,128]` capture. Do not capture confidential
workflows or unrelated processes. Traces contain input and output buffers.

```sh
# From the repository root. PY points to the validated Python environment.
E=benchmarks/experiments/h3-gpu-profile
G=benchmarks/experiments/h3-exact-attention/guard.py
mkdir -p /tmp/h3-gpu-capture
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$E/capture.mm" -o /tmp/h3-gpu-capture/capture
MTL_CAPTURE_ENABLED=1 "$PY" "$G" --output /tmp/h3-gpu-capture/guard.json -- \
  /tmp/h3-gpu-capture/capture /tmp/h3-profile /tmp/h3-gpu-capture
"$PY" "$G" --output /tmp/h3-gpu-profile-guard.json -- \
  "$PY" "$E/profile.py" --trace /tmp/h3-gpu-capture/exact.gputrace \
  --output /tmp/h3-gpu-profile --state high
```

Use a fresh output directory for capture; do not overwrite an existing trace.
The capture runner performs three warm-ups, then captures one dispatch and
saves `captured-output.bin`. Compare its SHA-256 with the original reference.
The input directory's shader and buffers determine the experiment; the
capture runner itself never changes attention arithmetic.

The profiler collects at a requested performance state and embeds the session
in the trace. Collection and reading use separate debugger sessions because
v1.0 can incorrectly report a newly collected, empty view as already loaded.
Missing nodes or counters are errors, never zero values. Check **Consistent
state** before comparing replay times. The tested tool reports **one sampled
core out of 80**, so this is not a per-core imbalance survey.

Profiler replay time is not a ComfyUI benchmark. Measure ordinary execution
separately, including layout copies and allocations. Hardware occupancy is
not the same metric as neural-accelerator utilization. Nonzero spilled bytes
alone do not prove spill-induced throttling; inspect pressure influences,
cache utilization, bandwidth and the matrix-unit counters together.

The guard checks system available memory and ComfyUI queue activity. Its
process-RSS field may exclude debugger helper processes; use the system-memory
floor as the safeguard for the complete profiler session.

The larger-contraction screen is also retained here:

```sh
"$PY" "$G" --output /tmp/h3-contractions-guard.json -- \
  "$PY" "$E/test_contractions.py" --capture /path/to/authored-attention.pt \
  --output /tmp/h3-contractions.json
```

It tests QK/PV contraction variants independently, checks output bytes before
timing, and randomizes ten measurement rounds. It does not enable any production
patch. The 32-row control additionally requires the `m32row2` shader from the
preceding exact-attention experiment and **64**, rather than 128, threads per
threadgroup in the native runner. `m32col2` is a different, rejected variant.
The slow 32-row trace returned valid timeline counters but no shader statistics;
the exporter's failure in that case is intentional. Inspect the available
records as partial data, never substitute zeros for missing register counts.
