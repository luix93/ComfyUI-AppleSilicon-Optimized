# Exact H3 attention profiling experiments

These files are offline research tools, never loaded by the custom node.
`candidates.py` contains exact-layout candidates; acceptance still requires
complete-workflow validation. `prepare.py` generates a **diagnostic** shader
with runtime QK/softmax/PV bypasses. Bypassed outputs are not valid attention
and must never be used for generation.

Use the validated M5 Ultra / PyTorch 2.14.1 environment, an authored public
`torch.save((q.cpu(), k.cpu(), v.cpu()), capture)` fixture, and an empty output
directory. Input must be contiguous BF16 `[1,56,N,128]`, N=30001..65536.
Do not use untrusted tensor archives. The native runner also allocates CPU
copies; use the guard even though its memory footprint is modest here.

```sh
# From the repository root; PY is your validated environment's Python.
E=benchmarks/experiments/h3-attention-profile
G=benchmarks/experiments/h3-exact-attention/guard.py
"$PY" "$G" --output /tmp/h3-prepare-guard.json -- \
  "$PY" "$E/prepare.py" --capture /path/to/public-qkv.pt --output /tmp/h3-profile
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  "$E/profile.mm" -o /tmp/h3-profile/runner
"$PY" "$G" --output /tmp/h3-profile/guard.json -- \
  /tmp/h3-profile/runner /tmp/h3-profile 8
"$PY" "$G" --output /tmp/h3-screen-guard.json -- \
  "$PY" "$E/screen.py" --capture /path/to/public-qkv.pt --output /tmp/h3-screen.json
```

The full phase-mask value 7 must match the original output hash before
interpreting ablations. GPU command-buffer timestamps provide the independent
clock check. Resolved timestamp-counter deltas on this tested device are in
nanoseconds; do not multiply them by the CPU mach timebase. The runner prints
both measurements. The runtime bypasses share one compiled pipeline, but their
cost differences are **not additive**, and the full instrumented kernel has
measurable overhead. These probes do not measure register occupancy or spills.

The kernel source is assembled from the installed PyTorch headers using the
existing query-cache helper. Its PyTorch provenance and license are covered
by the repository's third-party notices; generated shader binaries and test
captures are not shipped here. The old public Apple G13 disassembler was
investigated separately, could not decode this M5 binary, and is not bundled.
