"""Small, reusable building blocks for Metal kernel tournaments.

The native extensions in this package deliberately expose several compile-time
kernel variants.  This module keeps the measurement rules in one place so each
experiment does not invent a subtly different benchmark.

Nothing here autotunes during normal inference.  Development harnesses run a
paired tournament, validate the output, and may persist the winning static
configuration for later review/promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import statistics
import tempfile
import time
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class KernelVariant:
    """One named callable configuration in a tournament."""

    name: str
    config: tuple[Any, ...]


@dataclass(frozen=True)
class TournamentResult:
    name: str
    config: tuple[Any, ...]
    candidate_seconds: float | None
    baseline_seconds: float | None
    speedup: float | None
    exact: bool | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["config"] = list(self.config)
        return value


def _timed(
    fn: Callable[[], Any], synchronize: Callable[[], None], reps: int,
) -> tuple[float, Any]:
    synchronize()
    start = time.perf_counter()
    out = None
    for _ in range(reps):
        out = fn()
    synchronize()
    return (time.perf_counter() - start) / reps, out


def paired_tournament(
    variants: Iterable[KernelVariant],
    make_runner: Callable[[KernelVariant], Callable[[], Any]],
    *,
    baseline: KernelVariant,
    synchronize: Callable[[], None],
    rounds: int = 6,
    reps: int = 4,
    compare: Callable[[Any, Any], tuple[bool, float, float]] | None = None,
) -> list[TournamentResult]:
    """Benchmark every variant against a freshly interleaved baseline.

    The median speedup is computed from paired per-round ratios.  Candidate
    seconds and baseline seconds are their own direct medians; one is never
    reconstructed by dividing unrelated medians.  Alternating order cancels
    most clock-ramp and thermal drift, which is material on a long M5 sweep.
    """
    if rounds < 1 or reps < 1:
        raise ValueError("rounds and reps must be positive")
    variants = list(variants)
    if not any(v == baseline for v in variants):
        variants.insert(0, baseline)

    baseline_runner = make_runner(baseline)
    baseline_reference = baseline_runner()
    synchronize()
    results: list[TournamentResult] = []

    for variant in variants:
        try:
            runner = make_runner(variant)
            # Compile/build the pipeline and ramp it once outside measurement.
            candidate_reference = runner()
            synchronize()

            candidate_times: list[float] = []
            baseline_times: list[float] = []
            ratios: list[float] = []
            for round_index in range(rounds):
                if round_index & 1:
                    candidate_s, _ = _timed(runner, synchronize, reps)
                    baseline_s, _ = _timed(
                        baseline_runner, synchronize, reps)
                else:
                    baseline_s, _ = _timed(
                        baseline_runner, synchronize, reps)
                    candidate_s, _ = _timed(runner, synchronize, reps)
                candidate_times.append(candidate_s)
                baseline_times.append(baseline_s)
                ratios.append(baseline_s / candidate_s)

            exact = None
            max_abs = None
            max_rel = None
            if compare is not None:
                exact, max_abs, max_rel = compare(
                    baseline_reference, candidate_reference)
            results.append(TournamentResult(
                name=variant.name,
                config=variant.config,
                candidate_seconds=statistics.median(candidate_times),
                baseline_seconds=statistics.median(baseline_times),
                speedup=statistics.median(ratios),
                exact=exact,
                max_abs=max_abs,
                max_rel=max_rel,
            ))
        except Exception as exc:  # a bad compile-time candidate is one result
            results.append(TournamentResult(
                name=variant.name,
                config=variant.config,
                candidate_seconds=None,
                baseline_seconds=None,
                speedup=None,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
            ))
    return results


class JsonTuningCache:
    """Atomic JSON store for reviewed development-tournament results."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)

    def load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return {}
        if not isinstance(value, dict):
            raise ValueError("tuning cache root must be a JSON object")
        return value

    def update(self, key: str, value: Any) -> None:
        data = self.load()
        data[key] = value
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".metal-autotune-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
