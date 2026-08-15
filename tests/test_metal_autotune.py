import json

from _patches.metal_autotune import (
    JsonTuningCache,
    KernelVariant,
    paired_tournament,
)


def test_paired_tournament_reports_direct_times(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "_patches.metal_autotune.time.perf_counter", lambda: clock[0])
    base = KernelVariant("base", (0,))
    fast = KernelVariant("fast", (1,))

    def make_runner(variant):
        duration = 2.0 if variant == base else 1.0

        def run():
            clock[0] += duration
            return variant.config[0]

        return run

    results = paired_tournament(
        [fast], make_runner,
        baseline=base, synchronize=lambda: None, rounds=2, reps=1,
        compare=lambda a, b: (a == b, float(abs(a - b)), 0.0),
    )

    by_name = {result.name: result for result in results}
    assert by_name["base"].candidate_seconds == 2.0
    assert by_name["fast"].candidate_seconds == 1.0
    assert by_name["fast"].baseline_seconds == 2.0
    assert by_name["fast"].speedup == 2.0
    assert by_name["fast"].exact is False


def test_json_tuning_cache_merges_atomically(tmp_path):
    cache = JsonTuningCache(str(tmp_path / "nested" / "tuning.json"))
    cache.update("shape-a", {"winner": [16, 64]})
    cache.update("shape-b", {"winner": [8, 128]})
    assert cache.load() == {
        "shape-a": {"winner": [16, 64]},
        "shape-b": {"winner": [8, 128]},
    }
    json.loads((tmp_path / "nested" / "tuning.json").read_text())
