import subprocess
import sys
import types

import pytest

from _patches import super_resolution as sr


@pytest.fixture(autouse=True)
def free_memory(monkeypatch):
    monkeypatch.setattr(sr.psutil, "virtual_memory", lambda: types.SimpleNamespace(available=128 * sr.GIB))


def test_image_batch_budget_includes_all_returned_frames():
    with pytest.raises(ValueError, match="Video node"):
        sr._memory_budget(1920, 1080, 120, 8, True)


def test_video_budget_does_not_grow_with_frame_count():
    assert sr._memory_budget(1920, 1080, 1, 8, False) == sr._memory_budget(1920, 1080, 100000, 8, False)


def test_large_image_refused_before_execution():
    with pytest.raises(ValueError, match="Reduce the input size"):
        sr._memory_budget(6240, 4160, 1, 8, True)


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), 0, 33])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="Memory limit"):
        sr._memory_budget(64, 64, 1, budget, True)


def test_low_available_memory_refused(monkeypatch):
    monkeypatch.setattr(sr.psutil, "virtual_memory", lambda: types.SimpleNamespace(available=256 * 1024 ** 2))
    with pytest.raises(MemoryError, match="free memory"):
        sr._memory_budget(512, 512, 1, 8, True)


def test_neural_budget_accounts_for_both_upscales():
    # 1024 input -> 2048 output -> 4096 internal NR: refuse before allocating.
    sr._memory_budget(1024, 1024, 1, 8, True, nr_scale=1)
    with pytest.raises(ValueError, match="processing scale"):
        sr._memory_budget(1024, 1024, 1, 8, True, nr_scale=2)
    sr._memory_budget(1024, 1024, 1, 32, True, nr_scale=2)


def test_neural_video_budget_stays_bounded():
    assert sr._memory_budget(384, 256, 1, 8, False, 2) == sr._memory_budget(384, 256, 100000, 8, False, 2)


@pytest.mark.parametrize("model,intensity", [("disabled", 1), ("missing.dlssmodel", 0)])
def test_neural_bypass_does_not_load_model(monkeypatch, model, intensity):
    def unexpected(_):
        raise AssertionError("bypass must not need NR weights")
    monkeypatch.setattr(sr, "_neural_model_path", unexpected)
    assert sr._neural_arguments(model, "natural", intensity, 1, 1, 2) == []


def test_neural_command_order_and_temporal(monkeypatch):
    monkeypatch.setattr(sr, "_neural_model_path", lambda _: "/models/NeuralRendering.dlssmodel")
    args = sr._neural_arguments("NeuralRendering.dlssmodel", "natural", 0.8, 1.5, 0.5, 2, True)
    options = dict(zip(args[::2], args[1::2]))
    assert options["--vsr-first"] == "on"
    assert options["--temporal"] == "on"
    assert options["--processing-scale"] == "2"
    assert options["--detail-strength"] == "1.5"
    assert options["--colour-strength"] == "0.5"


@pytest.mark.parametrize("profile,intensity,detail,colour,scale", [
    ("unknown", 1, 1, 1, 1), ("natural", float("nan"), 1, 1, 1),
    ("natural", 1, float("inf"), 1, 1), ("natural", 1, 1, -1, 1),
    ("natural", 1, 1, 1, 4),
])
def test_invalid_neural_controls(profile, intensity, detail, colour, scale):
    with pytest.raises(ValueError):
        sr._neural_arguments("model.dlssmodel", profile, intensity, detail, colour, scale)


@pytest.fixture
def worker(monkeypatch):
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(sr, "_executable", lambda: sys.executable)
    started = []
    original = subprocess.Popen

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(sr.subprocess, "Popen", popen)
    return mm, started


def test_worker_success(worker):
    result = sr._run(["-c", "import json; print(json.dumps({'elapsedSeconds': 0.1}))"], sr.GIB)
    assert result["elapsedSeconds"] == 0.1
    assert worker[1][0].poll() == 0


def test_worker_failure_includes_diagnostic(worker):
    with pytest.raises(RuntimeError, match="fixture failure"):
        sr._run(["-c", "import sys; sys.stderr.write('fixture failure'); sys.exit(2)"], sr.GIB)
    assert worker[1][0].poll() == 2


def test_worker_memory_limit_terminates_process(worker):
    with pytest.raises(MemoryError, match="memory limit"):
        sr._run(["-c", "import time; time.sleep(30)"], 1)
    assert worker[1][0].poll() is not None


def test_cancellation_terminates_process(worker):
    calls = 0

    class Interrupted(Exception):
        pass

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Interrupted()

    worker[0].throw_exception_if_processing_interrupted = interrupt
    with pytest.raises(Interrupted):
        sr._run(["-c", "import time; time.sleep(30)"], sr.GIB)
    assert worker[1][0].poll() is not None
