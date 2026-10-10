"""Runtime setup must verify first, respect overrides and defer live updates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from _patches import torch_mpp_qcache as qc

spec = importlib.util.spec_from_file_location("asfp8_runtime_install", Path(__file__).parents[1] / "install.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def info(version="2.14.1"):
    tested = setup.profile()
    match = tested["supported"][version]
    return {"versions": {"torch": version, "torchvision": match["torchvision"], "mtlflashattn": "0.1.0", "ninja": "1.11.1"},
            "torch_git": match["git"], "headers": True, "mps_built": True}


@pytest.fixture
def apple_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(setup.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(setup.sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(setup, "STATE", tmp_path / "state")
    monkeypatch.setattr(setup, "probe", lambda: info())
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return tmp_path


def test_profile_builds_match_kernel_guards():
    tested = setup.profile()
    assert {name: row["git"] for name, row in tested["supported"].items()} == qc._TORCH_BUILDS
    assert tested["torch_git"] == qc._TORCH_BUILDS[tested["torch"]]


@pytest.mark.parametrize("version", ["2.14.0", "2.14.1"])
def test_existing_reviewed_build_is_kept_without_network_or_mutation(apple_setup, monkeypatch, version):
    monkeypatch.setattr(setup, "probe", lambda: info(version))
    monkeypatch.setattr(setup, "run", lambda *a, **k: pytest.fail("no subprocess/network expected"))
    assert setup.main([]) == 0
    assert (setup.STATE / "ready.json").is_file()


@pytest.mark.parametrize("change", [{"torch_git": "unreviewed"}, {"headers": False}, {"torchvision_error": "broken"}])
def test_same_version_unreviewed_or_incomplete_build_needs_setup(change):
    assert not setup.runtime_ready({**info(), **change}, setup.profile())


@pytest.mark.parametrize("ready,expected", [(True, 0), (False, 1)])
def test_check_never_queues_or_installs(apple_setup, monkeypatch, ready, expected):
    current = info()
    if not ready:
        current["torch_git"] = "unknown"
    monkeypatch.setattr(setup, "probe", lambda: current)
    monkeypatch.setattr(setup, "live_comfy", lambda **k: pytest.fail("read-only check"))
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("read-only check"))
    assert setup.main(["--check"]) == expected
    assert not setup.STATE.exists()


def test_live_comfy_queues_without_replacing_packages(apple_setup, monkeypatch):
    current = {**info(), "torch_git": "unknown"}
    monkeypatch.setattr(setup, "probe", lambda: current)
    monkeypatch.setattr(setup, "live_comfy", lambda **k: [123])
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("live runtime must be preserved"))
    assert setup.main([]) == 0
    assert json.loads(setup.pending_path().read_text()) == {"prefix": sys.prefix, "keep_torch": False}


def test_keep_torch_preserves_unreviewed_runtime(apple_setup, monkeypatch):
    monkeypatch.setattr(setup, "probe", lambda: {**info(), "torch_git": "custom"})
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("no changes needed"))
    assert setup.main(["--keep-torch"]) == 0


def test_manager_environment_opt_out_preserves_torch(apple_setup, monkeypatch):
    monkeypatch.setenv("ASFP8_SETUP_KEEP_TORCH", "1")
    monkeypatch.setattr(setup, "probe", lambda: {**info(), "torch_git": "custom"})
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("opt-out must preserve Torch"))
    assert setup.main([]) == 0


@pytest.mark.parametrize("disabled", [False, True])
def test_prestartup_preserves_memory_guard_without_ordinary_install(monkeypatch, disabled):
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    monkeypatch.setenv("APPLESILICON_FP8_MPS_WATERMARK", "off" if disabled else "auto")
    original = runpy.run_path
    hook = Path(__file__).parents[1] / "prestartup_script.py"
    calls = []
    def installer_only(path, *args, **kwargs):
        assert Path(path).name == "install.py"
        return {"startup_setup": lambda: calls.append("startup")}
    monkeypatch.setattr(runpy, "run_path", installer_only)
    original(str(hook))
    assert calls == ["startup"]
    assert os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") == (None if disabled else "0.8")
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.7"


def test_system_python_refused_before_installing(apple_setup, monkeypatch):
    monkeypatch.setattr(setup, "probe", lambda: {**info(), "headers": False})
    monkeypatch.setattr(setup.sys, "base_prefix", sys.prefix)
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("wrong environment"))
    with pytest.raises(RuntimeError, match="virtual-environment"):
        setup.main([])


@pytest.mark.parametrize("platform,machine", [("linux", "arm64"), ("win32", "AMD64"), ("darwin", "x86_64")])
def test_other_platforms_are_unchanged(monkeypatch, platform, machine):
    monkeypatch.setattr(setup.sys, "platform", platform)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    monkeypatch.setattr(setup, "probe", lambda: pytest.fail("Apple-only profile"))
    assert setup.main([]) == 0


def test_resume_refuses_loaded_torch(apple_setup, monkeypatch):
    setup.defer(False)
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setattr(setup, "main", lambda *a, **k: pytest.fail("Torch is already loaded"))
    setup.resume_pending()
    assert setup.pending_path().is_file()


def test_pending_environment_cannot_redirect_installer(apple_setup, monkeypatch):
    setup.defer(False)
    setup.pending_path().write_text(json.dumps({"prefix": "/other/environment", "keep_torch": False}))
    monkeypatch.setattr(setup, "main", lambda *a, **k: pytest.fail("wrong environment"))
    setup.resume_pending()


def test_resume_only_explicit_pending_request(apple_setup, monkeypatch):
    setup.defer(True)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    monkeypatch.setattr(setup, "main", lambda *a, **k: calls.append((a, k)))
    setup.resume_pending()
    assert calls == [((["--keep-torch"],), {"startup_pid": os.getpid()})]


def test_plain_clone_runs_first_launch_setup(apple_setup, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    monkeypatch.setattr(setup, "main", lambda *a, **k: calls.append((a, k)))
    setup.startup_setup()
    assert calls == [(([],), {"startup_pid": os.getpid()})]


def test_unchanged_launch_uses_stamp_without_probe_or_network(apple_setup, monkeypatch):
    setup.mark_ready(False)
    monkeypatch.setattr(setup, "probe", lambda: pytest.fail("unchanged launch needs no Torch probe"))
    monkeypatch.setattr(setup, "main", lambda *a, **k: pytest.fail("unchanged launch needs no installer"))
    setup.startup_setup()


@pytest.mark.parametrize("change", ["profile", "versions", "prefix", "headers"])
def test_changed_runtime_rechecks_setup(apple_setup, monkeypatch, change):
    setup.mark_ready(False)
    signature = setup.setup_signature()
    signature[change] = "changed"
    monkeypatch.setattr(setup, "setup_signature", lambda: signature)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    monkeypatch.setattr(setup, "main", lambda *a, **k: calls.append((a, k)))
    setup.startup_setup()
    assert len(calls) == 1


def test_first_launch_refuses_loaded_torch(apple_setup, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setattr(setup, "main", lambda *a, **k: pytest.fail("loaded Torch must not be replaced"))
    setup.startup_setup()
    assert not setup.STATE.exists()


def test_startup_finishes_pending_request_before_stamp(apple_setup, monkeypatch):
    setup.defer(False)
    calls = []
    monkeypatch.setattr(setup, "resume_pending", lambda: calls.append("pending"))
    monkeypatch.setattr(setup, "setup_signature", lambda: pytest.fail("pending takes priority"))
    setup.startup_setup()
    assert calls == ["pending"]


def test_recorded_keep_torch_survives_environment_changes(apple_setup, monkeypatch):
    setup.mark_ready(True)
    signature = setup.setup_signature()
    signature["versions"] = {}
    monkeypatch.setattr(setup, "setup_signature", lambda: signature)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    calls = []
    monkeypatch.setattr(setup, "main", lambda *a, **k: calls.append((a, k)))
    setup.startup_setup()
    assert calls == [((["--keep-torch"],), {"startup_pid": os.getpid()})]


def test_missing_pip_uses_disposable_runner_targeting_comfy_python(apple_setup, monkeypatch):
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)
    calls = []
    monkeypatch.setattr(setup, "run", lambda command, **k: calls.append(command))
    command = setup.pip_command(apple_setup)
    assert calls == [[sys.executable, "-m", "venv", str(apple_setup / "pip-runner")]]
    assert command[command.index("--python") + 1] == sys.executable
    assert command[0] == str(apple_setup / "pip-runner/bin/python")


def test_setup_signature_does_not_import_torch(apple_setup, monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    setup.setup_signature()
    assert "torch" not in sys.modules


def test_direct_installer_refuses_inprocess_torch_mutation(apple_setup, monkeypatch):
    monkeypatch.setattr(setup, "probe", lambda: {**info(), "headers": False})
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setattr(setup, "install_runtime", lambda *a, **k: pytest.fail("loaded Torch must be preserved"))
    with pytest.raises(RuntimeError, match="Torch is already loaded"):
        setup.main([])


def test_process_detection_scopes_environment_and_ignores_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.sys, "prefix", str(tmp_path / ".venv"))
    prefix = sys.prefix
    lines = f"11 {prefix}/bin/python main.py\n12 {prefix}/bin/python -m main\n13 {prefix}/bin/python install.py\n14 /another/.venv/bin/python main.py\n15 python main.py"
    monkeypatch.setattr(setup, "run", lambda *a, **k: SimpleNamespace(stdout=lines))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda pid: SimpleNamespace(cwd=lambda: str(tmp_path if pid == 15 else tmp_path / "other")))
    assert setup.live_comfy(ignore_pid=12) == [11, 15]


def test_other_absolute_python_is_excluded_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(setup, "run", lambda *a, **k: SimpleNamespace(stdout="42 /another/.venv/bin/python main.py"))
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: pytest.fail("another absolute environment is already identified"))
    assert setup.live_comfy() == []


def test_relative_python_uses_macos_cwd_when_psutil_missing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(setup.sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(setup, "run", lambda *a, **k: SimpleNamespace(stdout="42 python main.py"))
    calls = []
    def cwd_query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"p42\nn{tmp_path}\n")
    monkeypatch.setattr(setup.subprocess, "run", cwd_query)
    assert setup.live_comfy() == [42]
    assert calls == [["/usr/sbin/lsof", "-a", "-p", "42", "-d", "cwd", "-Fn"]]


def fake_wheels(directory, tested, *, revision=None, headers=True):
    tested = json.loads(json.dumps(tested))
    for name in ("torch", "torchvision"):
        filename = f"{name}-{tested[name]}-cp313-cp313-macosx_14_0_arm64.whl"
        path = directory / filename
        with zipfile.ZipFile(path, "w") as archive:
            if name == "torch":
                archive.writestr("torch/version.py", "git_version = '" + (revision or tested["torch_git"]) + "'")
                if headers:
                    archive.writestr("torch/include/torch/extension.h", "header")
        tested["wheel_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return tested


@pytest.mark.parametrize("issue", ["checksum", "revision", "headers", "missing"])
def test_bad_wheels_rejected_before_any_pip_install(tmp_path, issue):
    tested = fake_wheels(tmp_path, setup.profile(), revision="wrong" if issue == "revision" else None, headers=issue != "headers")
    if issue == "checksum":
        next(tmp_path.glob("torch-*.whl")).write_bytes(b"tampered")
    if issue == "missing":
        next(tmp_path.glob("torchvision-*.whl")).unlink()
    with pytest.raises(RuntimeError):
        setup.verify_wheels(tmp_path, tested)


def test_preflight_download_failure_does_not_install_or_snapshot(apple_setup, monkeypatch):
    def failed_download(command, **kwargs):
        assert "download" in command
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(setup, "run", failed_download)
    monkeypatch.setattr(setup, "package_snapshot", lambda: pytest.fail("download failed before mutation"))
    with pytest.raises(subprocess.CalledProcessError):
        setup.install_runtime({**info(), "headers": False}, setup.profile(), False)
    assert not list(setup.STATE.glob("*.json"))


def test_live_comfy_starting_during_preflight_defers_mutation(apple_setup, monkeypatch):
    monkeypatch.setattr(setup, "run", lambda *a, **k: None)
    monkeypatch.setattr(setup, "verify_wheels", lambda *a: None)
    monkeypatch.setattr(setup, "live_comfy", lambda **k: [123])
    monkeypatch.setattr(setup, "package_snapshot", lambda: pytest.fail("new live process must be preserved"))
    setup.install_runtime({**info(), "headers": False}, setup.profile(), False)
    assert setup.pending_path().is_file()


def test_success_verifies_and_records_the_same_python_environment(apple_setup, monkeypatch):
    calls = []
    before = {"torch": "2.13.0", "torchvision": "0.28.0", "torchaudio": "2.11.0"}
    snapshots = iter([before, {"torch": "2.14.1", "torchvision": "0.29.1", "torchaudio": "2.11.0"}])
    monkeypatch.setattr(setup, "run", lambda command, **k: calls.append(command))
    monkeypatch.setattr(setup, "verify_wheels", lambda *a: None)
    monkeypatch.setattr(setup, "live_comfy", lambda **k: [])
    monkeypatch.setattr(setup, "package_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(setup.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=""))
    setup.install_runtime({**info(), "headers": False}, setup.profile(), False)
    assert "download" in calls[0] and "install" in calls[1]
    assert all(command[0] == sys.executable for command in calls)
    assert "--no-index" in calls[1]
    assert not any("torchaudio==2.14.1" in command for command in calls)
    journal = json.loads(next(setup.STATE.glob("20*.json")).read_text())
    assert journal["completed"] and journal["before"] == before
    assert journal["after"]["torchaudio"] == "2.11.0"
