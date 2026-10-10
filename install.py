"""Install the reviewed Apple Silicon runtime; never run from node imports.

ComfyUI-Manager calls this installation hook. A plain clone runs setup at first
prestartup, before Torch is imported. Manual installs use ComfyUI's Python.
--check is read-only; --keep-torch installs only pack dependencies.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "_runtime" / "setup"
TAG = "[AppleSilicon-Optimized/setup]"


def profile():
    return json.loads((ROOT / "dev" / "tested_runtime.json").read_text())


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def probe():
    # Import in a disposable child: our installer must not retain loaded dylibs
    # while pip replaces their files. Versions alone do not identify a build.
    source = '''
import importlib.metadata as md,json,sys
from pathlib import Path
d={"prefix":sys.prefix,"versions":{}}
for name in ("torch","torchvision","torchaudio","mtlflashattn","ninja"):
    try:d["versions"][name]=md.version(name)
    except md.PackageNotFoundError:pass
try:
    import torch
    d.update(torch_git=torch.version.git_version,
             headers=(Path(torch.__file__).parent/"include/torch/extension.h").is_file(),
             mps_built=torch.backends.mps.is_built(),
             mps_available=torch.backends.mps.is_available())
except Exception as e:d["torch_error"]=str(e)
try:
    import torchvision
except Exception as e:d["torchvision_error"]=str(e)
print(json.dumps(d))
'''
    result = run([sys.executable, "-c", source], capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout.strip().splitlines()[-1])


def runtime_ready(info, tested):
    version = info["versions"].get("torch", "").split("+")[0]
    match = tested["supported"].get(version)
    return bool(match and info.get("torch_git") == match["git"]
                and info["versions"].get("torchvision", "").split("+")[0] == match["torchvision"]
                and info.get("headers") and info.get("mps_built")
                and not info.get("torch_error") and not info.get("torchvision_error"))


def dependencies_ready(info):
    def at_least(name, minimum):
        value = info["versions"].get(name, "")
        match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
        return bool(match and tuple(int(v or 0) for v in match.groups()) >= minimum)
    return at_least("mtlflashattn", (0, 1, 0)) and at_least("ninja", (1, 11, 0))


def live_comfy(ignore_pid=None):
    """Find common ComfyUI launch forms in this Python environment only."""
    result = run(["/bin/ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, command = int(fields[0]), fields[1]
        if pid in {os.getpid(), ignore_pid}:
            continue
        if re.search(r"(?:^|[ /])main\.py(?:\s|$)|\s-m\s+main(?:\s|$)", command):
            if sys.prefix + "/" not in command:
                try:
                    executable = Path(shlex.split(command)[0])
                except ValueError:
                    executable = Path("")
                if executable.is_absolute() and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable.name, re.IGNORECASE):
                    if executable.parent != Path(sys.executable).parent:
                        continue
                # An activated venv can launch as "python main.py", hiding
                # the interpreter path from ps. Check its ComfyUI directory.
                # Inaccessible process metadata defers conservatively.
                try:
                    import psutil
                except ImportError:
                    try:
                        result = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                                                capture_output=True, text=True, timeout=10)
                        cwd = next((Path(row[1:]) for row in result.stdout.splitlines() if row.startswith("n")), None)
                    except (OSError, subprocess.SubprocessError):
                        cwd = None
                else:
                    try:
                        cwd = Path(psutil.Process(pid).cwd())
                    except psutil.Error:
                        cwd = None
                if cwd is not None and cwd not in {Path(sys.prefix).parent, ROOT.parent.parent}:
                    continue
            found.append(pid)
    return found


def pending_path():
    return STATE / "pending.json"


def setup_signature():
    """Inspect installed files without importing Torch or touching the network."""
    versions = {}
    for name in ("torch", "torchvision", "torchaudio", "mtlflashattn", "ninja"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    torch_source, headers = None, False
    if "torch" in versions:
        distribution = importlib.metadata.distribution("torch")
        source = Path(distribution.locate_file("torch/version.py"))
        if source.is_file():
            torch_source = hashlib.sha256(source.read_bytes()).hexdigest()
        headers = Path(distribution.locate_file("torch/include/torch/extension.h")).is_file()
    return {"prefix": sys.prefix, "python": sys.executable, "versions": versions,
            "torch_source": torch_source, "headers": headers,
            "profile": hashlib.sha256((ROOT / "dev/tested_runtime.json").read_bytes()).hexdigest(),
            "requirements": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest()}


def mark_ready(keep_torch):
    STATE.mkdir(parents=True, exist_ok=True)
    staged = STATE / ("ready-" + uuid.uuid4().hex + ".tmp")
    staged.write_text(json.dumps({"signature": setup_signature(), "keep_torch": keep_torch}) + "\n")
    staged.replace(STATE / "ready.json")


def defer(keep_torch):
    STATE.mkdir(parents=True, exist_ok=True)
    staged = STATE / ("pending-" + uuid.uuid4().hex + ".tmp")
    staged.write_text(json.dumps({"prefix": sys.prefix, "keep_torch": keep_torch}) + "\n")
    staged.replace(pending_path())
    print(f"{TAG} ComfyUI is running. Setup queued for its next restart, before Torch loads.")


@contextmanager
def installation_lock():
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "install.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another runtime installer is running; wait for it to finish.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_wheels(directory, tested):
    """Verify official target checksums and the revision before any mutation."""
    for name in ("torch", "torchvision"):
        expected = tested[name]
        wheels = list(directory.glob(f"{name}-{expected}-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one {name} {expected} wheel; no packages were changed.")
        wheel = wheels[0]
        checksum = hashlib.sha256()
        with wheel.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        digest = checksum.hexdigest()
        if tested["wheel_sha256"].get(wheel.name) != digest:
            raise RuntimeError(f"Unreviewed {name} wheel checksum; no packages were changed.")
        if name == "torch":
            with zipfile.ZipFile(wheel) as archive:
                source = archive.read("torch/version.py").decode("utf-8")
                if tested["torch_git"] not in source:
                    raise RuntimeError("Torch wheel revision differs from the tested build.")
                if "torch/include/torch/extension.h" not in archive.namelist():
                    raise RuntimeError("Torch wheel is missing the native-extension headers.")


def package_snapshot():
    source = 'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'
    result = run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def pip_command(directory):
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    # uv/Desktop environments can omit pip. Bootstrap it in a disposable helper
    # venv, then target the exact ComfyUI Python without installing pip into it.
    helper = directory / "pip-runner"
    run([sys.executable, "-m", "venv", str(helper)], timeout=120)
    return [str(helper / "bin/python"), "-m", "pip", "--python", sys.executable,
            "--disable-pip-version-check"]


def install_runtime(info, tested, keep_torch, *, ignore_pid=None):
    pin_torch = not keep_torch and not runtime_ready(info, tested)
    with installation_lock(), tempfile.TemporaryDirectory(prefix="asfp8-runtime-") as temporary:
        stage = Path(temporary)
        constraints = stage / "constraints.txt"
        pins = ([f"torch=={tested['torch']}", f"torchvision=={tested['torchvision']}"] if pin_torch else
                [f"{name}=={info['versions'][name]}" for name in ("torch", "torchvision") if name in info["versions"]])
        if "torchaudio" in info["versions"]:
            pins.append("torchaudio==" + info["versions"]["torchaudio"])
        constraints.write_text("\n".join(pins) + "\n")
        requested = pins[:2] if pin_torch else []
        # Download and resolve the full requested dependency set before touching
        # the environment. Installation then uses only this local wheel set.
        pip = pip_command(stage)
        print(f"{TAG} Preparing {'tested Torch/vision and ' if pin_torch else ''}pack dependencies for {sys.executable}.")
        run([*pip, "download", "--only-binary=:all:", "--dest", str(stage),
             "--constraint", str(constraints), "--requirement", str(ROOT / "requirements.txt"), *requested], timeout=900)
        if pin_torch:
            verify_wheels(stage, tested)
        if live_comfy(ignore_pid=ignore_pid):
            defer(keep_torch)
            return
        before = package_snapshot()
        journal = STATE / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8] + ".json")
        state = {"python": sys.executable, "prefix": sys.prefix, "before": before,
                 "target": {"torch": tested["torch"], "torchvision": tested["torchvision"]}, "keep_torch": keep_torch,
                 "completed": False}
        journal.write_text(json.dumps(state, indent=2) + "\n")
        try:
            run([*pip, "install", "--no-index", "--find-links", str(stage),
                 "--constraint", str(constraints), "--requirement", str(ROOT / "requirements.txt"), *requested], timeout=900)
            # pip leaves a same-version custom/stripped build installed unless
            # explicitly replaced. Limit force-reinstallation to the two wheels.
            if pin_torch and not runtime_ready(probe(), tested):
                wheels = [str(next(stage.glob(f"{name}-{tested[name]}-*.whl"))) for name in ("torch", "torchvision")]
                run([*pip, "install", "--no-index", "--no-deps", "--force-reinstall", *wheels], timeout=900)
            after = probe()
            if (not keep_torch and not runtime_ready(after, tested)) or not dependencies_ready(after):
                raise RuntimeError("Installed runtime failed verification; inspect the setup record.")
            state["completed"] = True
            state["runtime"] = after
        finally:
            state["after"] = package_snapshot()
            journal.write_text(json.dumps(state, indent=2) + "\n")
            print(f"{TAG} Setup record: {journal}")
        check = subprocess.run([*pip, "check"], capture_output=True, text=True, timeout=60)
        if check.returncode:
            print(f"{TAG} Existing/custom-node dependency constraints need attention:\n{check.stdout.strip()}")
        pending_path().unlink(missing_ok=True)
        mark_ready(keep_torch)
        print(f"{TAG} Setup verified. Native acceleration is automatic at startup.")


def main(argv=None, *, startup_pid=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report this Python environment without installing or downloading.")
    parser.add_argument("--keep-torch", action="store_true", default=os.environ.get("ASFP8_SETUP_KEEP_TORCH") == "1",
                        help="Keep installed Torch/vision and install only pack dependencies (also ASFP8_SETUP_KEEP_TORCH=1).")
    options = parser.parse_args(argv)
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        print(f"{TAG} Tested runtime setup is for Apple Silicon macOS; this environment is unchanged.")
        return 0
    tested = profile()
    info = probe()
    print(f"{TAG} Python: {sys.executable}; installed: " + json.dumps(info["versions"], sort_keys=True))
    ready = runtime_ready(info, tested)
    print(f"{TAG} Reviewed Torch/vision build: {'ready' if ready else 'setup needed'}; preferred: {tested['torch']} / {tested['torchvision']}.")
    if options.check:
        return 0 if ready and dependencies_ready(info) else 1
    if (ready or options.keep_torch) and dependencies_ready(info):
        pending_path().unlink(missing_ok=True)
        mark_ready(options.keep_torch)
        print(f"{TAG} " + ("Already configured" if ready else "Torch kept by request; pack dependencies ready")
              + "; no packages changed.")
        return 0
    if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError("Run install.py with ComfyUI's virtual-environment Python, not the system Python.")
    if "torch" in sys.modules:
        raise RuntimeError("Torch is already loaded; stop ComfyUI and run install.py to change packages.")
    if live_comfy(ignore_pid=startup_pid):
        defer(options.keep_torch)
        return 0
    install_runtime(info, tested, options.keep_torch, ignore_pid=startup_pid)
    return 0


def resume_pending():
    if not pending_path().is_file():
        return
    request = json.loads(pending_path().read_text())
    if request.get("prefix") != sys.prefix:
        print(f"{TAG} Pending setup belongs to another Python environment; run install.py here.")
        return
    if "torch" in sys.modules:
        print(f"{TAG} Torch is already loaded; stop ComfyUI and run install.py to finish setup.")
        return
    args = ["--keep-torch"] if request.get("keep_torch") is True else []
    main(args, startup_pid=os.getpid())


def startup_setup():
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        return
    if pending_path().is_file():
        resume_pending()
        return
    ready_path = STATE / "ready.json"
    try:
        ready = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
    except (OSError, ValueError):
        ready = {}
    if ready.get("signature") == setup_signature():
        return
    if "torch" in sys.modules:
        print(f"{TAG} Torch is already loaded; stop ComfyUI and run install.py to finish first-launch setup.")
        return
    keep = ready.get("keep_torch") is True or os.environ.get("ASFP8_SETUP_KEEP_TORCH") == "1"
    print(f"{TAG} First-launch setup or changed runtime detected; checking this ComfyUI environment.")
    main(["--keep-torch"] if keep else [], startup_pid=os.getpid())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"{TAG} Setup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
