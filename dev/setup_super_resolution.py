#!/usr/bin/env python3
"""Build the optional, pinned MLX-DLSS runtime without changing ComfyUI packages."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REVISION = "0ca2deab092fe6f3e331bf4f616271dbc64521d0"
ROOT = Path(__file__).resolve().parents[1]
MEMORY_SETTINGS = '''
// ASFP8 SR memory control begin
import MLX
import Darwin

var srMemoryWatchdog: DispatchSourceTimer?
if let text = ProcessInfo.processInfo.environment["ASFP8_SR_MEMORY_BYTES"],
   let bytes = Int(text), bytes >= 268435456 {
    Memory.cacheLimit = min(bytes / 8, 536870912)
    Memory.memoryLimit = bytes
    let watchdog = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
    watchdog.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
    watchdog.setEventHandler { @Sendable in
        if Memory.activeMemory + Memory.cacheMemory > bytes {
            FileHandle.standardError.write(Data("VSR exceeded its Metal memory limit.\\n".utf8))
            Darwin._exit(3)
        }
    }
    watchdog.resume()
    srMemoryWatchdog = watchdog
}
// ASFP8 SR memory control end
'''
MEMORY_REPORT = '''
    if let watchdog = srMemoryWatchdog {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            watchdog.setCancelHandler { @Sendable in continuation.resume() }
            watchdog.cancel()
        }
        FileHandle.standardError.write(Data("ASFP8_SR_METAL_PEAK_BYTES=\\(Memory.peakMemory)\\n".utf8))
    }
'''


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing checkout at the supported revision")
    parser.add_argument("--metallib", type=Path, help="Existing MLX 0.31.1 mlx.metallib")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "_runtime/mlx-dlss")
    options = parser.parse_args()
    runtime = options.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    source = (options.source or runtime / "source").resolve()
    if not source.exists():
        run("git", "clone", "https://github.com/iamwavecut/MLX-DLSS.git", str(source))
        run("git", "checkout", REVISION, cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Expected MLX-DLSS {REVISION}, found {actual}")
    patch = ROOT / "dev/patches/vsr-neural-rendering.patch"
    applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                             cwd=source, capture_output=True).returncode == 0
    if not applied:
        run("git", "apply", "--check", str(patch), cwd=source)
        run("git", "apply", str(patch), cwd=source)
    main_file = source / "Sources/mlxdlss/main.swift"
    text = main_file.read_text()
    if "// ASFP8 SR memory control begin" in text:
        text = re.sub(r"// ASFP8 SR memory control begin.*?// ASFP8 SR memory control end",
                      lambda _: MEMORY_SETTINGS.strip(), text, count=1, flags=re.S)
        text = re.sub(r"    if let watchdog = srMemoryWatchdog \{.*?\n    \}",
                      lambda _: MEMORY_REPORT.strip("\n"), text, count=1, flags=re.S)
        main_file.write_text(text)
    elif "ASFP8_SR_MEMORY_BYTES" not in text:
        marker = 'setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", 0)\n'
        if marker not in text:
            raise RuntimeError("Upstream CLI layout changed")
        text = text.replace(marker, marker + MEMORY_SETTINGS, 1)
        marker = "    try await runCommand(Array(CommandLine.arguments.dropFirst()))\n"
        if marker not in text:
            raise RuntimeError("Upstream CLI entry point changed")
        main_file.write_text(text.replace(marker, marker + MEMORY_REPORT, 1))
    else:
        raise RuntimeError("Unrecognized existing VSR memory patch; use a clean source checkout.")

    shader = options.metallib
    if shader is None:
        # The binary wheel supplies the matching shaders; nothing is installed
        # into the active interpreter or ComfyUI environment.
        dependencies = runtime / "shader-package"
        shader = dependencies / "mlx/lib/mlx.metallib"
        if not shader.exists():
            run(sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                "--target", str(dependencies), "mlx-metal==0.31.1")
    shader = shader.resolve()
    if not shader.is_file():
        raise FileNotFoundError(shader)
    run("swift", "build", "--build-system", "native", "-c", "release", "--product", "mlxdlss", "--jobs", "4", cwd=source)
    binary_dir = Path(subprocess.check_output(
        ["swift", "build", "--build-system", "native", "-c", "release", "--show-bin-path"],
        cwd=source, text=True).strip())
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        staged = Path(temporary)
        shutil.copy2(binary_dir / "mlxdlss", staged / "mlxdlss")
        shutil.copy2(shader, staged / "mlx.metallib")
        (staged / "UPSTREAM.txt").write_text(
            f"MLX-DLSS {REVISION}\nhttps://github.com/iamwavecut/MLX-DLSS\n"
            "MLX shaders: mlx-metal 0.31.1\n"
            "Local changes: allocator/cache limits; optional VSR before neural rendering.\n")
        shutil.copy2(source / "LICENSE", staged / "LICENSE-MLX-DLSS.txt")
        shutil.copy2(source / "NOTICE", staged / "NOTICE-MLX-DLSS.txt")
        mlx_license = source / ".build/checkouts/mlx-swift/LICENSE"
        if mlx_license.exists():
            shutil.copy2(mlx_license, staged / "LICENSE-MLX-Swift.txt")
        for item in staged.iterdir():
            item.replace(runtime / item.name)
    print(f"Installed VSR runtime: {runtime}")


if __name__ == "__main__":
    main()
