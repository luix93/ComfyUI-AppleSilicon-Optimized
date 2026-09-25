"""JIT loader for the DiT hot-path MPS extension (fused qkv norm+rope).

Builds _patches/dit_ext/dit_kernels.mm via torch.utils.cpp_extension.load
(ObjC++, Metal). Opt-in (ASFP8_DIT_EXT=1), guarded, cached; returns None on any
failure so callers fall back to the eager path. Mirrors int8_ext/loader.py,
including the space-in-path workaround.
"""

import os
import shutil

_mod = None
_tried = False

from .._native_build import build_root

_NOSPACE_ROOT = build_root()


def _nospace_torch_lib():
    import torch.utils.cpp_extension as cpp
    real = cpp.TORCH_LIB_PATH
    if " " not in real:
        return real
    os.makedirs(_NOSPACE_ROOT, exist_ok=True)
    link = os.path.join(_NOSPACE_ROOT, "torchlib")
    try:
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(real, link)
    except OSError:
        return real
    return link


def module():
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True

    if os.environ.get("ASFP8_DIT_EXT") != "1":
        return None
    if shutil.which("xcrun") is None:
        print("[dit_ext] no Metal toolchain (xcrun); DiT kernels disabled.")
        return None

    if shutil.which("ninja") is None:
        try:
            import ninja
            bin_dir = getattr(ninja, "BIN_DIR", None) or os.path.join(
                os.path.dirname(ninja.__file__), "data", "bin"
            )
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"[dit_ext] ninja not importable ({e!r}); DiT kernels disabled.")
            return None
    if shutil.which("ninja") is None:
        print("[dit_ext] ninja binary not found on PATH; DiT kernels disabled.")
        return None

    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[dit_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "dit_kernels.mm")
    build_dir = os.path.join(_NOSPACE_ROOT, "dit_ext")
    os.makedirs(build_dir, exist_ok=True)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        _mod = cpp_load(
            name="asfp8_dit_kernels",
            sources=[src],
            extra_cflags=["-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[dit_ext] build failed; DiT kernels disabled: {e!r}")
        _mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return _mod


def available():
    return module() is not None
