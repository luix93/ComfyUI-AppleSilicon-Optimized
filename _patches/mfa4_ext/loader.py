"""JIT loader for the Metal 4.1 bf16 flash attention MPS extension.

Builds _patches/mfa4_ext/flash41.mm via torch.utils.cpp_extension.load (ObjC++,
Metal 4.1). Opt-in (ASFP8_ATTN_ROUTER=1), guarded, cached; returns None on any failure
so callers fall back to sub-quad attention.

Same space-in-path workaround and ninja-binary PATH shim as the sibling loaders
(fp8_ext, int8_ext).
"""

import os
import shutil

_mod = None
_tried = False

_NOSPACE_ROOT = "/tmp/asfp8_build"


def _nospace_torch_lib():
    """Return a no-space path to torch/lib (via symlink if the real path has a space)."""
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

    if os.environ.get("ASFP8_ATTN_ROUTER") != "1":
        return None
    if shutil.which("xcrun") is None:
        print("[mfa4_ext] no Metal toolchain (xcrun); flash41 disabled.")
        return None

    # torch's cpp_extension needs the `ninja` *binary* on PATH (not just the
    # python module). When ComfyUI is launched from the Desktop app, PATH often
    # lacks homebrew/venv bins, so add the ninja package's bundled binary dir.
    if shutil.which("ninja") is None:
        try:
            import ninja  # provides a bundled `ninja` executable
            bin_dir = getattr(ninja, "BIN_DIR", None) or os.path.join(
                os.path.dirname(ninja.__file__), "data", "bin"
            )
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"[mfa4_ext] ninja not importable ({e!r}); flash41 disabled.")
            return None
    if shutil.which("ninja") is None:
        print("[mfa4_ext] ninja binary not found on PATH; flash41 disabled "
              "(pip install ninja into the ComfyUI venv).")
        return None

    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[mfa4_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "flash41.mm")
    build_dir = os.path.join(_NOSPACE_ROOT, "mfa4_ext")
    os.makedirs(build_dir, exist_ok=True)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        _mod = cpp_load(
            name="asfp8_flash41",
            sources=[src],
            extra_cflags=["-std=c++17", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[mfa4_ext] build failed; flash41 disabled: {e!r}")
        _mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return _mod


def available():
    return module() is not None
