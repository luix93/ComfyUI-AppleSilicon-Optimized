"""JIT loader for the Metal 4.1 Sol block-sparse attention MPS extension.

Builds _patches/sol_ext/sol_attn.mm via torch.utils.cpp_extension.load (ObjC++,
Metal 4.1). Guarded, cached; returns None on any failure so the sparse attention
node can refuse to apply itself with a clear message instead of silently running
a slow fallback.

Unlike the other extensions here this one is NOT gated on an env switch: it is
only ever built when the user explicitly applies the experimental H3 sparse
attention node, which is opt-in by construction. ASFP8_SOL_EXT=0 force-disables
it (useful for A/B against the pure-PyTorch reference path).

Same space-in-path workaround and ninja-binary PATH shim as the sibling loaders.
"""

import os
import shutil

_mod = None
_tried = False

from .._native_build import build_root

_NOSPACE_ROOT = build_root()


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

    if os.environ.get("ASFP8_SOL_EXT") == "0":
        return None
    if shutil.which("xcrun") is None:
        print("[sol_ext] no Metal toolchain (xcrun); sparse attention disabled.")
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
            print(f"[sol_ext] ninja not importable ({e!r}); sparse attention disabled.")
            return None
    if shutil.which("ninja") is None:
        print("[sol_ext] ninja binary not found on PATH; sparse attention disabled "
              "(pip install ninja into the ComfyUI venv).")
        return None

    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[sol_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "sol_attn.mm")
    build_dir = os.path.join(_NOSPACE_ROOT, "sol_ext")
    os.makedirs(build_dir, exist_ok=True)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        _mod = cpp_load(
            name="asfp8_sol_attn",
            sources=[src],
            extra_cflags=["-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[sol_ext] build failed; sparse attention disabled: {e!r}")
        _mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return _mod


def available():
    return module() is not None
