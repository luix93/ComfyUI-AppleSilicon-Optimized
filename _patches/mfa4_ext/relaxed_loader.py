"""Build the optional H3 tail variant separately from the exact extension."""
import hashlib
from pathlib import Path

_modules = {}


def tail_source(source):
    """Guarded derivation avoids maintaining a second copy of FlashAttention.

    Only the synchronized long-H3 path peels the tail. The default extension
    and all source files remain unchanged. Refuse an unrecognized loop body.
    """
    start = source.index("kernel void flash_attn_fwd_na(")
    end = source.index("// ── na_fp8qk:", start)
    kernel = source[start:end]
    head = "    for (int c = 0; c < Lk; c += NABLK*NABC) {"
    tail_flag = "        const bool tail = (c + NABLK * NABC > Lk);"
    if kernel.count(head) != 1 or kernel.count(tail_flag) != 1:
        raise ValueError("unrecognized H3 attention loop; optional tail disabled")
    lo = kernel.index(head)
    hi = kernel.index("    device @DT@* Ob = O", lo)
    loop = kernel[lo:hi]
    main = loop.replace(head,
        "    for (int c = 0; c < (Lk / (NABLK*NABC)) * (NABLK*NABC); c += NABLK*NABC) {")
    main = main.replace(tail_flag, "        const bool tail = false;")
    tail = loop.replace(head,
        "    if (Lk % (NABLK*NABC) != 0) {\n        const int c = (Lk / (NABLK*NABC)) * (NABLK*NABC);")
    tail = tail.replace(tail_flag, "        const bool tail = true;")
    kernel = kernel[:lo] + "#if @NASYNC@\n" + main + tail + "#else\n" + loop + "#endif\n" + kernel[hi:]
    return source[:start] + kernel + source[end:]


def scheduled_source(source):
    """Permit the separately selected eight-SIMD-group, 96-column schedule."""
    replacements = (
        ("&& Lq == Lk && Lq >= 8192 && NSG == 16 && BC == 64 && BLK == 2",
         "&& Lq == Lk && Lq >= 8192"),
        ("if (c % 1024 == 0) threadgroup_barrier(mem_flags::mem_none);",
         "if (c % (8 * NABLK * NABC) == 0) threadgroup_barrier(mem_flags::mem_none);"),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("unrecognized H3 attention schedule; optional tuning disabled")
        source = source.replace(old, new)
    return source


def module(mode="tail"):
    if mode not in ("tail", "tuned"):
        raise ValueError("unknown relaxed H3 attention mode")
    if mode in _modules:
        return _modules[mode]
    _modules[mode] = None
    try:
        from . import loader
        # Reuse the exact loader's toolchain setup and platform checks.
        if loader.module() is None:
            return None
        import torch.utils.cpp_extension as cpp
        here = Path(__file__).resolve().parent
        source = (here / "flash41.mm").read_text()
        if mode == "tuned":
            source = scheduled_source(source)
        source = tail_source(source)
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        directory = Path(loader._NOSPACE_ROOT) / ("mfa4_" + mode) / digest
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "flash41.mm"
        if not path.exists() or path.read_text() != source:
            path.write_text(source)
        saved = cpp.TORCH_LIB_PATH
        try:
            cpp.TORCH_LIB_PATH = loader._nospace_torch_lib()
            _modules[mode] = cpp.load(name="asfp8_flash41_" + mode, sources=[str(path)],
                extra_cflags=["-ObjC++", "-I" + str(here)],
                extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
                build_directory=str(directory), verbose=False)
        finally:
            cpp.TORCH_LIB_PATH = saved
    except Exception as exc:
        print(f"[minimax_relaxed] optional {mode} attention unavailable; exact route retained ({str(exc)[:200]}).")
    return _modules[mode]
