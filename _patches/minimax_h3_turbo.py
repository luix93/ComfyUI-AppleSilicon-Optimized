"""MiniMax H3 LoRA loaders with an INT8-aware exact runtime path.

The released ComfyUI Turbo adapters update four Linear modules in each of the
50 DiT blocks and four modules in each token-refiner block. They deliberately
omit the 51 AdaLN projections, so the same 208-target layout works with both the
full/native H3 base and ComfyUI's pruned/curve base. Applying those updates
through ordinary weight merging is correct but rounds a small low-rank delta
back into the quantized base. Applying every update through ComfyUI's generic
bypass path is sharper, but H3 has two call sites which deliberately bypass
``Linear.forward`` (``linear_input_act`` for MLP fc2 and our direct INT8 output
projection).

The strict Turbo node and checkpoint-agnostic standard H3 node share three
application modes:

``fast_runtime``
    Keep every LoRA update in activation space. The INT8 backend applies QKV
    deltas before Q/K RMSNorm/RoPE, FC1 deltas before SwiGLU, and FC2/output
    deltas after their base GEMMs.
``reference_runtime``
    Use normal ComfyUI bypass adapters except for quantized fc2 weights, which
    are merged because stock ``linear_input_act`` does not call forward.
``merged``
    Use ordinary ModelPatcher weight patches for the lowest runtime memory.

All model mutation is owned by ModelPatcher injections/patches.  No module is
permanently modified and adapter tensors are returned to the patcher's offload
device when the injection is ejected.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn.functional as F


TAG = "[AppleSilicon-Optimized/H3Turbo]"
RUNTIME_ATTR = "_asfp8_runtime_lora"
INJECTION_KEY = "asfp8_h3_turbo"
MODES = ("fast_runtime", "reference_runtime", "merged")

_BLOCK_SUFFIXES = (
    "attn.out_proj",
    "attn.qkv_proj",
    "mlp.fc1",
    "mlp.fc2",
)
_REFINER_SUFFIXES = (
    "attn.out_proj",
    "attn.qkv_proj",
    "mlp.fc1",
    "mlp.fc2",
)


def _core_modules() -> set[str]:
    modules = {
        f"blocks.{block}.{suffix}"
        for block in range(50)
        for suffix in _BLOCK_SUFFIXES
    }
    modules.update(
        f"token_refiner.blocks.{block}.{suffix}"
        for block in range(2)
        for suffix in _REFINER_SUFFIXES
    )
    return modules


CORE_MODULES = frozenset(_core_modules())
ADALN_MODULES = frozenset(
    {f"blocks.{block}.adaln_proj.linear" for block in range(50)}
    | {"final_layer.adaln_proj.linear"}
)
FULL_MODULES = CORE_MODULES | ADALN_MODULES
# Public compatibility name retained for tests and external callers. It now
# describes the released ComfyUI adapter layout, not the hypothetical layout
# that also includes every AdaLN projection.
EXPECTED_MODULES = CORE_MODULES
LORA_PREFIX = "diffusion_model."


def _normalized_lora(lora: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept ComfyUI's standard ``diffusion_model.`` key prefix or no prefix."""
    normalized = {}
    for key, value in lora.items():
        canonical = key[len(LORA_PREFIX):] if key.startswith(LORA_PREFIX) else key
        if canonical in normalized:
            raise ValueError(
                f"H3 LoRA contains duplicate prefixed/unprefixed key '{canonical}'")
        normalized[canonical] = value
    return normalized


def _module_names(
    lora: dict[str, torch.Tensor], official: bool = True,
) -> set[str]:
    """Validate standard H3 linear A/B pairs and return their target modules."""
    lora = _normalized_lora(lora)
    pairs: dict[str, set[str]] = {}
    alphas: set[str] = set()
    for key, value in lora.items():
        if key.endswith(".alpha"):
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(f"H3 LoRA alpha '{key}' must be a scalar tensor")
            alphas.add(key[:-len(".alpha")])
            continue
        if key.endswith(".lora_A.weight"):
            module, side = key[:-len(".lora_A.weight")], "A"
        elif key.endswith(".lora_B.weight"):
            module, side = key[:-len(".lora_B.weight")], "B"
        else:
            raise ValueError(
                f"unsupported H3 LoRA tensor '{key}'; only linear lora_A/lora_B "
                "pairs (and optional alpha scalars) are accepted")
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"H3 LoRA tensor '{key}' must be a 2-D tensor")
        pairs.setdefault(module, set()).add(side)

    incomplete = sorted(name for name, sides in pairs.items() if sides != {"A", "B"})
    if incomplete:
        raise ValueError(
            "H3 LoRA has incomplete A/B pairs: " + ", ".join(incomplete[:5]))

    modules = set(pairs)
    if not modules:
        raise ValueError("H3 LoRA does not contain any linear A/B pairs")
    unknown_alphas = sorted(alphas - modules)
    if unknown_alphas:
        raise ValueError(
            "H3 LoRA has alpha values without matching A/B pairs: "
            + ", ".join(unknown_alphas[:5]))
    extra = sorted(modules - FULL_MODULES)
    if extra:
        raise ValueError(
            "H3 LoRA has unsupported targets: " + ", ".join(extra[:5]))
    if official:
        missing = sorted(CORE_MODULES - modules)
        adaln = modules & ADALN_MODULES
        partial_adaln = bool(adaln and adaln != ADALN_MODULES)
    else:
        missing = []
        partial_adaln = False
    if missing or partial_adaln:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} targets ({', '.join(missing[:3])})")
        if partial_adaln:
            omitted = sorted(ADALN_MODULES - adaln)
            detail.append(
                f"incomplete AdaLN target set ({len(omitted)} missing: "
                f"{', '.join(omitted[:3])})")
        raise ValueError(
            "this is not a supported MiniMax H3 Turbo adapter: "
            + "; ".join(detail))
    return modules


def _resolve_h3(model):
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if dm is None or len(getattr(dm, "blocks", ())) != 50:
        raise ValueError("H3 LoRA requires a 50-block MiniMax H3 MODEL")
    if len(getattr(getattr(dm, "token_refiner", None), "blocks", ())) != 2:
        raise ValueError("the connected model does not have H3's two-block token refiner")
    return dm


def _load_adapters(model, lora, modules: Iterable[str]):
    import comfy.lora
    import comfy.utils
    import comfy.weight_adapter

    lora = _normalized_lora(lora)
    modules = sorted(modules)
    key_map = {name: f"diffusion_model.{name}.weight" for name in modules}
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    if set(loaded) != set(key_map.values()):
        missing = sorted(set(key_map.values()) - set(loaded))
        raise ValueError(
            f"ComfyUI could not load {len(missing)} H3 LoRA targets "
            f"({', '.join(missing[:3])})")

    dm = model.model.diffusion_model
    for name in modules:
        key = key_map[name]
        adapter = loaded[key]
        if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter):
            raise ValueError(f"H3 target '{name}' did not decode as a linear LoRA")
        up, down, _alpha, mid, dora, reshape = adapter.weights
        if mid is not None or dora is not None or reshape is not None:
            raise ValueError(
                f"H3 target '{name}' uses an unsupported mid/DoRA/reshape adapter")
        module = comfy.utils.get_attr(dm, name)
        shape = tuple(module.weight.shape)
        if (up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]
                or (up.shape[0], down.shape[1]) != shape):
            raise ValueError(
                f"H3 target '{name}' has delta shape "
                f"{(up.shape[0], down.shape[1])}, model expects {shape}")
    return loaded, key_map


class H3RuntimeLoRA:
    """Linear LoRA adapter with an in-place, backend-callable additive path."""

    def __init__(self, weights, fast: bool):
        self.weights = tuple(weights)
        self.fast = bool(fast)
        self.multiplier = 1.0
        self.is_conv = False

    def _scale(self) -> float:
        down, alpha = self.weights[1], self.weights[2]
        return ((alpha / down.shape[0]) if alpha is not None else 1.0) * self.multiplier

    def linear_parts(self, x: torch.Tensor):
        """Return standard linear LoRA tensors in the activation's format."""
        up, down, _alpha, mid, dora, reshape = self.weights
        if (self.is_conv or mid is not None or dora is not None
                or reshape is not None or up.ndim != 2 or down.ndim != 2):
            raise ValueError("H3 exact runtime path requires a standard linear LoRA")
        return (
            up.to(device=x.device, dtype=x.dtype),
            down.to(device=x.device, dtype=x.dtype),
            self._scale(),
        )

    def add_to(self, base_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add ``B(A(x))`` to a fresh base output without extra full outputs."""
        up, down, scale = self.linear_parts(x)
        delta = F.linear(F.linear(x, down), up)
        return base_out.add_(delta, alpha=scale)

    def add_bhsd_to(self, base_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply an output-projection LoRA to native attention [B,H,S,D]."""
        batch, heads, seq, dim = x.shape
        merged = x.transpose(1, 2).reshape(batch, seq, heads * dim)
        return self.add_to(base_out, merged)

    def add_qkv_to(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   x: torch.Tensor, heads: int, head_dim: int):
        """Add a fused-QKV LoRA without materializing its full 3HD output."""
        up, down, scale = self.linear_parts(x)
        low_rank = F.linear(x, down)
        width = heads * head_dim
        if up.shape[0] != 3 * width:
            raise ValueError("H3 QKV Turbo adapter has an unexpected output width")
        for base, up_part in zip((q, k, v), up.split(width, dim=0)):
            delta = F.linear(low_rank, up_part).view(
                x.shape[0], heads, head_dim).permute(1, 0, 2).unsqueeze(0)
            base.add_(delta, alpha=scale)
            del delta
        return q, k, v

    def bypass_forward(self, original_forward, x, *args, **kwargs):
        base_out = original_forward(x, *args, **kwargs)
        return self.add_to(base_out, x)

    def move_to(self, device, dtype=None):
        moved = []
        for value in self.weights:
            if isinstance(value, torch.Tensor):
                kwargs = {"device": device}
                if dtype is not None and value.is_floating_point():
                    kwargs["dtype"] = dtype
                moved.append(value.to(**kwargs))
            else:
                moved.append(value)
        self.weights = tuple(moved)


def _is_quantized_int8(module) -> bool:
    weight = getattr(module, "weight", None)
    return (getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False))


def _runtime_injection(new_model, adapters, key_map, runtime_modules, strength, fast):
    import comfy.patcher_extension
    import comfy.utils
    from comfy.weight_adapter.bypass import BypassForwardHook

    dm = new_model.model.diffusion_model
    hooks = []
    for name in sorted(runtime_modules):
        module = comfy.utils.get_attr(dm, name)
        adapter = H3RuntimeLoRA(adapters[key_map[name]].weights, fast=fast)
        hooks.append(BypassForwardHook(module, adapter, multiplier=strength))

    def inject(_model_patcher):
        for hook in hooks:
            existing = getattr(hook.module, RUNTIME_ATTR, None)
            if existing is not None and existing is not hook.adapter:
                raise RuntimeError(
                    "another runtime LoRA already owns an H3 target")
        injected = []
        try:
            for hook in hooks:
                hook.inject()
                setattr(hook.module, RUNTIME_ATTR, hook.adapter)
                injected.append(hook)
        except Exception:
            for hook in reversed(injected):
                if getattr(hook.module, RUNTIME_ATTR, None) is hook.adapter:
                    delattr(hook.module, RUNTIME_ATTR)
                hook.eject()
            raise

    def eject(model_patcher):
        for hook in reversed(hooks):
            if getattr(hook.module, RUNTIME_ATTR, None) is hook.adapter:
                delattr(hook.module, RUNTIME_ATTR)
            hook.eject()
            hook.adapter.move_to(model_patcher.offload_device)

    new_model.set_injections(
        INJECTION_KEY,
        [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)],
    )
    return len(hooks)


def _apply_h3_lora(
    model, dm, loaded, key_map, modules, strength: float, mode: str,
    label: str,
):
    if INJECTION_KEY in getattr(model, "injections", {}):
        raise ValueError("an H3 runtime LoRA is already attached to this MODEL")
    if mode != "merged" and "bypass_lora" in getattr(model, "injections", {}):
        raise ValueError(
            "another bypass LoRA is already attached. Runtime H3 LoRA cannot "
            "safely compose nested bypass hooks at fused H3 call sites; remove "
            "the other runtime LoRA or use merged mode.")

    new_model = model.clone()
    if strength == 0:
        return new_model

    if mode == "merged":
        applied = new_model.add_patches(loaded, strength)
        if len(applied) != len(modules):
            raise ValueError(
                f"only {len(applied)}/{len(modules)} H3 LoRA weights were patched")
        logging.info(
            "%s base=%s merged %d adapters at strength %.3f (lowest runtime "
            "memory; quantized bases may soften the delta)", TAG,
            "curve" if getattr(dm, "use_adaln_curves", False) else "native",
            len(applied), strength)
        return new_model

    import comfy.utils
    quant_fc2 = {
        name for name in modules
        if name.endswith(".mlp.fc2")
        and _is_quantized_int8(comfy.utils.get_attr(dm, name))
    }
    if mode == "fast_runtime" and quant_fc2:
        import comfy.ops
        if not getattr(comfy.ops.linear_input_act, "_asfp8_wrapped", False):
            raise ValueError(
                "fast_runtime with an INT8 H3 base requires the Apple Silicon "
                "INT8 backend (set ASFP8_INT8_EXT=1 and restart ComfyUI). Without "
                "it, H3's fused fc2 call would bypass the LoRA update.")

    merged_modules = quant_fc2 if mode == "reference_runtime" else set()
    runtime_modules = modules - merged_modules
    runtime_count = _runtime_injection(
        new_model, loaded, key_map, runtime_modules, strength,
        fast=(mode == "fast_runtime"))
    if merged_modules:
        merged = {key_map[name]: loaded[key_map[name]] for name in merged_modules}
        applied = new_model.add_patches(merged, strength)
        if len(applied) != len(merged_modules):
            raise ValueError("failed to attach all reference-runtime fc2 patches")

    logging.info(
        "%s base=%s mode=%s strength=%.3f: %d activation-space adapters, "
        "%d merged fc2 (%s)", TAG,
        "curve" if getattr(dm, "use_adaln_curves", False) else "native",
        mode, strength, runtime_count, len(merged_modules), label)
    return new_model


def apply_h3_turbo(model, lora, strength: float, mode: str):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    dm = _resolve_h3(model)
    modules = _module_names(lora)
    if getattr(dm, "use_adaln_curves", False) and modules & ADALN_MODULES:
        raise ValueError(
            "this Turbo LoRA contains full-width AdaLN adapters, but the "
            "pruned/curve H3 base uses an 8-dimensional sampled basis. Select "
            "a released 208-target ComfyUI/pruned adapter or use the full/native "
            "H3 base.")
    loaded, key_map = _load_adapters(model, lora, modules)
    return _apply_h3_lora(
        model, dm, loaded, key_map, modules, strength, mode, "official Turbo")


def apply_h3_lora(model, lora, strength: float, mode: str):
    """Apply any shape-compatible subset of standard linear H3 LoRA targets."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    dm = _resolve_h3(model)
    modules = _module_names(lora, official=False)
    loaded, key_map = _load_adapters(model, lora, modules)
    return _apply_h3_lora(
        model, dm, loaded, key_map, modules, strength, mode, "standard H3")


class AppleSiliconH3TurboLoRA:
    CATEGORY = "Apple Silicon/approximate"
    DESCRIPTION = (
        "Apply a validated MiniMax H3 Turbo LoRA to native or pruned/curve H3. "
        "fast_runtime preserves LoRA "
        "deltas in activation space and cooperates with the fused INT8 backend; "
        "reference_runtime merges only INT8 fc2 updates for compatibility; "
        "merged uses the least runtime memory but can soften small deltas on a "
        "quantized base. This node does not change the sampler or step count."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "mode": (list(MODES), {"default": "fast_runtime"}),
            }
        }

    def apply(self, model, lora_name, strength, mode):
        import comfy.utils
        import folder_paths

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        return (apply_h3_turbo(model, lora, strength, mode),)


class AppleSiliconH3LoRA(AppleSiliconH3TurboLoRA):
    CATEGORY = "Apple Silicon/exact"
    DESCRIPTION = (
        "Apply any shape-compatible standard linear MiniMax H3 LoRA target "
        "subset. fast_runtime keeps every delta in activation space and uses "
        "the exact LoRA-aware INT8 QKV and tiled MLP hot paths when eligible; "
        "unsupported layouts fail clearly instead of changing model quality."
    )

    def apply(self, model, lora_name, strength, mode):
        import comfy.utils
        import folder_paths

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        return (apply_h3_lora(model, lora, strength, mode),)


NODE_CLASS_MAPPINGS = {
    "AppleSiliconH3TurboLoRA": AppleSiliconH3TurboLoRA,
    "AppleSiliconH3LoRA": AppleSiliconH3LoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AppleSiliconH3TurboLoRA": "H3 Turbo LoRA (Apple Silicon)",
    "AppleSiliconH3LoRA": "H3 LoRA (Apple Silicon, Exact Runtime)",
}
