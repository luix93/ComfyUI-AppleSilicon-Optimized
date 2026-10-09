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

The H3 LoRA node supports ordered adapter stacks, optional strict Turbo
validation for each entry, and three application modes:

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

import json
import logging
import math
from typing import Iterable

import torch
import torch.nn.functional as F


TAG = "[AppleSilicon-Optimized/H3LoRA]"
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


class H3RuntimeLoRAStack:
    """One hook per target; apply each adapter in selection order.

    Do not concatenate ranks or pre-scale B: that changes intermediate rounding.
    QKV uses scatter + ordered updates + the existing normalization/RoPE pass;
    the single-adapter fused QKV epilogue remains unchanged.
    """

    def __init__(self, adapters):
        self.adapters = tuple(adapters)
        self.fast = all(adapter.fast for adapter in self.adapters)

    def add_to(self, base_out, x):
        for adapter in self.adapters:
            adapter.add_to(base_out, x)
        return base_out

    def add_bhsd_to(self, base_out, x):
        batch, heads, seq, dim = x.shape
        return self.add_to(base_out, x.transpose(1, 2).reshape(batch, seq, heads * dim))

    def add_qkv_to(self, q, k, v, x, heads, head_dim):
        for adapter in self.adapters:
            adapter.add_qkv_to(q, k, v, x, heads, head_dim)
        return q, k, v

    def bypass_forward(self, original_forward, x, *args, **kwargs):
        return self.add_to(original_forward(x, *args, **kwargs), x)

    def move_to(self, device, dtype=None):
        for adapter in self.adapters:
            adapter.move_to(device, dtype)


def _is_quantized_int8(module) -> bool:
    weight = getattr(module, "weight", None)
    return (getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False))


def _runtime_injection(new_model, targets, fast):
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.utils
    from comfy.weight_adapter.bypass import BypassForwardHook

    dm = new_model.model.diffusion_model
    hooks = []
    for name, entries in sorted(targets.items()):
        module = comfy.utils.get_attr(dm, name)
        adapters = []
        for weights, strength in entries:
            adapter = H3RuntimeLoRA(weights, fast=fast)
            adapter.multiplier = strength
            adapters.append(adapter)
        adapter = adapters[0] if len(adapters) == 1 else H3RuntimeLoRAStack(adapters)
        hooks.append(BypassForwardHook(
            module, adapter, multiplier=entries[0][1] if len(entries) == 1 else 1.0))

    def inject(_model_patcher):
        for hook in hooks:
            existing = getattr(hook.module, RUNTIME_ATTR, None)
            if existing is not None and existing is not hook.adapter:
                raise RuntimeError(
                    "another runtime LoRA already owns an H3 target")
        injected = []
        try:
            for hook in hooks:
                injected.append(hook)
                # ComfyUI moves a single adapter's .weights itself. A stack owns
                # several tuples, so move those explicitly at injection time.
                if isinstance(hook.adapter, H3RuntimeLoRAStack):
                    dtype = getattr(hook.module.weight, "dtype", None)
                    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                        dtype = None
                    hook.adapter.move_to(comfy.model_management.get_torch_device(), dtype)
                hook.inject()
                setattr(hook.module, RUNTIME_ATTR, hook.adapter)
        except Exception:
            for hook in reversed(injected):
                if getattr(hook.module, RUNTIME_ATTR, None) is hook.adapter:
                    delattr(hook.module, RUNTIME_ATTR)
                hook.eject()
                hook.adapter.move_to(_model_patcher.offload_device)
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


def apply_h3_lora_stack(model, entries, mode):
    """Entries are (state_dict, strength, strict_turbo); validate before patching."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    active = []
    for lora, strength, strict in entries:
        if not math.isfinite(strength) or not -10 <= strength <= 10:
            raise ValueError("H3 LoRA strength must be finite and between -10 and 10")
        if strength != 0:
            active.append((lora, strength, strict))
    if not active:
        return model
    if INJECTION_KEY in getattr(model, "injections", {}):
        raise ValueError("an H3 runtime LoRA is already attached; add entries to the same H3 LoRA node")
    if mode != "merged" and "bypass_lora" in getattr(model, "injections", {}):
        raise ValueError(
            "another bypass LoRA is already attached. Remove it or use merged mode; "
            "nested hooks cannot safely compose at fused H3 call sites.")

    import comfy.utils
    dm = _resolve_h3(model)
    targets = {}
    merged_patches = []
    for lora, strength, strict in active:
        modules = _module_names(lora, official=strict)
        if strict and getattr(dm, "use_adaln_curves", False) and modules & ADALN_MODULES:
            raise ValueError(
                "this Turbo LoRA has full-width AdaLN targets. Use a released "
                "208-target ComfyUI adapter with the curve base, or a native H3 base.")
        loaded, key_map = _load_adapters(model, lora, modules)
        quant_fc2 = {
            name for name in modules if name.endswith(".mlp.fc2")
            and _is_quantized_int8(comfy.utils.get_attr(dm, name))
        }
        if mode == "fast_runtime" and quant_fc2:
            import comfy.ops
            if not getattr(comfy.ops.linear_input_act, "_asfp8_wrapped", False):
                raise ValueError(
                    "fast_runtime with an INT8 H3 base requires the Apple Silicon "
                    "INT8 backend (set ASFP8_INT8_EXT=1 and restart ComfyUI).")
        merged = modules if mode == "merged" else (
            quant_fc2 if mode == "reference_runtime" else set())
        if merged:
            merged_patches.append(({
                key_map[name]: loaded[key_map[name]] for name in merged}, strength))
        for name in sorted(modules - merged):
            targets.setdefault(name, []).append((loaded[key_map[name]].weights, strength))

    new_model = model.clone()
    for patches, strength in merged_patches:
        if len(new_model.add_patches(patches, strength)) != len(patches):
            raise ValueError("failed to attach all H3 LoRA weight patches")
    if targets:
        _runtime_injection(new_model, targets, fast=(mode == "fast_runtime"))
    logging.info("%s %d LoRA(s), mode=%s, %d runtime targets, %d merged updates",
                 TAG, len(active), mode, len(targets),
                 sum(len(patches) for patches, _ in merged_patches))
    return new_model


def apply_h3_turbo(model, lora, strength: float, mode: str):
    return apply_h3_lora_stack(model, [(lora, strength, True)], mode)


def apply_h3_lora(model, lora, strength: float, mode: str):
    return apply_h3_lora_stack(model, [(lora, strength, False)], mode)


def parse_lora_entries(value):
    """Shared UI/API payload. Disabled or zero-weight entries never load files."""
    try:
        entries = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("H3 LoRA selection must be a JSON list") from exc
    if not isinstance(entries, list):
        raise ValueError("H3 LoRA selection must be a JSON list")
    active = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict) or type(entry.get("enabled", True)) is not bool:
            raise ValueError(f"H3 LoRA entry {index}: expected an entry with a boolean enabled switch")
        if not entry.get("enabled", True):
            continue
        strength = entry.get("strength", 1.0)
        if (type(strength) not in (int, float) or not math.isfinite(strength)
                or not -10 <= strength <= 10):
            raise ValueError(f"H3 LoRA entry {index}: strength must be finite and between -10 and 10")
        if strength == 0:
            continue
        name = entry.get("lora_name")
        validation = entry.get("validation", "standard")
        if not isinstance(name, str) or not name:
            raise ValueError(f"H3 LoRA entry {index}: choose a LoRA file")
        if validation not in ("standard", "turbo"):
            raise ValueError(f"H3 LoRA entry {index}: validation must be standard or turbo")
        active.append((name, float(strength), validation == "turbo"))
    return active


class AppleSiliconH3LoRA:
    CATEGORY = "Apple Silicon/H3"
    DESCRIPTION = (
        "Apply one or more H3 LoRAs in list order, each with its own enable switch "
        "and strength. Standard validation accepts compatible linear H3 targets; "
        "Turbo validation requires the released complete Turbo layout. "
        "fast_runtime keeps deltas in activation space and cooperates with INT8 "
        "kernels. reference_runtime merges INT8 FC2; merged patches all weights. "
        "Does not change the sampler or step count."
    )
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "model": ("MODEL",),
            "loras": ("STRING", {
                "default": "[]", "h3_lora_files": folder_paths.get_filename_list("loras"),
            }),
            "mode": (list(MODES), {"default": "fast_runtime"}),
        }}

    def apply(self, model, loras="[]", mode="fast_runtime"):
        import comfy.utils
        import folder_paths
        selections = parse_lora_entries(loras)
        # Check against the current registry as well as resolving the path.
        # JSON/API input must not allow paths outside registered LoRA folders.
        available = set(folder_paths.get_filename_list("loras"))
        entries = []
        for name, strength, strict in selections:
            if name not in available:
                raise ValueError(f"H3 LoRA file is not in the LoRA folder: {name}")
            path = folder_paths.get_full_path_or_raise("loras", name)
            entries.append((comfy.utils.load_torch_file(path, safe_load=True), strength, strict))
        return (apply_h3_lora_stack(model, entries, mode),)


NODE_CLASS_MAPPINGS = {"AppleSiliconH3LoRA": AppleSiliconH3LoRA}
NODE_DISPLAY_NAME_MAPPINGS = {"AppleSiliconH3LoRA": "H3 LoRA (Apple Silicon)"}
