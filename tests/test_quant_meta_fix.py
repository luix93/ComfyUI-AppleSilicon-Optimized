"""Patch #22: missing comfy_quant 'format' defaults to int8_tensorwise (int8+scale only)."""
import json

import torch

from _patches import quant_meta_fix as qm


def _blob(conf):
    return torch.frombuffer(bytearray(json.dumps(conf).encode()), dtype=torch.uint8)


def _conf(state_dict, prefix="layer."):
    return json.loads(bytes(state_dict[f"{prefix}comfy_quant"].numpy().tobytes()))


def test_injects_format_for_int8_with_scale():
    sd = {
        "layer.comfy_quant": _blob({"convrot": True, "convrot_groupsize": 256, "per_row": True}),
        "layer.weight": torch.zeros(4, 4, dtype=torch.int8),
        "layer.weight_scale": torch.ones(4, 1),
    }
    qm._maybe_fix(sd, "layer.")
    conf = _conf(sd)
    assert conf["format"] == "int8_tensorwise"
    assert conf["convrot"] is True and conf["per_row"] is True  # rest preserved


def test_existing_format_untouched():
    sd = {
        "layer.comfy_quant": _blob({"format": "int8_tensorwise", "convrot": True}),
        "layer.weight": torch.zeros(4, 4, dtype=torch.int8),
        "layer.weight_scale": torch.ones(4, 1),
    }
    before = bytes(sd["layer.comfy_quant"].numpy().tobytes())
    qm._maybe_fix(sd, "layer.")
    assert bytes(sd["layer.comfy_quant"].numpy().tobytes()) == before


def test_non_int8_left_alone():
    sd = {
        "layer.comfy_quant": _blob({"convrot": True}),
        "layer.weight": torch.zeros(4, 4, dtype=torch.uint8),  # packed w4, not int8
        "layer.weight_scale": torch.ones(4, 1),
    }
    qm._maybe_fix(sd, "layer.")
    assert "format" not in _conf(sd)


def test_missing_scale_left_alone():
    sd = {
        "layer.comfy_quant": _blob({"convrot": True}),
        "layer.weight": torch.zeros(4, 4, dtype=torch.int8),
    }
    qm._maybe_fix(sd, "layer.")
    assert "format" not in _conf(sd)
