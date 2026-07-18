"""ASFP8_NATIVE=1 master switch: implies every opt-in var; explicit values win."""
from _patches import _common


def test_off_by_default():
    env = {}
    assert _common.apply_native_master_switch(env) == []
    assert env == {}


def test_native_zero_is_not_on():
    env = {"ASFP8_NATIVE": "0"}
    assert _common.apply_native_master_switch(env) == []
    assert "ASFP8_INT8_EXT" not in env


def test_native_implies_all_optins():
    env = {"ASFP8_NATIVE": "1"}
    implied = _common.apply_native_master_switch(env)
    assert set(implied) == set(_common.NATIVE_IMPLIES)
    for var in _common.NATIVE_IMPLIES:
        assert env[var] == "1"


def test_explicit_zero_wins_over_master():
    env = {"ASFP8_NATIVE": "1", "ASFP8_DIT_EXT": "0"}
    implied = _common.apply_native_master_switch(env)
    assert env["ASFP8_DIT_EXT"] == "0"          # kill switch preserved
    assert "ASFP8_DIT_EXT" not in implied
    assert env["ASFP8_INT8_EXT"] == "1"          # the rest still implied


def test_explicit_one_not_reported_as_implied():
    env = {"ASFP8_NATIVE": "1", "ASFP8_INT8_EXT": "1"}
    implied = _common.apply_native_master_switch(env)
    assert "ASFP8_INT8_EXT" not in implied       # was already set by the user
    assert env["ASFP8_INT8_EXT"] == "1"
