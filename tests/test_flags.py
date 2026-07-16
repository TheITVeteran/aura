"""Contracts for the C1 typed flag layer.

The layer exists so a knob has exactly one meaning, one type, one default,
and an enumerable live value — instead of 686 scattered untyped env reads.
"""
from __future__ import annotations

import pytest

from core.runtime.flags import (
    Flag,
    FlagKind,
    FlagSpec,
    aura_root_override,
    declare,
    declared_flags,
    flag_report,
    get_flag,
)

pytestmark = pytest.mark.unit


def _spec(name="AURA_TEST_FLAG_X", kind=FlagKind.BOOL, default=True):
    return FlagSpec(name=name, kind=kind, default=default, description="d", owner="tests")


class TestResolutionPrecedence:
    def test_env_wins_over_default(self, monkeypatch):
        flag = Flag(_spec(kind=FlagKind.FLOAT, default=1.5))
        monkeypatch.setenv("AURA_TEST_FLAG_X", "2.5")
        value, source = flag.value_with_source()
        assert value == 2.5 and source == "env"

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("AURA_TEST_FLAG_X", raising=False)
        flag = Flag(_spec(kind=FlagKind.FLOAT, default=1.5))
        value, source = flag.value_with_source()
        assert value == 1.5 and source == "default"

    def test_settings_layer_between_env_and_default(self, monkeypatch):
        flag = Flag(_spec(kind=FlagKind.INT, default=7))
        monkeypatch.delenv("AURA_TEST_FLAG_X", raising=False)
        monkeypatch.setattr(
            Flag, "_persisted_value", lambda self: 9
        )
        value, source = flag.value_with_source()
        assert value == 9 and source == "settings"

    def test_aura_root_override_is_typed_and_read_through(self, monkeypatch, tmp_path):
        expected = tmp_path / "aura-runtime"
        monkeypatch.setenv("AURA_ROOT", f" {expected} ")
        assert aura_root_override() == str(expected)
        assert get_flag("AURA_ROOT") is not None


class TestCoercion:
    def test_bool_parsing(self, monkeypatch):
        flag = Flag(_spec(kind=FlagKind.BOOL, default=True))
        for raw, expected in (
            ("1", True), ("true", True), ("ON", True), ("yes", True),
            ("0", False), ("false", False), ("OFF", False), ("no", False),
        ):
            monkeypatch.setenv("AURA_TEST_FLAG_X", raw)
            assert flag.value() is expected, raw

    def test_malformed_env_degrades_to_default_never_raises(self, monkeypatch):
        flag = Flag(_spec(kind=FlagKind.FLOAT, default=3.0))
        monkeypatch.setenv("AURA_TEST_FLAG_X", "not-a-number")
        value, source = flag.value_with_source()
        assert value == 3.0
        assert source == "default(malformed_env)"

    def test_int_and_string(self, monkeypatch):
        int_flag = Flag(_spec(kind=FlagKind.INT, default=1))
        monkeypatch.setenv("AURA_TEST_FLAG_X", " 42 ")
        assert int_flag.value() == 42
        str_flag = Flag(_spec(kind=FlagKind.STRING, default="a"))
        monkeypatch.setenv("AURA_TEST_FLAG_X", "enforce")
        assert str_flag.value() == "enforce"


class TestRegistryDiscipline:
    def test_identical_redeclaration_is_idempotent(self):
        a = declare(
            "AURA_TEST_FLAG_IDEM", kind=FlagKind.BOOL, default=True,
            description="d", owner="tests",
        )
        b = declare(
            "AURA_TEST_FLAG_IDEM", kind=FlagKind.BOOL, default=True,
            description="d", owner="tests",
        )
        assert a is b

    def test_conflicting_redeclaration_raises(self):
        declare(
            "AURA_TEST_FLAG_CONFLICT", kind=FlagKind.BOOL, default=True,
            description="d", owner="tests",
        )
        with pytest.raises(ValueError, match="different spec"):
            declare(
                "AURA_TEST_FLAG_CONFLICT", kind=FlagKind.BOOL, default=False,
                description="d", owner="other",
            )

    def test_names_must_be_namespaced(self):
        with pytest.raises(ValueError, match="AURA_"):
            declare("NOT_NAMESPACED", kind=FlagKind.BOOL, default=True,
                    description="d", owner="tests")


class TestEnumeration:
    def test_report_carries_value_and_source(self, monkeypatch):
        declare(
            "AURA_TEST_FLAG_REPORT", kind=FlagKind.FLOAT, default=5.0,
            description="report test", owner="tests",
        )
        monkeypatch.setenv("AURA_TEST_FLAG_REPORT", "6.0")
        entry = next(e for e in flag_report() if e["name"] == "AURA_TEST_FLAG_REPORT")
        assert entry["value"] == 6.0
        assert entry["source"] == "env"
        assert entry["default"] == 5.0
        assert entry["owner"] == "tests"

    def test_this_pass_reliability_flags_are_declared(self):
        """The first adopters: the K3/K4/K1/K2 flags declare at import."""
        import core.brain.lane_admission  # noqa: F401
        import core.runtime.lane_reconciler  # noqa: F401
        from core.runtime.health_contract import _startup_deadline_s

        _startup_deadline_s()  # lazy declaration
        declared = declared_flags()
        for name in (
            "AURA_LANE_BUDGET_GB",
            "AURA_LANE_BUDGET_FRACTION",
            "AURA_LANE_EVICTION_SHIELD_S",
            "AURA_LANE_ADMISSION",
            "AURA_CRASHLOOP_YOUNG_S",
            "AURA_CRASHLOOP_THRESHOLD",
            "AURA_CRASHLOOP_BREAKER",
            "AURA_LANE_RECONCILER",
            "AURA_LANE_RECONCILE_INTERVAL_S",
            "AURA_STARTUP_DEADLINE_S",
        ):
            assert name in declared, f"{name} not declared through the flag layer"
            assert get_flag(name) is not None
