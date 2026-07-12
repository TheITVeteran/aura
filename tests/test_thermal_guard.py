"""The thermal guard must actually work on macOS.

The old gate read psutil.sensors_temperatures(), which does not exist on
Darwin: on the primary deployment host the guard never fired and sustained
background load cooked the machine (operator-reported, July 2026).
"""
from __future__ import annotations

import pytest

import core.runtime.thermal as thermal
from core.runtime.thermal import ThermalReading, reset_thermal_cache, thermal_state


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_thermal_cache()
    yield
    reset_thermal_cache()


def test_nsprocessinfo_is_the_primary_source(monkeypatch):
    monkeypatch.setattr(
        thermal, "_read_nsprocessinfo", lambda: ThermalReading(2, "nsprocessinfo")
    )
    monkeypatch.setattr(thermal, "_read_pmset", lambda: ThermalReading(0, "pmset"))
    reading = thermal_state()
    assert reading.level == 2
    assert reading.source == "nsprocessinfo"


def test_fallback_chain_reaches_pmset_then_psutil(monkeypatch):
    monkeypatch.setattr(thermal, "_read_nsprocessinfo", lambda: None)
    monkeypatch.setattr(thermal, "_read_pmset", lambda: None)
    monkeypatch.setattr(
        thermal, "_read_psutil", lambda: ThermalReading(3, "psutil", "max_temp_c=91.0")
    )
    reading = thermal_state()
    assert (reading.level, reading.source) == (3, "psutil")


def test_blind_host_reports_itself_blind_not_cool(monkeypatch):
    monkeypatch.setattr(thermal, "_read_nsprocessinfo", lambda: None)
    monkeypatch.setattr(thermal, "_read_pmset", lambda: None)
    monkeypatch.setattr(thermal, "_read_psutil", lambda: None)
    reading = thermal_state()
    assert reading.level == 0
    assert reading.blind, "an unreadable host must be distinguishable from a cool one"


def test_reading_is_cached(monkeypatch):
    calls = []

    def counting():
        calls.append(1)
        return ThermalReading(0, "nsprocessinfo")

    monkeypatch.setattr(thermal, "_read_nsprocessinfo", counting)
    thermal_state()
    thermal_state()
    assert len(calls) == 1


@pytest.mark.host_observation
def test_live_reading_on_this_host_is_not_blind():
    """On macOS or Linux, at least one real source must answer."""
    reading = thermal_state()
    assert reading.level in (0, 1, 2, 3)
    assert not reading.blind, f"host has no working thermal source: {reading}"


def test_background_policy_defers_on_serious_heat(resource_observer):
    from core.runtime import background_policy

    resource_observer.configure_thermal(3, provider="nsprocessinfo")
    reason = background_policy._read_compute_pressure_reason()
    assert reason.startswith("thermal_pressure_level_3"), (
        f"critical heat must gate background work, got {reason!r}"
    )


def test_background_policy_allows_nominal_heat(resource_observer):
    from core.runtime import background_policy

    resource_observer.configure_thermal(0, provider="nsprocessinfo")
    reason = background_policy._read_compute_pressure_reason()
    assert not reason.startswith("thermal"), f"nominal heat must not gate: {reason!r}"
