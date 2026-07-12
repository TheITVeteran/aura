from __future__ import annotations

import asyncio

import pytest

from core.soma import resilience_engine as resilience_module
from core.soma.resilience_engine import ResilienceEngine


def test_pulse_uses_attributable_resource_telemetry(resource_observer):
    resource_observer.configure_compute(cpu_percent=77.0)
    resource_observer.configure_memory(percent=82.0)
    resource_observer.configure_thermal(2, provider="simulated-hot")

    engine = ResilienceEngine()

    pulse = asyncio.run(engine.pulse())

    assert pulse["cpu_pressure"] == pytest.approx(0.77)
    assert pulse["ram_pressure"] == pytest.approx(0.82)
    assert pulse["thermal_load"] == pytest.approx((82.0 - 45.0) / 55.0)
    assert pulse["resource_anxiety"] == pytest.approx(0.82)
    assert all(0.0 <= value <= 1.0 for value in pulse.values())


def test_body_snapshot_exposes_resource_pressure(resource_observer):
    resource_observer.configure_compute(cpu_percent=40.0)
    resource_observer.configure_memory(percent=92.0)
    resource_observer.configure_thermal(0)

    engine = ResilienceEngine()

    snapshot = engine.get_body_snapshot()

    assert snapshot["soma"]["cpu_pressure"] == pytest.approx(0.4)
    assert snapshot["soma"]["ram_pressure"] == pytest.approx(0.92)
    assert snapshot["soma"]["resource_anxiety"] == pytest.approx(0.92)
    assert snapshot["energy"] < 1.0
    assert snapshot["vitality"] < 1.0


def test_resilience_inputs_are_bounded():
    engine = ResilienceEngine()

    assert engine._clamp01(float("nan")) == pytest.approx(0.0)
    assert engine._clamp01("not-a-number") == pytest.approx(0.0)

    engine.record_failure("planning", severity=-5.0, stakes=3.0)
    assert engine.profile.frustration == pytest.approx(0.0)
    assert engine.profile.depletion == pytest.approx(0.0)
    assert engine.profile.failure_history[-1].severity == pytest.approx(0.0)
    assert engine.profile.failure_history[-1].stakes == pytest.approx(1.0)

    engine.record_failure("planning", severity=4.0, stakes=4.0)
    assert engine.profile.frustration == pytest.approx(0.4)
    assert engine.profile.depletion == pytest.approx(0.15)
    assert engine.profile.failure_history[-1].severity == pytest.approx(1.0)
    assert engine.profile.failure_history[-1].stakes == pytest.approx(1.0)

    engine.profile.frustration = 0.4
    engine.record_success("planning", stakes=-2.0)
    assert engine.profile.frustration == pytest.approx(0.4)

    engine.profile.depletion = 0.25
    engine.record_rest(-120.0)
    assert engine.profile.depletion == pytest.approx(0.25)


def test_decay_loop_returns_when_shutdown_is_requested(monkeypatch):
    monkeypatch.setattr(resilience_module, "is_shutdown_requested", lambda: True)
    engine = ResilienceEngine()

    asyncio.run(asyncio.wait_for(engine._decay_loop(), timeout=0.1))
