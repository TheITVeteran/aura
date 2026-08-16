"""CP126 contract tests for the ontology genesis engine.

The headline finding was that "autonomous formation of cognitive laws" was a
sixty-second sleep, reporting `active: true` the whole time. The module was
made to say so, and these pinned that it did.

The step is written now, so the four tests that pinned its absence pin its
presence instead — including the part of the CP126 protection that still
matters most: a cycle that discovers nothing must never look like one that
did. The admission-gate tests below are unchanged.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain import ontology_genesis as module
from core.brain.ontology_genesis import (
    ANXIETY_THRESHOLD_ADMISSION,
    ANXIETY_THRESHOLD_RUNNING,
    ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION,
    UNKNOWN_ANXIETY,
    OntologyGenesisEngine,
)


@pytest.fixture()
def engine() -> OntologyGenesisEngine:
    return OntologyGenesisEngine()


@pytest.fixture()
def services(monkeypatch):
    registry: dict = {}
    monkeypatch.setattr(
        module, "get_runtime_service",
        lambda name, default=None: registry.get(name, default),
    )
    return registry


class _Homeostasis:
    def __init__(self, anxiety):
        self.anxiety = anxiety


class _Kernel:
    def __init__(self, volition):
        self.volition_level = volition


# --- c0a3c26e: the module does not claim discovery it cannot do ---------


def test_discovery_is_declared_implemented():
    """And the flag is not the evidence — the step it names has to be there."""
    assert module.DISCOVERY_IMPLEMENTED is True
    assert callable(getattr(module.OntologyGenesisEngine, "run_discovery_cycle", None))
    assert callable(getattr(module, "degradation_observations", None))


def test_start_is_admitted_once_the_step_exists(engine, services):
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(5)

    assert asyncio.run(engine.start_discovery()) is True
    assert engine.get_status()["last_refusal"] == ""
    asyncio.run(engine.stop_discovery())


def test_status_reports_implemented_with_separate_counters(engine):
    status = engine.get_status()

    assert status["implemented"] is True
    assert status["active"] is False
    assert status["discoveries"] == 0
    # Cycles and discoveries are separate numbers. Most cycles find nothing,
    # so a single "discoveries: 0" cannot distinguish an engine that is
    # working from one that never ran, which is the CP126 confusion in a new
    # costume.
    assert status["cycles"] == 0
    assert status["integrated"] == 0
    assert status["last_law"] is None


def test_active_is_still_gated_on_implementation(engine, services):
    services["homeostasis"] = _Homeostasis(0.0)
    engine._active = True
    monkey = module.DISCOVERY_IMPLEMENTED
    try:
        module.DISCOVERY_IMPLEMENTED = False
        assert engine.get_status()["active"] is False
    finally:
        module.DISCOVERY_IMPLEMENTED = monkey


def test_an_empty_cycle_records_a_refusal_and_no_discovery(engine):
    """The CP126 defect in its most dangerous form: nothing, reported as work."""
    outcome = asyncio.run(engine.run_discovery_cycle(observations=[]))

    assert outcome.found is False
    status = engine.get_status()
    assert status["discoveries"] == 0
    assert status["integrated"] == 0
    assert status["cycles"] == 1
    assert status["last_refusal"]


# --- 72c94940: mode alone grants nothing --------------------------------


def test_deep_research_string_does_not_bypass_volition(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(0)

    assert asyncio.run(engine.start_discovery(mode="deep_research")) is False
    assert engine.get_status()["last_refusal"] == "insufficient_volition"


def test_a_forged_token_does_not_authorize_deep_research(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(0)

    started = asyncio.run(
        engine.start_discovery(mode="deep_research", capability_token="made-up")
    )

    assert started is False


def test_sufficient_volition_admits(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(2)

    assert asyncio.run(engine.start_discovery()) is True
    asyncio.run(engine.stop_discovery())


def test_a_valid_token_authorizes_deep_research(engine, services, monkeypatch):
    from core.agency.capability_token import get_token_store

    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(0)
    token = get_token_store().issue(
        origin="test", scope="unit", ttl_seconds=30.0,
        domain="self_modification", requested_action="deep_research",
        approver="cp126-test", parent_receipt="test-receipt",
    )
    token_str = getattr(token, "token", None) or getattr(token, "token_str", "")

    started = asyncio.run(
        engine.start_discovery(mode="deep_research", capability_token=token_str)
    )

    assert started is True
    asyncio.run(engine.stop_discovery())


# --- d52e6a29: missing telemetry defers, never admits -------------------


def test_missing_homeostasis_reports_maximum_pressure(engine, services):
    anxiety, measured = engine.resource_anxiety()

    assert anxiety == UNKNOWN_ANXIETY
    assert measured is False


def test_missing_telemetry_refuses_admission(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["aura_kernel"] = _Kernel(5)

    assert asyncio.run(engine.start_discovery()) is False
    assert engine.get_status()["last_refusal"] == "resource_telemetry_unavailable"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "high", None])
def test_an_unusable_anxiety_reading_is_unmeasured(engine, services, bad):
    services["homeostasis"] = _Homeostasis(bad)

    anxiety, measured = engine.resource_anxiety()

    assert measured is False
    assert anxiety == UNKNOWN_ANXIETY


def test_high_pressure_refuses_admission(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.9)
    services["aura_kernel"] = _Kernel(5)

    assert asyncio.run(engine.start_discovery()) is False
    assert "resource_pressure" in engine.get_status()["last_refusal"]


def test_a_raising_homeostasis_is_unmeasured(engine, services):
    class Boom:
        @property
        def anxiety(self):
            raise RuntimeError("probe down")

    services["homeostasis"] = Boom()

    assert engine.resource_anxiety() == (UNKNOWN_ANXIETY, False)


# --- 96cb9483: the reported threshold is the real one -------------------


def test_status_reports_the_thresholds_actually_in_force(engine, services):
    services["aura_kernel"] = _Kernel(0)

    status = engine.get_status()

    assert status["anxiety_threshold_admission"] == ANXIETY_THRESHOLD_ADMISSION
    assert status["anxiety_threshold_running"] == ANXIETY_THRESHOLD_RUNNING


def test_the_running_threshold_tracks_volition(engine, services):
    assert engine.running_threshold(0) == ANXIETY_THRESHOLD_RUNNING
    assert engine.running_threshold(5) == ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION


def test_high_volition_status_reports_the_higher_threshold(engine, services):
    services["aura_kernel"] = _Kernel(5)

    assert engine.get_status()["anxiety_threshold_running"] == (
        ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION
    )


def test_status_no_longer_advertises_a_threshold_nothing_uses(engine):
    """The old status hardcoded 0.2 while nothing used 0.2."""
    status = engine.get_status()

    assert 0.2 not in {
        status["anxiety_threshold_admission"], status["anxiety_threshold_running"],
    }


# --- 478804d4: the task is supervised -----------------------------------


def test_a_failed_task_clears_the_active_flag(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(5)

    async def scenario():
        async def boom(volition=0):
            raise RuntimeError("loop exploded")

        monkeypatch.setattr(engine, "_discovery_loop", boom)
        await engine.start_discovery()
        await asyncio.sleep(0.05)
        return engine._active

    assert asyncio.run(scenario()) is False


def test_a_stale_active_flag_does_not_block_restart(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(5)

    async def scenario():
        async def quick(volition=0):
            return None

        monkeypatch.setattr(engine, "_discovery_loop", quick)
        await engine.start_discovery()
        await asyncio.sleep(0.05)
        # Simulate the stale flag the old code left behind.
        engine._active = True
        restarted = await engine.start_discovery()
        await engine.stop_discovery()
        return restarted

    assert asyncio.run(scenario()) is True


def test_stop_is_safe_when_nothing_is_running(engine):
    asyncio.run(engine.stop_discovery())

    assert engine._active is False


def test_status_exposes_task_completion(engine, services, monkeypatch):
    monkeypatch.setattr(module, "DISCOVERY_IMPLEMENTED", True)
    services["homeostasis"] = _Homeostasis(0.0)
    services["aura_kernel"] = _Kernel(5)

    async def scenario():
        async def quick(volition=0):
            return None

        monkeypatch.setattr(engine, "_discovery_loop", quick)
        await engine.start_discovery()
        await asyncio.sleep(0.05)
        return engine.get_status()

    assert asyncio.run(scenario())["task_done"] is True
