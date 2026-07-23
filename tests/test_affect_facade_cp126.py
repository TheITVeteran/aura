"""CP126 contract tests for the affect facade.

The governing rule: an absent affect engine must never be presented as a calm
one. Numbers only exist when something measured them.
"""
from __future__ import annotations

import asyncio

import pytest

from core.affect import affect_facade as facade_module
from core.affect.affect_facade import (
    MODIFIER_AVAILABLE_KEY,
    MODIFIER_DEGRADED_KEY,
    AffectFacade,
)


@pytest.fixture(autouse=True)
def _no_registry(monkeypatch):
    """Default to 'nothing registered' so tests opt in to an engine."""
    registry: dict = {}
    monkeypatch.setattr(
        facade_module,
        "get_runtime_service",
        lambda name, default=None: registry.get(name, default),
    )
    return registry


class _Engine:
    def __init__(self, **overrides):
        self._overrides = overrides

    def get_status(self):
        return {"mood": "Curious", "valence": 0.4, "arousal": 0.5, **self._overrides}

    def get_context_injection(self):
        return "Mood: Curious | Valence: 0.40"

    def receive_qualia_echo(self, q_norm, pri, trend):
        return {"applied": True, "q": q_norm}

    async def react(self, trigger, context=None):
        return {"applied": True, "trigger": trigger}

    async def get_behavioral_modifiers(self):
        return {"creativity": 1.4, "patience": 0.8}


# --- 73b36873: unavailable must not be quantified --------------------------


def test_unavailable_status_invents_no_telemetry():
    status = AffectFacade().get_status()

    assert status["available"] is False
    assert status["status"] == "unavailable"
    assert status["reason"]
    for invented in ("curiosity", "stability", "energy", "mood", "valence", "arousal"):
        assert invented not in status


def test_live_status_is_forwarded_and_flagged(_no_registry):
    _no_registry["affect_engine"] = _Engine()

    status = AffectFacade().get_status()

    assert status["available"] is True
    assert status["mood"] == "Curious"
    assert status["source"] == "_Engine"


def test_raising_engine_status_reports_unavailable(_no_registry):
    class Boom:
        def get_status(self):
            raise RuntimeError("engine exploded")

    _no_registry["affect_engine"] = Boom()

    status = AffectFacade().get_status()

    assert status["available"] is False
    assert "RuntimeError" in status["reason"]


def test_non_mapping_status_is_not_passed_through(_no_registry):
    class Weird:
        def get_status(self):
            return "fine"

    _no_registry["affect_engine"] = Weird()

    assert AffectFacade().get_status()["available"] is False


# --- 422f28c5: no fabricated 72bpm body claim ------------------------------


def test_context_injection_is_empty_when_nothing_is_felt():
    injection = AffectFacade().get_context_injection()

    assert injection == ""
    assert "72bpm" not in injection


def test_context_injection_is_forwarded_when_live(_no_registry):
    _no_registry["affect_engine"] = _Engine()
    assert "Curious" in AffectFacade().get_context_injection()


def test_raising_context_injection_yields_no_prompt_text(_no_registry):
    class Boom:
        def get_context_injection(self):
            raise ValueError("nope")

    _no_registry["affect_engine"] = Boom()
    assert AffectFacade().get_context_injection() == ""


# --- e9f2af3b: dropped updates must be reported ----------------------------


def test_react_reports_that_it_did_not_apply():
    result = asyncio.run(AffectFacade().react("praise"))

    assert result["applied"] is False
    assert result["reason"]
    assert result["trigger"] == "praise"


def test_react_forwards_to_a_live_engine(_no_registry):
    _no_registry["affect_engine"] = _Engine()
    assert asyncio.run(AffectFacade().react("praise"))["applied"] is True


def test_qualia_echo_reports_that_it_did_not_apply():
    result = AffectFacade().receive_qualia_echo(0.5, 0.5, 0.0)
    assert result["applied"] is False
    assert result["reason"]


def test_qualia_echo_forwards_to_a_live_engine(_no_registry):
    _no_registry["affect_engine"] = _Engine()
    assert AffectFacade().receive_qualia_echo(0.5, 0.5, 0.0)["applied"] is True


def test_unavailability_is_recorded_as_a_degradation(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        facade_module,
        "record_degradation",
        lambda *args, **kwargs: recorded.append((args, kwargs)),
    )

    AffectFacade().receive_qualia_echo(0.5, 0.5, 0.0)

    assert recorded
    assert recorded[0][0][0] == "affect_facade"


def test_degradation_receipts_are_rate_limited(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        facade_module,
        "record_degradation",
        lambda *args, **kwargs: recorded.append(args),
    )
    accessor = AffectFacade()

    for _ in range(5):
        accessor.receive_qualia_echo(0.5, 0.5, 0.0)

    assert len(recorded) == 1


# --- 29db4cc1: the engine reference must not go stale ----------------------


def test_engine_is_re_resolved_after_registration(_no_registry):
    accessor = AffectFacade()
    assert accessor.get_status()["available"] is False

    _no_registry["affect_engine"] = _Engine()

    assert accessor.get_status()["available"] is True


def test_replaced_engine_is_picked_up(_no_registry):
    accessor = AffectFacade()
    _no_registry["affect_engine"] = _Engine(mood="First")
    assert accessor.get_status()["mood"] == "First"

    _no_registry["affect_engine"] = _Engine(mood="Second")

    assert accessor.get_status()["mood"] == "Second"


def test_deregistered_engine_stops_being_used(_no_registry):
    accessor = AffectFacade()
    _no_registry["affect_engine"] = _Engine()
    assert accessor.get_status()["available"] is True

    _no_registry.pop("affect_engine")

    assert accessor.get_status()["available"] is False


def test_a_retired_engine_is_treated_as_unavailable(_no_registry):
    engine = _Engine()
    engine.is_stopped = True
    _no_registry["affect_engine"] = engine

    assert AffectFacade().get_status()["available"] is False


def test_a_running_false_engine_is_treated_as_unavailable(_no_registry):
    engine = _Engine()
    engine.running = False
    _no_registry["affect_engine"] = engine

    assert AffectFacade().get_status()["available"] is False


def test_self_registration_does_not_recurse(_no_registry):
    accessor = AffectFacade()
    _no_registry["affect_engine"] = accessor

    assert accessor.get_status()["available"] is False
    assert accessor.is_ready() is False


# --- db6b6b69: readiness probes must be normalized -------------------------


def test_async_is_ready_is_not_reported_as_ready(_no_registry):
    class AsyncProbe(_Engine):
        async def is_ready(self):
            return True

    _no_registry["affect_engine"] = AsyncProbe()

    assert AffectFacade().is_ready() is False


def test_raising_is_ready_does_not_escape_the_health_probe(_no_registry):
    class Boom(_Engine):
        def is_ready(self):
            raise RuntimeError("probe exploded")

    _no_registry["affect_engine"] = Boom()

    assert AffectFacade().is_ready() is False


def test_sync_is_ready_is_honoured(_no_registry):
    class Ready(_Engine):
        def is_ready(self):
            return True

    _no_registry["affect_engine"] = Ready()

    assert AffectFacade().is_ready() is True


def test_status_derived_readiness_still_works(_no_registry):
    _no_registry["affect_engine"] = _Engine()
    assert AffectFacade().is_ready() is True


def test_status_derived_readiness_rejects_out_of_range_values(_no_registry):
    _no_registry["affect_engine"] = _Engine(valence=42.0)
    assert AffectFacade().is_ready() is False


# --- 79c270b6: neutral modifiers must carry provenance ---------------------


def test_neutral_modifiers_declare_that_affect_was_unavailable():
    modifiers = asyncio.run(AffectFacade().get_behavioral_modifiers())

    assert modifiers["creativity"] == 1.0
    assert modifiers[MODIFIER_AVAILABLE_KEY] == 0.0
    assert modifiers[MODIFIER_DEGRADED_KEY] == 1.0


def test_live_modifiers_declare_availability(_no_registry):
    _no_registry["affect_engine"] = _Engine()

    modifiers = asyncio.run(AffectFacade().get_behavioral_modifiers())

    assert modifiers["creativity"] == 1.4
    assert modifiers[MODIFIER_AVAILABLE_KEY] == 1.0
    assert modifiers[MODIFIER_DEGRADED_KEY] == 0.0


def test_raising_modifiers_degrade_with_provenance(_no_registry):
    class Boom(_Engine):
        async def get_behavioral_modifiers(self):
            raise RuntimeError("nope")

    _no_registry["affect_engine"] = Boom()

    modifiers = asyncio.run(AffectFacade().get_behavioral_modifiers())

    assert modifiers[MODIFIER_AVAILABLE_KEY] == 0.0
    assert modifiers["patience"] == 1.0


# --- get_state_sync wiring -------------------------------------------------


def test_state_sync_forwards_a_live_engine_reading(_no_registry):
    class Sync(_Engine):
        def get_state_sync(self):
            return {"valence": 0.7, "arousal": 0.2}

    _no_registry["affect_engine"] = Sync()

    state = AffectFacade().get_state_sync()

    assert state["valence"] == 0.7
    assert state["available"] is True


def test_state_sync_normalizes_a_dataclass_reading(_no_registry):
    class State:
        valence, arousal, engagement, dominant_emotion = 0.3, 0.4, 0.5, "calm"

    class Sync(_Engine):
        def get_state_sync(self):
            return State()

    _no_registry["affect_engine"] = Sync()

    state = AffectFacade().get_state_sync()

    assert state["valence"] == 0.3
    assert state["dominant_emotion"] == "calm"


def test_state_sync_is_explicitly_unavailable_without_an_engine():
    state = AffectFacade().get_state_sync()
    assert state["available"] is False
    assert "valence" not in state
