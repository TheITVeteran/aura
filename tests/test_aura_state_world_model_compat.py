from __future__ import annotations

from types import SimpleNamespace

from core.morality import deception_guard as deception_guard_module
from core.morality.deception_guard import DeceptionGuard, _runtime_world_facts
from core.state.aura_state import AuraState, WorldModel


def test_aura_state_world_model_is_mutable_canonical_fact_view():
    state = AuraState.default()

    state.world_model["sensor_blackout"] = True

    assert state.world.facts["sensor_blackout"] is True
    assert _runtime_world_facts(state)["sensor_blackout"] is True


def test_aura_state_world_model_setter_preserves_structured_world():
    state = AuraState.default()
    world = WorldModel()
    world.known_entities["Bryan"] = {"role": "operator"}

    state.world_model = world
    state.world_model["last_verification"] = {"channel": "test"}

    assert state.world.known_entities["Bryan"]["role"] == "operator"
    assert state.world.facts["last_verification"]["channel"] == "test"


def test_deception_guard_uses_aura_state_world_model_without_degradation(monkeypatch):
    state = AuraState.default()
    state.world_model["sensor_blackout"] = True
    repository = SimpleNamespace(_current=state)
    degradations = []

    monkeypatch.setattr(
        "core.runtime.service_access.resolve_state_repository",
        lambda default=None: repository,
    )
    monkeypatch.setattr(
        deception_guard_module,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    reply = DeceptionGuard().filter_text_claims("I see the menu bar clearly.")

    assert "sensors are offline" in reply.lower()
    assert degradations == []
