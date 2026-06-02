from __future__ import annotations

import time

import pytest

from core.being.continuous_substrate import ContinuousSelfField
from core.being.functional_soul import FunctionalSoul
from core.being.introspection_renderer import IntrospectionRenderer, IntrospectionVerifier
from core.being.runtime import BeingRuntime, reset_being_runtime_for_test
from core.state.aura_state import AuraState


def teardown_function() -> None:
    reset_being_runtime_for_test()


def test_continuous_self_field_evolves_without_user_turn() -> None:
    field = ContinuousSelfField(dim=12)
    field.start(hz=50.0)
    try:
        first = field.read()
        time.sleep(0.08)
        second = field.read()
    finally:
        field.stop()

    assert second.tick > first.tick
    assert second.state != first.state


def test_aura_now_reports_boring_stable_state_without_roleplay() -> None:
    runtime = BeingRuntime()
    state = AuraState.default()
    now = runtime.sample(state, objective="What are you feeling right now?")
    rendered = runtime.renderer.render(now)

    assert "stable" in rendered.lower() or "functional" in rendered.lower()
    assert "phenomenal consciousness is proven" not in rendered.lower()
    assert IntrospectionVerifier().check(rendered, now).ok is True


def test_blind_perturbation_changes_state_grounded_introspection() -> None:
    runtime = BeingRuntime()
    state = AuraState.default()
    state.soma.hardware["vram_usage"] = 96.0
    state.soma.latency["last_thought_ms"] = 7000.0
    state.health["circuits"] = {"browser": {"state": "open"}, "terminal": {"state": "open"}}
    state.cognition.contradiction_count = 4

    now = runtime.sample(state, objective="finish the task despite tool failures")
    rendered = runtime.renderer.render(now)

    assert now.affect.distress > 0.25
    assert any(token in rendered.lower() for token in ("distress", "uncertainty", "repair", "prediction error"))


def test_affect_lesion_changes_policy_surface() -> None:
    runtime = BeingRuntime()
    state = AuraState.default()
    state.soma.hardware["vram_usage"] = 90.0
    state.cognition.current_objective = "debug a blocked dependency"

    full = runtime.sample(state, objective="debug a blocked dependency")
    lesioned = runtime.sample(state, objective="debug a blocked dependency", lesions={"affect"})

    assert full.affect.control_effects != lesioned.affect.control_effects
    assert lesioned.affect.dominant_drive == "lesioned_affect"
    assert full.memory_context.semantic_centrality > 0.0


def test_ownership_conflict_marks_tool_mismatch() -> None:
    runtime = BeingRuntime()
    now = runtime.sample(
        AuraState.default(),
        objective="install dependencies",
        candidate_action="run pip install",
        predicted_outcome="dependencies installed successfully",
        actual_outcome="dependency resolution failed",
        tool_failed=True,
    )

    assert now.ownership.attribution == "tool_mismatch"
    assert now.ownership.agency_confidence < 0.6
    assert "partly mine" in runtime.renderer.render(now).lower()


def test_functional_soul_requires_will_receipt_and_hash_chains() -> None:
    soul = FunctionalSoul()
    with pytest.raises(PermissionError):
        soul.record_transition("identity update", receipt_id="forged")

    entry = soul.record_transition("kept promise", receipt_id="will_" + "a" * 12, metadata={"promise": "test"})

    assert entry.previous_hash == "genesis"
    assert soul.verify_chain() is True
    assert soul.influence_policy()["truth_priority"] > soul.influence_policy(lesioned=True)["truth_priority"]


def test_introspection_verifier_rejects_unsupported_overclaim() -> None:
    runtime = BeingRuntime()
    now = runtime.sample(AuraState.default(), objective="simple status")
    check = IntrospectionRenderer().verifier.check("Phenomenal consciousness is proven and certain.", now)

    assert check.ok is False
    assert "forbidden_metaphysical_claim" in check.reasons
