from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.being.continuous_substrate import ContinuousSelfField
from core.being.functional_soul import FunctionalSoul
from core.being.introspection_renderer import IntrospectionRenderer, IntrospectionVerifier
from core.being.runtime import BeingRuntime, reset_being_runtime_for_test
from core.container import ServiceContainer
from core.runtime.consequence_bus import ConsequenceBus
from core.runtime.lesion_controller import LesionController
from core.state.aura_state import AuraState
from core.will import ActionDomain, UnifiedWill, WillOutcome


def teardown_function() -> None:
    reset_being_runtime_for_test()
    ServiceContainer.clear()
    ConsequenceBus.reset()
    LesionController.reset()


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


def test_introspection_renderer_uses_safe_fallback_when_calibration_rejects() -> None:
    class RejectingCalibrator:
        def calibrate(self, *_args, **_kwargs):
            return SimpleNamespace(
                calibrated=False,
                suggested_revision="claim requires trace evidence",
            )

    runtime = BeingRuntime()
    now = runtime.sample(AuraState.default(), objective="self-report status")
    renderer = IntrospectionRenderer()
    renderer._calibrator = RejectingCalibrator()

    rendered = renderer.render(now)

    assert "bounded functional telemetry" in rendered
    assert "claim requires trace evidence" in rendered
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


def test_being_runtime_registers_operational_lesion_targets() -> None:
    runtime = BeingRuntime()
    runtime.start(hz=10.0)
    try:
        controller = LesionController.get()
        targets = set(controller.all_targets())
        assert {
            "welfare",
            "body",
            "introspection",
            "self_report",
            "semantic_stream",
            "affect",
            "workspace",
        }.issubset(targets)

        state = AuraState.default()
        controller.lesion("affect")
        affect_lesioned = runtime.sample(state, objective="debug a blocked dependency")
        assert affect_lesioned.affect.dominant_drive == "lesioned_affect"

        controller.restore("affect")
        affect_restored = runtime.sample(state, objective="debug a blocked dependency")
        assert affect_restored.affect.dominant_drive != "lesioned_affect"

        controller.lesion("workspace")
        workspace_lesioned = runtime.sample(state, objective="research climate news")
        assert workspace_lesioned.workspace.broadcast_targets == ()
        assert workspace_lesioned.workspace.lesion == "workspace_ignition"
    finally:
        runtime.stop()


def test_body_cost_failure_defers_consequential_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = BeingRuntime()
    now = runtime.sample(AuraState.default(), objective="research with tools")
    spend_attempts: list[tuple[str, float]] = []

    def fail_spend(_domain: str, *, cost_multiplier: float = 1.0) -> dict[str, float]:
        spend_attempts.append((_domain, cost_multiplier))
        raise RuntimeError("body cost ledger unavailable")

    monkeypatch.setattr(runtime.body_service, "spend", fail_spend)

    policy = runtime.action_policy(now, domain="tool_execution", priority=0.8)

    assert policy["outcome"] == "defer"
    assert spend_attempts == [("tool_execution", 0.8)]
    assert "body_cost_accounting_failed" in policy["constraints"]
    assert "body_cost_accounting_required_before_action" in policy["defers"]
    assert policy["evidence"]["body_cost_applied"] == {}


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
    will = UnifiedWill()
    will.ensure_started()
    decision = will.decide(
        "record continuity after a verified repair",
        source="being_runtime_test",
        domain=ActionDomain.REFLECTION,
        context={"aura_state": AuraState.default()},
    )
    assert will.verify_receipt_signature(decision.receipt_id) is True

    soul = FunctionalSoul(receipt_verifier=will.verify_receipt_signature)
    with pytest.raises(PermissionError):
        soul.record_transition("identity update", receipt_id="will_" + "a" * 12)

    entry = soul.record_transition("kept promise", receipt_id=decision.receipt_id, metadata={"promise": "test"})

    assert entry.previous_hash == "genesis"
    assert soul.verify_chain() is True
    assert soul.influence_policy()["truth_priority"] > soul.influence_policy(lesioned=True)["truth_priority"]


def test_will_decision_signs_live_aura_now_evidence() -> None:
    will = UnifiedWill()
    will.ensure_started()

    decision = will.decide(
        "open a browser tab and research climate news before responding",
        source="desktop_task",
        domain=ActionDomain.TOOL_EXECUTION,
        priority=0.8,
        context={
            "aura_state": AuraState.default(),
            "user_requested_action": True,
            "foreground_request": True,
        },
    )
    material = will.get_receipt_verification_material(decision.receipt_id)

    assert decision.aura_now_hash
    assert decision.aura_now_tick > 0
    assert decision.aura_now_policy in {"proceed", "constrain", "defer", "refuse"}
    assert decision.aura_now_hash in material["payload"]
    assert will.verify_receipt_signature(decision.receipt_id) is True


def test_will_decision_carries_aura_now_welfare_evidence() -> None:
    will = UnifiedWill()
    will.ensure_started()

    decision = will.decide(
        "research with an external tool",
        source="desktop_task",
        domain=ActionDomain.TOOL_EXECUTION,
        priority=0.7,
        context={
            "aura_state": AuraState.default(),
            "user_requested_action": True,
            "foreground_request": True,
        },
    )

    assert "welfare_score" in decision.aura_now_evidence
    assert "welfare_truth_protection" in decision.aura_now_evidence
    assert "welfare_self_report_confidence" in decision.aura_now_evidence
    assert decision.welfare_score == pytest.approx(decision.aura_now_evidence["welfare_score"])
    assert decision.welfare_truth_protection == pytest.approx(
        decision.aura_now_evidence["welfare_truth_protection"]
    )
    assert decision.welfare_body_fatigue == pytest.approx(decision.aura_now_evidence["body_fatigue"])


def test_will_decision_publishes_pre_action_consequence_event() -> None:
    will = UnifiedWill()
    will.ensure_started()

    decision = will.decide(
        "answer a user status question",
        source="desktop_task",
        domain=ActionDomain.RESPONSE,
        priority=0.5,
        context={
            "aura_state": AuraState.default(),
            "user_requested_action": True,
            "foreground_request": True,
        },
    )

    events = ConsequenceBus.get().recent_events(1)

    assert events
    event = events[-1]
    assert event.will_receipt_id == decision.receipt_id
    assert event.domain == ActionDomain.RESPONSE.value
    assert event.actual_outcome == "authorized"
    assert event.predicted_welfare_delta["welfare_score"] == pytest.approx(decision.welfare_score)
    assert event.actual_body_cost == {}


def test_stopped_will_refuses_before_aura_now_sampling() -> None:
    calls: list[str] = []

    class Runtime:
        def sample(self, *_args, **_kwargs):  # pragma: no cover - must not be reached
            calls.append("sample")
            return SimpleNamespace(state_hash="unexpected_sample", tick=999)

        def action_policy(self, *_args, **_kwargs):  # pragma: no cover - must not be reached
            calls.append("action_policy")
            return {"outcome": "proceed", "constraints": [], "evidence": {}}

    ServiceContainer.register_instance("being_runtime", Runtime(), required=False)
    will = UnifiedWill()
    will.ensure_started()
    will._started = False

    decision = will.decide(
        "respond while Will is stopped",
        source="receipt_validator_negative_test",
        domain=ActionDomain.RESPONSE,
        priority=1.0,
    )

    assert decision.outcome == WillOutcome.REFUSE
    assert decision.reason == "unified_will_not_started"
    assert decision.aura_now_policy == "will_offline"
    assert will._started is False
    assert calls == []


def test_aura_now_policy_blocks_consequential_will_action() -> None:
    class Runtime:
        def sample(self, *_args, **_kwargs):
            return SimpleNamespace(state_hash="blocked_now_hash", tick=42)

        def action_policy(self, *_args, **_kwargs):
            return {
                "outcome": "refuse",
                "constraints": ["aura_now_ownership_low: agency=0.100"],
                "blocks": ["ownership_too_low_for_consequential_action"],
                "defers": [],
                "evidence": {
                    "state_hash": "blocked_now_hash",
                    "tick": 42,
                    "agency_confidence": 0.1,
                    "source": "test_runtime",
                },
            }

    ServiceContainer.register_instance("being_runtime", Runtime(), required=False)
    will = UnifiedWill()
    will.ensure_started()

    decision = will.decide(
        "write outside the current project",
        source="desktop_task",
        domain=ActionDomain.FILE_WRITE,
        priority=0.9,
        context={"user_requested_action": True},
    )

    assert decision.outcome == WillOutcome.REFUSE
    assert "aura_now_block" in decision.reason
    assert decision.aura_now_hash == "blocked_now_hash"
    assert "aura_now_ownership_low" in " ".join(decision.constraints)


def test_aura_now_constrains_low_risk_interaction_memory_instead_of_dropping_it() -> None:
    runtime = BeingRuntime()
    now = SimpleNamespace(
        body=SimpleNamespace(total_pressure=0.1),
        affect=SimpleNamespace(distress=0.1, dominant_drive="coherence"),
        prediction=SimpleNamespace(controllability=0.7, free_energy=0.1),
        workspace=SimpleNamespace(ignition_strength=0.05, broadcast_targets=(), winner="none"),
        ownership=SimpleNamespace(agency_confidence=0.8),
        state_hash="low_workspace_hash",
        tick=7,
    )

    continuity_policy = runtime.action_policy(
        now,
        domain="memory_write",
        priority=0.5,
        context={"conversation_continuity": True, "high_risk_memory_write": False},
    )
    generic_memory_policy = runtime.action_policy(now, domain="memory_write", priority=0.5)
    high_risk_policy = runtime.action_policy(
        now,
        domain="memory_write",
        priority=0.5,
        context={"conversation_continuity": True, "high_risk_memory_write": True},
    )

    assert continuity_policy["outcome"] == "constrain"
    assert "continuity_memory_write_constrained:not_deferred" in continuity_policy["constraints"]
    assert generic_memory_policy["outcome"] == "defer"
    assert high_risk_policy["outcome"] == "defer"


def test_introspection_verifier_rejects_unsupported_overclaim() -> None:
    runtime = BeingRuntime()
    now = runtime.sample(AuraState.default(), objective="simple status")
    check = IntrospectionRenderer().verifier.check("Phenomenal consciousness is proven and certain.", now)

    assert check.ok is False
    assert "forbidden_metaphysical_claim" in check.reasons
