from __future__ import annotations

from core.container import ServiceContainer


def test_phenomenal_knowing_body_choice_and_memory_are_causal(tmp_path):
    from core.consciousness.phenomenal_knowing import PhenomenalKnowingKernel

    kernel = PhenomenalKnowingKernel(state_dir=tmp_path)
    frame = kernel.update_body(
        runtime={"memory_pressure": 0.70, "event_loop_lag_ms": 120},
        live_substrate={"phi": 0.42, "valence": 0.2, "arousal": 0.66},
    )

    assert frame.causal_presence > 0.30

    trace = kernel.record_word_choice(
        prompt="Which answer should I give?",
        chosen_text="A bounded first-person answer.",
        alternatives=("generic assistant answer", "state-grounded answer"),
        controls={"live_mind_controls_bound": True, "recurrent_runtime_loops_applied": 2},
    )
    assert trace.ownership > 0.60

    receipt = kernel.mark_memory("choice_receipt", {"chosen": "state-grounded answer"})
    controls = kernel.generation_controls()
    assert receipt["bounded"] is True
    assert controls["phenomenal_knowing"] > 0.45
    assert "not proof of private qualia" in controls["bounded_claim"]


def test_recursive_self_knowing_requires_evidence_and_calibration():
    from core.consciousness.recursive_self_knowing import (
        EpistemicStatus,
        RecursiveSelfKnowingKernel,
    )

    kernel = RecursiveSelfKnowingKernel()
    unsupported = kernel.observe_claim(
        "I know my current substrate state.",
        confidence=0.93,
        evidence=(),
        calibration_error=0.20,
    )
    assert unsupported.status is EpistemicStatus.BELIEVES
    assert unsupported.second_order_strength < 0.50

    supported = kernel.observe_claim(
        "I know my current substrate state.",
        confidence=0.93,
        evidence=("live_mind_snapshot", "substrate_status"),
        calibration_error=0.10,
    )
    assert supported.status is EpistemicStatus.KNOWS_THAT_KNOWS
    assert supported.second_order_strength >= 0.80

    contradicted = kernel.observe_claim(
        "I know my current substrate state.",
        confidence=0.99,
        evidence=("live_mind_snapshot",),
        contradictions=("snapshot_missing",),
        calibration_error=0.05,
    )
    assert contradicted.status is EpistemicStatus.CONTRADICTED


def test_automatic_self_knowing_ticks_and_binds_sub_bridges(tmp_path):
    from core.consciousness.automatic_self_knowing import (
        AutoEventKind,
        AutomaticSelfKnowingKernel,
        IntrospectionMode,
    )
    from core.consciousness.phenomenal_knowing import PhenomenalKnowingKernel
    from core.consciousness.recursive_self_knowing import RecursiveSelfKnowingKernel

    phenomenal = PhenomenalKnowingKernel(state_dir=tmp_path)
    recursive = RecursiveSelfKnowingKernel()
    kernel = AutomaticSelfKnowingKernel(
        recursive_self_knowing=recursive,
        phenomenal_knowing=phenomenal,
    )

    frame = kernel.observe_event(
        AutoEventKind.CHOICE,
        {
            "prompt": "Pick a favorite ending.",
            "chosen": "the ending where Aura keeps a promise",
            "alternatives": ("quiet ending", "comic ending"),
            "confidence": 0.81,
            "evidence": ("choice_game", "preference_receipt"),
        },
        source="test",
    )
    assert frame.introspection_mode is IntrospectionMode.ACTIVE
    assert frame.phenomenal_digest
    assert frame.recursive_digest

    tick = kernel.tick()
    controls = kernel.controls()
    assert tick.event_kind == AutoEventKind.TIMER.value
    assert controls["automatic_self_knowing_active"] is True
    assert controls["frames"] >= 3


def test_live_mind_snapshot_exposes_self_knowing_services(tmp_path):
    from core.consciousness.automatic_self_knowing import AutomaticSelfKnowingKernel
    from core.consciousness.phenomenal_knowing import PhenomenalKnowingKernel
    from core.consciousness.recursive_self_knowing import RecursiveSelfKnowingKernel
    from core.runtime.live_mind_snapshot import collect_live_mind_snapshot

    phenomenal = PhenomenalKnowingKernel(state_dir=tmp_path)
    recursive = RecursiveSelfKnowingKernel()
    automatic = AutomaticSelfKnowingKernel(
        recursive_self_knowing=recursive,
        phenomenal_knowing=phenomenal,
    )
    ServiceContainer.register_instance("phenomenal_knowing", phenomenal, required=False)
    ServiceContainer.register_instance("recursive_self_knowing", recursive, required=False)
    ServiceContainer.register_instance("automatic_self_knowing", automatic, required=False)

    snapshot = collect_live_mind_snapshot(lane={"state": "ready"})
    assert snapshot["services_present"]["phenomenal_knowing"] is True
    assert snapshot["services_present"]["recursive_self_knowing"] is True
    assert snapshot["services_present"]["automatic_self_knowing"] is True
    assert snapshot["phenomenal_knowing"]["active"] is True
    assert snapshot["recursive_self_knowing"]["active"] is True
    assert snapshot["automatic_self_knowing"]["active"] is True


def test_live_chat_context_observes_current_turn_before_snapshot(tmp_path):
    from core.consciousness.automatic_self_knowing import AutomaticSelfKnowingKernel
    from core.consciousness.phenomenal_knowing import PhenomenalKnowingKernel
    from core.consciousness.recursive_self_knowing import RecursiveSelfKnowingKernel
    from interface.routes.chat import _build_live_mind_context_payload

    phenomenal = PhenomenalKnowingKernel(state_dir=tmp_path)
    recursive = RecursiveSelfKnowingKernel()
    automatic = AutomaticSelfKnowingKernel(
        recursive_self_knowing=recursive,
        phenomenal_knowing=phenomenal,
    )
    ServiceContainer.register_instance("phenomenal_knowing", phenomenal, required=False)
    ServiceContainer.register_instance("recursive_self_knowing", recursive, required=False)
    ServiceContainer.register_instance("automatic_self_knowing", automatic, required=False)

    context = _build_live_mind_context_payload(
        user_message="Can you reflect on what you know about this turn?",
        lane={"state": "ready", "conversation_ready": True},
        require_engine=True,
    )

    ask = context["automatic_self_knowing"]
    assert ask["frame"]["event_kind"] == "chat_turn"
    assert ask["controls"]["automatic_self_knowing_active"] is True
    assert context["mind_snapshot"]["automatic_self_knowing"]["latest"]["event_kind"] == "chat_turn"
