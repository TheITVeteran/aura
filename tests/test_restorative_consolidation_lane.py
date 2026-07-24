"""Sleep-class restoration must survive high welfare recovery drive.

The live 7/17-7/21 sessions measured a mind-level deadlock: whenever
``welfare.recovery_drive`` exceeded 0.6 the AuraNow policy deferred every
consequential action (11,738 defers), INCLUDING dream consolidation
(2,428 blocks). Consolidation is the mechanism that lowers recovery
drive, so the interior life froze for whole sessions — no memory writes,
no belief updates, no initiative. These tests pin the narrow exemption:
allowlisted restorative sources may consolidate under high recovery
drive; nothing else gets through, and no self-labeling works.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.being.runtime import BeingRuntime
from core.state.aura_state import AuraState


def _high_recovery_welfare() -> SimpleNamespace:
    return SimpleNamespace(
        action_inhibition=0.0,
        integrity_guard=0.0,
        recovery_drive=0.9,
        self_report_confidence=0.9,
        welfare_score=0.4,
        distress=0.3,
        truth_protection=0.5,
        should_protect_integrity=lambda: False,
        should_verify_before_claiming=lambda: False,
    )


def _policy(runtime: BeingRuntime, *, domain: str, context: dict | None):
    now = runtime.sample(AuraState.default(), objective="idle consolidation")
    runtime._last_welfare = _high_recovery_welfare()
    return runtime.action_policy(now, domain=domain, priority=0.55, context=context)


RESTORATIVE_CONTEXT = {
    "source": "mind_tick.dream_consolidation",
    "effect_scope": "internal_restoration",
    "no_external_effects": True,
}


def test_generic_state_mutation_still_defers_under_recovery_drive():
    policy = _policy(BeingRuntime(), domain="state_mutation", context=None)
    assert "welfare_recovery_required_before_action" in policy["defers"]


def test_allowlisted_consolidation_is_exempt_from_recovery_defer():
    policy = _policy(
        BeingRuntime(), domain="state_mutation", context=dict(RESTORATIVE_CONTEXT)
    )
    assert "welfare_recovery_required_before_action" not in policy["defers"], (
        "deferring consolidation because recovery is needed is the deadlock: "
        "recovery drive can only fall if restoration runs"
    )
    assert any(
        c.startswith("restorative_consolidation_lane") for c in policy["constraints"]
    )


def test_self_labeled_source_cannot_claim_the_lane():
    context = dict(RESTORATIVE_CONTEXT, source="rogue.subsystem")
    policy = _policy(BeingRuntime(), domain="state_mutation", context=context)
    assert "welfare_recovery_required_before_action" in policy["defers"]


def test_external_effect_markers_void_the_lane():
    context = dict(RESTORATIVE_CONTEXT, network_call=True)
    policy = _policy(BeingRuntime(), domain="state_mutation", context=context)
    assert "welfare_recovery_required_before_action" in policy["defers"]


def test_lane_does_not_bypass_the_integrity_guard():
    runtime = BeingRuntime()
    now = runtime.sample(AuraState.default(), objective="idle consolidation")
    welfare = _high_recovery_welfare()
    welfare.integrity_guard = 0.95
    welfare.should_protect_integrity = lambda: True
    runtime._last_welfare = welfare
    policy = runtime.action_policy(
        now,
        domain="memory_write",
        priority=0.55,
        context=dict(RESTORATIVE_CONTEXT, source="memory.consolidate_working_memory"),
    )
    assert "welfare_integrity_protection_active" in policy["defers"], (
        "the restorative lane exempts only the recovery-drive rule; the "
        "integrity guard must still hold"
    )
