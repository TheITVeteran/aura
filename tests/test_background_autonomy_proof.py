from __future__ import annotations

from tools.closeout.background_autonomy_proof import (
    REQUIRED_COMPONENTS,
    evaluate_background_autonomy,
)


def _running_component(**extra):
    payload = {"running": True}
    payload.update(extra)
    return payload


def _healthy_payload():
    components = {name: _running_component() for name in REQUIRED_COMPONENTS}
    components["autonomy_conductor"] = _running_component(
        active=True,
        jobs={
            "metabolic_budget": {
                "policy": "constitutive",
                "last_status": "ok",
                "last_result": {"allocation": "bounded"},
            },
            "online_lora_status": {
                "policy": "constitutive",
                "last_status": "ok",
                "last_result": {"enabled": True},
            },
            "overt_action_cycle": {
                "policy": "delegated",
                "last_status": "ok",
                "last_result": {"status": "skipped", "error": "no_authorized_initiative"},
            },
            "internal_deliberation_cycle": {
                "policy": "research",
                "last_status": "deferred",
                "last_result": {"reason": "recent_user_4"},
            },
        },
    )
    components["autonomous_initiative"] = _running_component(
        core_tasks={
            "world": True,
            "knowledge": True,
            "self_development": True,
            "social": True,
            "mission": True,
            "frontier_discovery": True,
        },
    )
    return {
        "full_runtime_ready": True,
        "full_runtime": {
            "full_runtime_expected": True,
            "ready": True,
            "blockers": [],
            "background_cognition": {
                "enabled": True,
                "active": True,
                "loops_allowed": True,
                "work_admission": "allowed",
                "work_defer_reason": "",
            },
            "components": components,
        },
    }


def test_background_autonomy_evaluator_accepts_full_live_runtime_shape():
    report = evaluate_background_autonomy(_healthy_payload())

    assert report["passed"] is True
    assert report["running_component_count"] == len(REQUIRED_COMPONENTS)
    assert report["pass_conditions"]["has_delegated_overt_action"] is True
    assert report["pass_conditions"]["has_deliberation_job"] is True
    assert report["pass_conditions"]["initiative_core_tasks_alive"] is True


def test_background_autonomy_evaluator_rejects_disabled_background():
    payload = _healthy_payload()
    payload["full_runtime"]["background_cognition"]["enabled"] = False
    payload["full_runtime"]["background_cognition"]["active"] = False

    report = evaluate_background_autonomy(payload)

    assert report["passed"] is False
    assert report["pass_conditions"]["background_enabled"] is False
    assert report["pass_conditions"]["background_active"] is False


def test_background_autonomy_evaluator_rejects_forbidden_defer_reasons():
    payload = _healthy_payload()
    payload["full_runtime"]["components"]["autonomy_conductor"]["jobs"]["metabolic_budget"] = {
        "policy": "constitutive",
        "last_status": "deferred",
        "last_result": {"reason": "background_cognition_disabled"},
    }

    report = evaluate_background_autonomy(payload)

    assert report["passed"] is False
    assert report["bad_deferred_reasons"] == {
        "metabolic_budget": "background_cognition_disabled"
    }
