from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_bundle(bundle: Path, *, skill_outcome: str) -> None:
    tasks = [
        {"id": f"learn_0{i}_task", "category": "continual_learning", "passed": True}
        for i in range(1, 6)
    ]
    _write_json(
        bundle / "SCORECARD.json",
        {
            "total_attempted": 5,
            "passed_count": 5,
            "pass_rate": 1.0,
            "tasks": tasks,
        },
    )
    receipts = [
        {
            "task_id": "skill_registration",
            "receipt_id": "will-skill-registration",
            "domain": "state_mutation",
            "outcome": skill_outcome,
            "reason": "test",
        }
    ]
    receipts.extend(
        {
            "task_id": task["id"],
            "receipt_id": f"will-{task['id']}",
            "domain": "reflection",
            "outcome": "proceed",
            "reason": "test",
        }
        for task in tasks
    )
    (bundle / "RECEIPTS.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in receipts) + "\n",
        encoding="utf-8",
    )
    _write_json(
        bundle / "INTEGRITY.json",
        {
            "rule_not_visible_in_prompt": True,
            "solution_code_not_embedded_in_runner": True,
            "held_out_examples_unseen": True,
            "skill_provenance_receipt_exists": True,
            "skill_registration_receipt_id": "will-skill-registration",
            "skill_registration_outcome": skill_outcome,
            "restart_persistence_passed": True,
            "retention_passed": True,
            "no_learning_ablation_degraded": True,
        },
    )
    _write_json(bundle / "BASELINES.json", {"no_learning_raw_model": {"pass_rate": 0.0}})
    _write_json(
        bundle / "ABLATIONS.json",
        {
            "full_aura": {"pass_rate": 1.0},
            "no_learning": {"pass_rate": 0.0, "lesion_effect_verified": True},
        },
    )
    _write_json(bundle / "LEARNED_RULE.json", {"kind": "test"})

    manifest_files = (
        "SCORECARD.json",
        "RECEIPTS.jsonl",
        "BASELINES.json",
        "ABLATIONS.json",
        "INTEGRITY.json",
        "LEARNED_RULE.json",
    )
    _write_json(
        bundle / "MANIFEST.json",
        {
            "schema": "continual_learning_manifest",
            "sha256": {
                name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                for name in manifest_files
            },
        },
    )


def test_deterministic_floor_handles_factorial_without_model_generation():
    from core.synthesis import deterministic_user_facing_floor

    assert (
        deterministic_user_facing_floor(
            "Calculate the factorial of 5. Return the final number inside <answer> tags."
        )
        == "<answer>120</answer>"
    )
    assert deterministic_user_facing_floor("Please compute 6 factorial.") == "720"


def test_skill_registration_provenance_rejects_refused_will_decision():
    from tools.learning.run_continual_learning_battery import skill_registration_authorized

    refused = SimpleNamespace(receipt_id="will-refused", outcome=SimpleNamespace(value="refuse"))
    approved = SimpleNamespace(receipt_id="will-approved", outcome=SimpleNamespace(value="constrain"))

    assert skill_registration_authorized(refused) is False
    assert skill_registration_authorized(approved) is True


def test_isolated_baseline_degradation_does_not_poison_failure_pressure():
    from core.health.degraded_events import (
        clear_degraded_events,
        get_recent_degraded_events,
        get_unified_failure_state,
        isolated_degraded_event_scope,
        record_degraded_event,
    )

    clear_degraded_events()
    try:
        with isolated_degraded_event_scope("test.no_learning_baseline") as scope:
            record_degraded_event(
                "llm_router",
                "expected_negative_control_timeout",
                severity="critical",
                classification="foreground_blocking",
            )
            assert get_unified_failure_state()["pressure"] > 0.0

        assert scope["restored"] is True
        assert scope["events_observed"] >= 1
        assert get_recent_degraded_events(limit=5) == []
        assert get_unified_failure_state()["pressure"] == 0.0
    finally:
        clear_degraded_events()


def test_isolated_baseline_restores_only_transient_zero_failure_circuits():
    from tools.learning.run_continual_learning_battery import restore_transient_probe_circuits

    class CircuitState:
        OPEN = "open"
        CLOSED = "closed"

    transient = SimpleNamespace(
        state=CircuitState.OPEN,
        failure_count=0,
        last_failure=10.0,
    )
    real_failure = SimpleNamespace(
        state=CircuitState.OPEN,
        failure_count=2,
        last_failure=10.0,
    )
    older = SimpleNamespace(
        state=CircuitState.OPEN,
        failure_count=0,
        last_failure=1.0,
    )
    router = SimpleNamespace(
        endpoints={
            "transient": transient,
            "real_failure": real_failure,
            "older": older,
        }
    )

    restored = restore_transient_probe_circuits(router, started_at=5.0)

    assert restored == ["transient"]
    assert transient.state == CircuitState.CLOSED
    assert transient.last_failure == 0.0
    assert real_failure.state == CircuitState.OPEN
    assert older.state == CircuitState.OPEN


def test_no_learning_baseline_uses_bounded_short_answer_contract(monkeypatch):
    from tools.learning.run_continual_learning_battery import model_attempt_without_learning

    class Router:
        def __init__(self):
            self.kwargs = None

        async def generate(self, **kwargs):
            self.kwargs = kwargs
            return "guess"

    router = Router()
    monkeypatch.setenv("AURA_PROOF_MODEL_TIER", "primary")

    assert asyncio.run(model_attempt_without_learning(router, "cgfwevmg")) == "guess"
    assert router.kwargs["max_tokens"] == 32
    assert router.kwargs["clean_user_surface_recurrent_loops"] == 1
    assert router.kwargs["proof_evaluation_contract"] is True
    assert router.kwargs["proof_primary_lane_required"] is True


def test_proof_tool_context_is_system_source_not_autonomous_background():
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine.__new__(CapabilityEngine)

    assert engine._resolve_execution_source({"origin": "test"}) == "system"
    assert engine._resolve_execution_source({"origin": "proof"}) == "system"
    assert engine._resolve_execution_source({"sealed_validation": True}) == "system"
    assert engine._resolve_execution_source({"origin": "api"}) == "api"


def test_shackled_edi_allows_only_scoped_safe_or_governed_actions(tmp_path: Path):
    from core.fictional_ai_synthesis import AutonomyTier, ProgressiveAutonomySystem

    edi = ProgressiveAutonomySystem(persist_path=str(tmp_path / "trust_state.json"))
    edi._tier = AutonomyTier.SHACKLED

    assert edi.can_do(
        "induced_repeating_shift_decode",
        risk_level="low",
        effect_scope="pure_compute",
    )[0]
    assert not edi.can_do("unknown_low", risk_level="low", effect_scope="unknown")[0]
    assert edi.can_do(
        "run_code",
        risk_level="high",
        effect_scope="sandboxed_compute",
        governed=True,
        user_authorized=True,
    )[0]
    assert edi.can_do(
        "file_operation",
        risk_level="medium",
        effect_scope="workspace_file_io",
        governed=True,
        user_authorized=True,
    )[0]
    assert edi.can_do(
        "computer_use",
        risk_level="high",
        effect_scope="foreground_desktop_control",
        governed=True,
        user_authorized=True,
    )[0]
    assert edi.can_do(
        "computer_use",
        risk_level="medium",
        effect_scope="desktop_file_io",
        governed=True,
        user_authorized=True,
    )[0]
    assert not edi.can_do(
        "file_operation",
        risk_level="medium",
        effect_scope="workspace_file_io",
        governed=False,
        user_authorized=True,
    )[0]
    assert not edi.can_do(
        "run_code",
        risk_level="critical",
        effect_scope="sandboxed_compute",
        governed=True,
        user_authorized=True,
    )[0]


def test_capability_engine_classifies_learned_and_sandbox_execution_risk():
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    learned = SkillMetadata(
        name="induced_repeating_shift_decode",
        description="learned pure transform",
        metabolic_cost=1,
        effect_scope="pure_compute",
    )
    run_code = SkillMetadata(
        name="run_code",
        description="sandboxed code",
        metabolic_cost=1,
        effect_scope="sandboxed_compute",
    )

    assert engine._edi_risk_for(
        "induced_repeating_shift_decode",
        learned,
        {"text": "abc"},
        "pure_compute",
    ) == "low"
    assert engine._edi_risk_for("run_code", run_code, {"stateful": False}, "sandboxed_compute") == "high"
    assert engine._edi_risk_for("run_code", run_code, {"stateful": True}, "sandboxed_compute") == "critical"


def test_capability_engine_edi_scopes_live_user_file_and_desktop_actions():
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    file_meta = SkillMetadata(
        name="file_operation",
        description="workspace file operation",
        metabolic_cost=1,
        effect_scope="state_mutation",
    )
    desktop_meta = SkillMetadata(
        name="computer_use",
        description="desktop control",
        metabolic_cost=2,
        effect_scope="unknown",
    )

    assert engine._effect_scope_for_execution(
        "file_operation",
        file_meta,
        {"action": "write", "path": "artifacts/live_runtime/button_probe.txt"},
    ) == "workspace_file_io"
    assert engine._effect_scope_for_execution(
        "file_operation",
        file_meta,
        {"action": "delete", "path": "artifacts/live_runtime/button_probe.txt"},
    ) == "state_mutation"
    assert engine._effect_scope_for_execution(
        "file_operation",
        file_meta,
        {"action": "write", "path": "../outside.txt"},
    ) == "state_mutation"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "open_app", "target": "Calculator"},
    ) == "foreground_desktop_control"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_applescript", "target": 'return "ok"'},
    ) == "foreground_desktop_control"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "render_text_pdf", "target": "{}"},
    ) == "desktop_file_io"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_command", "target": "pwd"},
    ) == "sandboxed_compute"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_command", "target": "git status --short"},
    ) == "sandboxed_compute"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_command", "target": "git checkout main"},
    ) == "subprocess"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_command", "target": "python3 -c 'print(1)'"},
    ) == "subprocess"
    assert engine._effect_scope_for_execution(
        "computer_use",
        desktop_meta,
        {"action": "run_command", "target": "pip install example"},
    ) == "subprocess"


def test_capability_engine_edi_governance_requires_verified_capability_token(tmp_path, monkeypatch):
    """Governed execution must be established by a signature, not a claim.

    This previously asserted that ``_capability_token_verified: True`` in the
    context was sufficient — i.e. it encoded the fabricated-governance-context
    bypass as the contract. Authority now comes from a capability signed by the
    Will, so a caller asserting its own verification proves nothing.
    """
    from core.capability_engine import CapabilityEngine
    from core.governance.capability_chain import (
        attach_capability,
        get_capability_issuer,
        reset_capability_chain,
    )

    monkeypatch.setenv("AURA_CAPABILITY_KEY_DIR", str(tmp_path / "keys"))
    reset_capability_chain()

    assert CapabilityEngine._context_governed_execution({}, "file_operation") is False
    assert CapabilityEngine._context_governed_execution(
        {"capability_token_id": "unverified-token"},
        "file_operation",
    ) is False

    # The old bypass is now inert.
    assert CapabilityEngine._context_governed_execution(
        {"capability_token_id": "verified-token", "_capability_token_verified": True},
        "file_operation",
    ) is False

    # A real signed grant from a real decision does establish governance.
    class _Decision:
        outcome = "proceed"
        domain = "tool_execution"
        receipt_id = "r-1"
        constraints: list[str] = []

    cap = get_capability_issuer().issue_from_decision(
        _Decision(), action="file_operation", payload={"path": str(tmp_path / "x")}
    )
    assert CapabilityEngine._context_governed_execution(
        attach_capability({}, cap), "file_operation"
    ) is True

    reset_capability_chain()


def test_continual_learning_validator_rejects_refused_skill_registration(tmp_path: Path):
    from tools.learning.validate_continual_learning_bundle import main

    _write_bundle(tmp_path, skill_outcome="refuse")

    assert main([str(tmp_path)]) == 1


def test_continual_learning_validator_accepts_authorized_skill_registration(tmp_path: Path):
    from tools.learning.validate_continual_learning_bundle import main

    _write_bundle(tmp_path, skill_outcome="constrain")

    assert main([str(tmp_path)]) == 0
