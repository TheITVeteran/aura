from __future__ import annotations

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


def test_proof_tool_context_is_system_source_not_autonomous_background():
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine.__new__(CapabilityEngine)

    assert engine._resolve_execution_source({"origin": "test"}) == "system"
    assert engine._resolve_execution_source({"origin": "proof"}) == "system"
    assert engine._resolve_execution_source({"sealed_validation": True}) == "system"
    assert engine._resolve_execution_source({"origin": "api"}) == "api"


def test_continual_learning_validator_rejects_refused_skill_registration(tmp_path: Path):
    from tools.learning.validate_continual_learning_bundle import main

    _write_bundle(tmp_path, skill_outcome="refuse")

    assert main([str(tmp_path)]) == 1


def test_continual_learning_validator_accepts_authorized_skill_registration(tmp_path: Path):
    from tools.learning.validate_continual_learning_bundle import main

    _write_bundle(tmp_path, skill_outcome="constrain")

    assert main([str(tmp_path)]) == 0
