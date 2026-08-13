from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex import frontier_tasks as ft
from tools import verify_rlc_reconciliation_campaign as verifier


def _result(task, response):
    correct = str(response).startswith("RIGHT")
    return ft.ScoreResult(
        schema=ft.SCORE_RESULT_SCHEMA,
        task_id=task.task_id,
        domain=task.domain,
        scorer_id=task.public.scorer_id,
        parsed=True,
        correct=correct,
        reason="correct" if correct else "incorrect_or_schema_mismatch",
        normalized_answer_sha256=hashlib.sha256(str(response).encode()).hexdigest(),
    )


def _write_complete_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seed = 909
    domains = tuple(ft.FRONTIER_DOMAINS)
    tasks = tuple(
        ft.generate_task_battery(
            [seed],
            domains=domains,
            difficulty=2,
            registry_version=ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
        )
    )
    commitment = ft.build_task_commitment(ft.build_task_manifest(tasks))
    config_path = tmp_path / "controller.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = {
        "campaign_id": "campaign",
        "out_dir": str(tmp_path),
        "source_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "campaign_stage": "component",
        "seed": seed,
        "per_domain": 1,
        "difficulty": 2,
        "domains": list(domains),
        "task_registry_version": ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
        "episode_wall_s": 1800.0,
        "model": "/model",
        "n_slots": 16,
        "max_tokens": 2048,
        "integrated_recurrent_max_tokens": 2048,
        "integrated_recurrent_package": {
            "package_id": "package",
            "manifest_sha256": "9" * 64,
            "controller_sha256": "8" * 64,
            "activation_sha256": "7" * 64,
        },
    }
    monkeypatch.setattr(
        verifier,
        "_verify_controller",
        lambda _path: {
            "config": config,
            "source_root": tmp_path,
            "source_manifest_sha256": "c" * 64,
            "model_manifest_sha256": "d" * 64,
            "package_files_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(verifier, "_verify_implementation_files", lambda *_args: "f" * 64)
    monkeypatch.setattr(verifier, "_reconstruct_runtime_evidence", lambda *_args: "1" * 64)
    monkeypatch.setattr(ft, "score_task", _result)
    fingerprint = {
        "schema": verifier.FINGERPRINT_SCHEMA,
        "requested_arms": list(verifier.EXPECTED_REQUESTED_ARMS),
        "required_arms": list(verifier.EXPECTED_REQUIRED_ARMS),
        "campaign_stage": "component",
        "resource_dominating_target_arm": verifier.TREATMENT_ARM,
        "implementation_files": {"x": "2" * 64},
        "implementation_sha256": "f" * 64,
        "task_commitment_sha256": commitment.commitment_sha256,
        "expected_task_ids": [task.task_id for task in tasks],
        "domains": list(domains),
        "difficulty": 2,
        "task_registry_version": ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
        "completion_budget_policy": "semantic_completion_floor.v1",
        "fast_weight_site": {"target": "o_proj", "layer_placement": "early"},
        "output_memory_diagnostic": False,
        "integrated_recurrent_package": {
            "package_id": "package",
            "manifest_sha256": "9" * 64,
            "controller_sha256": "8" * 64,
            "activation_sha256": "7" * 64,
        },
        "integrated_recurrent_max_tokens": 2048,
        "arm_max_tokens": {arm: 2048 for arm in verifier.EXPECTED_REQUIRED_ARMS},
        "task_max_tokens": {
            arm: {task.task_id: 2048 for task in tasks}
            for arm in verifier.EXPECTED_REQUIRED_ARMS
        },
    }
    fingerprints = {
        arm: verifier._decode_fingerprint(
            config=config,
            fingerprint=fingerprint,
            arm=arm,
            max_tokens=2048,
            implementation_sha256="f" * 64,
        )
        for arm in verifier.EXPECTED_REQUIRED_ARMS
    }
    fingerprint["decode_fingerprint"] = fingerprints
    (tmp_path / "decode_fingerprint.json").write_text(
        json.dumps(fingerprint),
        encoding="utf-8",
    )
    (tmp_path / "task_commitment.json").write_text(
        json.dumps(
            {
                "schema": "aura.rlc_reconciliation_sweep.v1",
                "seed": seed,
                "per_domain": 1,
                "difficulty": 2,
                "registry_version": ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
                "domains": list(domains),
                "task_count": len(tasks),
                "commitment_sha256": commitment.commitment_sha256,
            }
        ),
        encoding="utf-8",
    )
    records = []
    for task_index, task in enumerate(tasks):
        for arm in verifier.EXPECTED_REQUIRED_ARMS:
            correct = arm in {
                verifier.TREATMENT_ARM,
                "complete_system_recurrent_depth_lesion",
            }
            if arm == "complete_system_recurrent_depth_lesion" and task_index == 0:
                correct = False
            records.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "arm_profile": (
                        "ordinary" if arm == "vanilla" else "complete_closed_book"
                    ),
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "decode_fingerprint": fingerprints[arm],
                    "runtime_receipt_path": "runtime_receipts/x.json",
                    "runtime_receipt_sha256": "1" * 64,
                    "full_stack_evidence": {},
                    "complete_system_evidence": {},
                    "text": "RIGHT" if correct else "WRONG",
                    "error": "",
                }
            )
    (tmp_path / "journal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    scored = {
        arm: {
            task.task_id: record["text"] == "RIGHT"
            for task, record in zip(
                tasks,
                [candidate for candidate in records if candidate["arm"] == arm],
                strict=True,
            )
        }
        for arm in verifier.EXPECTED_REQUIRED_ARMS
    }
    adjudication = verifier._independent_adjudication(
        scored, [task.task_id for task in tasks]
    )
    (tmp_path / "verdict.json").write_text(
        json.dumps(
            {
                "primary_claim_target": "composed_recurrent_tissue",
                "decision": adjudication["decision"],
                "composed_recurrent_adjudication": {
                    "comparisons": adjudication["comparisons"]
                },
                "claims": {
                    "reasoning_gain_proven": False,
                    "fusion_authorized": False,
                    "frontier_level_proven": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path, tasks


def test_independent_verifier_reconstructs_positive_component_canary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, tasks = _write_complete_campaign(tmp_path, monkeypatch)

    result = verifier.verify(config_path=config_path, campaign_dir=tmp_path)

    assert result["verified"] is True
    assert result["cell_count"] == len(tasks) * len(verifier.EXPECTED_REQUIRED_ARMS)
    assert result["adjudication"]["bounded_learned_tissue_positive"] is True
    assert result["adjudication"]["recurrent_depth_positive"] is True
    assert result["required_next_gate"] == "fresh_powered_preregistered_replication"
    assert result["reasoning_gain_proven"] is False
    assert result["fusion_authorized"] is False
    assert result["wow_signal_authorized"] is False


def test_independent_verifier_rejects_incomplete_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _tasks = _write_complete_campaign(tmp_path, monkeypatch)
    records = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "journal.jsonl").write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(verifier.ReconciliationVerificationError, match="campaign_incomplete"):
        verifier.verify(config_path=config_path, campaign_dir=tmp_path)


def test_independent_verifier_rejects_duplicate_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _tasks = _write_complete_campaign(tmp_path, monkeypatch)
    records = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "journal.jsonl").write_text(
        "\n".join([*records, records[0]]) + "\n", encoding="utf-8"
    )

    with pytest.raises(verifier.ReconciliationVerificationError, match="duplicate_cell"):
        verifier.verify(config_path=config_path, campaign_dir=tmp_path)


def test_independent_adjudication_refuses_treatment_regression() -> None:
    task_ids = ["a", "b"]
    scored = {
        arm: {task_id: False for task_id in task_ids}
        for arm in verifier.EXPECTED_REQUIRED_ARMS
    }
    scored[verifier.TREATMENT_ARM] = {"a": True, "b": False}
    scored["complete_system_recurrent_initial_control"] = {"a": False, "b": True}
    scored["complete_system_closed_book"] = {"a": False, "b": False}
    scored["complete_system_recurrent_depth_lesion"] = {"a": False, "b": False}
    scored["vanilla"] = {"a": False, "b": False}

    result = verifier._independent_adjudication(scored, task_ids)

    assert result["bounded_learned_tissue_positive"] is False
    assert result["decision"] == "no_bounded_learned_tissue_gain"
