"""Model-free evidence contracts for broad parent-to-treatment transfer."""

from __future__ import annotations

import copy

import pytest

from core.brain.llm import unified_recurrent_transfer_canary as canary
from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS


def _plan() -> dict[str, object]:
    tasks = [
        {
            "task_id": f"task-{domain}",
            "domain": domain,
            "task_payload_sha256": "1" * 64,
            "answer_commitment_sha256": "2" * 64,
            "prompt_sha256": "3" * 64,
        }
        for domain in FRONTIER_DOMAINS
    ]
    return canary.seal_transfer_canary_plan(
        campaign_identity_sha256="4" * 64,
        parent_checkpoint_sha256="5" * 64,
        parent_controller_sha256="6" * 64,
        treatment_checkpoint_sha256="7" * 64,
        treatment_controller_sha256="8" * 64,
        recurrence_depth=4,
        max_tokens=128,
        tasks=tasks,
        source_binding={
            "git_commit": "9" * 40,
            "implementation_sha256s": {"core/example.py": "a" * 64},
        },
    )


def _candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, domain in enumerate(FRONTIER_DOMAINS):
        correctness = {
            "base_greedy": index in {0, 1},
            "parent_typed": index in {0, 1, 2},
            "treatment_typed": index in {0, 1, 2, 3, 4},
            "action_lesion": index in {0, 1, 2},
            "process_tape_lesion": index in {0, 1, 2},
        }
        for arm in canary.ARMS:
            rows.append(
                {
                    "task_id": f"task-{domain}",
                    "domain": domain,
                    "arm": arm,
                    "correct": correctness[arm],
                    "parsed": True,
                    "response_sha256": "b" * 64,
                    "generated_tokens": 12,
                    "stopped": True,
                    "latency_ms": 10,
                }
            )
    return rows


def test_transfer_canary_supports_gain_with_causal_lesion() -> None:
    plan = _plan()
    assert canary.transfer_canary_plan_errors(plan) == []
    result = canary.seal_transfer_canary_result(plan, _candidates())
    assert result["supported"] is True
    assert result["verdict"] == canary.SUPPORTED
    assert result["counts"] == {
        "base_greedy": 2,
        "parent_typed": 3,
        "treatment_typed": 5,
        "action_lesion": 3,
        "process_tape_lesion": 3,
    }
    assert canary.transfer_canary_result_errors(result, plan=plan) == []


def test_transfer_canary_refutes_regression_or_failed_lesion() -> None:
    rows = _candidates()
    for row in rows:
        if row["task_id"] == f"task-{FRONTIER_DOMAINS[2]}" and row["arm"] == "treatment_typed":
            row["correct"] = False
        if row["arm"] == "action_lesion":
            treatment = next(
                candidate
                for candidate in rows
                if candidate["task_id"] == row["task_id"]
                and candidate["arm"] == "treatment_typed"
            )
            row["correct"] = treatment["correct"]
    result = canary.seal_transfer_canary_result(_plan(), rows)
    assert result["conclusive"] is True
    assert result["supported"] is False
    assert result["verdict"] == canary.REFUTED
    assert result["checks"]["treatment_has_zero_parent_regressions"] is False
    assert result["checks"]["action_lesion_removes_treatment_gain"] is False


def test_transfer_canary_is_inconclusive_when_matrix_or_band_is_unusable() -> None:
    result = canary.seal_transfer_canary_result(_plan(), _candidates()[:-1])
    assert result["conclusive"] is False
    assert result["verdict"] == canary.INCONCLUSIVE


def test_transfer_canary_rejects_same_parent_and_treatment() -> None:
    plan = _plan()
    plan["treatment_controller_sha256"] = plan["parent_controller_sha256"]
    with pytest.raises(canary.UnifiedRecurrentTransferCanaryError):
        canary.seal_transfer_canary_result(plan, _candidates())


def test_transfer_canary_reopening_detects_tampering() -> None:
    plan = copy.deepcopy(_plan())
    plan["max_tokens"] = 129
    assert "transfer_canary_identity_invalid" in canary.transfer_canary_plan_errors(plan)
