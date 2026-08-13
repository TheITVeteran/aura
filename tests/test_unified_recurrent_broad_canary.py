from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm import unified_recurrent_broad_canary as broad
from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _plan():
    return broad.seal_broad_canary_plan(
        package_id="cp352-fixture",
        manifest_sha256="a" * 64,
        controller_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
        recurrence_depth=4,
        max_tokens=256,
        tasks=[
            {
                "task_id": f"task-{domain}",
                "domain": domain,
                "task_payload_sha256": _sha(f"payload:{domain}"),
                "answer_commitment_sha256": _sha(f"answer:{domain}"),
                "prompt_sha256": _sha(f"prompt:{domain}"),
            }
            for domain in FRONTIER_DOMAINS
        ],
    )


def _candidates():
    rows = []
    for index, domain in enumerate(FRONTIER_DOMAINS):
        task_id = f"task-{domain}"
        correctness = {
            "base_greedy": index < 3,
            "initial_t4": index < 2,
            "trained_t1": index < 4,
            "trained_t4": index < 6,
        }
        for arm in broad.ARMS:
            rows.append(
                {
                    "task_id": task_id,
                    "domain": domain,
                    "arm": arm,
                    "correct": correctness[arm],
                    "parsed": True,
                    "response_sha256": _sha(f"{task_id}:{arm}"),
                    "generated_tokens": 12,
                    "stopped": True,
                    "latency_ms": 10,
                }
            )
    return rows


def test_seals_answer_blind_broad_plan() -> None:
    plan = _plan()

    assert plan["arms"] == list(broad.ARMS)
    assert plan["domains"] == list(FRONTIER_DOMAINS)
    assert plan["typed_state_slots_enabled"] is False
    assert plan["terminal_grammar_enabled"] is False
    assert plan["answer_digit_pointer_enabled"] is False
    assert broad.broad_canary_plan_errors(plan) == []


def test_support_requires_gain_without_regression_and_recurrence_value() -> None:
    result = broad.seal_broad_canary_result(_plan(), _candidates())

    assert result["supported"] is True
    assert result["verdict"] == broad.SUPPORTED
    assert result["counts"] == {
        "base_greedy": 3,
        "initial_t4": 2,
        "trained_t1": 4,
        "trained_t4": 6,
    }
    assert result["transitions"]["initial_to_trained"] == {
        "wrong_to_right": 4,
        "right_to_wrong": 0,
    }
    assert broad.broad_canary_result_errors(result, plan=_plan()) == []


def test_refutes_one_base_regression() -> None:
    candidates = _candidates()
    row = next(
        item
        for item in candidates
        if item["task_id"] == f"task-{FRONTIER_DOMAINS[0]}"
        and item["arm"] == "trained_t4"
    )
    row["correct"] = False

    result = broad.seal_broad_canary_result(_plan(), candidates)

    assert result["supported"] is False
    assert result["checks"]["trained_preserves_base_successes"] is False


def test_refutes_missing_candidate_arm() -> None:
    result = broad.seal_broad_canary_result(_plan(), _candidates()[:-1])

    assert result["supported"] is False
    assert result["checks"]["complete_candidate_matrix"] is False


def test_rejects_plan_tampering() -> None:
    plan = copy.deepcopy(_plan())
    plan["terminal_grammar_enabled"] = True

    with pytest.raises(
        broad.UnifiedRecurrentBroadCanaryError,
        match="plan_identity_differs",
    ):
        broad.seal_broad_canary_result(plan, _candidates())
