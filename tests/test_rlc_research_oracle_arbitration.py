"""The hidden benchmark oracle diagnoses selection without becoming serving policy."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.latent_cortex.research_oracle_arbitration import (
    build_research_oracle_arbitration,
    build_research_oracle_assessment,
    validate_research_oracle_arbitration,
    validate_research_oracle_assessment,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _assessor(correct_text: str, *, task: str = "task-a"):
    def assess(candidate: str):
        correct = candidate == correct_text
        return build_research_oracle_assessment(
            candidate=candidate,
            task_id=f"rlc_frontier:unit:{task}",
            task_payload_sha256=_sha(f"payload:{task}"),
            answer_commitment_sha256=_sha(f"answer:{task}"),
            scorer_id="exact_unit_scorer",
            scorer_version="1",
            scorer_source_sha256=_sha("scorer-source"),
            parsed=True,
            correct=correct,
            reason="correct" if correct else "incorrect_or_schema_mismatch",
            normalized_answer_sha256=_sha(candidate),
        )

    return assess


def test_oracle_replaces_only_wrong_current_with_correct_selected_candidate() -> None:
    assess = _assessor("correct recurrent")
    receipt, tokens = build_research_oracle_arbitration(
        current_text="wrong current",
        current_tokens=[1, 2],
        recurrent_text="correct recurrent",
        recurrent_tokens=[3, 4],
        selected_branch=1,
        assess=assess,
    )

    assert receipt["decision"] == "replace"
    assert receipt["current_output"]["correct"] is False
    assert receipt["recurrent_output"]["correct"] is True
    assert receipt["serving_authority"] is False
    assert receipt["capability_claim_authority"] is False
    assert receipt["research_measurement_authority"] is True
    assert tokens == [3, 4]


@pytest.mark.parametrize(
    ("correct_text", "current", "recurrent"),
    [
        ("current", "current", "wrong recurrent"),
        ("neither", "wrong current", "wrong recurrent"),
        ("current", "current", "current"),
    ],
)
def test_oracle_retains_when_strict_correctness_dominance_is_absent(
    correct_text: str,
    current: str,
    recurrent: str,
) -> None:
    receipt, tokens = build_research_oracle_arbitration(
        current_text=current,
        current_tokens=[1],
        recurrent_text=recurrent,
        recurrent_tokens=[2],
        selected_branch=0,
        assess=_assessor(correct_text),
    )

    assert receipt["decision"] == "retain"
    assert tokens == [1]


def test_oracle_receipt_replay_binds_private_candidates_and_output() -> None:
    assess = _assessor("recurrent")
    left = assess("current")
    right = assess("recurrent")
    receipt, tokens = build_research_oracle_arbitration(
        current_text="current",
        current_tokens=[10],
        recurrent_text="recurrent",
        recurrent_tokens=[20],
        selected_branch=1,
        assess=assess,
    )

    rebuilt = validate_research_oracle_arbitration(
        receipt,
        current_text="current",
        current_tokens=[10],
        recurrent_text="recurrent",
        recurrent_tokens=[20],
        selected_branch=1,
        current_assessment=left,
        recurrent_assessment=right,
        expected_output_text="recurrent",
        expected_output_tokens=tokens,
    )
    assert rebuilt == receipt

    tampered = copy.deepcopy(receipt)
    tampered["serving_authority"] = True
    with pytest.raises(ValueError, match="reconstruction differs"):
        validate_research_oracle_arbitration(
            tampered,
            current_text="current",
            current_tokens=[10],
            recurrent_text="recurrent",
            recurrent_tokens=[20],
            selected_branch=1,
            current_assessment=left,
            recurrent_assessment=right,
        )


def test_oracle_assessments_for_different_tasks_cannot_be_combined() -> None:
    left = _assessor("current", task="left")("current")
    right = _assessor("recurrent", task="right")("recurrent")
    receipt, _tokens = build_research_oracle_arbitration(
        current_text="current",
        current_tokens=[1],
        recurrent_text="recurrent",
        recurrent_tokens=[2],
        selected_branch=0,
        assess=_assessor("recurrent", task="left"),
    )
    with pytest.raises(ValueError, match="different tasks"):
        validate_research_oracle_arbitration(
            receipt,
            current_text="current",
            current_tokens=[1],
            recurrent_text="recurrent",
            recurrent_tokens=[2],
            selected_branch=0,
            current_assessment=left,
            recurrent_assessment=right,
        )


def test_frozen_frontier_oracle_emits_hidden_answer_commitments_only() -> None:
    from core.brain.llm.latent_cortex import frontier_tasks as ft
    from tools.run_rlc_reconciliation_sweep import _OracleTaskVerifier

    task = ft.generate_task_battery([20260808], difficulty=2)[0]
    expected = task.reveal_for_verifier()["expected"]
    correct = "FINAL_ANSWER: " + json.dumps(expected, sort_keys=True, separators=(",", ":"))
    verifier = _OracleTaskVerifier(task)

    accepted = verifier.research_oracle_assessment(correct)
    rejected = verifier.research_oracle_assessment('FINAL_ANSWER: {"sequence":[],"checksum":0}')

    assert validate_research_oracle_assessment(accepted, candidate=correct)["correct"] is True
    assert rejected["correct"] is False
    assert "expected" not in json.dumps(accepted, sort_keys=True)
    assert accepted["answer_key_exposed"] is False


def test_engine_meter_preserves_and_charges_research_oracle_assessment() -> None:
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    class Budget:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def charge_verifier(self, operation, **kwargs) -> None:
            self.calls.append({"operation": operation, **kwargs})

    class Verifier:
        def __call__(self, text: str) -> float:
            return float(bool(text))

        def research_oracle_assessment(self, text: str) -> dict:
            return {"candidate": text, "correct": text == "right"}

    budget = Budget()
    metered = LatentCortexEngine._meter_verifier(Verifier(), budget)

    assert metered is not None
    assert metered("candidate") == 1.0
    assert metered.research_oracle_assessment("right") == {
        "candidate": "right",
        "correct": True,
    }
    assert [row["operation"] for row in budget.calls] == [
        "task_verifier",
        "task_verifier",
    ]
