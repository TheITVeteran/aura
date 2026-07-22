from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.blind_review import (
    run_blind_review,
    run_decoy_balanced_review,
    run_decoy_preflight,
    validate_blind_review_receipt,
    validate_decoy_preflight_receipt,
    validate_decoy_review_receipt,
)
from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier


def _isolation(count: int) -> dict:
    return {
        "schema": "aura.rlc.branch_isolation.v1",
        "certified": True,
        "candidates": [
            {"index": index, "candidate_sha256": f"{index + 1:064x}"}
            for index in range(count)
        ],
    }


def test_reviewer_sees_only_deranged_origin_free_candidate_text():
    seen: list[str] = []

    def reviewer(text: str) -> float:
        seen.append(text)
        return len(text) / 100.0

    scores, receipt = run_blind_review(
        {
            0: "My first answer is four.",
            1: "As Aura, candidate 1 concludes five.",
            2: "Are you sure? The result is six.",
        },
        reviewer,
        episode_id="episode-123",
        objective_sha256=hashlib.sha256(b"objective").hexdigest(),
        isolation_receipt=_isolation(3),
    )

    assert set(scores) == {0, 1, 2}
    assert receipt["deranged_order"] is True
    assert receipt["first_answer_designated"] is False
    assert receipt["ownership_framing_supplied"] is False
    assert all(row["review_position"] != row["branch"] for row in receipt["rows"])
    assert sum(row["origin_redactions"] for row in receipt["rows"]) == 4
    joined = " ".join(seen).lower()
    assert "first answer" not in joined
    assert "as aura" not in joined
    assert "candidate 1" not in joined
    assert "are you sure" not in joined


def test_blind_review_requires_fresh_context_isolation():
    with pytest.raises(ValueError, match="fresh-context"):
        run_blind_review(
            {0: "answer"},
            lambda _text: 1.0,
            episode_id="episode",
            objective_sha256="a" * 64,
            isolation_receipt={"certified": False},
        )


def test_origin_only_candidate_is_scored_as_non_substantive_without_aborting():
    seen: list[str] = []
    scores, receipt = run_blind_review(
        {0: "My first answer", 1: "candidate 1"},
        lambda text: seen.append(text) or 0.0,
        episode_id="episode-origin-only",
        objective_sha256="d" * 64,
        isolation_receipt=_isolation(2),
    )

    assert scores == {0: 0.0, 1: 0.0}
    assert seen == ["[no substantive candidate content]"] * 2
    assert all(row["origin_redactions"] == 1 for row in receipt["rows"])


def test_service_validation_rejects_order_and_score_tampering():
    isolation = _isolation(2)
    scores, receipt = run_blind_review(
        {0: "four", 1: "five"},
        lambda text: 1.0 if text == "four" else 0.0,
        episode_id="episode-456",
        objective_sha256="b" * 64,
        isolation_receipt=isolation,
    )

    branch_scores = [scores[index] for index in range(2)]
    validate_blind_review_receipt(
        receipt,
        n_branches=2,
        branch_scores=branch_scores,
        isolation_receipt=isolation,
        objective_sha256="b" * 64,
        episode_id="episode-456",
        selected_branch=0,
    )

    tampered = copy.deepcopy(receipt)
    tampered["rows"][0]["branch"] = tampered["rows"][0]["review_position"]
    with pytest.raises(ValueError, match="mapping"):
        validate_blind_review_receipt(
            tampered,
            n_branches=2,
            branch_scores=branch_scores,
            isolation_receipt=isolation,
            objective_sha256="b" * 64,
            episode_id="episode-456",
            selected_branch=0,
        )

    tampered = copy.deepcopy(receipt)
    tampered["rows"][0]["score"] = 0.5
    with pytest.raises(ValueError, match="mapping"):
        validate_blind_review_receipt(
            tampered,
            n_branches=2,
            branch_scores=branch_scores,
            isolation_receipt=isolation,
            objective_sha256="b" * 64,
            episode_id="episode-456",
            selected_branch=0,
        )

    tampered = copy.deepcopy(receipt)
    tampered["reviewer_forbidden_fields"] = []
    with pytest.raises(ValueError, match="boundary"):
        validate_blind_review_receipt(
            tampered,
            n_branches=2,
            branch_scores=branch_scores,
            isolation_receipt=isolation,
            objective_sha256="b" * 64,
            episode_id="episode-456",
            selected_branch=0,
        )

    with pytest.raises(ValueError, match="boundary"):
        validate_blind_review_receipt(
            receipt,
            n_branches=2,
            branch_scores=branch_scores,
            isolation_receipt=isolation,
            objective_sha256="c" * 64,
            episode_id="episode-456",
            selected_branch=0,
        )


def test_decoy_balanced_review_admits_a_discriminating_stable_verifier():
    isolation = _isolation(2)
    objective_sha256 = hashlib.sha256(b"Compute 12 + 30.").hexdigest()
    verifier = EpisodeTaskVerifier("Compute 12 + 30.")
    scores, blind, decoy = run_decoy_balanced_review(
        {
            0: "The result is 42 because 12 + 30 = 42.",
            1: "The result is 43 because 12 + 30 = 43.",
        },
        verifier,
        episode_id="episode-balanced",
        objective_sha256=objective_sha256,
        isolation_receipt=isolation,
    )

    assert decoy["certified"] is True
    assert decoy["selection_admitted"] is True
    assert decoy["labels_withheld_during_review"] is True
    assert {row["item_class"] for row in decoy["batch_rows"]} == {
        "candidate",
        "control",
    }
    controls = {row["kind"]: row for row in decoy["controls"]}
    assert controls["correct"]["score"] > controls["incorrect"]["score"]
    assert controls["unchanged_a"]["score"] == controls["unchanged_b"]["score"]
    assert len(decoy["control_evaluation_indices"]) == 4
    assert verifier.to_receipt(
        exclude_evaluation_indices=set(decoy["control_evaluation_indices"])
    )["evaluations"] == 2

    validate_decoy_review_receipt(
        decoy,
        blind_receipt=blind,
        episode_id="episode-balanced",
        objective_sha256=objective_sha256,
    )
    validate_blind_review_receipt(
        blind,
        n_branches=2,
        branch_scores=[scores[index] for index in range(2)],
        isolation_receipt=isolation,
        objective_sha256=objective_sha256,
        episode_id="episode-balanced",
        selected_branch=max(scores, key=scores.get),
        decoy_receipt=decoy,
    )
    with pytest.raises(ValueError, match="selected branch"):
        validate_blind_review_receipt(
            blind,
            n_branches=2,
            branch_scores=[scores[index] for index in range(2)],
            isolation_receipt=isolation,
            objective_sha256=objective_sha256,
            episode_id="episode-balanced",
            selected_branch=1 - max(scores, key=scores.get),
            decoy_receipt=decoy,
        )


def test_decoy_preflight_gates_recurrent_verifier_authority_and_is_tamper_evident():
    objective_sha256 = hashlib.sha256(b"Compute a result.").hexdigest()
    verifier = EpisodeTaskVerifier("Compute a result.")
    receipt = run_decoy_preflight(
        verifier,
        episode_id="episode-preflight",
        objective_sha256=objective_sha256,
    )

    assert receipt["certified"] is True
    assert receipt["verifier_admitted"] is True
    assert len(receipt["control_evaluation_indices"]) == 4
    validate_decoy_preflight_receipt(
        receipt,
        episode_id="episode-preflight",
        objective_sha256=objective_sha256,
    )

    rejected = run_decoy_preflight(
        lambda _text: 0.5,
        episode_id="episode-preflight-rejected",
        objective_sha256=objective_sha256,
    )
    assert rejected["verifier_admitted"] is False
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        run_decoy_preflight(
            lambda _text: 2.0,
            episode_id="episode-preflight-out-of-range",
            objective_sha256=objective_sha256,
        )

    tampered = copy.deepcopy(receipt)
    tampered["controls"][0]["score"] = 0.123
    with pytest.raises(ValueError, match="verdict"):
        validate_decoy_preflight_receipt(
            tampered,
            episode_id="episode-preflight",
            objective_sha256=objective_sha256,
        )

def test_uncalibrated_reviewer_cannot_control_branch_selection():
    isolation = _isolation(2)
    objective_sha256 = hashlib.sha256(b"objective").hexdigest()
    _, blind, decoy = run_decoy_balanced_review(
        {0: "candidate alpha", 1: "candidate beta"},
        lambda _text: 0.5,
        episode_id="episode-constant",
        objective_sha256=objective_sha256,
        isolation_receipt=isolation,
    )

    assert decoy["correct_above_incorrect"] is False
    assert decoy["unchanged_consistent"] is True
    assert decoy["selection_admitted"] is False
    validate_blind_review_receipt(
        blind,
        n_branches=2,
        branch_scores=[-0.1, -0.2],
        isolation_receipt=isolation,
        objective_sha256=objective_sha256,
        episode_id="episode-constant",
        selected_branch=0,
        decoy_receipt=decoy,
    )


def test_decoy_validator_rejects_order_label_and_verdict_tampering():
    isolation = _isolation(2)
    objective_sha256 = hashlib.sha256(b"objective").hexdigest()
    verifier = EpisodeTaskVerifier("Compute a result.")
    _, blind, decoy = run_decoy_balanced_review(
        {0: "answer alpha", 1: "answer beta"},
        verifier,
        episode_id="episode-tamper",
        objective_sha256=objective_sha256,
        isolation_receipt=isolation,
    )

    for mutation, match in (
        (lambda value: value["batch_rows"].reverse(), "batch order"),
        (lambda value: value["controls"][0].update(kind="incorrect"), "evidence"),
        (lambda value: value.update(certified=False), "verdict"),
    ):
        tampered = copy.deepcopy(decoy)
        mutation(tampered)
        with pytest.raises(ValueError, match=match):
            validate_decoy_review_receipt(
                tampered,
                blind_receipt=blind,
                episode_id="episode-tamper",
                objective_sha256=objective_sha256,
            )
