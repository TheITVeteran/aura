from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.blind_review import (
    run_blind_review,
    validate_blind_review_receipt,
)


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
        )

    with pytest.raises(ValueError, match="boundary"):
        validate_blind_review_receipt(
            receipt,
            n_branches=2,
            branch_scores=branch_scores,
            isolation_receipt=isolation,
            objective_sha256="c" * 64,
        )
