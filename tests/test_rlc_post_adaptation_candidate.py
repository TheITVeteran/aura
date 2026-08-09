from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.post_adaptation_candidate import (
    advance_post_adaptation_candidate,
    build_post_adaptation_candidate_receipt,
    validate_post_adaptation_candidate_receipt,
)


def _evidence() -> dict[str, object]:
    return {
        "latent_opt_attempts": 4,
        "latent_opt_accepted_steps": 3,
        "fast_weight_disposition": "not_applied",
        "fast_weight_decode_active": False,
        "slot_ablation_applied": False,
    }


def test_post_adaptation_candidate_replaces_stale_probe_with_strict_candidate():
    prior = 'FINAL_ANSWER: {"answer": 3}'
    observed = 'FINAL_ANSWER: {"answer": 4}'

    transition, candidate = advance_post_adaptation_candidate(
        selected_branch=1,
        prior_candidate=prior,
        observed_candidate=observed,
        stage="post_final_adaptation",
        strict_answer_contract=True,
        response_contract='{"answer":int}',
        adaptation_evidence=_evidence(),
    )
    receipt = build_post_adaptation_candidate_receipt([transition])

    assert candidate == observed
    assert receipt["candidate_pool_effect"] == "replaced"
    assert receipt["final_candidate_available"] is True
    assert receipt["correctness_authority"] == "none"
    assert receipt["answer_selection_authority"] == "none"
    assert validate_post_adaptation_candidate_receipt(receipt) == receipt


def test_post_adaptation_candidate_discloses_bounded_schema_coercion():
    transition, candidate = advance_post_adaptation_candidate(
        selected_branch=0,
        prior_candidate=(
            'FINAL_ANSWER: {"items":[[1,2,3]],"kind":"fixed"}'
        ),
        observed_candidate="[1, 2, 3]",
        stage="post_final_adaptation",
        strict_answer_contract=True,
        response_contract='{"items":list[list[int]],"kind":"fixed"}',
        adaptation_evidence=_evidence(),
    )

    assert candidate == (
        'FINAL_ANSWER: {"items":[[1,2,3]],"kind":"fixed"}'
    )
    assert transition["disposition"] == "serialization_coercion_admitted"
    assert transition["candidate_pool_effect"] == "retained"


def test_post_adaptation_candidate_removes_contract_invalid_observation():
    transition, candidate = advance_post_adaptation_candidate(
        selected_branch=0,
        prior_candidate='FINAL_ANSWER: {"answer": 3}',
        observed_candidate="I am still thinking",
        stage="post_final_adaptation",
        strict_answer_contract=True,
        response_contract='{"answer":int}',
        adaptation_evidence=_evidence(),
    )
    receipt = build_post_adaptation_candidate_receipt([transition])

    assert candidate is None
    assert receipt["candidate_pool_effect"] == "removed"
    assert receipt["final_candidate_available"] is False
    assert receipt["final_candidate_sha256"] == ""


def test_research_refresh_can_add_a_final_candidate_without_a_stale_prior():
    transition, candidate = advance_post_adaptation_candidate(
        selected_branch=1,
        prior_candidate=None,
        observed_candidate='FINAL_ANSWER: {"answer": 9}',
        stage="post_final_adaptation",
        strict_answer_contract=True,
        response_contract='{"answer":int}',
        adaptation_evidence=_evidence(),
    )
    receipt = build_post_adaptation_candidate_receipt([transition])

    assert candidate == 'FINAL_ANSWER: {"answer": 9}'
    assert transition["prior_candidate_available"] is False
    assert receipt["candidate_pool_effect"] == "added"
    assert receipt["final_candidate_available"] is True


def test_post_adaptation_receipt_rejects_tampering():
    transition, _candidate = advance_post_adaptation_candidate(
        selected_branch=0,
        prior_candidate="old",
        observed_candidate="new",
        stage="post_final_adaptation",
        strict_answer_contract=False,
        adaptation_evidence=_evidence(),
    )
    receipt = build_post_adaptation_candidate_receipt([transition])
    tampered = copy.deepcopy(receipt)
    tampered["transitions"][0]["observed_candidate_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        validate_post_adaptation_candidate_receipt(tampered)
