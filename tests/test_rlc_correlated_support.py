from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.correlated_support import (
    MIN_PAIRED_OUTCOMES,
    BranchCorrelationLedger,
    build_correlated_support_receipt,
    build_correlation_evidence,
    initial_exchange_weights,
    validate_correlated_support_receipt,
)

ROLES = ["constructive_solution", "counterexample_search", "constraint_checking"]


def _task(index: int) -> str:
    return hashlib.sha256(f"fresh-task-{index}".encode()).hexdigest()


def _checked_rows(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        shared = index % 3 != 0
        rows.append(
            {
                "checked": True,
                "task_sha256": _task(index),
                "correct_by_role": {
                    ROLES[0]: shared,
                    ROLES[1]: shared,
                    ROLES[2]: index % 2 == 0,
                },
            }
        )
    return rows


def _structure(*, duplicate_first_pair: bool = False) -> dict:
    hashes = ["a" * 64, "b" * 64, "c" * 64]
    distances = [0.8, 1.0, 1.0]
    duplicate_groups = []
    if duplicate_first_pair:
        hashes[1] = hashes[0]
        distances[0] = 0.0
        duplicate_groups = [[0, 1]]
    return {
        "branches": [
            {
                "index": index,
                "role_path": [role],
                "structural_sha256": hashes[index],
            }
            for index, role in enumerate(ROLES)
        ],
        "pairwise": [
            {"left": 0, "right": 1, "distance": distances[0]},
            {"left": 0, "right": 2, "distance": distances[1]},
            {"left": 1, "right": 2, "distance": distances[2]},
        ],
        "duplicate_groups": duplicate_groups,
    }


def test_checked_error_history_estimates_and_shrinks_positive_correlation():
    evidence = build_correlation_evidence(
        bucket="logic|heldout",
        roles=ROLES,
        checked_outcomes=_checked_rows(MIN_PAIRED_OUTCOMES * 2),
    )

    correlated = next(row for row in evidence["pairs"] if set((row["left"], row["right"])) == set(ROLES[:2]))
    assert evidence["evidence_state"] == "measured"
    assert correlated["phi"] == 1.0
    assert 0.0 < correlated["positive_shrunk_correlation"] < correlated["phi"]
    assert correlated["enough_evidence"] is True


def test_unpowered_history_cannot_discount_support_empirically():
    evidence = build_correlation_evidence(
        bucket="logic|bootstrap",
        roles=ROLES,
        checked_outcomes=_checked_rows(MIN_PAIRED_OUTCOMES - 1),
    )

    assert evidence["evidence_state"] == "bootstrap_unmeasured"
    assert all(row["positive_shrunk_correlation"] == 0.0 for row in evidence["pairs"])
    assert all(row["enough_evidence"] is False for row in evidence["pairs"])


def test_correlated_paths_reduce_effective_support_and_exchange_weight():
    evidence = build_correlation_evidence(
        bucket="logic|heldout",
        roles=ROLES,
        checked_outcomes=_checked_rows(MIN_PAIRED_OUTCOMES * 2),
    )

    receipt = build_correlated_support_receipt(
        structural_diversity=_structure(),
        correlation_evidence=evidence,
    )
    exchange = {row["branch"]: row["weight"] for row in receipt["exchange_weights_applied"]}

    assert receipt["empirical_correlation_applied"] is True
    assert receipt["effective_support_count"] < receipt["raw_support_count"]
    assert receipt["confidence_multiplier"] < 1.0
    assert exchange[0] < 1.0 and exchange[1] < 1.0
    assert exchange[2] == 1.0


def test_duplicate_programs_collapse_without_waiting_for_history():
    weights = initial_exchange_weights(roles=["direct_derivation", "simplification"], correlation_evidence=None)
    assert weights == {0: 0.5, 1: 0.5}

    receipt = build_correlated_support_receipt(
        structural_diversity=_structure(duplicate_first_pair=True),
        correlation_evidence=None,
    )
    assert receipt["duplicate_votes_collapsed"] is True
    assert receipt["evidence_state"] == "bootstrap_unmeasured"
    assert receipt["effective_support_count"] < 3.0


def test_correlated_support_claim_is_exactly_reconstructed():
    evidence = build_correlation_evidence(
        bucket="logic|heldout",
        roles=ROLES,
        checked_outcomes=_checked_rows(MIN_PAIRED_OUTCOMES),
    )
    structure = _structure()
    receipt = build_correlated_support_receipt(
        structural_diversity=structure,
        correlation_evidence=evidence,
    )
    validate_correlated_support_receipt(
        receipt,
        structural_diversity=structure,
        correlation_evidence=evidence,
    )

    tampered = copy.deepcopy(receipt)
    tampered["confidence_multiplier"] = 1.0
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_correlated_support_receipt(
            tampered,
            structural_diversity=structure,
            correlation_evidence=evidence,
        )


def test_checked_outcome_ledger_persists_restores_and_rejects_replay(tmp_path):
    path = tmp_path / "checked.jsonl"
    ledger = BranchCorrelationLedger(path)
    outcomes = {role: index % 2 == 0 for index, role in enumerate(ROLES)}

    assert ledger.record_checked(
        bucket="logic|durable",
        task_sha256=_task(999),
        correct_by_role=outcomes,
    )
    with pytest.raises(ValueError, match="already recorded"):
        ledger.record_checked(
            bucket="logic|durable",
            task_sha256=_task(999),
            correct_by_role=outcomes,
        )

    restored = BranchCorrelationLedger(path)
    evidence = restored.evidence(bucket="logic|durable", roles=ROLES)
    assert evidence["checked_tasks"] == 1
    assert restored.status()["restore_errors"] == 0
