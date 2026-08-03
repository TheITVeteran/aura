"""Contracts for recurrent checkpoint promotion and exact rollback."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.learning.recurrent_checkpoint_promotion import (
    FAIL,
    MINIMUM_PROBES,
    PASS,
    PROMOTE,
    REQUIRED_GATES,
    RETAIN_PARENT,
    ROLLBACK_AND_HALT,
    UNMEASURED,
    RecurrentCheckpointPromotionError,
    apply_checkpoint_decision,
    checkpoint_candidate,
    checkpoint_registry,
    evaluate_checkpoint_candidate,
    evidence_gate,
    stage_checkpoint_candidate,
    validate_registry,
)

PARENT = "a" * 64
CANDIDATE = "b" * 64
SOURCE = "c" * 64
GRADE = "d" * 64
INDEPENDENT = "e" * 64
COMMIT = "1" * 40


def _verifiers() -> dict[str, str]:
    return {gate: f"{index + 10:064x}" for index, gate in enumerate(REQUIRED_GATES)}


def _gates(
    *, failed: str | None = None, unmeasured: str | None = None, underpowered: str | None = None
) -> list[dict[str, object]]:
    verifiers = _verifiers()
    rows = []
    for gate in REQUIRED_GATES:
        minimum = MINIMUM_PROBES[gate]
        status = FAIL if gate == failed else UNMEASURED if gate == unmeasured else PASS
        probes = 0 if status == UNMEASURED else minimum - 1 if gate == underpowered else minimum
        passed = probes if status == PASS else max(0, probes - 1)
        rows.append(
            evidence_gate(
                gate=gate,
                status=status,
                probes_graded=probes,
                probes_passed=passed,
                evidence_sha256=f"{100 + len(rows):064x}",
                verifier_sha256=verifiers[gate],
            )
        )
    return rows


def _candidate(
    *,
    active: bool = False,
    failed: str | None = None,
    unmeasured: str | None = None,
    underpowered: str | None = None,
    artifact: str = CANDIDATE,
) -> dict[str, object]:
    return checkpoint_candidate(
        candidate_id="candidate-2",
        parent_id="parent-1",
        candidate_artifact_sha256=artifact,
        parent_artifact_sha256=PARENT,
        candidate_active=active,
        source_commit=COMMIT,
        source_closure_sha256=SOURCE,
        campaign_grade_sha256=GRADE,
        independent_verdict_sha256=INDEPENDENT,
        gates=_gates(failed=failed, unmeasured=unmeasured, underpowered=underpowered),
    )


def _registry() -> dict[str, object]:
    return checkpoint_registry(
        promoted_id="parent-1",
        promoted_artifact_sha256=PARENT,
        source_commit="2" * 40,
        source_closure_sha256="3" * 64,
    )


def _apply(
    registry: dict[str, object],
    decision: dict[str, object],
    candidate: dict[str, object],
    *,
    restored_artifact_sha256: str | None = None,
) -> dict[str, object]:
    return apply_checkpoint_decision(
        registry,
        decision,
        candidate=candidate,
        expected_verifiers=_verifiers(),
        restored_artifact_sha256=restored_artifact_sha256,
    )


def test_all_required_powered_gates_promote_shadow_candidate() -> None:
    candidate = _candidate()
    decision = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    staged = stage_checkpoint_candidate(_registry(), candidate)
    applied = _apply(staged, decision, candidate)

    assert decision["outcome"] == PROMOTE
    assert decision["reasons"] == []
    assert staged["latest"] == staged["pending"]
    assert staged["promoted"]["checkpoint_id"] == "parent-1"
    assert applied["pending"] is None
    assert applied["promoted"]["checkpoint_id"] == "candidate-2"
    assert validate_registry(applied) == applied


@pytest.mark.parametrize("status", ["failed", "unmeasured", "underpowered"])
def test_incomplete_or_negative_gate_retains_parent(status: str) -> None:
    kwargs = {status: "positive_interaction"}
    candidate = _candidate(**kwargs)
    decision = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    applied = _apply(stage_checkpoint_candidate(_registry(), candidate), decision, candidate)

    assert decision["outcome"] == RETAIN_PARENT
    assert decision["reasons"][0]["gate"] == "positive_interaction"
    assert applied["promoted"]["checkpoint_id"] == "parent-1"
    assert applied["latest"]["checkpoint_id"] == "candidate-2"
    assert applied["rejected_decisions"] == [decision["decision_sha256"]]


def test_active_regression_requires_exact_rollback_and_halts_candidate() -> None:
    candidate = _candidate(active=True, failed="vanilla_no_regression")
    decision = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    staged = stage_checkpoint_candidate(_registry(), candidate)

    assert decision["outcome"] == ROLLBACK_AND_HALT
    with pytest.raises(RecurrentCheckpointPromotionError, match="rollback_not_exact"):
        _apply(staged, decision, candidate, restored_artifact_sha256="f" * 64)

    applied = _apply(staged, decision, candidate, restored_artifact_sha256=PARENT)
    assert applied["promoted"]["artifact_sha256"] == PARENT
    assert applied["halted_candidates"] == ["candidate-2"]


def test_active_candidate_cannot_be_retroactively_promoted() -> None:
    decision = evaluate_checkpoint_candidate(
        _candidate(active=True), expected_verifiers=_verifiers()
    )

    assert decision["outcome"] == ROLLBACK_AND_HALT
    assert {row["gate"] for row in decision["reasons"]} == {"activation_order"}


def test_candidate_equal_to_parent_is_not_a_gain() -> None:
    decision = evaluate_checkpoint_candidate(
        _candidate(artifact=PARENT), expected_verifiers=_verifiers()
    )

    assert decision["outcome"] == RETAIN_PARENT
    assert decision["reasons"] == [
        {"gate": "artifact_identity", "reason": "candidate_equals_parent"}
    ]


def test_missing_duplicate_and_wrong_verifier_evidence_fail_closed() -> None:
    verifiers = _verifiers()
    missing = _candidate()
    missing["gates"] = missing["gates"][:-1]
    body = dict(missing)
    body.pop("candidate_sha256")
    missing["candidate_sha256"] = "0" * 64
    with pytest.raises(RecurrentCheckpointPromotionError, match="candidate_identity_invalid"):
        evaluate_checkpoint_candidate(missing, expected_verifiers=verifiers)

    wrong_verifier = _candidate()
    wrong_verifier["gates"][0]["verifier_sha256"] = "f" * 64
    body = dict(wrong_verifier)
    body.pop("candidate_sha256")
    # The public builder is the only supported way to produce a new candidate;
    # mutation must fail at the outer identity before inner evidence is trusted.
    with pytest.raises(RecurrentCheckpointPromotionError, match="candidate_identity_invalid"):
        evaluate_checkpoint_candidate(wrong_verifier, expected_verifiers=verifiers)

    duplicate = _gates()
    duplicate[-1] = copy.deepcopy(duplicate[0])
    candidate = checkpoint_candidate(
        candidate_id="candidate-2",
        parent_id="parent-1",
        candidate_artifact_sha256=CANDIDATE,
        parent_artifact_sha256=PARENT,
        candidate_active=False,
        source_commit=COMMIT,
        source_closure_sha256=SOURCE,
        campaign_grade_sha256=GRADE,
        independent_verdict_sha256=INDEPENDENT,
        gates=duplicate,
    )
    with pytest.raises(RecurrentCheckpointPromotionError, match="gate_duplicate"):
        evaluate_checkpoint_candidate(candidate, expected_verifiers=verifiers)


def test_stage_refuses_wrong_parent_and_parallel_pending_candidate() -> None:
    candidate = _candidate()
    wrong_parent = copy.deepcopy(candidate)
    wrong_parent["parent_artifact_sha256"] = "f" * 64
    with pytest.raises(RecurrentCheckpointPromotionError, match="parent_mismatch"):
        stage_checkpoint_candidate(_registry(), wrong_parent)

    staged = stage_checkpoint_candidate(_registry(), candidate)
    with pytest.raises(RecurrentCheckpointPromotionError, match="pending_candidate_exists"):
        stage_checkpoint_candidate(staged, candidate)


def test_apply_refuses_stale_or_tampered_decision() -> None:
    candidate = _candidate()
    staged = stage_checkpoint_candidate(_registry(), candidate)
    decision = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    tampered = copy.deepcopy(decision)
    tampered["candidate_id"] = "other-candidate"
    with pytest.raises(RecurrentCheckpointPromotionError, match="decision_identity_invalid"):
        _apply(staged, tampered, candidate)

    other = _registry()
    with pytest.raises(RecurrentCheckpointPromotionError, match="decision_stale"):
        _apply(other, decision, candidate)


def test_apply_replays_and_rejects_a_rehashed_forged_promotion() -> None:
    candidate = _candidate(failed="recurrent_gain")
    staged = stage_checkpoint_candidate(_registry(), candidate)
    forged = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    forged["outcome"] = PROMOTE
    forged["reasons"] = []
    material = dict(forged)
    material.pop("decision_sha256")
    forged["decision_sha256"] = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(RecurrentCheckpointPromotionError, match="decision_replay_mismatch"):
        _apply(staged, forged, candidate)


def test_unexpected_rollback_evidence_is_rejected() -> None:
    candidate = _candidate(failed="branch_specialization")
    decision = evaluate_checkpoint_candidate(candidate, expected_verifiers=_verifiers())
    staged = stage_checkpoint_candidate(_registry(), candidate)

    with pytest.raises(RecurrentCheckpointPromotionError, match="unexpected_rollback"):
        _apply(staged, decision, candidate, restored_artifact_sha256=PARENT)


def test_registry_digest_detects_pointer_and_history_tampering() -> None:
    registry = _registry()
    registry["promoted"]["artifact_sha256"] = CANDIDATE
    with pytest.raises(RecurrentCheckpointPromotionError, match="registry_identity_invalid"):
        validate_registry(registry)


def test_gate_count_contract_never_turns_unmeasured_into_pass() -> None:
    with pytest.raises(RecurrentCheckpointPromotionError, match="counts_invalid"):
        evidence_gate(
            gate="recurrent_gain",
            status=UNMEASURED,
            probes_graded=1,
            probes_passed=0,
            evidence_sha256="4" * 64,
            verifier_sha256="5" * 64,
        )
    with pytest.raises(RecurrentCheckpointPromotionError, match="counts_invalid"):
        evidence_gate(
            gate="recurrent_gain",
            status=PASS,
            probes_graded=40,
            probes_passed=39,
            evidence_sha256="4" * 64,
            verifier_sha256="5" * 64,
        )
