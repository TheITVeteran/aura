"""Contracts for the strict RLC epistemic state and transaction authority."""
from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from core.brain.llm.latent_cortex.epistemic_state import (
    AcceptedAnswer,
    ClaimRecord,
    ClaimStatus,
    ComputeBudgetState,
    EpistemicState,
    EpistemicStateError,
    EpistemicStateMachine,
    EvidenceKind,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    ProbabilityInterval,
    ProblemFrame,
    StaleEpistemicTransactionError,
    text_sha256,
)


def interval(point: float = 0.5) -> ProbabilityInterval:
    return ProbabilityInterval(
        lower=max(0.0, point - 0.1),
        point=point,
        upper=min(1.0, point + 0.1),
        method="test_wilson",
        evidence_count=1,
    )


def evidence(evidence_id: str, *, supports=(), kind=EvidenceKind.CALCULATION):
    summary = f"Evidence for {evidence_id}"
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        summary=summary,
        content_sha256=text_sha256(summary),
        source="unit_test",
        observed_at=1.0,
        receipt_sha256=text_sha256(f"receipt:{evidence_id}"),
        supports=tuple(supports),
    )


def genesis() -> EpistemicState:
    problem = ProblemFrame.create("Determine whether the deployment is safe.")
    problem_evidence = evidence("ev.problem", kind=EvidenceKind.IMMUTABLE_PROBLEM)
    return EpistemicState.genesis(
        episode_id="episode.test",
        problem=problem,
        budget=ComputeBudgetState(total=100.0, tool_calls_total=4),
        evidence=(problem_evidence,),
    )


def test_genesis_is_canonical_deeply_immutable_and_content_addressed():
    state = genesis()
    assert state.version == 0 and state.parent_sha256 == ""
    assert state.problem.immutable_evidence_ids == ("ev.problem",)
    assert len(state.state_sha256) == 64
    assert EpistemicState.genesis(
        episode_id="episode.test",
        problem=ProblemFrame.create("Determine whether the deployment is safe."),
        budget=ComputeBudgetState(total=100.0, tool_calls_total=4),
        evidence=(evidence("ev.problem", kind=EvidenceKind.IMMUTABLE_PROBLEM),),
    ).state_sha256 == state.state_sha256
    with pytest.raises(FrozenInstanceError):
        state.version = 4  # type: ignore[misc]
    with pytest.raises(TypeError):
        state.claims[0] = None  # type: ignore[index]


def test_transaction_commits_typed_claim_hypothesis_evidence_operation_and_answer():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord(
            claim_id="claim.root",
            text="The health gate passed.",
            status=ClaimStatus.VERIFIED,
            uncertainty=interval(0.9),
            evidence_ids=("ev.health",),
            answer_relevant=True,
        )
    )
    tx.add_evidence(evidence("ev.health", supports=("claim.root",)))
    tx.add_hypothesis(
        HypothesisRecord(
            hypothesis_id="hyp.safe",
            statement="Deployment is safe.",
            posterior=interval(0.85),
            status=HypothesisStatus.FAVORED,
            claim_ids=("claim.root",),
        )
    )
    tx.add_operation(
        OperationRecord(
            operation_id="op.verify",
            kind=OperationKind.CHECK_ASSUMPTION,
            outcome=OperationOutcome.SUCCEEDED,
            input_state_sha256=tx.base.state_sha256,
            cost=3.0,
            affected_claim_ids=("claim.root",),
            evidence_gained=("ev.health",),
        )
    )
    tx.set_budget(
        ComputeBudgetState(
            total=100.0,
            used=3.0,
            tool_calls_total=4,
            tool_calls_used=1,
        )
    )
    answer_text = "The deployment is safe under the verified health gate."
    tx.accept_answer(
        AcceptedAnswer(
            text=answer_text,
            text_sha256=text_sha256(answer_text),
            claim_ids=("claim.root",),
            evidence_ids=("ev.health",),
            confidence=interval(0.85),
        )
    )
    committed = machine.commit(tx)
    assert committed.version == 1
    assert committed.parent_sha256 == genesis().state_sha256
    assert committed.accepted_answer is not None
    assert committed.operations[0].input_state_sha256 == committed.parent_sha256
    assert machine.snapshot() is committed

    canonical = committed.to_canonical_json()
    restored = EpistemicState.from_canonical_json(canonical)
    assert restored == committed
    assert restored.to_canonical_json() == canonical


def test_canonical_state_hash_is_independent_of_record_insertion_order():
    def commit_in_order(claim_ids: tuple[str, ...]) -> EpistemicState:
        machine = EpistemicStateMachine(genesis())
        tx = machine.begin()
        for claim_id in claim_ids:
            tx.add_claim(
                ClaimRecord(claim_id, claim_id, ClaimStatus.PROPOSED, interval())
            )
        return machine.commit(tx)

    left = commit_in_order(("claim.b", "claim.a"))
    right = commit_in_order(("claim.a", "claim.b"))
    assert left == right
    assert left.state_sha256 == right.state_sha256


def test_deserialization_rejects_unknown_fields_tampering_and_wrong_wire_types():
    payload = genesis().to_dict()
    with_unknown = copy.deepcopy(payload)
    with_unknown["unexpected"] = True
    with pytest.raises(EpistemicStateError, match="unknown"):
        EpistemicState.from_dict(with_unknown)

    tampered = copy.deepcopy(payload)
    tampered["problem"]["objective"] = "Different objective."
    with pytest.raises(EpistemicStateError, match="digest"):
        EpistemicState.from_dict(tampered)

    tampered_hash = copy.deepcopy(payload)
    tampered_hash["state_sha256"] = "f" * 64
    with pytest.raises(EpistemicStateError, match="state hash"):
        EpistemicState.from_dict(tampered_hash)

    wrong_collection = copy.deepcopy(payload)
    wrong_collection["claims"] = "not-an-array"
    with pytest.raises(EpistemicStateError, match="array"):
        EpistemicState.from_dict(wrong_collection)

    wrong_kind = copy.deepcopy(payload)
    wrong_kind["evidence"][0]["kind"] = "imagined"
    with pytest.raises(EpistemicStateError, match="supported value"):
        EpistemicState.from_dict(wrong_kind)

    noncanonical = genesis().to_canonical_json().replace(",", ", ", 1)
    with pytest.raises(EpistemicStateError, match="not canonical"):
        EpistemicState.from_canonical_json(noncanonical)


def test_invalid_transaction_is_atomic_and_leaves_current_state_unchanged():
    machine = EpistemicStateMachine(genesis())
    before = machine.snapshot()
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord(
            claim_id="claim.orphan",
            text="Unsupported claim.",
            status=ClaimStatus.SUPPORTED,
            uncertainty=interval(),
            evidence_ids=("ev.missing",),
        )
    )
    with pytest.raises(EpistemicStateError, match="unknown evidence"):
        machine.commit(tx)
    assert machine.snapshot() is before


def test_claim_cycles_and_asymmetric_contradictions_are_rejected():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord("claim.a", "A", ClaimStatus.PROPOSED, interval(), premises=("claim.b",))
    )
    tx.add_claim(
        ClaimRecord("claim.b", "B", ClaimStatus.PROPOSED, interval(), premises=("claim.a",))
    )
    with pytest.raises(EpistemicStateError, match="cycle"):
        machine.commit(tx)

    tx = machine.begin()
    tx.add_claim(
        ClaimRecord("claim.a", "A", ClaimStatus.PROPOSED, interval(), contradictions=("claim.b",))
    )
    tx.add_claim(ClaimRecord("claim.b", "B", ClaimStatus.PROPOSED, interval()))
    with pytest.raises(EpistemicStateError, match="symmetric"):
        machine.commit(tx)


def test_answer_cannot_depend_on_rejected_claim():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord("claim.bad", "Bad", ClaimStatus.REJECTED, interval(0.1))
    )
    tx.accept_answer(
        AcceptedAnswer(
            text="Bad",
            text_sha256=text_sha256("Bad"),
            claim_ids=("claim.bad",),
            evidence_ids=(),
            confidence=interval(0.1),
        )
    )
    with pytest.raises(EpistemicStateError, match="rejected or contradicted"):
        machine.commit(tx)


def test_stale_transactions_cannot_overwrite_newer_state():
    machine = EpistemicStateMachine(genesis())
    first = machine.begin()
    stale = machine.begin()
    first.add_claim(ClaimRecord("claim.first", "First", ClaimStatus.PROPOSED, interval()))
    machine.commit(first)
    stale.add_claim(ClaimRecord("claim.stale", "Stale", ClaimStatus.PROPOSED, interval()))
    with pytest.raises(StaleEpistemicTransactionError):
        machine.commit(stale)
    assert {claim.claim_id for claim in machine.snapshot().claims} == {"claim.first"}


def test_budget_caps_are_monotonic_and_cannot_be_changed_by_transaction():
    tx = EpistemicStateMachine(genesis()).begin()
    with pytest.raises(EpistemicStateError, match="budget caps"):
        tx.set_budget(ComputeBudgetState(total=200.0, tool_calls_total=4))
    machine = EpistemicStateMachine(genesis())
    advanced = machine.begin()
    advanced.set_budget(
        ComputeBudgetState(total=100.0, used=4.0, tool_calls_total=4)
    )
    machine.commit(advanced)
    with pytest.raises(EpistemicStateError, match="cannot refund"):
        machine.begin().set_budget(
            ComputeBudgetState(total=100.0, used=3.0, tool_calls_total=4)
        )


def test_operation_must_bind_transaction_base_hash():
    tx = EpistemicStateMachine(genesis()).begin()
    with pytest.raises(EpistemicStateError, match="input hash"):
        tx.add_operation(
            OperationRecord(
                operation_id="op.wrong",
                kind=OperationKind.COMPARE,
                outcome=OperationOutcome.UNKNOWN,
                input_state_sha256="f" * 64,
                cost=1.0,
            )
        )


def test_bounds_and_strict_types_reject_malformed_state_components():
    with pytest.raises(EpistemicStateError, match="interval"):
        ProbabilityInterval(0.8, 0.2, 0.9, "bad", 1)
    with pytest.raises(EpistemicStateError, match="boolean ground truth|numeric"):
        ProbabilityInterval(False, 0.5, 0.6, "bad", 1)  # type: ignore[arg-type]
    with pytest.raises(EpistemicStateError, match="control characters"):
        ProblemFrame.create("bad\x00objective")
    with pytest.raises(EpistemicStateError, match="sequence"):
        ProblemFrame("objective", text_sha256("objective"), "wrong", ())
    with pytest.raises(EpistemicStateError, match="sequence"):
        ClaimRecord("claim.bad", "Bad", ClaimStatus.PROPOSED, interval(), premises="x")
    with pytest.raises(EpistemicStateError, match="digest"):
        EvidenceRecord(
            "ev.bad",
            EvidenceKind.OBSERVATION,
            "summary",
            "not-a-digest",
            "source",
            1.0,
        )
