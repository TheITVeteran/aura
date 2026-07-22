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
    EvidenceProvenance,
    EvidencePurpose,
    EvidenceRecord,
    EvidenceScope,
    EvidenceVerification,
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

EPISODE_ID = "episode.test"
OBJECTIVE = "Determine whether the deployment is safe."
OBJECTIVE_SHA256 = text_sha256(OBJECTIVE)


def interval(point: float = 0.5) -> ProbabilityInterval:
    return ProbabilityInterval(
        lower=max(0.0, point - 0.1),
        point=point,
        upper=min(1.0, point + 0.1),
        method="test_wilson",
        evidence_count=1,
    )


def evidence(
    evidence_id: str,
    *,
    supports=(),
    contradicts=(),
    kind=EvidenceKind.CALCULATION,
    verification=EvidenceVerification.SOURCE_BOUND,
    purpose: EvidencePurpose | None = None,
    observed_at: float = 1.0,
    expires_at: float | None = None,
    episode_id: str = EPISODE_ID,
    objective_sha256: str = OBJECTIVE_SHA256,
):
    summary = f"Evidence for {evidence_id}"
    claim_ids = tuple(sorted((*supports, *contradicts)))
    if purpose is None:
        if kind is EvidenceKind.IMMUTABLE_PROBLEM:
            purpose = EvidencePurpose.IMMUTABLE_PROBLEM
        elif claim_ids:
            purpose = EvidencePurpose.CLAIM_TEST
        else:
            purpose = EvidencePurpose.CONTEXT_ONLY
    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        summary=summary,
        content_sha256=text_sha256(summary),
        provenance=EvidenceProvenance(
            source_id="unit_test",
            source_version="v1",
            invocation_sha256=text_sha256(f"invocation:{evidence_id}"),
            receipt_sha256=text_sha256(f"receipt:{evidence_id}"),
            verification=verification,
            verifier_id=(
                "independent_verifier" if verification is EvidenceVerification.INDEPENDENT else ""
            ),
            verifier_version=("v1" if verification is EvidenceVerification.INDEPENDENT else ""),
            verification_receipt_sha256=(
                text_sha256(f"verification:{evidence_id}")
                if verification is EvidenceVerification.INDEPENDENT
                else ""
            ),
        ),
        scope=EvidenceScope(
            episode_id=episode_id,
            objective_sha256=objective_sha256,
            claim_ids=claim_ids,
            purpose=purpose,
        ),
        observed_at=observed_at,
        expires_at=expires_at,
        supports=tuple(supports),
        contradicts=tuple(contradicts),
    )


def genesis() -> EpistemicState:
    problem = ProblemFrame.create(OBJECTIVE)
    problem_evidence = evidence("ev.problem", kind=EvidenceKind.IMMUTABLE_PROBLEM)
    return EpistemicState.genesis(
        episode_id=EPISODE_ID,
        problem=problem,
        budget=ComputeBudgetState(total=100.0, tool_calls_total=4),
        evidence=(problem_evidence,),
    )


def test_genesis_is_canonical_deeply_immutable_and_content_addressed():
    state = genesis()
    assert state.version == 0 and state.parent_sha256 == ""
    assert state.problem.immutable_evidence_ids == ("ev.problem",)
    assert len(state.state_sha256) == 64
    assert (
        EpistemicState.genesis(
            episode_id=EPISODE_ID,
            problem=ProblemFrame.create(OBJECTIVE),
            budget=ComputeBudgetState(total=100.0, tool_calls_total=4),
            evidence=(evidence("ev.problem", kind=EvidenceKind.IMMUTABLE_PROBLEM),),
        ).state_sha256
        == state.state_sha256
    )
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
            accepted_at=2.0,
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
            tx.add_claim(ClaimRecord(claim_id, claim_id, ClaimStatus.PROPOSED, interval()))
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
    tx.add_claim(ClaimRecord("claim.bad", "Bad", ClaimStatus.REJECTED, interval(0.1)))
    tx.accept_answer(
        AcceptedAnswer(
            text="Bad",
            text_sha256=text_sha256("Bad"),
            claim_ids=("claim.bad",),
            evidence_ids=(),
            confidence=interval(0.1),
            accepted_at=2.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="rejected or contradicted"):
        machine.commit(tx)


def test_claim_invalidation_revokes_descendants_hypothesis_and_answer_atomically():
    machine = EpistemicStateMachine(genesis())
    setup = machine.begin()
    setup.add_claim(
        ClaimRecord(
            "claim.root",
            "Root",
            ClaimStatus.VERIFIED,
            interval(0.9),
            answer_relevant=True,
        )
    )
    setup.add_claim(
        ClaimRecord(
            "claim.child",
            "Child",
            ClaimStatus.SUPPORTED,
            interval(0.8),
            premises=("claim.root",),
            answer_relevant=True,
        )
    )
    setup.add_claim(
        ClaimRecord(
            "claim.grandchild",
            "Grandchild",
            ClaimStatus.VERIFIED,
            interval(0.8),
            premises=("claim.child",),
            answer_relevant=True,
        )
    )
    setup.add_claim(
        ClaimRecord(
            "claim.independent",
            "Independent",
            ClaimStatus.SUPPORTED,
            interval(0.7),
        )
    )
    setup.add_hypothesis(
        HypothesisRecord(
            "hyp.main",
            "Main",
            interval(0.8),
            HypothesisStatus.FAVORED,
            ("claim.grandchild",),
        )
    )
    setup.accept_answer(
        AcceptedAnswer(
            "Grandchild",
            text_sha256("Grandchild"),
            ("claim.grandchild",),
            (),
            interval(0.8),
            2.0,
        )
    )
    base = machine.commit(setup)
    assert base.claim_descendants("claim.root") == (
        "claim.child",
        "claim.grandchild",
    )

    tx = machine.begin()
    affected = tx.invalidate_claim(
        "claim.root",
        operation_id="op.invalidate.root",
        status=ClaimStatus.CONTRADICTED,
        cost=2.0,
    )
    assert affected == ("claim.child", "claim.grandchild", "claim.root")
    state = machine.commit(tx)
    statuses = {claim.claim_id: claim.status for claim in state.claims}
    assert statuses == {
        "claim.child": ClaimStatus.UNRESOLVED,
        "claim.grandchild": ClaimStatus.UNRESOLVED,
        "claim.independent": ClaimStatus.SUPPORTED,
        "claim.root": ClaimStatus.CONTRADICTED,
    }
    assert state.hypotheses[0].status is HypothesisStatus.UNRESOLVED
    assert state.accepted_answer is None
    assert state.budget.used == 2.0
    assert state.operations[0].affected_claim_ids == affected


def test_unestablished_premises_cannot_retain_supported_descendants_or_favored_hypotheses():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.root", "Root", ClaimStatus.REJECTED, interval()))
    tx.add_claim(
        ClaimRecord(
            "claim.child",
            "Child",
            ClaimStatus.SUPPORTED,
            interval(),
            premises=("claim.root",),
        )
    )
    with pytest.raises(EpistemicStateError, match="unestablished premise"):
        machine.commit(tx)

    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.root", "Root", ClaimStatus.REJECTED, interval()))
    tx.add_hypothesis(
        HypothesisRecord(
            "hyp.bad",
            "Bad",
            interval(),
            HypothesisStatus.FAVORED,
            ("claim.root",),
        )
    )
    with pytest.raises(EpistemicStateError, match="favored hypothesis"):
        machine.commit(tx)

    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.root", "Root", ClaimStatus.PROPOSED, interval()))
    tx.add_claim(
        ClaimRecord(
            "claim.child",
            "Child",
            ClaimStatus.VERIFIED,
            interval(),
            premises=("claim.root",),
        )
    )
    with pytest.raises(EpistemicStateError, match="unestablished premise"):
        machine.commit(tx)


def test_mutually_contradictory_claims_cannot_both_be_established():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord(
            "claim.a",
            "A",
            ClaimStatus.VERIFIED,
            interval(),
            contradictions=("claim.b",),
        )
    )
    tx.add_claim(
        ClaimRecord(
            "claim.b",
            "B",
            ClaimStatus.SUPPORTED,
            interval(),
            contradictions=("claim.a",),
        )
    )
    with pytest.raises(EpistemicStateError, match="both be established"):
        machine.commit(tx)


def test_answer_requires_supported_answer_relevant_claims():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(
        ClaimRecord(
            "claim.proposed",
            "Proposed",
            ClaimStatus.PROPOSED,
            interval(),
            answer_relevant=True,
        )
    )
    tx.accept_answer(
        AcceptedAnswer(
            "Proposed",
            text_sha256("Proposed"),
            ("claim.proposed",),
            (),
            interval(),
            2.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="unresolved claims"):
        machine.commit(tx)

    with pytest.raises(EpistemicStateError, match="at least one claim"):
        AcceptedAnswer("No claim", text_sha256("No claim"), (), (), interval(), 2.0)


def test_claim_replacement_requires_successful_operation_receipt():
    machine = EpistemicStateMachine(genesis())
    setup = machine.begin()
    setup.add_claim(ClaimRecord("claim.one", "One", ClaimStatus.PROPOSED, interval()))
    machine.commit(setup)

    tx = machine.begin()
    tx.replace_claim(ClaimRecord("claim.one", "Revised", ClaimStatus.PROPOSED, interval()))
    with pytest.raises(EpistemicStateError, match="lack a successful operation"):
        machine.commit(tx)

    tx = machine.begin()
    tx.replace_claim(ClaimRecord("claim.one", "Revised", ClaimStatus.PROPOSED, interval()))
    tx.add_operation(
        OperationRecord(
            "op.revise",
            OperationKind.COMPARE,
            OperationOutcome.SUCCEEDED,
            tx.base.state_sha256,
            0.0,
            affected_claim_ids=("claim.one",),
        )
    )
    assert machine.commit(tx).claims[0].text == "Revised"


def test_failed_invalidation_does_not_partially_mutate_transaction():
    machine = EpistemicStateMachine(genesis())
    setup = machine.begin()
    setup.add_claim(ClaimRecord("claim.one", "One", ClaimStatus.PROPOSED, interval()))
    machine.commit(setup)
    tx = machine.begin()
    with pytest.raises(EpistemicStateError, match="exceeds compute budget"):
        tx.invalidate_claim(
            "claim.one",
            operation_id="op.too-expensive",
            cost=101.0,
        )
    tx.invalidate_claim("claim.one", operation_id="op.valid", cost=1.0)
    state = machine.commit(tx)
    assert state.operations[0].operation_id == "op.valid"
    assert state.budget.used == 1.0


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
    advanced.set_budget(ComputeBudgetState(total=100.0, used=4.0, tool_calls_total=4))
    machine.commit(advanced)
    with pytest.raises(EpistemicStateError, match="cannot refund"):
        machine.begin().set_budget(ComputeBudgetState(total=100.0, used=3.0, tool_calls_total=4))


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


def test_evidence_provenance_and_scope_are_strict_and_content_addressed():
    state = genesis()
    payload = state.to_dict()
    provenance = payload["evidence"][0]["provenance"]
    scope = payload["evidence"][0]["scope"]
    assert provenance == {
        "source_id": "unit_test",
        "source_version": "v1",
        "invocation_sha256": text_sha256("invocation:ev.problem"),
        "receipt_sha256": text_sha256("receipt:ev.problem"),
        "verification": "source_bound",
        "verifier_id": "",
        "verifier_version": "",
        "verification_receipt_sha256": "",
    }
    assert scope == {
        "episode_id": EPISODE_ID,
        "objective_sha256": OBJECTIVE_SHA256,
        "claim_ids": [],
        "purpose": "immutable_problem",
    }

    with_extra_provenance = copy.deepcopy(payload)
    with_extra_provenance["evidence"][0]["provenance"]["trust_me"] = True
    with pytest.raises(EpistemicStateError, match="unknown"):
        EpistemicState.from_dict(with_extra_provenance)

    with_extra_scope = copy.deepcopy(payload)
    with_extra_scope["evidence"][0]["scope"]["global"] = True
    with pytest.raises(EpistemicStateError, match="unknown"):
        EpistemicState.from_dict(with_extra_scope)


@pytest.mark.parametrize(
    ("episode_id", "objective_sha256", "message"),
    (
        ("episode.other", OBJECTIVE_SHA256, "another episode"),
        (EPISODE_ID, text_sha256("another objective"), "another objective"),
    ),
)
def test_evidence_cannot_cross_episode_or_objective_scope(
    episode_id: str,
    objective_sha256: str,
    message: str,
):
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_evidence(
        evidence(
            "ev.foreign",
            episode_id=episode_id,
            objective_sha256=objective_sha256,
        )
    )
    with pytest.raises(EpistemicStateError, match=message):
        machine.commit(tx)


def test_evidence_claim_scope_and_links_must_be_exact_and_bidirectional():
    with pytest.raises(EpistemicStateError, match="exactly match"):
        EvidenceRecord(
            evidence_id="ev.misdeclared",
            kind=EvidenceKind.CALCULATION,
            summary="Misdeclared scope",
            content_sha256=text_sha256("Misdeclared scope"),
            provenance=EvidenceProvenance(
                "calculator",
                "v1",
                text_sha256("invocation"),
                text_sha256("receipt"),
                EvidenceVerification.SOURCE_BOUND,
            ),
            scope=EvidenceScope(
                EPISODE_ID,
                OBJECTIVE_SHA256,
                ("claim.a",),
                EvidencePurpose.CLAIM_TEST,
            ),
            observed_at=1.0,
            supports=("claim.b",),
        )

    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.a", "A", ClaimStatus.PROPOSED, interval()))
    tx.add_evidence(evidence("ev.a", supports=("claim.a",)))
    with pytest.raises(EpistemicStateError, match="bidirectional"):
        machine.commit(tx)

    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_evidence(evidence("ev.context"))
    tx.add_claim(
        ClaimRecord(
            "claim.a",
            "A",
            ClaimStatus.PROPOSED,
            interval(),
            evidence_ids=("ev.context",),
        )
    )
    with pytest.raises(EpistemicStateError, match="bidirectional"):
        machine.commit(tx)


def test_memory_and_unverified_context_cannot_be_promoted_to_fact():
    with pytest.raises(EpistemicStateError, match="memory evidence is context-only"):
        evidence("ev.memory", kind=EvidenceKind.MEMORY, supports=("claim.a",))

    with pytest.raises(EpistemicStateError, match="unverified evidence"):
        evidence(
            "ev.unverified",
            supports=("claim.a",),
            verification=EvidenceVerification.UNVERIFIED,
        )

    recalled = evidence(
        "ev.recalled",
        kind=EvidenceKind.MEMORY,
        verification=EvidenceVerification.UNVERIFIED,
    )
    assert recalled.scope.purpose is EvidencePurpose.CONTEXT_ONLY
    assert recalled.supports == recalled.contradicts == ()

    reverified = evidence(
        "ev.reverified",
        kind=EvidenceKind.OBSERVATION,
        supports=("claim.a",),
        verification=EvidenceVerification.INDEPENDENT,
    )
    assert reverified.provenance.verification is EvidenceVerification.INDEPENDENT
    assert reverified.provenance.verifier_id == "independent_verifier"

    with pytest.raises(EpistemicStateError, match="requires verifier identity"):
        EvidenceProvenance(
            "producer",
            "v1",
            text_sha256("invocation"),
            text_sha256("receipt"),
            EvidenceVerification.INDEPENDENT,
        )
    with pytest.raises(EpistemicStateError, match="must differ"):
        EvidenceProvenance(
            "producer",
            "v1",
            text_sha256("invocation"),
            text_sha256("receipt"),
            EvidenceVerification.INDEPENDENT,
            "producer",
            "v2",
            text_sha256("verification"),
        )


def test_answer_requires_fresh_transitive_evidence_closure():
    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_evidence(
        evidence(
            "ev.root",
            supports=("claim.root",),
            expires_at=10.0,
        )
    )
    tx.add_evidence(evidence("ev.child", supports=("claim.child",)))
    tx.add_claim(
        ClaimRecord(
            "claim.root",
            "Root",
            ClaimStatus.VERIFIED,
            interval(0.9),
            evidence_ids=("ev.root",),
        )
    )
    tx.add_claim(
        ClaimRecord(
            "claim.child",
            "Child",
            ClaimStatus.SUPPORTED,
            interval(0.8),
            premises=("claim.root",),
            evidence_ids=("ev.child",),
            answer_relevant=True,
        )
    )
    tx.accept_answer(
        AcceptedAnswer(
            "Child",
            text_sha256("Child"),
            ("claim.child",),
            ("ev.child",),
            interval(0.8),
            2.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="transitive claim dependencies"):
        machine.commit(tx)

    tx.accept_answer(
        AcceptedAnswer(
            "Child",
            text_sha256("Child"),
            ("claim.child",),
            ("ev.child", "ev.root"),
            interval(0.8),
            11.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="stale"):
        machine.commit(tx)

    tx.accept_answer(
        AcceptedAnswer(
            "Child",
            text_sha256("Child"),
            ("claim.child",),
            ("ev.child", "ev.root"),
            interval(0.8),
            2.0,
        )
    )
    assert machine.commit(tx).accepted_answer is not None


def test_answer_rejects_future_or_unresolved_counterevidence():
    future = evidence(
        "ev.future",
        supports=("claim.future",),
        observed_at=5.0,
    )
    assert not future.is_fresh(4.0)
    assert future.is_fresh(5.0)

    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_evidence(future)
    tx.add_claim(
        ClaimRecord(
            "claim.future",
            "Future",
            ClaimStatus.VERIFIED,
            interval(0.9),
            evidence_ids=("ev.future",),
            answer_relevant=True,
        )
    )
    tx.accept_answer(
        AcceptedAnswer(
            "Future",
            text_sha256("Future"),
            ("claim.future",),
            ("ev.future",),
            interval(0.9),
            4.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="not-yet-observed"):
        machine.commit(tx)

    machine = EpistemicStateMachine(genesis())
    tx = machine.begin()
    tx.add_evidence(evidence("ev.counter", contradicts=("claim.a",)))
    tx.add_claim(
        ClaimRecord(
            "claim.a",
            "A",
            ClaimStatus.VERIFIED,
            interval(0.8),
            evidence_ids=("ev.counter",),
            answer_relevant=True,
        )
    )
    tx.accept_answer(
        AcceptedAnswer(
            "A",
            text_sha256("A"),
            ("claim.a",),
            ("ev.counter",),
            interval(0.8),
            2.0,
        )
    )
    with pytest.raises(EpistemicStateError, match="contradictory evidence"):
        machine.commit(tx)


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
            evidence_id="ev.bad",
            kind=EvidenceKind.OBSERVATION,
            summary="summary",
            content_sha256="not-a-digest",
            provenance=EvidenceProvenance(
                "unit_test",
                "v1",
                text_sha256("invocation"),
                text_sha256("receipt"),
                EvidenceVerification.SOURCE_BOUND,
            ),
            scope=EvidenceScope(
                EPISODE_ID,
                OBJECTIVE_SHA256,
                (),
                EvidencePurpose.CONTEXT_ONLY,
            ),
            observed_at=1.0,
        )
