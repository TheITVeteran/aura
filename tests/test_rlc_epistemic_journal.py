"""Crash recovery contracts for the RLC epistemic write-ahead journal."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.epistemic_calibration import (
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationProfile,
)
from core.brain.llm.latent_cortex.epistemic_journal import (
    EPISTEMIC_JOURNAL_SCHEMA,
    EpistemicJournalError,
    EpistemicStateJournal,
)
from core.brain.llm.latent_cortex.epistemic_state import (
    ClaimRecord,
    ClaimStatus,
    ComputeBudgetState,
    EpistemicState,
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
    text_sha256,
)
from core.runtime.file_write_gateway import FileWriteGateway


def interval() -> ProbabilityInterval:
    return ProbabilityInterval(0.4, 0.5, 0.6, "uncalibrated_test_interval", 0)


def calibration_profile() -> CalibrationProfile:
    rows = tuple(
        CalibrationObservation(
            observation_id=f"journal.obs.{index}",
            domain="general",
            predicted_probability=0.1 if index < 2 else 0.9,
            outcome=index >= 2,
            prediction_receipt_sha256=text_sha256(f"journal.prediction:{index}"),
            outcome_receipt_sha256=text_sha256(f"journal.outcome:{index}"),
            outcome_verifier_id="journal_exact_grader",
            outcome_verifier_version="v1",
            observed_at=float(index + 1),
        )
        for index in range(4)
    )
    return CalibrationProfile.fit(
        profile_id="cal.journal.v1",
        estimator_id="journal_claim_head",
        estimator_version="v1",
        domain="general",
        dataset_sha256=text_sha256("journal-dataset"),
        split_manifest_sha256=text_sha256("journal-split"),
        trained_at=10.0,
        expires_at=100.0,
        observations=rows,
        policy=CalibrationPolicy(
            bins=2,
            min_samples=4,
            min_bin_samples=2,
            max_brier=0.1,
            max_ece=0.11,
            support_lower_bound=0.5,
        ),
    )


def genesis(*, objective: str = "Recover the exact epistemic state.") -> EpistemicState:
    summary = "Immutable problem observation"
    problem = ProblemFrame.create(objective)
    return EpistemicState.genesis(
        episode_id="episode.journal",
        problem=problem,
        budget=ComputeBudgetState(total=100.0, tool_calls_total=4),
        evidence=(
            EvidenceRecord(
                "ev.problem",
                EvidenceKind.IMMUTABLE_PROBLEM,
                summary,
                text_sha256(summary),
                EvidenceProvenance(
                    "unit_test",
                    "v1",
                    text_sha256("invocation:problem"),
                    text_sha256("receipt:problem"),
                    EvidenceVerification.SOURCE_BOUND,
                ),
                EvidenceScope(
                    "episode.journal",
                    problem.objective_sha256,
                    (),
                    EvidencePurpose.IMMUTABLE_PROBLEM,
                ),
                1.0,
            ),
        ),
    )


def add_claim(machine: EpistemicStateMachine, claim_id: str) -> EpistemicState:
    tx = machine.begin()
    tx.add_claim(ClaimRecord(claim_id, claim_id, ClaimStatus.PROPOSED, interval()))
    return machine.commit(tx)


def test_gateway_write_if_absent_never_replaces_the_winner(tmp_path: Path):
    gateway = FileWriteGateway()
    target = tmp_path / "winner.bin"
    assert gateway.write_bytes_if_absent(target, b"first", source="unit") is True
    assert gateway.write_bytes_if_absent(target, b"second", source="unit") is False
    assert target.read_bytes() == b"first"


def test_journal_initializes_recovers_and_continues_exact_state(tmp_path: Path):
    path = tmp_path / "epistemic.jsonl"
    initial = genesis()
    journal = EpistemicStateJournal(path)
    machine = EpistemicStateMachine(initial, persistence=journal)
    committed = add_claim(machine, "claim.first")

    lines = path.read_bytes().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["schema"] == EPISTEMIC_JOURNAL_SCHEMA
    assert first["sequence"] == 0
    assert second["sequence"] == 1
    assert second["previous_entry_sha256"] == first["entry_sha256"]
    assert second["state_sha256"] == committed.state_sha256

    recovered_journal = EpistemicStateJournal(path)
    recovered = EpistemicStateMachine(initial, persistence=recovered_journal)
    assert recovered.snapshot() == committed
    assert recovered_journal.last_recovery is not None
    assert recovered_journal.last_recovery.entry_count == 2
    continued = add_claim(recovered, "claim.second")
    assert continued.version == 2
    assert {claim.claim_id for claim in continued.claims} == {
        "claim.first",
        "claim.second",
    }


def test_journal_persists_calibration_and_rejects_history_rewrite(tmp_path: Path):
    path = tmp_path / "calibration.jsonl"
    initial = genesis()
    journal = EpistemicStateJournal(path)
    machine = EpistemicStateMachine(initial, persistence=journal)
    tx = machine.begin()
    tx.add_calibration(calibration_profile())
    committed = machine.commit(tx)
    assert committed.calibrations[0].profile_id == "cal.journal.v1"

    recovered = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    assert recovered.snapshot() == committed

    candidate = EpistemicState._build(
        episode_id=committed.episode_id,
        version=committed.version + 1,
        parent_sha256=committed.state_sha256,
        problem=committed.problem,
        calibrations=(),
        evidence=committed.evidence,
        hypotheses=committed.hypotheses,
        claims=committed.claims,
        operations=committed.operations,
        budget=committed.budget,
        accepted_answer=committed.accepted_answer,
    )
    with pytest.raises(
        EpistemicJournalError,
        match="journal_calibration_history_rewritten",
    ):
        journal.append(expected_base=committed, candidate=candidate)


def test_journal_rejects_hypothesis_revision_without_successful_operation(tmp_path: Path):
    path = tmp_path / "hypothesis-rewrite.jsonl"
    initial = genesis()
    journal = EpistemicStateJournal(path)
    machine = EpistemicStateMachine(initial, persistence=journal)
    tx = machine.begin()
    tx.add_hypothesis(
        HypothesisRecord(
            "hyp.one", "One", ProbabilityInterval(0.9, 1.0, 1.0, "test", 0), HypothesisStatus.ACTIVE
        )
    )
    base = machine.commit(tx)
    revised = HypothesisRecord(
        "hyp.one",
        "One",
        ProbabilityInterval(0.9, 1.0, 1.0, "test", 0),
        HypothesisStatus.UNRESOLVED,
    )
    candidate = EpistemicState._build(
        episode_id=base.episode_id,
        version=base.version + 1,
        parent_sha256=base.state_sha256,
        problem=base.problem,
        calibrations=base.calibrations,
        evidence=base.evidence,
        hypotheses=(revised,),
        claims=base.claims,
        operations=base.operations,
        budget=base.budget,
        accepted_answer=base.accepted_answer,
    )
    with pytest.raises(
        EpistemicJournalError,
        match="journal_hypothesis_revision_without_operation",
    ):
        journal.append(expected_base=base, candidate=candidate)

    covered = EpistemicState._build(
        episode_id=base.episode_id,
        version=base.version + 1,
        parent_sha256=base.state_sha256,
        problem=base.problem,
        calibrations=base.calibrations,
        evidence=base.evidence,
        hypotheses=(revised,),
        claims=base.claims,
        operations=(
            OperationRecord(
                "op.revise.hypothesis",
                OperationKind.COMPARE,
                OperationOutcome.SUCCEEDED,
                base.state_sha256,
                0.0,
                affected_hypothesis_ids=("hyp.one",),
            ),
        ),
        budget=base.budget,
        accepted_answer=base.accepted_answer,
    )
    journal.append(expected_base=base, candidate=covered)
    recovered = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    assert recovered.snapshot() == covered


def test_journal_rejects_unreceipted_addition_to_existing_portfolio(tmp_path: Path):
    path = tmp_path / "hypothesis-addition.jsonl"
    initial = genesis()
    journal = EpistemicStateJournal(path)
    machine = EpistemicStateMachine(initial, persistence=journal)
    tx = machine.begin()
    tx.add_hypothesis(
        HypothesisRecord(
            "hyp.one",
            "One",
            ProbabilityInterval(0.9, 1.0, 1.0, "test", 0),
            HypothesisStatus.ACTIVE,
        )
    )
    base = machine.commit(tx)
    candidate = EpistemicState._build(
        episode_id=base.episode_id,
        version=base.version + 1,
        parent_sha256=base.state_sha256,
        problem=base.problem,
        calibrations=base.calibrations,
        evidence=base.evidence,
        hypotheses=(
            HypothesisRecord(
                "hyp.one",
                "One",
                ProbabilityInterval(0.9, 1.0, 1.0, "test", 0),
                HypothesisStatus.ACTIVE,
            ),
            HypothesisRecord(
                "hyp.two",
                "Two",
                ProbabilityInterval(0.0, 0.0, 0.0, "test", 0),
                HypothesisStatus.REFUTED,
                ("claim.blocked",),
            ),
        ),
        claims=(ClaimRecord("claim.blocked", "Blocked", ClaimStatus.REJECTED, interval()),),
        operations=base.operations,
        budget=base.budget,
        accepted_answer=base.accepted_answer,
    )
    with pytest.raises(
        EpistemicJournalError,
        match="journal_hypothesis_addition_without_operation",
    ):
        journal.append(expected_base=base, candidate=candidate)


@pytest.mark.asyncio
async def test_async_commit_keeps_fsync_off_the_event_loop(tmp_path: Path):
    machine = EpistemicStateMachine(
        genesis(),
        persistence=EpistemicStateJournal(tmp_path / "async.jsonl"),
    )
    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.async", "Async", ClaimStatus.PROPOSED, interval()))
    committed = await machine.commit_async(tx)
    assert committed.version == 1
    assert machine.snapshot() is committed


def test_partial_append_never_publishes_and_is_repaired_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "partial.jsonl"
    journal = EpistemicStateJournal(path)
    machine = EpistemicStateMachine(genesis(), persistence=journal)
    before = machine.snapshot()
    tx = machine.begin()
    tx.add_claim(ClaimRecord("claim.retry", "Retry", ClaimStatus.PROPOSED, interval()))
    original = journal._write_record

    def torn_write(handle, payload):
        handle.write(payload[:37])
        handle.flush()
        os.fsync(handle.fileno())
        raise EpistemicJournalError("injected_partial_append")

    monkeypatch.setattr(journal, "_write_record", torn_write)
    with pytest.raises(EpistemicJournalError, match="injected_partial_append"):
        machine.commit(tx)
    assert machine.snapshot() is before

    monkeypatch.setattr(journal, "_write_record", original)
    committed = machine.commit(tx)
    assert committed.version == 1
    assert journal.last_recovery is not None
    assert journal.last_recovery.repaired_torn_tail_bytes == 37
    assert path.read_bytes().endswith(b"\n")


def test_torn_tail_recovery_can_be_required_to_fail_closed(tmp_path: Path):
    path = tmp_path / "strict-tail.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    with path.open("ab") as handle:
        handle.write(b'{"partial":')
    strict = EpistemicStateJournal(path, repair_torn_tail=False)
    with pytest.raises(EpistemicJournalError, match="journal_torn_tail"):
        EpistemicStateMachine(genesis(), persistence=strict)


def test_complete_corruption_is_never_rolled_back_as_a_torn_tail(tmp_path: Path):
    path = tmp_path / "corrupt.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    with path.open("ab") as handle:
        handle.write(b"{}\n")
    before = path.read_bytes()
    with pytest.raises(EpistemicJournalError, match="journal_entry_shape_invalid"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    assert path.read_bytes() == before


def test_corrupt_complete_prefix_is_not_mutated_when_a_torn_tail_follows(
    tmp_path: Path,
):
    path = tmp_path / "corrupt-before-tail.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    with path.open("ab") as handle:
        handle.write(b"{}\npartial")
    before = path.read_bytes()
    with pytest.raises(EpistemicJournalError, match="journal_entry_shape_invalid"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    assert path.read_bytes() == before


def test_noncanonical_crlf_framing_is_rejected(tmp_path: Path):
    path = tmp_path / "crlf.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(EpistemicJournalError, match="journal_entry_noncanonical"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))


def test_entry_tampering_and_wrong_external_genesis_are_rejected(tmp_path: Path):
    path = tmp_path / "tampered.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))
    record = json.loads(path.read_text().splitlines()[0])
    record["state"]["problem"]["objective"] = "Rewritten trust root."
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(EpistemicJournalError, match="journal_entry_hash_mismatch"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(path))

    clean = tmp_path / "wrong-genesis.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(clean))
    with pytest.raises(EpistemicJournalError, match="journal_genesis_mismatch"):
        EpistemicStateMachine(
            genesis(objective="Different objective."),
            persistence=EpistemicStateJournal(clean),
        )


def test_rehashed_forgery_cannot_break_chain_or_rewrite_history(tmp_path: Path):
    path = tmp_path / "forged-chain.jsonl"
    initial = genesis()
    machine = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    add_claim(machine, "claim.first")
    lines = path.read_text().splitlines()
    forged = json.loads(lines[1])
    forged["previous_entry_sha256"] = "0" * 64
    base = {key: value for key, value in forged.items() if key != "entry_sha256"}
    encoded_base = json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    forged["entry_sha256"] = hashlib.sha256(encoded_base).hexdigest()
    lines[1] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(EpistemicJournalError, match="journal_hash_chain_invalid"):
        EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))

    history_path = tmp_path / "rewritten-history.jsonl"
    history_journal = EpistemicStateJournal(history_path)
    history_machine = EpistemicStateMachine(initial, persistence=history_journal)
    tx = history_machine.begin()
    summary = "Transient evidence"
    tx.add_evidence(
        EvidenceRecord(
            "ev.transient",
            EvidenceKind.OBSERVATION,
            summary,
            text_sha256(summary),
            EvidenceProvenance(
                "unit_test",
                "v1",
                text_sha256("invocation:transient"),
                text_sha256("receipt:transient"),
                EvidenceVerification.SOURCE_BOUND,
            ),
            EvidenceScope(
                "episode.journal",
                initial.problem.objective_sha256,
                (),
                EvidencePurpose.CONTEXT_ONLY,
            ),
            2.0,
        )
    )
    base_state = history_machine.commit(tx)
    candidate = EpistemicState._build(
        episode_id=base_state.episode_id,
        version=base_state.version + 1,
        parent_sha256=base_state.state_sha256,
        problem=base_state.problem,
        calibrations=base_state.calibrations,
        evidence=(base_state.evidence[0],),
        hypotheses=base_state.hypotheses,
        claims=base_state.claims,
        operations=base_state.operations,
        budget=base_state.budget,
        accepted_answer=base_state.accepted_answer,
    )
    with pytest.raises(
        EpistemicJournalError,
        match="journal_evidence_history_rewritten",
    ):
        history_journal.append(expected_base=base_state, candidate=candidate)


def test_journal_rejects_symlinks_without_touching_target(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"unchanged")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(EpistemicJournalError, match="journal_path_symlink_rejected"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(link))
    assert target.read_bytes() == b"unchanged"

    valid = tmp_path / "valid.jsonl"
    EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(valid))
    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"unchanged")
    lock_path = valid.with_name(f".{valid.name}.lock")
    lock_path.unlink()
    lock_path.symlink_to(lock_target)
    with pytest.raises(EpistemicJournalError, match="journal_lock_failed"):
        EpistemicStateMachine(genesis(), persistence=EpistemicStateJournal(valid))
    assert lock_target.read_bytes() == b"unchanged"

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EpistemicJournalError, match="journal_path_symlink_rejected"):
        EpistemicStateMachine(
            genesis(),
            persistence=EpistemicStateJournal(linked_parent / "escaped.jsonl"),
        )
    assert not (real_parent / "escaped.jsonl").exists()


def test_stale_second_writer_cannot_fork_the_durable_chain(tmp_path: Path):
    path = tmp_path / "single-head.jsonl"
    initial = genesis()
    first = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    second = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    add_claim(first, "claim.first")

    stale = second.begin()
    stale.add_claim(ClaimRecord("claim.stale", "Stale", ClaimStatus.PROPOSED, interval()))
    with pytest.raises(EpistemicJournalError, match="journal_base_is_not_head"):
        second.commit(stale)
    assert second.snapshot().version == 0

    recovered = EpistemicStateMachine(initial, persistence=EpistemicStateJournal(path))
    assert {claim.claim_id for claim in recovered.snapshot().claims} == {"claim.first"}


def test_append_requires_a_successful_trusted_bootstrap(tmp_path: Path):
    journal = EpistemicStateJournal(tmp_path / "not-open.jsonl")
    initial = genesis()
    local = EpistemicStateMachine(initial)
    candidate = add_claim(local, "claim.local")
    with pytest.raises(EpistemicJournalError, match="journal_not_bootstrapped"):
        journal.append(expected_base=initial, candidate=candidate)
