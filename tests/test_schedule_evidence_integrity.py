"""Promotion evidence must say what it actually proves.

bf285ef2 — the binding is an UNKEYED hash over caller-supplied values, so
anyone who can write the ledger can change an outcome and recompute it. That
is an integrity checksum, not tamper evidence, and it used to be described as
the latter.

78c85746 — the binding named the candidate but not the baseline, so trials
against DIFFERENT defaults aggregated as one comparator.

487e7c0a — held_out and contamination_scan_passed are self-asserted booleans
and receipt fields were only checked to LOOK like digests; nothing was ever
resolved.
"""
from __future__ import annotations

import hashlib

import pytest

from core.brain.llm.latent_cortex.schedules import (
    PairedScheduleOutcome,
    ScheduleComputeReceipt,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _outcome(**overrides):
    kwargs = dict(
        schedule_hash=_digest("candidate"),
        domain="math",
        task_id="task-1",
        task_commitment_sha256=_digest("commit-1"),
        candidate_success=True,
        default_success=False,
        candidate_compute=ScheduleComputeReceipt(100, _digest("est")),
        default_compute=ScheduleComputeReceipt(100, _digest("est")),
        run_order="candidate_first",
        held_out=True,
        contamination_scan_passed=True,
        scorer_receipt_sha256=_digest("scorer"),
        verifier_receipt_sha256=_digest("verifier"),
        evaluation_run_id="run-1",
        evaluator_build_sha256=_digest("build"),
        model_checkpoint_sha256=_digest("ckpt"),
        evidence_protocol_sha256=_digest("protocol"),
        default_schedule_hash=_digest("baseline"),
    )
    kwargs.update(overrides)
    return PairedScheduleOutcome.create(**kwargs)


# --- 78c85746: the baseline is identified --------------------------------


def test_the_baseline_is_part_of_the_evidence():
    outcome = _outcome()

    assert outcome.default_schedule_hash == _digest("baseline")
    assert "default_schedule_hash" in outcome.to_dict()


def test_an_outcome_without_a_baseline_is_refused():
    with pytest.raises(ValueError, match="default_schedule_hash"):
        _outcome(default_schedule_hash="")


def test_a_schedule_cannot_be_paired_against_itself():
    """Otherwise it is credited with wins over its own results."""
    same = _digest("same")
    with pytest.raises(ValueError, match="identical"):
        _outcome(schedule_hash=same, default_schedule_hash=same)


def test_the_baseline_is_covered_by_the_binding():
    outcome = _outcome()
    tampered = outcome.to_dict()
    tampered["default_schedule_hash"] = _digest("a-different-baseline")

    with pytest.raises(ValueError, match="binding does not match"):
        PairedScheduleOutcome.from_dict(
            tampered, schedule_hash=_digest("candidate"), domain="math"
        )


# --- 487e7c0a: asserted is distinguished from verified -------------------


def test_receipts_are_unresolved_when_no_resolver_is_supplied():
    """Nothing was checked, and the outcome says so rather than implying it."""
    outcome = _outcome()

    assert outcome.receipts_resolved is False
    assert outcome.verified_provenance() is False


def test_a_resolver_that_accepts_marks_the_receipts_resolved():
    outcome = _outcome(receipt_resolver=lambda _kind, _digest_value: True)

    assert outcome.receipts_resolved is True


def test_a_resolver_that_rejects_leaves_them_unresolved():
    outcome = _outcome(receipt_resolver=lambda _kind, _digest_value: False)

    assert outcome.receipts_resolved is False


def test_a_broken_resolver_does_not_count_as_resolution():
    def _boom(_kind, _digest_value):
        raise RuntimeError("store offline")

    assert _outcome(receipt_resolver=_boom).receipts_resolved is False


def test_self_asserted_flags_are_still_required_but_prove_nothing():
    """They stop an outcome that ADMITS contamination and nothing else."""
    with pytest.raises(ValueError, match="held out"):
        _outcome(held_out=False)
    with pytest.raises(ValueError, match="contamination"):
        _outcome(contamination_scan_passed=False)


# --- bf285ef2: the binding does not claim more than it is ----------------


def test_the_binding_is_labelled_a_checksum_not_tamper_evidence():
    assert _outcome().evidence_authenticity == "unsigned_checksum"


def test_unsigned_evidence_is_never_verified_provenance():
    """An unkeyed hash the producer can recompute is not third-party proof."""
    outcome = _outcome(receipt_resolver=lambda _k, _d: True)

    assert outcome.receipts_resolved is True
    assert outcome.verified_provenance() is False   # still unsigned


def test_verification_metadata_is_not_part_of_the_binding():
    """Otherwise a producer could alter the binding by claiming it verified
    itself."""
    outcome = _outcome()
    payload = outcome.to_dict()
    payload["receipts_resolved"] = True
    payload["evidence_authenticity"] = "signed"

    restored = PairedScheduleOutcome.from_dict(
        payload, schedule_hash=_digest("candidate"), domain="math"
    )

    assert restored.receipts_resolved is True   # carried, not bound


def test_the_class_docstring_no_longer_claims_tamper_evidence():
    """The SUMMARY line is the claim; the body necessarily quotes the phrase
    it retracts in order to explain it."""
    summary = (PairedScheduleOutcome.__doc__ or "").strip().splitlines()[0]

    assert "tamper-evident" not in summary
