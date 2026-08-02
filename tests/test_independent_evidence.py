"""Aura's own testimony is not proof that Aura works.

The facade could check a worker receipt was well-formed and internally
consistent. That is a real check and it is not independence — the worker
and the facade share state and incentives, so "I did well" was graded by
something with a stake in the answer.

The refusal is the feature. There must be no path through this module that
returns "certified" because nothing objected.
"""
from __future__ import annotations

import pytest

from core.brain.verification.independent_evidence import (
    ClaimClass,
    ControlResult,
    EvidenceAdjudicationError,
    EvidenceBundle,
    VerdictStatus,
    VerificationGrade,
    VerifierExecution,
    adjudicate,
    required_evidence,
    summarize,
    verdict_signature_valid,
)


def _promotable(**over) -> EvidenceBundle:
    """A bundle that satisfies ADAPTER_PROMOTION, so tests can remove one thing."""
    base = dict(
        claim=ClaimClass.ADAPTER_PROMOTION,
        subject_identity="worker-a",
        model_hashes={
            "base_model": "aaa",
            "adapter": "bbb",
            "tokenizer": "ccc",
            "runtime": "ddd",
        },
        signed_attestation="sig",
        attestation_verified=True,
        raw_candidate="the model's actual output",
        verifier_identity="grader-b",
        verifier_code_hash="v1hash",
        verifier_execution=VerifierExecution.SEPARATE_PROCESS,
        verifier_score=0.90,
        treatment_compute_tokens=1000,
        controls=(
            ControlResult("vanilla", 0.70, 1000),
            ControlResult("equal_compute", 0.72, 1000),
        ),
        held_out_score=0.81,
        held_out_baseline=0.75,
    )
    base.update(over)
    return EvidenceBundle(**base)


# ------------------------------------------------------------------- refusal


def test_an_empty_bundle_never_certifies():
    for claim in ClaimClass:
        verdict = adjudicate(EvidenceBundle(claim=claim))
        assert verdict.status is not VerdictStatus.CERTIFIED, claim
        assert verdict.grade is VerificationGrade.NONE


def test_the_verdict_names_exactly_what_is_missing():
    """An operator has to know what to go and produce."""
    verdict = adjudicate(EvidenceBundle(claim=ClaimClass.ADAPTER_PROMOTION))
    assert set(verdict.missing) == set(required_evidence(ClaimClass.ADAPTER_PROMOTION))


def test_a_fully_evidenced_promotion_certifies():
    """The control: this must be reachable, or the module only ever says no."""
    verdict = adjudicate(_promotable())
    assert verdict.status is VerdictStatus.CERTIFIED
    assert verdict.grade is VerificationGrade.COUNTERFACTUALLY_VERIFIED


@pytest.mark.parametrize(
    "removal",
    [
        {"model_hashes": None},
        {"signed_attestation": None},
        {"raw_candidate": None},
        {"verifier_code_hash": None},
        {"verifier_execution": VerifierExecution.IN_PROCESS},
        {"controls": ()},
        {"held_out_score": None},
    ],
)
def test_removing_any_single_requirement_blocks_certification(removal):
    """No requirement is inferred from its neighbours passing."""
    assert adjudicate(_promotable(**removal)).status is not VerdictStatus.CERTIFIED


# --------------------------------------------------------------- independence


def test_the_subject_may_not_grade_itself():
    """The single most important line in the module."""
    verdict = adjudicate(
        _promotable(subject_identity="worker-a", verifier_identity="worker-a")
    )
    assert verdict.status is VerdictStatus.REFUTED
    assert "subject_graded_itself" in verdict.contradictions


def test_an_in_process_verifier_is_never_promotion_grade():
    verdict = adjudicate(_promotable(verifier_execution=VerifierExecution.IN_PROCESS))
    assert "out_of_process_verifier" in verdict.missing


def test_a_score_without_the_raw_candidate_is_not_a_regrade():
    """A supplied score is the worker's claim; only an output can be graded again."""
    assert "raw_candidate" in adjudicate(_promotable(raw_candidate="  ")).missing


def test_an_unchecked_signature_is_just_a_string():
    verdict = adjudicate(_promotable(attestation_verified=False))
    assert "signed_attestation" in verdict.missing


def test_partial_model_hashes_do_not_count_as_pinned():
    """Worse than none: reads as pinned while the changed thing is unrecorded."""
    verdict = adjudicate(_promotable(model_hashes={"base_model": "aaa"}))
    assert "model_hashes" in verdict.missing


# ------------------------------------------------------------------ controls


def test_a_control_given_less_compute_is_not_a_control():
    verdict = adjudicate(
        _promotable(
            controls=(ControlResult("vanilla", 0.70, 200), ControlResult("equal_compute", 0.70, 200)),
            treatment_compute_tokens=1000,
        )
    )
    assert "equal_compute_control" in verdict.missing


def test_not_beating_vanilla_is_refuted_not_merely_insufficient():
    """Somebody showed the opposite. That is a different fact from silence."""
    verdict = adjudicate(
        _promotable(
            verifier_score=0.60,
            controls=(ControlResult("vanilla", 0.70, 1000), ControlResult("equal_compute", 0.70, 1000)),
        )
    )
    assert verdict.status is VerdictStatus.REFUTED
    assert any("did_not_beat_vanilla" in reason for reason in verdict.contradictions)


def test_held_out_not_improving_is_refuted():
    verdict = adjudicate(_promotable(held_out_score=0.70, held_out_baseline=0.75))
    assert verdict.status is VerdictStatus.REFUTED


def test_missing_and_refuted_are_never_merged():
    """'We did not look' and 'we looked and it broke' call for different responses."""
    missing = adjudicate(EvidenceBundle(claim=ClaimClass.ADAPTER_PROMOTION))
    refuted = adjudicate(_promotable(held_out_score=0.10))
    assert missing.status is VerdictStatus.INSUFFICIENT
    assert refuted.status is VerdictStatus.REFUTED


# ------------------------------------------------------------------- tiering


def test_ordinary_turns_are_not_held_to_the_promotion_bar():
    """A check people route around protects nothing."""
    verdict = adjudicate(
        EvidenceBundle(
            claim=ClaimClass.TURN_QUALITY,
            subject_identity="worker-a",
            raw_candidate="an answer",
            verifier_identity="grader-b",
            verifier_score=0.8,
        )
    )
    assert verdict.status is VerdictStatus.CERTIFIED
    assert verdict.grade is VerificationGrade.POSTCONDITION_VERIFIED


def test_the_bar_rises_with_the_consequence():
    turn = set(required_evidence(ClaimClass.TURN_QUALITY))
    durable = set(required_evidence(ClaimClass.DURABLE_LEARNING))
    promotion = set(required_evidence(ClaimClass.ADAPTER_PROMOTION))
    assert turn < durable < promotion


def test_a_benchmark_claim_earns_the_highest_grade():
    assert adjudicate(_promotable(claim=ClaimClass.BENCHMARK_CLAIM)).grade is (
        VerificationGrade.EXTERNALLY_VERIFIED
    )


# ------------------------------------------------------------------- signing


def test_a_verdict_cannot_be_edited_into_a_pass():
    verdict = adjudicate(EvidenceBundle(claim=ClaimClass.TURN_QUALITY))
    assert verdict_signature_valid(verdict)

    from dataclasses import replace

    forged = replace(verdict, status=VerdictStatus.CERTIFIED)
    assert not verdict_signature_valid(forged)


def test_an_unsigned_verdict_is_not_valid():
    """Deleting a signature is what an attacker would do; it is not a pass."""
    from dataclasses import replace

    verdict = adjudicate(_promotable())
    assert not verdict_signature_valid(replace(verdict, signature=""))


# ------------------------------------------------------------------- misuse


def test_a_malformed_submission_is_rejected_not_adjudicated():
    with pytest.raises(EvidenceAdjudicationError):
        adjudicate({"claim": "adapter_promotion"})  # type: ignore[arg-type]


def test_the_summary_reports_what_evidence_is_habitually_absent():
    verdicts = [adjudicate(EvidenceBundle(claim=ClaimClass.ADAPTER_PROMOTION)) for _ in range(3)]
    summary = summarize(verdicts)
    assert summary["verdicts"]["insufficient"] == 3
    assert summary["most_common_missing_evidence"][0][1] == 3


# ------------------------------------------------- the seam into governance
#
# The evidence service is only worth building if something REFUSES on its
# verdict. These pin that it gates the two strongest durable-learning
# grades, so a caller can no longer simply assert them.


def _learning_update(**over):
    from core.governance.durable_learning import LearningUpdate

    base = dict(
        subsystem="mycelium",
        key="route-1",
        operation="reinforce",
        success=True,
        grade=VerificationGrade.COUNTERFACTUALLY_VERIFIED,
        verifier="grader-b",
        evidence_id="ev-1",
        inverse={"operation": "set_confidence", "confidence": 0.5},
    )
    base.update(over)
    return LearningUpdate(**base)


def test_a_top_grade_asserted_without_a_verdict_stays_session_local():
    from core.governance.durable_learning import DurableLearningGate, LearningScope

    admission = DurableLearningGate().admit(_learning_update(verdict=None))
    assert admission.scope is LearningScope.SESSION
    assert admission.reason == "top_grade_claimed_without_an_independent_verdict"


def test_a_top_grade_backed_by_a_certified_verdict_persists():
    """The control: the strongest tier must remain reachable."""
    from core.governance.durable_learning import DurableLearningGate, LearningScope

    verdict = adjudicate(_promotable())
    assert verdict.status is VerdictStatus.CERTIFIED
    admission = DurableLearningGate().admit(_learning_update(verdict=verdict))
    assert admission.scope is LearningScope.DURABLE


def test_a_forged_verdict_does_not_unlock_the_top_grade():
    from dataclasses import replace

    from core.governance.durable_learning import DurableLearningGate, LearningScope

    insufficient = adjudicate(EvidenceBundle(claim=ClaimClass.ADAPTER_PROMOTION))
    forged = replace(insufficient, status=VerdictStatus.CERTIFIED)
    admission = DurableLearningGate().admit(_learning_update(verdict=forged))
    assert admission.scope is LearningScope.SESSION
    assert admission.reason == "verdict_signature_invalid"


def test_a_verdict_that_proves_less_than_claimed_does_not_unlock_the_grade():
    from core.governance.durable_learning import DurableLearningGate, LearningScope

    weaker = adjudicate(
        EvidenceBundle(
            claim=ClaimClass.TURN_QUALITY,
            subject_identity="worker-a",
            raw_candidate="an answer",
            verifier_identity="grader-b",
            verifier_score=0.8,
        )
    )
    assert weaker.status is VerdictStatus.CERTIFIED
    admission = DurableLearningGate().admit(
        _learning_update(grade=VerificationGrade.EXTERNALLY_VERIFIED, verdict=weaker)
    )
    assert admission.scope is LearningScope.SESSION
    assert admission.reason == "verdict_grade_below_the_claimed_grade"


def test_the_lower_tiers_are_unaffected_and_still_need_no_verdict():
    """Requiring a verdict everywhere would make the gate something to route around."""
    from core.governance.durable_learning import DurableLearningGate, LearningScope

    admission = DurableLearningGate().admit(
        _learning_update(grade=VerificationGrade.POSTCONDITION_VERIFIED, verdict=None)
    )
    assert admission.scope is LearningScope.DURABLE
