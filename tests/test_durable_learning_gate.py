"""Aura may not learn permanently from her own testimony.

The failures behind this were one shape: a component's report about
itself, accepted as a checked result. Mycelium strengthened a route
because nothing threw, even when the tool returned a failure — so the
route became durably more trusted for producing a failed outcome.
"""
from __future__ import annotations

import pytest

from core.governance.durable_learning import (
    Admission,
    DurableLearningGate,
    LearningScope,
    LearningUpdate,
    VerificationGrade,
    grade_from_evidence,
)


def _update(**kwargs) -> LearningUpdate:
    base = {
        "subsystem": "mycelium",
        "key": "route-1",
        "operation": "reinforce",
        "success": True,
        "grade": VerificationGrade.POSTCONDITION_VERIFIED,
        "verifier": "postcondition_checker",
        "evidence_id": "ev-1",
        "inverse": {"operation": "set_confidence", "confidence": 0.5},
    }
    base.update(kwargs)
    return LearningUpdate(**base)


# ----------------------------------------------------------------- the core rule


def test_an_unverified_success_never_becomes_durable():
    """'Nothing threw' is the claim under test, not evidence for it."""
    gate = DurableLearningGate()
    admission = gate.admit(_update(grade=VerificationGrade.ASSERTED))
    assert admission.scope is LearningScope.SESSION
    assert admission.applies_now, "it should still steer this session"
    assert not gate.durable_updates()


def test_an_observed_success_is_still_not_enough_to_persist():
    gate = DurableLearningGate()
    assert gate.admit(_update(grade=VerificationGrade.OBSERVED)).scope is (
        LearningScope.SESSION
    )


def test_a_postcondition_verified_success_may_persist():
    """The control: verified learning must actually work."""
    gate = DurableLearningGate()
    admission = gate.admit(_update())
    assert admission.scope is LearningScope.DURABLE
    assert len(gate.durable_updates()) == 1


def test_failures_are_held_to_a_lower_bar_than_successes():
    """Asymmetric on purpose: a broken route must not survive on ceremony."""
    gate = DurableLearningGate()
    failure = gate.admit(_update(success=False, grade=VerificationGrade.OBSERVED))
    success = gate.admit(_update(success=True, grade=VerificationGrade.OBSERVED))
    assert failure.scope is LearningScope.DURABLE
    assert success.scope is LearningScope.SESSION


def test_an_unattributed_durable_update_is_downgraded():
    """A belief nobody signed cannot be invalidated, so it is not believed."""
    gate = DurableLearningGate()
    admission = gate.admit(_update(verifier=None))
    assert admission.scope is LearningScope.SESSION
    assert admission.reason == "durable_grade_but_no_named_verifier"


def test_an_irreversible_durable_update_is_downgraded():
    gate = DurableLearningGate()
    admission = gate.admit(_update(inverse=None))
    assert admission.scope is LearningScope.SESSION
    assert "no_inverse" in admission.reason


# ------------------------------------------------------------------- quarantine


def test_data_that_trains_a_judge_is_quarantined_however_well_verified():
    """A bad datum that teaches the judge certifies its own successors."""
    gate = DurableLearningGate()
    admission = gate.admit(
        _update(subsystem="calibration", grade=VerificationGrade.EXTERNALLY_VERIFIED)
    )
    assert admission.scope is LearningScope.QUARANTINE
    assert not admission.applies_now, "quarantined data must not be applied"
    assert len(gate.quarantined()) == 1


def test_release_from_quarantine_requires_a_named_reviewer():
    gate = DurableLearningGate()
    admission = gate.admit(_update(subsystem="verifier_training"))
    with pytest.raises(ValueError, match="named reviewer"):
        gate.release_from_quarantine(admission.record_id, reviewer="")
    released = gate.release_from_quarantine(admission.record_id, reviewer="bryan")
    assert released["released_by"] == "bryan"


# -------------------------------------------------------------- malformed input


def test_a_non_bool_success_is_refused_at_the_type_level():
    """The "false" -> True coercion bug, refused rather than coerced."""
    gate = DurableLearningGate()
    admission = gate.admit(_update(success="false"))
    assert admission.scope is LearningScope.REJECTED
    assert admission.reason == "success_is_not_a_bool"
    assert not admission.applies_now


def test_a_non_grade_is_refused():
    gate = DurableLearningGate()
    assert gate.admit(_update(grade="verified")).scope is LearningScope.REJECTED


@pytest.mark.parametrize("field", ["subsystem", "key"])
def test_unidentifiable_updates_are_refused(field):
    gate = DurableLearningGate()
    assert gate.admit(_update(**{field: "  "})).scope is LearningScope.REJECTED


# ---------------------------------------------------------------------- rollback


def test_invalidating_evidence_withdraws_everything_that_rested_on_it():
    gate = DurableLearningGate()
    gate.admit(_update(key="route-1", evidence_id="ev-9"))
    gate.admit(_update(key="route-2", evidence_id="ev-9"))
    gate.admit(_update(key="route-3", evidence_id="ev-other"))

    affected = gate.invalidate_evidence("ev-9")
    assert {entry["key"] for entry in affected} == {"route-1", "route-2"}
    assert len(gate.durable_updates()) == 1, "the unrelated update must survive"


def test_a_rollback_hands_back_the_inverse_to_apply():
    """The gate knows WHAT to undo, not how; the owner applies it."""
    gate = DurableLearningGate()
    gate.admit(_update(evidence_id="ev-3", inverse={"operation": "set_confidence", "confidence": 0.42}))
    affected = gate.invalidate_evidence("ev-3")
    assert affected[0]["inverse"] == {"operation": "set_confidence", "confidence": 0.42}


def test_invalidating_twice_does_not_double_undo():
    gate = DurableLearningGate()
    gate.admit(_update(evidence_id="ev-4"))
    assert len(gate.invalidate_evidence("ev-4")) == 1
    assert gate.invalidate_evidence("ev-4") == []


def test_learning_on_already_invalidated_evidence_is_refused():
    gate = DurableLearningGate()
    gate.invalidate_evidence("ev-poisoned")
    admission = gate.admit(_update(evidence_id="ev-poisoned"))
    assert admission.scope is LearningScope.REJECTED
    assert admission.reason == "evidence_already_invalidated"


# ------------------------------------------------------------- evidence grading


def test_nothing_throwing_grades_as_asserted():
    assert grade_from_evidence(None, success=True) is VerificationGrade.ASSERTED
    assert grade_from_evidence(True, success=True) is VerificationGrade.ASSERTED
    assert grade_from_evidence("done", success=True) is VerificationGrade.ASSERTED


def test_a_result_carrying_an_agreeing_outcome_grades_observed():
    assert grade_from_evidence({"ok": True}, success=True) is VerificationGrade.OBSERVED


def test_evidence_that_contradicts_the_caller_is_not_evidence_for_the_caller():
    """The live bug: tool returned {'ok': False}, caller said success=True."""
    assert grade_from_evidence({"ok": False}, success=True) is VerificationGrade.ASSERTED


def test_a_declared_grade_is_honoured_when_the_outcome_agrees():
    evidence = {"ok": True, "verification_grade": "postcondition_verified"}
    assert grade_from_evidence(evidence, success=True) is (
        VerificationGrade.POSTCONDITION_VERIFIED
    )


def test_a_declared_grade_cannot_launder_a_contradicting_outcome():
    evidence = {"ok": False, "verification_grade": "externally_verified"}
    assert grade_from_evidence(evidence, success=True) is VerificationGrade.ASSERTED


# ------------------------------------------------------------------- the report


def test_the_report_names_which_verifiers_justified_durable_learning():
    gate = DurableLearningGate()
    gate.admit(_update(verifier="postcondition_checker"))
    gate.admit(_update(key="r2", verifier="external_grader"))
    report = gate.report()
    assert report["durable_updates"] == 2
    assert report["verifiers"] == ["external_grader", "postcondition_checker"]


def test_the_report_counts_what_was_refused_and_what_awaits_review():
    gate = DurableLearningGate()
    gate.admit(_update(grade=VerificationGrade.ASSERTED))
    gate.admit(_update(success="nope"))
    gate.admit(_update(subsystem="calibration"))
    report = gate.report()
    assert report["admissions"]["session"] == 1
    assert report["admissions"]["rejected"] == 1
    assert report["quarantine_awaiting_review"] == 1


def test_the_ledger_survives_a_round_trip(tmp_path):
    path = tmp_path / "durable_learning.json"
    gate = DurableLearningGate(ledger_path=path)
    gate.admit(_update(evidence_id="ev-persist"))

    import json

    path.write_text(
        json.dumps(
            {
                "version": 1,
                "ledger": list(gate.durable_updates()),
                "quarantine": [],
                "invalidated": [],
            }
        )
    )
    restored = DurableLearningGate(ledger_path=path)
    assert restored.load()
    assert len(restored.durable_updates()) == 1
    # Rollback must still find it after a restart, or durability is a
    # one-way door.
    assert len(restored.invalidate_evidence("ev-persist")) == 1
