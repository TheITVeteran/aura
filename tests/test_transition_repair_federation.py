"""Contract tests: transition grading, on-policy repair, teacher federation.

The training-program spine for long-horizon capability:
- every consequential transition is graded on named dimensions and
  reliability compounds multiplicatively;
- the repair protocol locates the EARLIEST causal error by replay, trains
  the correction from the exact reached state, and retains only transfers;
- the federation lets the verifier — never prestige — pick the teacher.
"""
from __future__ import annotations

import pytest

from core.learning.on_policy_repair import (
    RepairCandidate,
    TrajectoryRecord,
    locate_first_causal_error,
    repair_intake,
    validate_repair,
)
from core.learning.teacher_federation import (
    Teacher,
    TeacherFederation,
)
from core.learning.transition_grading import (
    Transition,
    grade_trajectory,
    grade_transition,
)


def _transition(index: int, **overrides) -> Transition:
    defaults = dict(
        index=index,
        action=f"step {index}",
        preconditions_checked=True,
        expected_effect="file created with 3 entries",
        observed_effect="file created with 3 entries",
        consequential=True,
        reversible=False,
        checkpoint_created=True,
        verified_outcome=True,
        preserved_completed_work=True,
    )
    defaults.update(overrides)
    return Transition(**defaults)


# ── Transition grading ──────────────────────────────────────────────────


def test_clean_transition_earns_full_marks():
    grade = grade_transition(_transition(0))
    assert grade.composite == pytest.approx(1.0)
    assert grade.training_weight == pytest.approx(1.0)
    assert grade.prediction_method == "numeric_overlap"
    assert grade.reasons == []


def test_named_defects_are_individually_visible():
    grade = grade_transition(
        _transition(
            1,
            preconditions_checked=False,
            checkpoint_created=False,
            observed_effect="file missing",
            verified_outcome=False,
            preserved_completed_work=False,
        )
    )
    assert grade.dimensions["preconditions"] == 0.0
    assert grade.dimensions["recovery_placement"] == 0.0
    assert grade.dimensions["work_preservation"] == 0.0
    assert grade.dimensions["verification"] == 0.0
    assert "irreversible_action_without_recovery_point" in grade.reasons
    assert "discarded_completed_valid_work" in grade.reasons
    # A verified failure never becomes positive training signal.
    assert grade.training_weight == 0.0


def test_unverified_outcome_is_mostly_unknown_not_half_credit():
    grade = grade_transition(_transition(0, verified_outcome=None))
    assert grade.dimensions["verification"] == pytest.approx(0.25)
    assert "outcome_unverified" in grade.reasons


def test_recovery_from_failure_earns_bounded_bonus():
    plain = grade_transition(_transition(0))
    recovered = grade_transition(_transition(0, recovered_from_failure=True))
    assert recovered.composite == pytest.approx(min(1.0, plain.composite + 0.10))


def test_reliability_compounds_multiplicatively():
    strong = grade_trajectory(
        "proj-strong",
        [_transition(i) for i in range(3)],
        final_success=True,
    )
    weak = grade_trajectory(
        "proj-weak",
        [
            _transition(0),
            _transition(1, verified_outcome=None, preconditions_checked=False),
            _transition(2),
        ],
        final_success=True,
    )
    assert strong.reliability_estimate == pytest.approx(1.0)
    assert weak.reliability_estimate < strong.reliability_estimate
    receipt = weak.to_receipt()
    assert len(receipt["transitions"]) == 3


def test_repair_queue_targets_failed_and_weak_transitions():
    graded = grade_trajectory(
        "proj",
        [
            _transition(0),
            _transition(1, verified_outcome=False, observed_effect="crash"),
            _transition(2, verified_outcome=None, preconditions_checked=False,
                        checkpoint_created=False),
        ],
        final_success=False,
    )
    assert graded.first_failure_index == 1
    assert 1 in graded.repair_queue()
    assert 2 in graded.repair_queue()
    assert 0 not in graded.repair_queue()


def test_trajectory_validation_rejects_disorder():
    with pytest.raises(ValueError):
        grade_trajectory("p", [], final_success=True)
    with pytest.raises(ValueError):
        grade_trajectory(
            "p",
            [_transition(1), _transition(0)],
            final_success=True,
        )


# ── On-policy repair ────────────────────────────────────────────────────


def _record(n: int = 5, fail_at: int = 3) -> TrajectoryRecord:
    transitions = []
    for i in range(n):
        transitions.append(
            _transition(
                i,
                verified_outcome=(False if i == fail_at else True),
                observed_effect=("crash" if i == fail_at else "file created with 3 entries"),
            )
        )
    return TrajectoryRecord(
        task_id="task-1",
        family="repo_fix",
        transitions=tuple(transitions),
        final_success=False,
    )


def test_causal_error_located_by_replay_bisect():
    record = _record(n=6, fail_at=3)

    # Ground truth: continuing fresh from any prefix <= 3 succeeds (the agent
    # avoids the bad step); keeping the recorded bad step (prefix >= 4) fails.
    def replay(prefix_length: int) -> bool:
        return prefix_length <= 3

    index, evidence = locate_first_causal_error(record, replay)
    assert index == 3
    assert evidence["reason"] == "causal_flip_located"
    assert evidence["replays"] <= 5


def test_successful_trajectory_needs_no_repair():
    record = TrajectoryRecord(
        task_id="ok",
        family="repo_fix",
        transitions=(_transition(0),),
        final_success=True,
    )
    index, evidence = locate_first_causal_error(record, lambda _p: True)
    assert index is None
    assert evidence["reason"] == "trajectory_succeeded"


def test_setup_level_failure_is_not_blamed_on_a_step():
    record = _record()
    index, evidence = locate_first_causal_error(record, lambda _p: False)
    assert index is None
    assert evidence["reason"] == "fresh_rerun_fails_no_single_step_causal"


def test_replay_budget_is_enforced():
    record = _record(n=5)
    with pytest.raises(RuntimeError, match="replay budget"):
        locate_first_causal_error(record, lambda p: p <= 2, max_replays=1)


def test_repair_retained_only_with_rerun_and_transfer_majority():
    record = _record()
    candidate = RepairCandidate(
        error_index=3, corrected_action="run migration inside transaction",
        corrector="aura_full_system",
    )
    outcome = validate_repair(
        record,
        candidate,
        rerun_with_repair_fn=lambda _r: True,
        transfer_tasks=["fresh-1", "fresh-2", "fresh-3"],
        run_transfer_fn=lambda task, _r: task != "fresh-3",
    )
    assert outcome.accepted is True
    assert outcome.stage == "retained"
    unit = outcome.training_unit
    assert unit["best_operation"] == "run migration inside transaction"
    assert unit["verified_outcome"] is True
    assert "step 3" in unit["possible_operations"]
    assert unit["transfer_evidence"] == {
        "fresh-1": True,
        "fresh-2": True,
        "fresh-3": False,
    }


def test_repair_rejected_when_rerun_still_fails_or_no_transfer():
    record = _record()
    candidate = RepairCandidate(
        error_index=3, corrected_action="fix", corrector="math_solver"
    )
    still_fails = validate_repair(
        record,
        candidate,
        rerun_with_repair_fn=lambda _r: False,
        transfer_tasks=["a", "b"],
        run_transfer_fn=lambda *_a: True,
    )
    assert still_fails.accepted is False
    assert still_fails.receipt["refusal"] == "rerun_still_fails"

    no_transfer = validate_repair(
        record,
        candidate,
        rerun_with_repair_fn=lambda _r: True,
        transfer_tasks=["a", "b"],
        run_transfer_fn=lambda *_a: False,
    )
    assert no_transfer.accepted is False
    assert no_transfer.receipt["refusal"] == "no_transfer_majority"

    starved = validate_repair(
        record,
        candidate,
        rerun_with_repair_fn=lambda _r: True,
        transfer_tasks=["only-one"],
        run_transfer_fn=lambda *_a: True,
    )
    assert starved.accepted is False
    assert starved.receipt["refusal"] == "insufficient_transfer_tasks"


def test_repair_intake_flows_from_the_grading_spine():
    assert 3 in repair_intake(_record())


# ── Teacher federation ──────────────────────────────────────────────────


def _teacher(name: str, kind: str, answer: str | None) -> Teacher:
    return Teacher(name=name, kind=kind, propose=lambda _task: answer)


def test_verifier_beats_prestige():
    federation = TeacherFederation(
        [
            _teacher("big_frontier", "frontier_generalist", "wrong answer"),
            _teacher("small_tool", "tool_execution", "right answer"),
        ]
    )
    selection = federation.select(
        "compute the thing",
        verifier=lambda text: text == "right answer",
    )
    assert selection.tier == "verified"
    assert selection.selected_teacher == "small_tool"
    # The failed frontier proposal became a negative example.
    assert selection.negative_examples[0]["teacher"] == "big_frontier"
    assert selection.negative_examples[0]["verified_outcome"] is False
    # Ledgers recorded both verdicts.
    assert federation.ledgers["small_tool"].successes == 1
    assert federation.ledgers["big_frontier"].successes == 0


def test_reliability_ledger_breaks_verified_ties():
    federation = TeacherFederation(
        [
            _teacher("veteran", "math_solver", "42"),
            _teacher("rookie", "math_solver", "42"),
        ]
    )
    # Seed a track record: the veteran has earned reliability.
    for _ in range(5):
        federation.ledgers["veteran"].record(True)
    selection = federation.select("q", verifier=lambda text: text == "42")
    assert selection.tier == "verified"
    assert selection.selected_teacher == "veteran"


def test_all_failures_yield_no_winner_but_negatives():
    federation = TeacherFederation(
        [_teacher("a", "simulator", "x"), _teacher("b", "formal_solver", "y")]
    )
    selection = federation.select("q", verifier=lambda _t: False)
    assert selection.tier == "none"
    assert selection.receipt["decision"] == "all_candidates_failed_verification"
    assert len(selection.negative_examples) == 2


def test_unverifiable_consensus_is_honestly_tiered():
    federation = TeacherFederation(
        [
            _teacher("a", "frontier_generalist", "the sky reads blue today"),
            _teacher("b", "human_demonstration", "The sky reads   blue today"),
            _teacher("c", "simulator", "something different"),
        ]
    )
    selection = federation.select("describe")
    assert selection.tier == "consensus_unverified"
    assert selection.receipt["consensus_group_size"] == 2
    lone = TeacherFederation(
        [
            _teacher("a", "frontier_generalist", "answer one"),
            _teacher("b", "simulator", "answer two"),
        ]
    )
    no_consensus = lone.select("describe")
    assert no_consensus.tier == "none"
    assert no_consensus.receipt["decision"] == "no_consensus_without_verifier"


def test_broken_and_abstaining_teachers_are_receipted_not_fatal():
    def explode(_task: str) -> str:
        raise RuntimeError("teacher offline")

    federation = TeacherFederation(
        [
            Teacher(name="broken", kind="simulator", propose=explode),
            _teacher("quiet", "human_demonstration", None),
            _teacher("worker", "tool_execution", "done"),
        ]
    )
    selection = federation.select("q", verifier=lambda t: t == "done")
    assert selection.tier == "verified"
    statuses = {row["teacher"]: row["status"] for row in selection.receipt["proposals"]}
    assert statuses["broken"].startswith("proposal_error")
    assert statuses["quiet"] == "abstained"


def test_verifier_fault_crowns_no_one():
    federation = TeacherFederation([_teacher("a", "simulator", "x")])

    def bad_verifier(_text: str) -> bool:
        raise ValueError("verifier crashed")

    selection = federation.select("q", verifier=bad_verifier)
    assert selection.tier == "none"
    assert selection.receipt["decision"].startswith("verifier_error")


def test_federation_construction_validates():
    with pytest.raises(ValueError):
        TeacherFederation([])
    with pytest.raises(ValueError):
        TeacherFederation(
            [_teacher("dup", "simulator", "x"), _teacher("dup", "simulator", "y")]
        )
    with pytest.raises(ValueError):
        Teacher(name="x", kind="wizard", propose=lambda _t: "").validated()
