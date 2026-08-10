"""Contracts for the frozen certified-recurrence behavioral discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_certified_recurrence_behavioral_gate import (
    _load_journal,
    _score_row,
    build_public_preregistration,
    build_tasks,
    exact_paired_pvalue,
    execute_certified_arm,
    execute_t1_lesion,
)


def test_public_preregistration_contains_no_private_answer_or_trace() -> None:
    tasks = build_tasks(depths=(1, 4, 8), seeds=(41, 42))
    prereg = build_public_preregistration(
        tasks,
        source_commit="a" * 40,
        model_path=Path("/tmp/model"),
        best_of_n=3,
        max_tokens=192,
    )
    wire = json.dumps(prereg, sort_keys=True)

    assert prereg["task_count"] == 12
    assert "answer" not in wire
    assert "expected" not in wire
    assert "transition_trace" not in wire
    assert all(task.answer not in wire for task in tasks)


def test_certified_arm_is_exact_and_t1_lesion_is_causal() -> None:
    tasks = build_tasks(depths=(4, 8), seeds=(51, 52, 53, 54))
    treatment_correct = 0
    lesion_correct = 0
    changed = 0
    for task in tasks:
        treatment, receipt = execute_certified_arm(task.prompt)
        lesion, lesion_receipt = execute_t1_lesion(task.prompt)
        treatment_correct += int(task.grade(treatment)["correct"])
        lesion_correct += int(task.grade(lesion)["correct"])
        changed += int(treatment != lesion)
        assert receipt["execution"]["student_rollin"]["transition_count"] == task.depth
        assert lesion_receipt["executed_transitions"] == 1

    assert treatment_correct == len(tasks)
    assert changed >= 12
    assert lesion_correct < treatment_correct


def test_exact_paired_test_counts_direction_and_rejects_bad_shapes() -> None:
    result = exact_paired_pvalue(
        (True,) * 12,
        (False,) * 10 + (True,) * 2,
    )
    assert result == {
        "treatment_only_correct": 10,
        "control_only_correct": 0,
        "discordant": 10,
        "two_sided_exact_p": pytest.approx(0.001953125),
    }
    with pytest.raises(ValueError, match="paired outcomes"):
        exact_paired_pvalue((True,), ())


def test_journal_resume_reconstructs_rows_and_rejects_tampering(tmp_path: Path) -> None:
    task = build_tasks(depths=(4,), seeds=(61,))[0]
    candidate, _receipt = execute_certified_arm(task.prompt)
    row = _score_row(
        task,
        arm="certified_recurrence",
        candidate=candidate,
        detail={},
    )
    journal = tmp_path / "observations.jsonl"
    journal.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows, indexed = _load_journal(
        journal,
        tasks=(task,),
        allowed_arms=("certified_recurrence",),
    )
    assert rows == [row]
    assert indexed[(task.task_id, "certified_recurrence")] == row

    row["correct"] = False
    journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed reconstruction"):
        _load_journal(
            journal,
            tasks=(task,),
            allowed_arms=("certified_recurrence",),
        )
