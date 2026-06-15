"""tests/test_ablation_harness.py
=====================================
The ablation benchmark is the keystone honesty test (does the cognitive
architecture actually beat a stateless wrapper?). It replaced a FABRICATED
benchmark that hardcoded baseline scores and asserted victory without running
anything. These tests pin the new harness's honesty:

  - scores come only from grading real responder outputs against answer keys,
  - the architecture "beats" a baseline ONLY on real CI separation,
  - when the architecture has no real advantage the verdict is False (it cannot
    fake a pass),
  - graders and bootstrap CIs behave correctly.
"""
from __future__ import annotations

from core.evaluation.ablation_harness import (
    FULL,
    PROMPTED,
    RAW,
    AblationHarness,
    AblationTask,
    bootstrap_ci,
    grade,
)

_TASKS = [
    AblationTask("t1", "memory_recall", ["codeword is amber-79204", "ok", "what was the codeword?"], "amber-79204"),
    AblationTask("t2", "memory_recall", ["my name is Petrova", "ok", "what is my name?"], "Petrova"),
    AblationTask("t3", "continuity", ["allergic to peanuts", "ok", "what am I allergic to?"], "peanuts"),
    AblationTask("t4", "continuity", ["deploy window is Tuesday", "ok", "when is deploy?"], "Tuesday"),
    AblationTask("t5", "continuity", ["favourite colour teal", "ok", "favourite colour?"], "teal"),
]


def _memory_aware_responder(condition, task, turn_index, history):
    """Stateful only under FULL: recalls the answer from history; stateless
    conditions get no history and cannot answer."""
    if turn_index < len(task.turns) - 1:
        return "ok"
    if condition == FULL:
        joined = " ".join(history).lower()
        return task.answer_key if task.answer_key.lower() in joined else "unknown"
    return "I don't have that information."


def _omniscient_responder(condition, task, turn_index, history):
    """Every condition answers correctly → no real architectural advantage."""
    if turn_index < len(task.turns) - 1:
        return "ok"
    return task.answer_key


def _useless_responder(condition, task, turn_index, history):
    """Nothing ever answers → no advantage for anyone."""
    return "nope"


def test_architecture_wins_only_when_it_really_recalls():
    harness = AblationHarness()
    report = harness.report(_memory_aware_responder, _TASKS)
    assert report["verdict"]["architecture_beats_stateless"] is True
    conds = report["conditions"]
    assert conds[FULL]["mean_score"] == 1.0
    assert conds[RAW]["mean_score"] == 0.0
    assert conds[PROMPTED]["mean_score"] == 0.0
    # CI separation is real, not asserted.
    assert report["verdict"]["comparisons"][RAW]["ci_separated"] is True
    assert report["verdict"]["comparisons"][RAW]["delta_mean"] == 1.0


def test_no_fake_pass_when_stateless_also_succeeds():
    harness = AblationHarness()
    report = harness.report(_omniscient_responder, _TASKS)
    # Everyone scores 1.0 → no separation → honest verdict is False.
    assert report["conditions"][FULL]["mean_score"] == 1.0
    assert report["conditions"][RAW]["mean_score"] == 1.0
    assert report["verdict"]["architecture_beats_stateless"] is False


def test_no_fake_pass_when_nothing_works():
    harness = AblationHarness()
    report = harness.report(_useless_responder, _TASKS)
    assert report["conditions"][FULL]["mean_score"] == 0.0
    assert report["verdict"]["architecture_beats_stateless"] is False


def test_report_has_no_hardcoded_scores_and_real_n():
    harness = AblationHarness()
    report = harness.report(_memory_aware_responder, _TASKS)
    assert report["tasks_evaluated"] == len(_TASKS)
    # per-task scores are present and real (the old version had none).
    assert set(report["conditions"][FULL]["per_task"]) == {t.task_id for t in _TASKS}
    assert "methodology" in report and "No hardcoded scores" in report["methodology"]


def test_grade_substring_token_and_exact():
    sub = AblationTask("g1", "f", ["x"], "amber-7", "recall_substring")
    assert grade("the code is AMBER-7 ok", sub) == 1.0
    assert grade("no idea", sub) == 0.0

    tok = AblationTask("g2", "f", ["x"], "14, 27, 88", "token_overlap")
    assert grade("the numbers were 14 and 88", tok) == 2 / 3
    assert grade("nothing", tok) == 0.0

    ex = AblationTask("g3", "f", ["x"], "teal", "exact")
    assert grade("  Teal ", ex) == 1.0
    assert grade("teal blue", ex) == 0.0


def test_bootstrap_ci_deterministic_and_bounded():
    a = bootstrap_ci([1.0, 1.0, 1.0])
    assert a == (1.0, 1.0)
    b = bootstrap_ci([0.0, 0.0, 0.0])
    assert b == (0.0, 0.0)
    mixed1 = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], seed=7)
    mixed2 = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0], seed=7)
    assert mixed1 == mixed2  # deterministic
    assert 0.0 <= mixed1[0] <= mixed1[1] <= 1.0


def test_report_from_results_matches_run():
    harness = AblationHarness()
    results = harness.run(_memory_aware_responder, _TASKS)
    report = harness.report_from_results(results, tasks_evaluated=len(_TASKS))
    assert report["verdict"]["architecture_beats_stateless"] is True
    assert report["tasks_evaluated"] == len(_TASKS)
