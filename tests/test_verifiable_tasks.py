"""Programmatic graders for reasoning tasks (CP228).

A verifier that can be fooled is worse than no verifier: policy gradient
optimizes whatever the grader actually measures, so a lenient grader
teaches the model to game it. These tests pin the two properties that
matter -- tolerant on FORM, strict on VALUE -- and the split hygiene that
keeps a held-out number from being a training-set number.
"""
from __future__ import annotations

import pytest

from core.learning.verifiable_tasks import (
    KNOWLEDGE_FREE,
    VerifiableTask,
    build_task_set,
    disjoint_split,
    grade_boolean,
    grade_exact_set,
    grade_json,
    grade_numeric,
    scaling_report,
)

# ── Tolerant on form ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "response",
    [
        "FINAL_ANSWER: 42",
        "**FINAL_ANSWER**: 42",
        "The answer is 42",
        "answer = 42",
        r"so we get \boxed{42}",
        "lots of reasoning here\n42",
        "FINAL_ANSWER: 42.",
        "FINAL_ANSWER: $42",
        "FINAL_ANSWER: 4,2".replace("4,2", "42"),
    ],
)
def test_numeric_grader_accepts_any_reasonable_form(response):
    assert grade_numeric(response, 42, {})["correct"] is True


def test_equivalent_numbers_are_the_same_answer():
    """2/4 and 0.5 are one answer, not two."""
    assert grade_numeric("FINAL_ANSWER: 2/4", 0.5, {})["correct"] is True
    assert grade_numeric("FINAL_ANSWER: 0.5", 0.5, {})["correct"] is True
    assert grade_numeric("FINAL_ANSWER: 1000", 1000, {})["correct"] is True


# ── Strict on value ─────────────────────────────────────────────────────


def test_perfect_format_with_a_wrong_number_fails():
    """The failure mode that matters: rewarding compliance over correctness."""
    assert grade_numeric("FINAL_ANSWER: 41", 42, {})["correct"] is False
    assert grade_numeric("FINAL_ANSWER: 42.5", 42, {})["correct"] is False


def test_unparseable_output_is_wrong_not_crashing():
    for junk in ("", "I don't know", "FINAL_ANSWER: banana", None):
        verdict = grade_numeric(junk, 42, {})
        assert verdict["correct"] is False


def test_tolerance_is_configurable_but_not_unbounded():
    assert grade_numeric("FINAL_ANSWER: 3.14159", 3.14159265, {"tolerance": 1e-4})[
        "correct"
    ] is True
    assert grade_numeric("FINAL_ANSWER: 3.1", 3.14159265, {"tolerance": 1e-6})[
        "correct"
    ] is False


def test_ordered_grader_enforces_order():
    """An ordering task graded by set equality marks every permutation
    correct, teaching the model that order does not matter."""
    from core.learning.verifiable_tasks import grade_ordered

    assert grade_ordered("FINAL_ANSWER: a, b, c", ["a", "b", "c"], {})["correct"]
    assert not grade_ordered("FINAL_ANSWER: c, b, a", ["a", "b", "c"], {})["correct"]


def test_ordering_tasks_actually_use_the_ordered_grader():
    tasks = build_task_set(
        domains=["constraint_order"], depths=[3], per_cell=3, seed=2
    )
    for task in tasks:
        assert task.grader == "ordered"
        scrambled = ", ".join(reversed(task.expected))
        assert not task.grade(f"FINAL_ANSWER: {scrambled}")["correct"]


def test_asking_for_more_tasks_than_exist_is_refused():
    """Silent duplicates would inflate every score computed over them."""
    with pytest.raises(ValueError, match="distinct tasks"):
        build_task_set(
            domains=["constraint_order"], depths=[1], per_cell=5000, seed=1
        )


def test_set_grader_ignores_order_not_membership():
    assert grade_exact_set("FINAL_ANSWER: b, a, c", ["a", "b", "c"], {})["correct"]
    assert not grade_exact_set("FINAL_ANSWER: a, b", ["a", "b", "c"], {})["correct"]


def test_boolean_and_json_graders():
    assert grade_boolean("FINAL_ANSWER: yes", True, {})["correct"] is True
    assert grade_boolean("FINAL_ANSWER: false", True, {})["correct"] is False
    assert grade_boolean("FINAL_ANSWER: maybe", True, {})["correct"] is False
    assert grade_json('FINAL_ANSWER: {"b":2,"a":1}', {"a": 1, "b": 2}, {})["correct"]
    assert not grade_json("FINAL_ANSWER: {broken", {"a": 1}, {})["correct"]


# ── Generated tasks are solvable, graded, and depth-stratified ──────────


def test_generated_tasks_grade_their_own_ground_truth_as_correct():
    """If a task cannot mark its own answer correct, every score computed
    from it is noise -- the CP223 calibration lesson, applied upstream."""
    tasks = build_task_set(
        domains=["arithmetic_chain", "constraint_order", "program_trace"],
        depths=[1, 3, 5], per_cell=4, seed=11,
    )
    assert len(tasks) == 3 * 3 * 4
    for task in tasks:
        gold = task.expected
        rendered = (
            ", ".join(str(x) for x in gold)
            if isinstance(gold, list)
            else str(gold)
        )
        verdict = task.grade(f"FINAL_ANSWER: {rendered}")
        assert verdict["correct"] is True, f"{task.task_id} cannot grade its own gold"


def test_generation_is_deterministic_and_duplicate_free():
    a = build_task_set(domains=["arithmetic_chain"], depths=[3], per_cell=8, seed=5)
    b = build_task_set(domains=["arithmetic_chain"], depths=[3], per_cell=8, seed=5)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert len({t.prompt for t in a}) == 8


def test_depth_is_recorded_so_scaling_can_be_measured():
    tasks = build_task_set(
        domains=["program_trace"], depths=[1, 2, 8], per_cell=2, seed=3
    )
    assert sorted({t.depth for t in tasks}) == [1, 2, 8]
    assert all(t.knowledge == KNOWLEDGE_FREE for t in tasks)


# ── Split hygiene ───────────────────────────────────────────────────────


def test_heldout_shares_no_prompt_with_train():
    """A leaking split reports a training number as generalization, and
    that is not detectable afterwards from the score alone."""
    train, holdout = disjoint_split(
        domains=["arithmetic_chain", "program_trace"], depths=[2, 4],
        train_per_cell=16, holdout_per_cell=8, seed=99,
    )
    assert holdout
    assert not ({t.prompt for t in train} & {t.prompt for t in holdout})
    assert not ({t.task_id for t in train} & {t.task_id for t in holdout})
    assert len({t.task_id for t in (*train, *holdout)}) == len(train) + len(holdout)


# ── The scaling curve is the criterion ──────────────────────────────────


def test_scaling_report_exposes_depth_falloff():
    tasks = build_task_set(
        domains=["arithmetic_chain"], depths=[1, 8], per_cell=2, seed=1
    )
    shallow = [t for t in tasks if t.depth == 1]
    deep = [t for t in tasks if t.depth == 8]
    results = [(t, True) for t in shallow] + [(t, False) for t in deep]
    report = scaling_report(results)
    assert report["accuracy_by_depth"] == {1: 1.0, 8: 0.0}
    assert report["depth_falloff"] == 1.0, "collapse at depth must be visible"

    flat = [(t, True) for t in tasks]
    assert scaling_report(flat)["depth_falloff"] == 0.0


def test_invalid_configuration_fails_closed():
    with pytest.raises(ValueError, match="unknown generators"):
        build_task_set(domains=["nope"], depths=[1], per_cell=1, seed=1)
    with pytest.raises(ValueError, match="depths"):
        build_task_set(domains=["program_trace"], depths=[0], per_cell=1, seed=1)
    with pytest.raises(ValueError, match="per_cell"):
        build_task_set(domains=["program_trace"], depths=[1], per_cell=0, seed=1)
    with pytest.raises(ValueError, match="unknown grader"):
        VerifiableTask(
            task_id="x", prompt="p", domain="d", depth=1,
            knowledge=KNOWLEDGE_FREE, grader="bogus", expected=1, metadata={},
        )
