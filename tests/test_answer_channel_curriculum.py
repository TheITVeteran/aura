"""Answer-channel curriculum contracts.

This curriculum is a bridge into verifier-driven RL, not a reasoning-gain
benchmark. Its tests keep that boundary explicit while proving the task source
is deterministic, strict, and train/holdout disjoint.
"""

from __future__ import annotations

import pytest

from core.learning.answer_channel_curriculum import (
    ANSWER_CHANNEL_FAMILIES,
    TASK_GENERATORS,
    disjoint_task_split,
    task_battery,
)
from tools.train_grpo import _build_task_split


def test_answer_channel_task_grades_strict_final_answer_json():
    task = TASK_GENERATORS["json_copy"](2, 17)

    assert task.metadata["claim_boundary"] == (
        "format_parseability_only_not_reasoning_gain"
    )
    assert task.grade(task.answer)["correct"] is True
    assert task.grade("FINAL_ANSWER: not json")["reason"] == "unparseable"
    assert task.grade(task.answer + "\ntrailing")["correct"] is False


def test_answer_channel_battery_is_deterministic_and_unique():
    first = task_battery(ANSWER_CHANNEL_FAMILIES, (1, 2), 2, seed=91)
    second = task_battery(ANSWER_CHANNEL_FAMILIES, (1, 2), 2, seed=91)

    assert first == second
    assert len({task.task_id for task in first}) == len(first)
    assert len({task.prompt for task in first}) == len(first)
    assert {task.family for task in first} == set(ANSWER_CHANNEL_FAMILIES)


def test_answer_channel_split_is_disjoint():
    train, holdout = disjoint_task_split(
        families=ANSWER_CHANNEL_FAMILIES,
        depths=(1, 2),
        train_per_cell=2,
        holdout_per_cell=1,
        seed=103,
    )

    assert {task.task_id for task in train}.isdisjoint(
        {task.task_id for task in holdout}
    )
    assert {task.prompt for task in train}.isdisjoint(
        {task.prompt for task in holdout}
    )


def test_answer_channel_split_matches_resident_preflight_dimensions():
    train, holdout = disjoint_task_split(
        families=("json_copy", "typed_boolean", "key_selection"),
        depths=(1, 2),
        train_per_cell=2,
        holdout_per_cell=1,
        seed=2026072413,
    )

    assert len(train) == 12
    assert len(holdout) == 6
    assert {task.task_id for task in train}.isdisjoint(
        {task.task_id for task in holdout}
    )
    assert {task.prompt for task in train}.isdisjoint(
        {task.prompt for task in holdout}
    )


def test_trainer_can_bind_answer_channel_task_source():
    train, holdout, source = _build_task_split(
        task_source="answer_channel_curriculum",
        domains=["json_copy", "typed_boolean"],
        depths=[1, 2],
        train_per_cell=2,
        holdout_per_cell=1,
        seed=17,
    )

    assert source.name == "answer_channel_curriculum.py"
    assert len(train) == 8
    assert len(holdout) == 4
    assert {task.metadata["source"] for task in train} == {
        "answer_channel_curriculum"
    }


def test_answer_channel_rejects_unknown_family():
    with pytest.raises(ValueError, match="families"):
        task_battery(["reasoning_claim"], (1,), 1, seed=1)
