from __future__ import annotations

import pytest

from core.learning.recurrent_process_curriculum import (
    PROCESS_REWARD_SCHEMA,
    boolean_process_chain,
    modular_process_chain,
    process_task_battery,
)


@pytest.mark.parametrize("factory", [boolean_process_chain, modular_process_chain])
def test_process_task_exact_answer_is_the_only_full_reward(factory) -> None:
    task = factory(3, 17)

    receipt = task.process_reward(task.answer)

    assert task.grade(task.answer)["correct"] is True
    assert receipt["schema"] == PROCESS_REWARD_SCHEMA
    assert receipt["correct"] is True
    assert receipt["reward"] == 1.0


def test_process_reward_is_longest_verified_prefix_not_numeric_closeness() -> None:
    task = modular_process_chain(3, 29)
    expected = task.expected
    first = expected["trace"][0]
    second = expected["trace"][1]
    partial = (
        'FINAL_ANSWER: {"residue":999,"trace":'
        f"[{first},{second},999,999]"
        "}"
    )
    no_prefix = (
        'FINAL_ANSWER: {"residue":999,"trace":'
        f"[999,{second},{expected['trace'][2]},{expected['trace'][3]}]"
        "}"
    )

    partial_receipt = task.process_reward(partial)
    no_prefix_receipt = task.process_reward(no_prefix)

    assert partial_receipt["matched_prefix_steps"] == 2
    assert no_prefix_receipt["matched_prefix_steps"] == 0
    assert 0.0 < no_prefix_receipt["reward"] < partial_receipt["reward"] < 1.0


def test_terminal_only_success_remains_partial_without_the_process() -> None:
    task = boolean_process_chain(3, 41)
    response = (
        'FINAL_ANSWER: {"trace":[9,9,9,9],"value":'
        f"{task.expected['value']}"
        "}"
    )

    receipt = task.process_reward(response)

    assert receipt["correct"] is False
    assert receipt["terminal_correct"] is True
    assert receipt["matched_prefix_steps"] == 0
    assert receipt["reward"] < 0.3


def test_unparseable_process_response_gets_no_credit() -> None:
    task = boolean_process_chain(2, 53)

    receipt = task.process_reward("the first state is probably 1")

    assert receipt["parsed"] is False
    assert receipt["reward"] == 0.0


def test_process_battery_is_deterministic_unique_and_covers_cells() -> None:
    first = process_task_battery(["boolean", "modular"], [2, 3], 2, seed=67)
    second = process_task_battery(["boolean", "modular"], [2, 3], 2, seed=67)

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.prompt for task in first] == [task.prompt for task in second]
    assert len(first) == 8
    assert len({task.task_id for task in first}) == len(first)


def test_process_reward_configuration_is_bounded() -> None:
    task = boolean_process_chain(2, 71)

    with pytest.raises(ValueError, match="format credit"):
        task.process_reward(task.answer, format_credit=0.11)
