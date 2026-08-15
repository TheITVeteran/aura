"""Contracts for the bounded recurrent mathematics work memory."""

from __future__ import annotations

import ast
import re

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from core.learning.recurrent_work_memory import (
    MATHEMATICS_WORK_MEMORY_CAPACITY,
    MathematicsWorkMemory,
    brute_force_mathematics_result,
    compile_mathematics_work_memory,
)

_OBJECTIVE = re.compile(
    r"From the set (?P<values>\[.*?\]), choose exactly (?P<choose>\d+) distinct "
    r"values\. Adjacent values in sorted chosen order must differ by at least "
    r"(?P<gap>\d+), and the chosen sum must be from (?P<low>-?\d+) through "
    r"(?P<high>-?\d+), inclusive\."
)


def _public_literals(prompt: str) -> dict[str, object]:
    match = _OBJECTIVE.search(prompt)
    assert match is not None
    return {
        "values": tuple(ast.literal_eval(match.group("values"))),
        "choose": int(match.group("choose")),
        "gap": int(match.group("gap")),
        "low": int(match.group("low")),
        "high": int(match.group("high")),
    }


@pytest.mark.parametrize("difficulty", (1, 2, 3))
def test_addressable_memory_matches_independent_oracle_across_registry(
    difficulty: int,
) -> None:
    for seed in range(40):
        task = generate_task("mathematics", seed=seed, difficulty=difficulty)
        literals = _public_literals(task.public.prompt)
        trace = compile_mathematics_work_memory(**literals)
        expected = task.reveal_for_verifier()["expected"]

        assert trace.states[-1].result() == (
            expected["count"],
            tuple(expected["witness"]),
        )
        assert trace.states[-1].result() == brute_force_mathematics_result(
            **literals
        )


def test_addressable_memory_is_one_stationary_markov_transition() -> None:
    state = MathematicsWorkMemory.empty(choose=3, gap=2, low=15, high=30)
    left = state.apply_value(3).apply_value(8)
    right = state.apply_value(3).apply_value(8)

    assert left == right
    assert left.state_sha256 == right.state_sha256
    assert left.apply_value(11) == right.apply_value(11)


def test_addressable_memory_refuses_reuse_and_capacity_overflow() -> None:
    state = MathematicsWorkMemory.empty(choose=3, gap=1, low=0, high=100)
    with pytest.raises(ValueError, match="input order"):
        state.apply_value(4).apply_value(4)

    constrained = MathematicsWorkMemory.empty(
        choose=3,
        gap=1,
        low=0,
        high=100,
        capacity=1,
    )
    with pytest.raises(OverflowError, match="capacity"):
        constrained.apply_value(4)


def test_public_commitment_discloses_shape_but_not_memory_contents() -> None:
    trace = compile_mathematics_work_memory(
        choose=3,
        gap=2,
        low=15,
        high=30,
        values=(3, 8, 11, 15),
    )
    commitment = trace.public_commitment()
    serialized = repr(commitment)

    assert commitment["capacity"] == MATHEMATICS_WORK_MEMORY_CAPACITY
    assert commitment["steps"] == 5
    assert commitment["configuration_steps"] == 1
    assert commitment["maximum_live_cells"] > 1
    assert commitment["runtime_teacher_available"] is False
    assert commitment["private_witnesses_exposed"] is False
    assert "multiplicity" not in serialized
    assert "cells" not in commitment
    assert "processed_values" not in serialized
