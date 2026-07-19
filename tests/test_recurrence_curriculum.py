"""Independent contracts for the recurrence-native training curriculum."""

from __future__ import annotations

import ast
import itertools
import json
import re
from fractions import Fraction

import pytest

from core.learning.recurrence_curriculum import (
    MAX_TRAINING_DEPTH,
    RECURRENCE_TRAINING_FAMILIES,
    TASK_GENERATORS,
    RecurrenceTrainingTask,
    task_battery,
)


def _payload(task: RecurrenceTrainingTask) -> dict[str, object]:
    prefix = "FINAL_ANSWER: "
    assert task.answer.startswith(prefix)
    value = json.loads(task.answer.removeprefix(prefix))
    assert isinstance(value, dict)
    return value


def _evaluate_boolean(expression: str) -> bool:
    tokens = re.findall(r"\(|\)|not|and|or|xor|0|1", expression)

    def parse(cursor: int) -> tuple[bool, int]:
        token = tokens[cursor]
        if token in {"0", "1"}:
            return token == "1", cursor + 1
        assert token == "("
        if tokens[cursor + 1] == "not":
            value, cursor = parse(cursor + 2)
            assert tokens[cursor] == ")"
            return not value, cursor + 1
        left, cursor = parse(cursor + 1)
        operation = tokens[cursor]
        right, cursor = parse(cursor + 1)
        assert tokens[cursor] == ")"
        if operation == "and":
            value = left and right
        elif operation == "or":
            value = left or right
        else:
            assert operation == "xor"
            value = left != right
        return value, cursor + 1

    result, cursor = parse(0)
    assert cursor == len(tokens)
    return result


def test_registry_and_battery_are_deterministic_unique_and_complete():
    assert len(RECURRENCE_TRAINING_FAMILIES) == 12
    assert set(RECURRENCE_TRAINING_FAMILIES) == set(TASK_GENERATORS)
    first = task_battery(
        RECURRENCE_TRAINING_FAMILIES,
        (1, 4),
        2,
        seed=1777,
    )
    second = task_battery(
        RECURRENCE_TRAINING_FAMILIES,
        (1, 4),
        2,
        seed=1777,
    )
    assert first == second
    assert len(first) == 48
    assert len({task.seed for task in first}) == len(first)
    assert len({task.prompt for task in first}) == len(first)
    assert {task.family for task in first} == set(RECURRENCE_TRAINING_FAMILIES)
    assert {task.depth for task in first} == {1, 4}
    assert first != task_battery(
        RECURRENCE_TRAINING_FAMILIES,
        (1, 4),
        2,
        seed=1778,
    )


def test_sample_identity_is_invariant_to_family_order_and_subset():
    forward = task_battery(("khop", "boolean", "modular"), (2, 4), 4, seed=99)
    reversed_order = task_battery(("modular", "boolean", "khop"), (2, 4), 4, seed=99)
    subset = task_battery(("boolean",), (2, 4), 4, seed=99)
    normalized = lambda tasks: sorted(
        (task.family, task.depth, task.seed, task.prompt, task.answer) for task in tasks
    )
    assert normalized(forward) == normalized(reversed_order)
    assert normalized(subset) == [
        record for record in normalized(forward) if record[0] == "boolean"
    ]


def test_low_depth_default_cell_count_contains_no_duplicate_examples():
    tasks = task_battery(RECURRENCE_TRAINING_FAMILIES, (2,), 64, seed=1777)
    assert len(tasks) == 64 * len(RECURRENCE_TRAINING_FAMILIES)
    assert len({task.prompt for task in tasks}) == len(tasks)
    assert all(_payload(task) for task in tasks)


def test_generator_registry_is_immutable():
    with pytest.raises(TypeError):
        TASK_GENERATORS["khop"] = TASK_GENERATORS["boolean"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("families", "depths", "per_cell", "seed"),
    [
        ((), (1,), 1, 1),
        (("khop", "khop"), (1,), 1, 1),
        (("unknown",), (1,), 1, 1),
        (("khop",), (), 1, 1),
        (("khop",), (1, 1), 1, 1),
        (("khop",), (True,), 1, 1),
        (("khop",), (MAX_TRAINING_DEPTH + 1,), 1, 1),
        (("khop",), (1,), 0, 1),
        (("khop",), (1,), 1, -1),
    ],
)
def test_battery_rejects_ambiguous_or_unbounded_coordinates(
    families: tuple[object, ...],
    depths: tuple[object, ...],
    per_cell: int,
    seed: int,
):
    with pytest.raises(ValueError):
        task_battery(families, depths, per_cell, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("family", RECURRENCE_TRAINING_FAMILIES)
def test_each_generator_rejects_invalid_coordinates(family: str):
    generator = TASK_GENERATORS[family]
    with pytest.raises(ValueError):
        generator(0, 1)
    with pytest.raises(ValueError):
        generator(1, -1)


def test_khop_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["khop"](8, 11)
    edges = {int(left): int(right) for left, right in re.findall(r"(\d+)->(\d+)", task.prompt)}
    match = re.search(r"Start at (\d+) and follow exactly (\d+) edges", task.prompt)
    assert match
    node, depth = map(int, match.groups())
    for _ in range(depth):
        node = edges[node]
    assert _payload(task) == {"node": node}


def test_boolean_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["boolean"](12, 12)
    expression = task.prompt.split(": ", 1)[1].rsplit(". Return", 1)[0]
    assert _payload(task) == {"value": 1 if _evaluate_boolean(expression) else 0}


def test_modular_answer_uses_the_published_initial_value():
    task = TASK_GENERATORS["modular"](8, 13)
    match = re.search(
        r"modulo (\d+): start=(\d+)\. Operations: ([+*\-, 0-9]+)\.",
        task.prompt,
    )
    assert match
    modulus, value = map(int, match.groups()[:2])
    for operation in match.group(3).split(", "):
        operand = int(operation[1:])
        if operation[0] == "+":
            value += operand
        elif operation[0] == "-":
            value -= operand
        else:
            assert operation[0] == "*"
            value *= operand
        value %= modulus
    assert _payload(task) == {"residue": value}


def test_register_trace_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["register_trace"](8, 14)
    match = re.search(
        r"r0=(\d+), r1=(\d+), r2=(\d+)\. Apply in order: (.+)\. End",
        task.prompt,
    )
    assert match
    registers = list(map(int, match.groups()[:3]))
    for operation in match.group(4).split("; "):
        parsed = re.fullmatch(
            r"r(\d)=\(r(\d)\+(\d+)\*r(\d)\+(\d+)\) mod (\d+)",
            operation,
        )
        assert parsed
        destination, left, multiplier, right, offset, modulus = map(int, parsed.groups())
        registers[destination] = (
            registers[left] + multiplier * registers[right] + offset
        ) % modulus
    assert _payload(task) == dict(zip(("r0", "r1", "r2"), registers, strict=True))


def test_stack_trace_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["stack_trace"](11, 15)
    match = re.search(r"Begin with list (\[[^]]+\])\. Apply in order: (.+)\. Return", task.prompt)
    assert match
    state = list(ast.literal_eval(match.group(1)))
    for operation in match.group(2).split("; "):
        if operation.startswith("append "):
            state.append(int(operation.split()[-1]))
        elif operation == "rotate left by one":
            state = state[1:] + state[:1]
        elif operation.startswith("set index "):
            index, value = map(int, re.findall(r"\d+", operation))
            state[index] = value
        else:
            assert operation == "remove the last value"
            state.pop()
    assert _payload(task) == {"state": state}


def test_constraint_order_answer_is_the_unique_topological_order():
    task = TASK_GENERATORS["constraint_order"](6, 16)
    body = task.prompt.split(": ", 1)[1].rsplit(". Return", 1)[0]
    edges = re.findall(r"([A-Z]) is before ([A-Z])", body)
    nodes = set(itertools.chain.from_iterable(edges))
    remaining = set(nodes)
    order: list[str] = []
    while remaining:
        candidates = sorted(
            node
            for node in remaining
            if not any(right == node and left in remaining for left, right in edges)
        )
        assert len(candidates) == 1
        order.append(candidates[0])
        remaining.remove(candidates[0])
    assert _payload(task) == {"order": order}


def test_causal_intervention_answer_replaces_not_conditions_equation():
    task = TASK_GENERATORS["causal_intervention"](8, 17)
    equations_text = task.prompt.split("Binary structural equations: ", 1)[1].split(
        ". Intervene", 1
    )[0]
    equations = equations_text.split("; ")
    root = int(equations[0].split("=")[1])
    flips = [int(equation.rsplit(" ", 1)[1]) for equation in equations[1:]]
    intervention = re.search(r"do\(x(\d+)=(\d)\)", task.prompt)
    assert intervention
    intervention_index, intervention_value = map(int, intervention.groups())
    values = [root]
    for index, flip in enumerate(flips, start=1):
        values.append(intervention_value if index == intervention_index else values[-1] ^ flip)
    assert _payload(task) == {f"x{len(values) - 1}": values[-1]}


def test_bayes_answer_is_exact_and_recomputable():
    task = TASK_GENERATORS["bayes_update"](5, 18)
    fractions = [Fraction(value) for value in re.findall(r"\d+/\d+", task.prompt)]
    prior, likelihood_h, likelihood_not_h = fractions
    repeats = int(re.search(r"exactly (\d+) times", task.prompt).group(1))  # type: ignore[union-attr]
    posterior = prior
    for _ in range(repeats):
        numerator = posterior * likelihood_h
        posterior = numerator / (numerator + (1 - posterior) * likelihood_not_h)
    assert _payload(task) == {
        "choice": "H" if posterior >= Fraction(1, 2) else "not_H",
        "posterior": f"{posterior.numerator}/{posterior.denominator}",
    }


def test_budget_answer_is_global_optimum_with_stated_tiebreaks():
    task = TASK_GENERATORS["budget_plan"](7, 19)
    budget = int(re.search(r"under budget (\d+)", task.prompt).group(1))  # type: ignore[union-attr]
    jobs = [
        (identifier, int(cost), int(reward))
        for identifier, cost, reward in re.findall(
            r"([A-Z])\(cost=(\d+),reward=(\d+)\)", task.prompt
        )
    ]
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for mask in range(1 << len(jobs)):
        selected = tuple(jobs[index][0] for index in range(len(jobs)) if mask & (1 << index))
        cost = sum(jobs[index][1] for index in range(len(jobs)) if mask & (1 << index))
        reward = sum(jobs[index][2] for index in range(len(jobs)) if mask & (1 << index))
        if cost <= budget:
            candidates.append((reward, cost, selected))
    reward, cost, selected = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[0]
    assert _payload(task) == {
        "cost": cost,
        "reward": reward,
        "selected": list(selected),
    }


def test_symbolic_rewrite_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["symbolic_rewrite"](8, 20)
    match = re.search(r"Start with string ([A-D]+)\. Apply in order: (.+)\. Return", task.prompt)
    assert match
    state = match.group(1)
    for operation in match.group(2).split("; "):
        if operation == "rotate left":
            state = state[1:] + state[:1]
        elif operation == "reverse":
            state = state[::-1]
        else:
            source, target = re.fullmatch(r"replace every ([A-D]) with ([A-D])", operation).groups()  # type: ignore[union-attr]
            state = state.replace(source, target)
    assert _payload(task) == {"state": state}


def test_premise_audit_answer_identifies_the_only_false_claim():
    task = TASK_GENERATORS["premise_audit"](8, 21)
    table_text = task.prompt.split("Ground-truth table: ", 1)[1].split(". Exactly", 1)[0]
    values = {
        int(index): int(value) for index, value in re.findall(r"item (\d+)=(\d+)", table_text)
    }
    claims = {
        int(index): (int(item), int(value))
        for index, item, value in re.findall(
            r"claim (\d+): item (\d+) has value (\d+)", task.prompt
        )
    }
    false_claims = [index for index, (item, value) in claims.items() if values[item] != value]
    assert len(false_claims) == 1
    assert _payload(task) == {"false_claim": false_claims[0]}


def test_code_trace_answer_is_recomputable_from_prompt():
    task = TASK_GENERATORS["code_trace"](8, 22)
    values = list(ast.literal_eval(re.search(r"values=(\[[^]]+\])", task.prompt).group(1)))  # type: ignore[union-attr]
    accumulator = int(re.search(r": acc=(\d+);", task.prompt).group(1))  # type: ignore[union-attr]
    modulus = int(re.search(r"mod (\d+) == 0", task.prompt).group(1))  # type: ignore[union-attr]
    for index, value in enumerate(values):
        accumulator += 2 * value if (index + value) % modulus == 0 else -value
        accumulator += index % 2
    assert _payload(task) == {"acc": accumulator}
