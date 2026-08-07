"""Deterministic training-only curriculum for recurrence-native adapters.

The frontier campaign generators remain a disjoint evaluation registry. This
module supplies broader compositional supervision without copying their task
templates or answer payloads. Every task is closed-form, seeded, short enough
for resident training, and scales through a common depth control.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Never

CURRICULUM_SCHEMA = "aura.recurrence_training_curriculum.v1"
CURRICULUM_VERSION = "2026.07.18.1"
MAX_TRAINING_DEPTH = 32
_TERMINAL_FINAL_CUE = re.compile(
    r"(?im)^[ \t]*(?:\*\*)?final(?:_| )answer(?:\*\*)?[ \t]*:[ \t]*"
)
_TERMINAL_JSON_FENCE = re.compile(
    r"```(?:json)?[ \t]*\r?\n(?P<payload>\{.*\})[ \t]*\r?\n```[ \t]*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _parse_semantic_terminal_answer(response: str) -> dict[str, Any]:
    """Canonicalize only unambiguous terminal JSON into the strict parser."""

    from core.brain.llm.latent_cortex.frontier_tasks import (
        FrontierTaskError,
        parse_final_answer,
    )

    try:
        return parse_final_answer(response)
    except FrontierTaskError as strict_error:
        stripped = response.strip()
        candidates: list[str] = []
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)
        fence = _TERMINAL_JSON_FENCE.search(stripped)
        if fence is not None:
            candidates.append(fence.group("payload").strip())
        cues = list(_TERMINAL_FINAL_CUE.finditer(stripped))
        if len(cues) == 1:
            payload = stripped[cues[0].end() :].strip()
            fenced_payload = _TERMINAL_JSON_FENCE.fullmatch(payload)
            if fenced_payload is not None:
                payload = fenced_payload.group("payload").strip()
            candidates.append(payload)
        unique = tuple(dict.fromkeys(candidates))
        if len(unique) != 1:
            raise strict_error
        try:
            def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                decoded: dict[str, Any] = {}
                for key, item in pairs:
                    if key in decoded:
                        raise ValueError("duplicate terminal JSON key")
                    decoded[key] = item
                return decoded

            def reject_float(_value: str) -> Never:
                raise ValueError("floating-point terminal JSON is unsupported")

            decoded = json.loads(
                unique[0],
                object_pairs_hook=reject_duplicates,
                parse_float=reject_float,
                parse_constant=reject_float,
            )
            canonical = json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            raise strict_error from None
        return parse_final_answer(f"FINAL_ANSWER: {canonical}")


@dataclass(frozen=True, slots=True)
class RecurrenceTrainingTask:
    prompt: str
    answer: str
    depth: int
    family: str
    seed: int

    def __post_init__(self) -> None:
        if not self.prompt or self.prompt != self.prompt.strip() or "\x00" in self.prompt:
            raise ValueError("training task prompt is invalid")
        if not self.answer or self.answer != self.answer.strip() or "\x00" in self.answer:
            raise ValueError("training task answer is invalid")
        if type(self.depth) is not int or self.depth <= 0:
            raise ValueError("training task depth is invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training task seed is invalid")
        if not self.family or not self.family.replace("_", "").isalnum():
            raise ValueError("training task family is invalid")

    @property
    def task_id(self) -> str:
        """Globally stable sample identity, including its generator seed."""
        return f"recurrence-{self.family}-d{self.depth}-s{self.seed}"

    @property
    def domain(self) -> str:
        return self.family

    @property
    def knowledge(self) -> str:
        return "parametric"

    @property
    def grader(self) -> str:
        return "exact_json"

    @property
    def expected(self) -> dict[str, Any]:
        prefix = "FINAL_ANSWER: "
        if not self.answer.startswith(prefix):
            raise ValueError("training task answer contract is invalid")
        value = json.loads(self.answer.removeprefix(prefix))
        if not isinstance(value, dict):
            raise ValueError("training task answer must be a JSON object")
        return value

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": "recurrence_curriculum",
            "curriculum_version": CURRICULUM_VERSION,
            "generation_seed": self.seed,
        }

    def grade(self, response: str) -> dict[str, Any]:
        """Grade one unambiguous terminal JSON answer against exact content."""
        try:
            from core.brain.llm.latent_cortex.frontier_tasks import (
                FrontierTaskError,
            )

            produced = _parse_semantic_terminal_answer(response)
        except (FrontierTaskError, TypeError, ValueError):
            return {
                "correct": False,
                "parsed": None,
                "expected": self.expected,
                "reason": "unparseable",
            }
        expected = self.expected
        return {
            "correct": produced == expected,
            "parsed": produced,
            "expected": expected,
        }


def _json_answer(value: dict[str, Any]) -> str:
    return "FINAL_ANSWER: " + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _terminal_contract(*keys: str) -> str:
    rendered = ", ".join(keys)
    return (
        " You may reason before answering. Finish with exactly one final line using the "
        "envelope FINAL_ANSWER: <JSON object>. FINAL_ANSWER is the envelope, not a "
        f"JSON key. The JSON object must contain exactly these keys: {rendered}."
    )


def _rng(family: str, depth: int, seed: int) -> random.Random:
    material = f"{CURRICULUM_VERSION}:{family}:{depth}:{seed}"
    return random.Random(int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()))


def _validate_coordinates(depth: int, seed: int) -> None:
    if type(depth) is not int or not 1 <= depth <= MAX_TRAINING_DEPTH:
        raise ValueError(f"training depth must be inside [1, {MAX_TRAINING_DEPTH}]")
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed is invalid")


def khop_reachability(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("khop", depth, seed)
    n_nodes = min(24, max(8, depth + 6))
    successor = {index: rng.randrange(n_nodes) for index in range(n_nodes)}
    start = rng.randrange(n_nodes)
    node = start
    for _ in range(depth):
        node = successor[node]
    edges = ", ".join(f"{left}->{right}" for left, right in sorted(successor.items()))
    return RecurrenceTrainingTask(
        prompt=(
            f"A functional directed graph has these edges: {edges}. Start at {start} "
            f"and follow exactly {depth} edges."
            + _terminal_contract("node")
        ),
        answer=_json_answer({"node": node}),
        depth=depth,
        family="khop",
        seed=seed,
    )


def nested_boolean(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("boolean", depth, seed)
    value = rng.random() < 0.5
    expression = "1" if value else "0"
    for _ in range(depth):
        operation = rng.choice(("and", "or", "not", "xor"))
        if operation == "not":
            expression = f"(not {expression})"
            value = not value
            continue
        right_value = rng.random() < 0.5
        right = "1" if right_value else "0"
        expression = f"({expression} {operation} {right})"
        if operation == "and":
            value = value and right_value
        elif operation == "or":
            value = value or right_value
        else:
            value = value != right_value
    return RecurrenceTrainingTask(
        prompt=(
            f"Evaluate this {depth}-operation expression with 1=true, 0=false, and xor "
            f"meaning exactly one operand is true: {expression}. Return a value of 1 or 0."
            + _terminal_contract("value")
        ),
        answer=_json_answer({"value": 1 if value else 0}),
        depth=depth,
        family="boolean",
        seed=seed,
    )


def modular_chain(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("modular", depth, seed)
    modulus = rng.choice((13, 17, 19, 23))
    initial = rng.randrange(modulus)
    value = initial
    operations: list[str] = []
    for _ in range(depth):
        operation = rng.choice(("+", "*", "-"))
        operand = rng.randrange(1, modulus)
        if operation == "+":
            value = (value + operand) % modulus
        elif operation == "-":
            value = (value - operand) % modulus
        else:
            value = (value * operand) % modulus
        operations.append(f"{operation}{operand}")
    return RecurrenceTrainingTask(
        prompt=(
            f"Start at the given value and apply each operation modulo {modulus}: "
            f"start={initial}. Operations: {', '.join(operations)}."
            + _terminal_contract("residue")
        ),
        answer=_json_answer({"residue": value}),
        depth=depth,
        family="modular",
        seed=seed,
    )


def register_trace(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("register_trace", depth, seed)
    modulus = 29
    initial = [rng.randrange(modulus) for _ in range(3)]
    registers = list(initial)
    operations: list[str] = []
    for step in range(depth):
        destination = step % 3
        left = (step + 1) % 3
        right = (step + 2) % 3
        multiplier = rng.choice((1, 2, 3))
        offset = rng.randrange(1, 8)
        registers[destination] = (
            registers[left] + multiplier * registers[right] + offset
        ) % modulus
        operations.append(f"r{destination}=(r{left}+{multiplier}*r{right}+{offset}) mod {modulus}")
    return RecurrenceTrainingTask(
        prompt=(
            f"Trace three registers from r0={initial[0]}, r1={initial[1]}, "
            f"r2={initial[2]}. Apply in order: {'; '.join(operations)}. End"
            + _terminal_contract("r0", "r1", "r2")
        ),
        answer=_json_answer({"r0": registers[0], "r1": registers[1], "r2": registers[2]}),
        depth=depth,
        family="register_trace",
        seed=seed,
    )


def stack_trace(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("stack_trace", depth, seed)
    initial = [rng.randrange(1, 10) for _ in range(3)]
    state = list(initial)
    operations: list[str] = []
    for step in range(depth):
        choice = step % 4
        if choice == 0:
            value = rng.randrange(1, 10)
            state.append(value)
            operations.append(f"append {value}")
        elif choice == 1:
            state = state[1:] + state[:1]
            operations.append("rotate left by one")
        elif choice == 2:
            value = rng.randrange(1, 10)
            index = rng.randrange(len(state))
            state[index] = value
            operations.append(f"set index {index} to {value}")
        else:
            state.pop()
            operations.append("remove the last value")
            if not state:
                state.append(rng.randrange(1, 10))
    return RecurrenceTrainingTask(
        prompt=(
            f"Begin with list {initial}. Apply in order: {'; '.join(operations)}. Return"
            + _terminal_contract("state")
        ),
        answer=_json_answer({"state": state}),
        depth=depth,
        family="stack_trace",
        seed=seed,
    )


def constraint_order(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("constraint_order", depth, seed)
    size = min(9, max(4, depth + 3))
    order = [chr(ord("A") + index) for index in range(size)]
    rng.shuffle(order)
    constraints = [f"{order[index]} is before {order[index + 1]}" for index in range(size - 1)]
    rng.shuffle(constraints)
    return RecurrenceTrainingTask(
        prompt=(
            "Find the unique total order satisfying every constraint: "
            f"{'; '.join(constraints)}."
            + _terminal_contract("order")
        ),
        answer=_json_answer({"order": order}),
        depth=depth,
        family="constraint_order",
        seed=seed,
    )


def causal_intervention(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("causal_intervention", depth, seed)
    count = min(12, max(4, depth + 3))
    root = rng.randrange(2)
    flips = [rng.randrange(2) for _ in range(1, count)]
    values = [root]
    for flip in flips:
        values.append(values[-1] ^ flip)
    intervention_index = rng.randrange(1, count - 1)
    intervention_value = 1 - values[intervention_index]
    intervened = values[:intervention_index] + [intervention_value]
    for index in range(intervention_index + 1, count):
        intervened.append(intervened[-1] ^ flips[index - 1])
    equations = [f"x0={root}"] + [
        f"x{index}=x{index - 1} xor {flips[index - 1]}" for index in range(1, count)
    ]
    return RecurrenceTrainingTask(
        prompt=(
            f"Binary structural equations: {'; '.join(equations)}. Intervene with "
            f"do(x{intervention_index}={intervention_value}), replacing that equation."
            + _terminal_contract(f"x{count - 1}")
        ),
        answer=_json_answer({f"x{count - 1}": intervened[-1]}),
        depth=depth,
        family="causal_intervention",
        seed=seed,
    )


def bayes_update(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("bayes_update", depth, seed)
    prior = Fraction(rng.randrange(1, 9), 10)
    likelihood_h = Fraction(rng.randrange(2, 10), 10)
    likelihood_not_h = Fraction(rng.randrange(1, 9), 10)
    if likelihood_h == likelihood_not_h:
        likelihood_not_h = Fraction(1, 10) if likelihood_h != Fraction(1, 10) else Fraction(2, 10)
    posterior = prior
    for _ in range(depth):
        numerator = posterior * likelihood_h
        denominator = numerator + (1 - posterior) * likelihood_not_h
        posterior = numerator / denominator
    posterior_text = f"{posterior.numerator}/{posterior.denominator}"
    return RecurrenceTrainingTask(
        prompt=(
            f"P(H)={prior.numerator}/{prior.denominator}, P(E|H)="
            f"{likelihood_h.numerator}/{likelihood_h.denominator}, and P(E|not H)="
            f"{likelihood_not_h.numerator}/{likelihood_not_h.denominator}. Observe "
            f"the same conditionally independent evidence E exactly {depth} times. "
            "The choice value must be H or not_H."
            + _terminal_contract("posterior", "choice")
        ),
        answer=_json_answer(
            {
                "choice": "H" if posterior >= Fraction(1, 2) else "not_H",
                "posterior": posterior_text,
            }
        ),
        depth=depth,
        family="bayes_update",
        seed=seed,
    )


def budget_plan(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("budget_plan", depth, seed)
    count = min(10, max(4, depth + 3))
    jobs = [
        (chr(ord("A") + index), rng.randrange(1, 8), rng.randrange(2, 15)) for index in range(count)
    ]
    budget = max(3, sum(cost for _identifier, cost, _reward in jobs) // 2)
    best_ids: tuple[str, ...] = ()
    best_reward = -1
    best_cost = 0
    for mask in range(1 << count):
        selected = tuple(jobs[index][0] for index in range(count) if mask & (1 << index))
        cost = sum(jobs[index][1] for index in range(count) if mask & (1 << index))
        reward = sum(jobs[index][2] for index in range(count) if mask & (1 << index))
        if cost > budget:
            continue
        if reward > best_reward or (
            reward == best_reward
            and (cost < best_cost or (cost == best_cost and selected < best_ids))
        ):
            best_ids, best_reward, best_cost = selected, reward, cost
    job_text = ", ".join(
        f"{identifier}(cost={cost},reward={reward})" for identifier, cost, reward in jobs
    )
    return RecurrenceTrainingTask(
        prompt=(
            f"Choose any subset of jobs under budget {budget}: {job_text}. Maximize "
            "reward, then minimize cost, then choose the lexicographically smallest "
            "ID list."
            + _terminal_contract("selected", "cost", "reward")
        ),
        answer=_json_answer({"cost": best_cost, "reward": best_reward, "selected": list(best_ids)}),
        depth=depth,
        family="budget_plan",
        seed=seed,
    )


def symbolic_rewrite(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("symbolic_rewrite", depth, seed)
    initial = "".join(rng.choice("ABCD") for _ in range(max(4, depth + 2)))
    state = initial
    operations: list[str] = []
    for step in range(depth):
        choice = step % 3
        if choice == 0:
            state = state[1:] + state[:1]
            operations.append("rotate left")
        elif choice == 1:
            source, target = rng.sample(list("ABCD"), 2)
            state = state.replace(source, target)
            operations.append(f"replace every {source} with {target}")
        else:
            state = state[::-1]
            operations.append("reverse")
    return RecurrenceTrainingTask(
        prompt=(
            f"Start with string {initial}. Apply in order: {'; '.join(operations)}. Return"
            + _terminal_contract("state")
        ),
        answer=_json_answer({"state": state}),
        depth=depth,
        family="symbolic_rewrite",
        seed=seed,
    )


def premise_audit(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("premise_audit", depth, seed)
    count = min(12, max(4, depth + 3))
    values = [rng.randrange(1, 20) for _ in range(count)]
    false_index = rng.randrange(count)
    claims = []
    for index, value in enumerate(values):
        claimed = value if index != false_index else value + rng.choice((-2, -1, 1, 2))
        claims.append(f"claim {index}: item {index} has value {claimed}")
    table = ", ".join(f"item {index}={value}" for index, value in enumerate(values))
    return RecurrenceTrainingTask(
        prompt=(
            f"Ground-truth table: {table}. Exactly one claim conflicts with the table: "
            f"{'; '.join(claims)}."
            + _terminal_contract("false_claim")
        ),
        answer=_json_answer({"false_claim": false_index}),
        depth=depth,
        family="premise_audit",
        seed=seed,
    )


def code_trace(depth: int, seed: int) -> RecurrenceTrainingTask:
    _validate_coordinates(depth, seed)
    rng = _rng("code_trace", depth, seed)
    values = [rng.randrange(1, 10) for _ in range(max(4, depth + 2))]
    modulus = rng.choice((3, 4, 5))
    accumulator = rng.randrange(1, 8)
    initial = accumulator
    for index, value in enumerate(values):
        if (index + value) % modulus == 0:
            accumulator += value * 2
        else:
            accumulator -= value
        accumulator += index % 2
    return RecurrenceTrainingTask(
        prompt=(
            f"Trace this pseudocode exactly with values={values}: acc={initial}; for each "
            f"(index,value), if (index+value) mod {modulus} == 0 then acc += 2*value "
            "else acc -= value; then acc += index mod 2. Indexing starts at zero. "
            + _terminal_contract("acc")
        ),
        answer=_json_answer({"acc": accumulator}),
        depth=depth,
        family="code_trace",
        seed=seed,
    )


TASK_GENERATORS: MappingProxyType[
    str,
    Callable[[int, int], RecurrenceTrainingTask],
] = MappingProxyType(
    {
        "khop": khop_reachability,
        "boolean": nested_boolean,
        "modular": modular_chain,
        "register_trace": register_trace,
        "stack_trace": stack_trace,
        "constraint_order": constraint_order,
        "causal_intervention": causal_intervention,
        "bayes_update": bayes_update,
        "budget_plan": budget_plan,
        "symbolic_rewrite": symbolic_rewrite,
        "premise_audit": premise_audit,
        "code_trace": code_trace,
    }
)
RECURRENCE_TRAINING_FAMILIES = tuple(TASK_GENERATORS)


def _sample_seed(
    *,
    root_seed: int,
    family: str,
    depth: int,
    cell: int,
    attempt: int,
) -> int:
    coordinates = f"{CURRICULUM_VERSION}:{root_seed}:{family}:{depth}:{cell}:{attempt}"
    # Keep public task coordinates portable across strict signed-64-bit JSON
    # implementations while retaining more than the preregistered 60 bits of
    # generation entropy.
    return int.from_bytes(
        hashlib.sha256(coordinates.encode("ascii")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)


def task_battery(
    families: Sequence[str],
    depths: Sequence[int],
    per_cell: int,
    *,
    seed: int,
    excluded_prompts: Sequence[str] = (),
    excluded_task_ids: Sequence[str] = (),
) -> list[RecurrenceTrainingTask]:
    if (
        isinstance(families, (str, bytes))
        or not families
        or any(not isinstance(family, str) or not family for family in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("training families must be nonempty and unique")
    if any(family not in TASK_GENERATORS for family in families):
        raise ValueError("training family is unknown")
    if (
        isinstance(depths, (str, bytes))
        or not depths
        or len(set(depths)) != len(depths)
        or any(type(depth) is not int or not 1 <= depth <= MAX_TRAINING_DEPTH for depth in depths)
    ):
        raise ValueError("training depths are invalid")
    if type(per_cell) is not int or per_cell <= 0:
        raise ValueError("training per_cell is invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed is invalid")
    if (
        isinstance(excluded_prompts, (str, bytes))
        or any(not isinstance(prompt, str) for prompt in excluded_prompts)
        or isinstance(excluded_task_ids, (str, bytes))
        or any(not isinstance(task_id, str) for task_id in excluded_task_ids)
    ):
        raise ValueError("training exclusions are invalid")
    tasks: dict[tuple[str, int, int], RecurrenceTrainingTask] = {}
    all_prompts = set(excluded_prompts)
    all_ids = set(excluded_task_ids)
    for family in families:
        generator = TASK_GENERATORS[family]
        for depth in depths:
            coordinate_prompts: set[str] = set()
            for cell in range(per_cell):
                for attempt in range(1_024):
                    sample_seed = _sample_seed(
                        root_seed=seed,
                        family=family,
                        depth=depth,
                        cell=cell,
                        attempt=attempt,
                    )
                    task = generator(depth, sample_seed)
                    if (
                        task.prompt not in coordinate_prompts
                        and task.prompt not in all_prompts
                        and task.task_id not in all_ids
                    ):
                        break
                else:  # pragma: no cover - finite generator exhaustion guard
                    raise RuntimeError(
                        f"training generator exhausted unique prompts: {family}/{depth}"
                    )
                coordinate_prompts.add(task.prompt)
                all_prompts.add(task.prompt)
                all_ids.add(task.task_id)
                tasks[(family, depth, cell)] = task

    # Execute one example from every family/depth stratum before repeating a
    # stratum. The per-round order is seed-bound, which removes contiguous
    # curriculum blocks without introducing process-global RNG state.
    strata = [(family, depth) for family in families for depth in depths]
    scheduled: list[RecurrenceTrainingTask] = []
    for cell in range(per_cell):
        round_order = list(strata)
        round_rng = random.Random(
            int.from_bytes(
                hashlib.sha256(
                    f"schedule|{seed}|{cell}".encode("ascii")
                ).digest()[:8],
                "big",
            )
        )
        round_rng.shuffle(round_order)
        scheduled.extend(tasks[(family, depth, cell)] for family, depth in round_order)
    return scheduled


def disjoint_task_split(
    *,
    families: Sequence[str],
    depths: Sequence[int],
    train_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[RecurrenceTrainingTask], list[RecurrenceTrainingTask]]:
    """Mint a globally disjoint train/holdout split from separate roots."""
    train = task_battery(families, depths, train_per_cell, seed=seed)
    holdout = task_battery(
        families,
        depths,
        holdout_per_cell,
        seed=seed + 7_919,
        excluded_prompts=tuple(task.prompt for task in train),
        excluded_task_ids=tuple(task.task_id for task in train),
    )
    train_prompts = {task.prompt for task in train}
    train_ids = {task.task_id for task in train}
    holdout_prompts = {task.prompt for task in holdout}
    holdout_ids = {task.task_id for task in holdout}
    if train_prompts & holdout_prompts:
        raise RuntimeError("recurrence train and holdout prompts overlap")
    if train_ids & holdout_ids:
        raise RuntimeError("recurrence train and holdout identities overlap")
    if len(train_ids) != len(train) or len(holdout_ids) != len(holdout):
        raise RuntimeError("recurrence task identity is not globally unique")
    return train, holdout


__all__ = [
    "CURRICULUM_SCHEMA",
    "CURRICULUM_VERSION",
    "MAX_TRAINING_DEPTH",
    "RECURRENCE_TRAINING_FAMILIES",
    "RecurrenceTrainingTask",
    "TASK_GENERATORS",
    "disjoint_task_split",
    "task_battery",
]
