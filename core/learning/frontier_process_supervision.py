"""Training-only verified process traces for the broad frontier registry.

The public prompt is the only model-visible input. Exact answers, actions, and
states remain issuer-side teaching evidence and are represented publicly only
by commitments. This prevents the executable teacher from becoming an answer
producer during held-out inference.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from core.brain.llm.latent_cortex.frontier_tasks import (
    CONTAMINATION_SAFE_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
    FrontierTask,
    generate_task,
)
from core.learning.public_frontier_action_compiler import (
    compile_public_frontier_raw_actions,
)
from core.learning.recurrence_curriculum import (
    RecurrenceTrainingTask,
    StructuredTransitionProgram,
    StructuredTransitionTrace,
)
from core.learning.recurrent_action_schema import (
    ACTION_NULL,
    OP_CAUSAL_CHAIN,
    OP_PAIR_ADD,
    OP_PAIR_COPY,
    OP_PAIR_DIV,
    OP_PAIR_EUCLID_STEP,
    OP_PAIR_MUL_IMMEDIATE,
    OP_PAIR_PRODUCT,
    OP_PAIR_SET,
    OP_PAIR_SIGNED_SUB_IMMEDIATE,
    OP_PAIR_SUB_IMMEDIATE,
    OP_RANKED_COMMIT,
    OP_RATIO_BAND,
    OP_RATIO_CHOICE,
    OP_SET_SCALAR,
    OP_SIGNED_PAIR_ADD_IMMEDIATE,
    OP_SIGNED_RANKED_GREATER,
)
from core.learning.recurrent_work_memory import (
    MathematicsWorkMemoryTrace,
    compile_mathematics_work_memory,
)

FRONTIER_PROCESS_SCHEMA: Final = "aura.frontier_process_supervision.v2"
PROCESS_RADIX: Final = 31
MAX_PROCESS_INTEGER: Final = PROCESS_RADIX**2 - 1

_MATH_RE = re.compile(
    r"From the set (?P<values>\[.*?\]), choose exactly (?P<choose>\d+) distinct "
    r"values\. Adjacent values in sorted chosen order must differ by at least "
    r"(?P<gap>\d+), and the chosen sum must be from (?P<low>-?\d+) through "
    r"(?P<high>-?\d+), inclusive\."
)
_CODING_RE = re.compile(
    r"The two inputs, in order, are (?P<cases>\[\[.*?\]\])\. Return each result",
    re.DOTALL,
)
_SCIENCE_RE = re.compile(
    r"baseline values (?P<a>[a-z][a-z0-9_]*)=(?P<a_value>-?\d+), "
    r"(?P<b>[a-z][a-z0-9_]*)=(?P<b_value>-?\d+), "
    r"(?P<c>[a-z][a-z0-9_]*)=(?P<c_value>-?\d+)\. "
    r"Independent interventions produced these changes relative to baseline: "
    r"setting [a-z][a-z0-9_]* up by (?P<root_step>\d+) changed "
    r"[a-z][a-z0-9_]* by \+(?P<root_mediator_change>\d+) and "
    r"[a-z][a-z0-9_]* by \+(?P<root_downstream_change>\d+);.*?"
    r"predict the absolute value of [a-z][a-z0-9_]* when "
    r"[a-z][a-z0-9_]* is set (?P<query_step>\d+) above baseline",
    re.DOTALL,
)
_PLANNING_RE = re.compile(
    r"overall horizon (?P<horizon>\d+)\. Tasks are (?P<tasks>\[\{.*?\}\])\. "
    r"Maximize total reward",
    re.DOTALL,
)
_CALIBRATION_RE = re.compile(
    r"Before evidence E, hypothesis H has probability (?P<prior>\d+/\d+)\. "
    r"The likelihood of E is (?P<likelihood_h>\d+/\d+) if H is true and "
    r"(?P<likelihood_not_h>\d+/\d+) if H is false\."
)
_PREMISE_RE = re.compile(
    r"The data are (?P<rows>\[\{.*?\}\])\. The claim says project "
    r"(?P<claim>[A-Z][A-Z0-9_]*) has the highest score",
    re.DOTALL,
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _digits(value: int) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= MAX_PROCESS_INTEGER:
        raise ValueError("frontier process integer is outside the two-slot vocabulary")
    return value % PROCESS_RADIX, value // PROCESS_RADIX


def _signed_digits(value: int) -> tuple[int, int]:
    if type(value) is not int:
        raise ValueError("frontier process signed integer is invalid")
    encoded = value * 2 if value >= 0 else (-value * 2) - 1
    return _digits(encoded)


def _literal(text: str, *, role: str) -> Any:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise ValueError(f"{role} literal is invalid") from None
    return value


def _program(
    *,
    family: str,
    field_names: tuple[str, ...],
    states: list[tuple[int, ...]],
    action_field_names: tuple[str, ...],
    actions: list[tuple[int, ...]],
) -> StructuredTransitionProgram:
    depth = len(actions)
    if not depth or len(states) != depth + 1:
        raise ValueError("frontier process trajectory is incomplete")
    trace = StructuredTransitionTrace(
        family=family,
        depth=depth,
        field_names=field_names,
        states=tuple(states),
    )
    return StructuredTransitionProgram(
        state_trace=trace,
        action_field_names=action_field_names,
        actions=tuple(actions),
    )


def _semantic_micro_states(
    *,
    family: str,
    initial_state: tuple[int, ...],
    actions: tuple[tuple[int, ...], ...],
) -> list[tuple[int, ...]]:
    """Execute the private reference semantics for answer-blind micro-actions."""

    if len(initial_state) < 5 or initial_state[0] != 0 or initial_state[-1] != 0 or not actions:
        raise ValueError("semantic micro-program initial state is invalid")
    states = [initial_state]
    values = list(initial_state[1:-1])

    def pair(low_slot: int) -> int:
        if not 0 <= low_slot < len(values) - 1:
            raise ValueError("semantic micro-program pair address is invalid")
        return values[low_slot] + PROCESS_RADIX * values[low_slot + 1]

    def write_pair(low_slot: int, value: int) -> None:
        if not 0 <= low_slot < len(values) - 1 or not 0 <= value <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic micro-program pair result is outside its bank")
        values[low_slot], values[low_slot + 1] = _digits(value)

    def signed_decode(encoded: int) -> int:
        return encoded // 2 if encoded % 2 == 0 else -((encoded + 1) // 2)

    def signed_encode(value: int) -> int:
        encoded = value * 2 if value >= 0 else (-value * 2) - 1
        if encoded > MAX_PROCESS_INTEGER:
            raise ValueError("semantic micro-program signed result is outside its bank")
        return encoded

    for step, action in enumerate(actions, 1):
        if len(action) != 7 or any(
            type(value) is not int or not 0 <= value < ACTION_NULL for value in action
        ):
            raise ValueError("semantic micro-program action is invalid")
        opcode, arg0, arg1, arg2, arg3, arg4, arg5 = action
        if opcode == OP_PAIR_SET:
            write_pair(arg0, arg1 + PROCESS_RADIX * arg2)
        elif opcode == OP_PAIR_ADD:
            write_pair(arg0, pair(arg1) + pair(arg2))
        elif opcode == OP_PAIR_MUL_IMMEDIATE:
            write_pair(arg0, pair(arg0) * arg1)
        elif opcode == OP_PAIR_PRODUCT:
            write_pair(arg0, arg1 * arg2)
        elif opcode == OP_PAIR_SUB_IMMEDIATE:
            result = pair(arg0) - arg1
            if result < 0:
                raise ValueError("semantic unsigned subtraction underflowed")
            write_pair(arg0, result)
        elif opcode == OP_PAIR_SIGNED_SUB_IMMEDIATE:
            write_pair(arg0, signed_encode(pair(arg0) - arg1))
        elif opcode == OP_PAIR_COPY:
            write_pair(arg0, pair(arg1))
        elif opcode == OP_PAIR_EUCLID_STEP:
            left = pair(arg0)
            right = pair(arg1)
            write_pair(arg0, right if right else left)
            write_pair(arg1, left % right if right else 0)
        elif opcode == OP_PAIR_DIV:
            numerator = pair(arg1)
            denominator = pair(arg2)
            if denominator < 1 or numerator % denominator:
                raise ValueError("semantic pair division is not exact")
            write_pair(arg0, numerator // denominator)
        elif opcode in {OP_RATIO_CHOICE, OP_RATIO_BAND}:
            if not 0 <= arg0 < len(values):
                raise ValueError("semantic ratio destination is invalid")
            numerator = pair(arg1)
            denominator = pair(arg2)
            if denominator < 1:
                raise ValueError("semantic ratio denominator is invalid")
            if opcode == OP_RATIO_CHOICE:
                values[arg0] = int(2 * numerator >= denominator) + 1
            else:
                percentage = (100 * numerator) // denominator
                values[arg0] = (
                    1 if percentage < 50 else 2 if percentage < 70 else 3 if percentage < 90 else 4
                )
        elif opcode == OP_SIGNED_PAIR_ADD_IMMEDIATE:
            delta = signed_decode(arg1)
            write_pair(arg0, signed_encode(signed_decode(pair(arg0)) + delta))
        elif opcode == OP_SIGNED_RANKED_GREATER:
            if not all(0 <= slot < len(values) for slot in (arg0, arg4, arg5)):
                raise ValueError("semantic ranked comparison address is invalid")
            candidate = signed_decode(pair(arg1))
            incumbent = signed_decode(pair(arg2))
            values[arg0] = int(
                values[arg5] == 0
                or candidate > incumbent
                or (candidate == incumbent and arg3 < values[arg4])
            )
        elif opcode == OP_RANKED_COMMIT:
            if not 0 <= arg0 < len(values):
                raise ValueError("semantic ranked commit flag is invalid")
            if values[arg0]:
                values[0] = arg2
                write_pair(1, pair(arg1))
                values[3] = arg3
            values[4] = 1
        elif opcode == OP_SET_SCALAR:
            if not 0 <= arg0 < len(values):
                raise ValueError("semantic scalar destination is invalid")
            values[arg0] = arg1
        elif opcode == OP_CAUSAL_CHAIN:
            change = arg3 + PROCESS_RADIX * arg4
            if arg0 <= 2 and arg1 <= 2:
                if values[8] == 0:
                    if arg0 == arg1 or arg5 != 0:
                        raise ValueError("causal root first edge is invalid")
                    values[:] = [arg0, arg1, arg2, arg3, arg4, 0, 0, 0, 1]
                elif values[8] == 1:
                    if (
                        arg0 != values[0]
                        or arg1 in {arg0, values[1]}
                        or arg2 != values[2]
                        or arg5 != 1
                    ):
                        raise ValueError("causal root second edge is invalid")
                    values[:] = [
                        arg0,
                        values[1],
                        values[3],
                        values[4],
                        arg1,
                        arg3,
                        arg4,
                        arg2,
                        2,
                    ]
                elif values[8] == 2:
                    if (
                        arg0 == values[0]
                        or arg0 not in {values[1], values[4]}
                        or arg1 == arg0
                        or arg1 not in {values[1], values[4]}
                        or arg5 != 1
                    ):
                        raise ValueError("causal mediator edge is invalid")
                    root_mediator_change = (
                        values[2] + PROCESS_RADIX * values[3]
                        if arg0 == values[1]
                        else values[5] + PROCESS_RADIX * values[6]
                    )
                    if root_mediator_change >= ACTION_NULL or change >= ACTION_NULL:
                        raise ValueError("causal gain numerator exceeds scalar state")
                    values[:] = [
                        values[0],
                        arg0,
                        arg1,
                        root_mediator_change,
                        values[7],
                        change,
                        arg2,
                        0,
                        3,
                    ]
                else:
                    raise ValueError("causal intervention arrived out of order")
            elif arg0 <= 2 and arg1 == 3:
                if (
                    values[8] != 3
                    or arg0 != values[2]
                    or any(value != 0 for value in (arg2, arg3, arg4))
                    or arg5 != 1
                ):
                    raise ValueError("causal downstream null intervention is invalid")
                values[8] = 4
            elif arg0 == 3 and arg1 == 3:
                if (
                    values[8] != 4
                    or not 1 <= arg2 < ACTION_NULL
                    or any(value != 0 for value in (arg3, arg4))
                    or arg5 != 1
                ):
                    raise ValueError("causal prediction query is invalid")
                if values[3] % values[4] or values[5] % values[6]:
                    raise ValueError("causal public effects are not exact")
                effect = arg2 * (values[3] // values[4]) * (values[5] // values[6])
                values[3], values[4] = _digits(effect)
                values[5:8] = [0, 0, 0]
                values[8] = 5
            elif arg0 == 4 and arg1 == 4:
                if (
                    values[8] not in {5, 6}
                    or any(value != 0 for value in (arg2, arg3, arg4))
                    or arg5 != 1
                ):
                    raise ValueError("causal baseline commit is invalid")
                if values[7] == values[2]:
                    write_pair(3, pair(3) + pair(5))
                    values[8] = 6
            else:
                raise ValueError("causal chain instruction is invalid")
        else:
            raise ValueError(f"semantic micro-program opcode {opcode} is unsupported")
        if any(not 0 <= value < ACTION_NULL for value in values):
            raise ValueError("semantic micro-program state left the categorical bank")
        states.append((step, *values, int(step == len(actions))))
    return states


def _semantic_micro_program(
    *,
    family: str,
    public_prompt: str,
    field_names: tuple[str, ...],
    initial_state: tuple[int, ...],
) -> StructuredTransitionProgram:
    action_field_names, actions = compile_public_frontier_raw_actions(
        public_prompt,
        family,
    )
    states = _semantic_micro_states(
        family=family,
        initial_state=initial_state,
        actions=actions,
    )
    return _program(
        family=family,
        field_names=field_names,
        states=states,
        action_field_names=action_field_names,
        actions=list(actions),
    )


def _novel_algorithms(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    sequence = expected.get("sequence")
    checksum = expected.get("checksum")
    values_match = re.search(r"original position order, are (?P<values>\[.*?\])\.", prompt)
    if (
        not isinstance(sequence, list)
        or not sequence
        or any(type(value) is not int for value in sequence)
        or type(checksum) is not int
        or values_match is None
    ):
        raise ValueError("frontier traversal evidence is invalid")
    values = _literal(values_match.group("values"), role="traversal values")
    if (
        not isinstance(values, list)
        or len(values) != len(sequence)
        or sorted(values) != sorted(sequence)
    ):
        raise ValueError("frontier traversal sequence differs from public values")
    states = [(0, 0, len(sequence), 0, 0)]
    actions: list[tuple[int, ...]] = []
    running = 0
    unused = set(range(len(values)))
    for step, value in enumerate(sequence, 1):
        index = min(index for index in unused if values[index] == value)
        unused.remove(index)
        running += step * value
        value_lo, value_hi = _digits(value)
        checksum_lo, checksum_hi = _digits(running % (MAX_PROCESS_INTEGER + 1))
        actions.append((index, value_lo, value_hi, checksum_lo, checksum_hi))
        states.append(
            (step, index, len(sequence) - step, running % PROCESS_RADIX, int(step == len(sequence)))
        )
    if running != checksum:
        raise ValueError("frontier traversal checksum differs from its sequence")
    return _program(
        family="frontier_novel_algorithms",
        field_names=("pc", "current_index", "remaining", "checksum_mod", "done"),
        states=states,
        action_field_names=("selected_index", "value_lo", "value_hi", "checksum_lo", "checksum_hi"),
        actions=actions,
    )


def _mathematics(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _MATH_RE.search(prompt)
    count = expected.get("count")
    witness = expected.get("witness")
    if match is None or type(count) is not int or not isinstance(witness, list):
        raise ValueError("frontier mathematics evidence is invalid")
    values = _literal(match.group("values"), role="mathematics values")
    choose = int(match.group("choose"))
    gap = int(match.group("gap"))
    low = int(match.group("low"))
    high = int(match.group("high"))
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError("frontier mathematics values are invalid")
    witness_tuple = tuple(witness)
    states = [(0, 0, 0, 0, 0)]
    low_lo, low_hi = _signed_digits(low)
    high_lo, high_hi = _signed_digits(high)
    actions: list[tuple[int, ...]] = [(choose, gap, low_lo, low_hi, high_lo, high_hi)]
    states.append((1, 0, 0, 0, 0))
    final_valid: list[tuple[int, ...]] = []
    for index, value in enumerate(values):
        prefix = values[: index + 1]
        valid = [
            combo
            for combo in itertools.combinations(prefix, choose)
            if all(right - left >= gap for left, right in zip(combo, combo[1:], strict=False))
            and low <= sum(combo) <= high
        ]
        current = len(valid)
        value_lo, value_hi = _signed_digits(value)
        actions.append((index, value_lo, value_hi, 0, 0, 0))
        count_lo, count_hi = _digits(current)
        current_witness_head = values.index(min(valid)[0]) + 1 if valid else 0
        states.append(
            (
                index + 2,
                count_lo,
                count_hi,
                current_witness_head,
                int(index + 1 == len(values)),
            )
        )
        final_valid = valid
    if len(final_valid) != count or (witness_tuple and witness_tuple not in final_valid):
        raise ValueError("frontier mathematics process differs from verified answer")
    return _program(
        family="frontier_mathematics",
        field_names=("pc", "count_lo", "count_hi", "witness_head", "done"),
        states=states,
        action_field_names=(
            "arg0",
            "arg1",
            "arg2",
            "arg3",
            "arg4",
            "arg5",
        ),
        actions=actions,
    )


def _coding(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _CODING_RE.search(prompt)
    returns = expected.get("returns")
    if match is None or not isinstance(returns, list) or len(returns) != 2:
        raise ValueError("frontier coding evidence is invalid")
    cases = _literal(match.group("cases"), role="coding cases")
    if not isinstance(cases, list) or len(cases) != len(returns):
        raise ValueError("frontier coding cases are invalid")
    names = sorted({event[0] for case in cases for event in case})
    if len(names) > 4:
        raise ValueError("frontier coding process exceeds its four balance registers")

    def balance_registers(values: dict[str, int]) -> tuple[int, ...]:
        encoded: list[int] = []
        for name in names:
            encoded.extend(_signed_digits(values.get(name, 0)))
        encoded.extend((0, 0) * (4 - len(names)))
        return tuple(encoded)

    balances: dict[str, int] = {}
    for case_index, case in enumerate(cases):
        balances = {}
        pressures: list[int] = []
        for event in case:
            if not isinstance(event, tuple) or len(event) != 2:
                raise ValueError("frontier coding event is invalid")
            name, delta = event
            balances[name] = balances.get(name, 0) + delta
            if balances[name] == 0:
                del balances[name]
            pressure = sum(abs(value) for value in balances.values())
            pressures.append(pressure)
        expected_return = returns[case_index]
        expected_state = [[name, balances[name]] for name in sorted(balances)]
        if expected_return != {"state": expected_state, "pressure": pressures}:
            raise ValueError("frontier coding process differs from verified answer")
    program = _semantic_micro_program(
        family="frontier_coding",
        public_prompt=prompt,
        field_names=(
            "pc",
            "case_index",
            "balance0_lo",
            "balance0_hi",
            "balance1_lo",
            "balance1_hi",
            "balance2_lo",
            "balance2_hi",
            "balance3_lo",
            "balance3_hi",
            "done",
        ),
        initial_state=(0, 0, *balance_registers({}), 0),
    )
    expected_terminal = (
        program.state_trace.depth,
        len(cases) - 1,
        *balance_registers(balances),
        1,
    )
    if program.state_trace.states[-1] != expected_terminal:
        raise RuntimeError("frontier coding micro-program differs from public execution")
    return program


def _scientific(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    program = _semantic_micro_program(
        family="frontier_scientific_inference",
        public_prompt=prompt,
        field_names=(
            "pc",
            "root_index",
            "mediator_index",
            "downstream_index",
            "prediction_lo",
            "prediction_hi",
            "scratch_lo",
            "scratch_hi",
            "variable_index",
            "stage",
            "done",
        ),
        initial_state=(0,) * 11,
    )
    terminal = program.state_trace.states[-1]
    match = _SCIENCE_RE.search(prompt)
    if match is None:
        raise ValueError("frontier scientific evidence is invalid")
    labels = [match.group("a"), match.group("b"), match.group("c")]
    try:
        observed = {
            "root": labels[terminal[1]],
            "mediator": labels[terminal[2]],
            "downstream": labels[terminal[3]],
            "predicted_downstream": terminal[4] + PROCESS_RADIX * terminal[5],
        }
    except (IndexError, TypeError):
        raise ValueError("frontier scientific terminal state is invalid") from None
    if terminal[9] != 6 or observed != expected:
        raise ValueError("frontier scientific process differs from verified answer")
    return program


def _planning(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _PLANNING_RE.search(prompt)
    order = expected.get("order")
    if match is None or not isinstance(order, list) or not order:
        raise ValueError("frontier planning evidence is invalid")
    tasks = _literal(match.group("tasks"), role="planning tasks")
    by_name = {task["name"]: task for task in tasks}
    names = sorted(by_name)
    elapsed = 0
    reward = 0
    completed: set[str] = set()
    states = [(0, 0, 0, 0, 0)]
    actions: list[tuple[int, ...]] = []
    for step, name in enumerate(order, 1):
        task = by_name[name]
        if not set(task["requires"]).issubset(completed):
            raise ValueError("frontier planning order violates dependencies")
        elapsed += task["duration"]
        reward += task["reward"]
        if elapsed > task["deadline"] or elapsed > int(match.group("horizon")):
            raise ValueError("frontier planning order violates a deadline")
        dependency = names.index(task["requires"][0]) + 1 if task["requires"] else 0
        actions.append(
            (names.index(name), task["duration"], task["deadline"], task["reward"], dependency)
        )
        reward_lo, reward_hi = _digits(reward)
        states.append((step, elapsed, reward_lo, reward_hi, int(step == len(order))))
        completed.add(name)
    if reward != expected.get("reward") or elapsed != expected.get("makespan"):
        raise ValueError("frontier planning process differs from verified answer")
    return _program(
        family="frontier_long_horizon_planning",
        field_names=("pc", "elapsed", "reward_lo", "reward_hi", "done"),
        states=states,
        action_field_names=("task_index", "duration", "deadline", "reward", "dependency_code"),
        actions=actions,
    )


def _calibration(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    try:
        posterior = Fraction(expected["posterior"])
        choice = 1 if expected["choice"] == "H" else 0
        band = ("below_50", "50_to_69", "70_to_89", "90_to_100").index(expected["confidence_band"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("frontier calibration evidence is invalid") from None
    match = _CALIBRATION_RE.search(prompt)
    if match is None:
        raise ValueError("frontier calibration public evidence is invalid")
    try:
        prior = Fraction(match.group("prior"))
        likelihood_h = Fraction(match.group("likelihood_h"))
        likelihood_not_h = Fraction(match.group("likelihood_not_h"))
        public_posterior = (likelihood_h * prior) / (
            likelihood_h * prior + likelihood_not_h * (1 - prior)
        )
    except (ValueError, ZeroDivisionError):
        raise ValueError("frontier calibration public evidence is invalid") from None
    if public_posterior != posterior:
        raise ValueError("frontier calibration public evidence differs from verified answer")
    num_lo, num_hi = _digits(posterior.numerator)
    den_lo, den_hi = _digits(posterior.denominator)
    program = _semantic_micro_program(
        family="frontier_calibration",
        public_prompt=prompt,
        field_names=(
            "pc",
            "numerator_lo",
            "numerator_hi",
            "denominator_lo",
            "denominator_hi",
            "choice",
            "confidence_band",
            "scratch_lo",
            "scratch_hi",
            "done",
        ),
        initial_state=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    expected_terminal = (
        program.state_trace.depth,
        num_lo,
        num_hi,
        den_lo,
        den_hi,
        choice + 1,
        band + 1,
        0,
        0,
        1,
    )
    if program.state_trace.states[-1] != expected_terminal:
        raise RuntimeError("frontier calibration micro-program differs from Bayes evidence")
    return program


def _premise(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _PREMISE_RE.search(prompt)
    if match is None:
        raise ValueError("frontier premise evidence is invalid")
    rows = _literal(match.group("rows"), role="premise rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frontier premise rows are invalid")
    best_index = 0
    names = sorted(row["name"] for row in rows)
    best_score = -10_000
    for index, row in enumerate(rows):
        score = row["impact"] * row["reliability"] - row["cost"]
        if score > best_score or (score == best_score and row["name"] < rows[best_index]["name"]):
            best_index = index
            best_score = score
    winner = rows[best_index]["name"]
    if winner != expected.get("actual_winner") or best_score != expected.get("actual_score"):
        raise ValueError("frontier premise process differs from verified answer")
    if bool(match.group("claim") == winner) is not expected.get("premise_valid"):
        raise ValueError("frontier premise validity differs from verified answer")
    program = _semantic_micro_program(
        family="frontier_misleading_premise",
        public_prompt=prompt,
        field_names=(
            "pc",
            "winner_index",
            "score_lo",
            "score_hi",
            "winner_name_rank",
            "has_winner",
            "candidate_score_lo",
            "candidate_score_hi",
            "candidate_wins",
            "done",
        ),
        initial_state=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )
    best_lo, best_hi = _signed_digits(best_score)
    expected_prefix = (
        program.state_trace.depth,
        best_index,
        best_lo,
        best_hi,
        names.index(winner),
        1,
    )
    if program.state_trace.states[-1][: len(expected_prefix)] != expected_prefix:
        raise RuntimeError("frontier premise micro-program differs from public ranking")
    return program


_COMPILERS = {
    "novel_algorithms": _novel_algorithms,
    "mathematics": _mathematics,
    "coding": _coding,
    "scientific_inference": _scientific,
    "long_horizon_planning": _planning,
    "calibration": _calibration,
    "misleading_premise": _premise,
}


@dataclass(frozen=True, slots=True)
class FrontierProcessSupervision:
    """Private process teacher plus a value-free public runtime contract."""

    source_task_id: str
    public_prompt: str
    answer: str
    program: StructuredTransitionProgram
    work_memory_trace: MathematicsWorkMemoryTrace | None = None

    @property
    def public_commitment(self) -> dict[str, Any]:
        body = {
            "schema": FRONTIER_PROCESS_SCHEMA,
            "source_task_id": self.source_task_id,
            "public_prompt_sha256": hashlib.sha256(self.public_prompt.encode("utf-8")).hexdigest(),
            "family": self.program.state_trace.family,
            "depth": self.program.state_trace.depth,
            "state_field_names": list(self.program.state_trace.field_names),
            "action_field_names": list(self.program.action_field_names),
            "program_sha256": self.program.program_sha256,
            "work_memory": (
                self.work_memory_trace.public_commitment()
                if self.work_memory_trace is not None
                else None
            ),
            "private_state_action_values_exposed": False,
            "final_answer_exposed": False,
            "runtime_teacher_available": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}

    def public_inference_request(self) -> dict[str, Any]:
        """Return exactly what a teacher-removed recurrent runtime may consume."""

        return {
            "schema": FRONTIER_PROCESS_SCHEMA,
            "public_prompt": self.public_prompt,
            "runtime_teacher_available": False,
        }

    def to_training_task(self) -> RecurrenceTrainingTask:
        seed = int(hashlib.sha256(self.source_task_id.encode("ascii")).hexdigest()[:15], 16)
        return RecurrenceTrainingTask(
            prompt=self.public_prompt,
            answer=self.answer,
            depth=self.program.state_trace.depth,
            family=self.program.state_trace.family,
            seed=seed,
            solution=self.answer,
            transition_trace=self.program.state_trace,
            transition_program=self.program,
            work_memory_trace=self.work_memory_trace,
        )


def compile_frontier_process_supervision(task: FrontierTask) -> FrontierProcessSupervision:
    """Compile one full issuer task into private state/action teaching evidence."""

    if not isinstance(task, FrontierTask):
        raise TypeError("frontier process source has the wrong type")
    verifier_payload = task.reveal_for_verifier()
    expected = verifier_payload.get("expected")
    if (
        not isinstance(expected, dict)
        or verifier_payload.get("domain") != task.domain
        or verifier_payload.get("generator_id") != task.public.generator_id
        or verifier_payload.get("scorer_id") != task.public.scorer_id
    ):
        raise RuntimeError("frontier verifier payload differs from its public task")
    answer = "FINAL_ANSWER: " + json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    verdict = task.score(answer)
    if not verdict.correct:
        raise RuntimeError("frontier verifier rejected its own canonical answer")
    compiler = _COMPILERS.get(task.domain)
    if compiler is None:
        raise ValueError("frontier process domain is unsupported")
    program = compiler(expected, task.public.prompt)
    work_memory_trace = None
    if task.domain == "mathematics":
        match = _MATH_RE.search(task.public.prompt)
        if match is None:
            raise RuntimeError("frontier mathematics public objective disappeared")
        values = _literal(match.group("values"), role="mathematics values")
        if not isinstance(values, list) or any(type(value) is not int for value in values):
            raise RuntimeError("frontier mathematics public values changed")
        work_memory_trace = compile_mathematics_work_memory(
            choose=int(match.group("choose")),
            gap=int(match.group("gap")),
            low=int(match.group("low")),
            high=int(match.group("high")),
            values=tuple(values),
        )
        memory_count, memory_witness = work_memory_trace.states[-1].result()
        if memory_count != expected.get("count") or memory_witness != tuple(
            expected.get("witness", ())
        ):
            raise RuntimeError("frontier mathematics work memory differs from verifier")
    return FrontierProcessSupervision(
        source_task_id=task.task_id,
        public_prompt=task.public.prompt,
        answer=answer,
        program=program,
        work_memory_trace=work_memory_trace,
    )


def frontier_process_task_battery(
    domains: Sequence[str],
    difficulties: Sequence[int],
    per_cell: int,
    *,
    seed: int,
    registry_version: str = CONTAMINATION_SAFE_REGISTRY_VERSION,
    excluded_prompts: Collection[str] = (),
) -> list[RecurrenceTrainingTask]:
    """Build one deterministic, globally addressable broad-process cohort."""

    if (
        isinstance(domains, (str, bytes))
        or not domains
        or len(set(domains)) != len(domains)
        or any(domain not in FRONTIER_DOMAINS for domain in domains)
        or isinstance(difficulties, (str, bytes))
        or not difficulties
        or len(set(difficulties)) != len(difficulties)
        or any(type(value) is not int or value not in (1, 2, 3) for value in difficulties)
        or type(per_cell) is not int
        or per_cell < 1
        or type(seed) is not int
        or seed < 0
        or isinstance(excluded_prompts, (str, bytes))
        or any(not isinstance(prompt, str) or not prompt for prompt in excluded_prompts)
    ):
        raise ValueError("frontier process battery contract is invalid")
    tasks: list[RecurrenceTrainingTask] = []
    cursor = seed
    seen_prompts = set(excluded_prompts)
    for domain in domains:
        for difficulty in difficulties:
            accepted = 0
            attempts = 0
            while accepted < per_cell:
                attempts += 1
                if attempts > max(1_000, per_cell * 1_000):
                    raise RuntimeError(
                        f"frontier process cell cannot provide unique prompts: {domain}:{difficulty}"
                    )
                source = generate_task(
                    domain,
                    seed=cursor,
                    difficulty=difficulty,
                    registry_version=registry_version,
                )
                cursor += 1
                if source.public.prompt in seen_prompts:
                    continue
                tasks.append(compile_frontier_process_supervision(source).to_training_task())
                seen_prompts.add(source.public.prompt)
                accepted += 1
    return tasks


__all__ = [
    "FRONTIER_PROCESS_SCHEMA",
    "FrontierProcessSupervision",
    "compile_frontier_process_supervision",
    "frontier_process_task_battery",
]
