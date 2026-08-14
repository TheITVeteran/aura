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
from core.learning.recurrence_curriculum import (
    RecurrenceTrainingTask,
    StructuredTransitionProgram,
    StructuredTransitionTrace,
)

FRONTIER_PROCESS_SCHEMA: Final = "aura.frontier_process_supervision.v1"
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
    witness_head = values.index(witness[0]) + 1 if witness else 0
    states = [(0, 0, 0, 0, 0)]
    actions: list[tuple[int, ...]] = []
    previous = 0
    for index, value in enumerate(values):
        prefix = values[: index + 1]
        valid = [
            combo
            for combo in itertools.combinations(prefix, choose)
            if all(right - left >= gap for left, right in zip(combo, combo[1:], strict=False))
            and low <= sum(combo) <= high
        ]
        current = len(valid)
        added_lo, added_hi = _digits(current - previous)
        value_lo, value_hi = _signed_digits(value)
        actions.append(
            (
                index,
                value_lo,
                value_hi,
                added_lo,
                added_hi,
                witness_head if valid else 0,
            )
        )
        count_lo, count_hi = _digits(current)
        states.append(
            (
                index + 1,
                count_lo,
                count_hi,
                witness_head if valid else 0,
                int(index + 1 == len(values)),
            )
        )
        previous = current
    if previous != count or (witness_tuple and witness_tuple not in valid):
        raise ValueError("frontier mathematics process differs from verified answer")
    return _program(
        family="frontier_mathematics",
        field_names=("pc", "count_lo", "count_hi", "witness_head", "done"),
        states=states,
        action_field_names=(
            "input_index",
            "value_lo",
            "value_hi",
            "valid_added_lo",
            "valid_added_hi",
            "next_witness_head",
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
    states = [(0, 0, 0, 0, 0)]
    actions: list[tuple[int, ...]] = []
    balances: dict[str, int] = {}
    step = 0
    for case_index, case in enumerate(cases):
        balances = {}
        pressures: list[int] = []
        for event_index, event in enumerate(case):
            if not isinstance(event, tuple) or len(event) != 2:
                raise ValueError("frontier coding event is invalid")
            name, delta = event
            balances[name] = balances.get(name, 0) + delta
            if balances[name] == 0:
                del balances[name]
            pressure = sum(abs(value) for value in balances.values())
            pressures.append(pressure)
            step += 1
            actions.append(
                (
                    case_index,
                    names.index(name),
                    delta + 3,
                    pressure,
                    len(balances),
                    int(event_index + 1 == len(case)),
                )
            )
            states.append((step, case_index, len(balances), pressure, 0))
        expected_return = returns[case_index]
        expected_state = [[name, balances[name]] for name in sorted(balances)]
        if expected_return != {"state": expected_state, "pressure": pressures}:
            raise ValueError("frontier coding process differs from verified answer")
    states[-1] = (*states[-1][:-1], 1)
    return _program(
        family="frontier_coding",
        field_names=("pc", "case_index", "active_count", "pressure", "done"),
        states=states,
        action_field_names=(
            "case_index",
            "name_index",
            "signed_delta",
            "pressure",
            "active_count",
            "case_terminal",
        ),
        actions=actions,
    )


def _scientific(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _SCIENCE_RE.search(prompt)
    if match is None:
        raise ValueError("frontier scientific evidence is invalid")
    labels = [match.group("a"), match.group("b"), match.group("c")]
    try:
        root = labels.index(expected["root"])
        mediator = labels.index(expected["mediator"])
        downstream = labels.index(expected["downstream"])
        prediction_lo, prediction_hi = _signed_digits(expected["predicted_downstream"])
        baselines = [
            int(match.group("a_value")),
            int(match.group("b_value")),
            int(match.group("c_value")),
        ]
        root_step = int(match.group("root_step"))
        root_mediator_change = int(match.group("root_mediator_change"))
        root_downstream_change = int(match.group("root_downstream_change"))
        query_step = int(match.group("query_step"))
    except (KeyError, TypeError, ValueError):
        raise ValueError("frontier scientific answer is invalid") from None
    if (
        root_step < 1
        or root_mediator_change % root_step
        or root_mediator_change < 1
        or root_downstream_change % root_mediator_change
    ):
        raise ValueError("frontier scientific public effects are not exact")
    mediator_gain = root_mediator_change // root_step
    downstream_gain = root_downstream_change // root_mediator_change
    downstream_lo, downstream_hi = _digits(baselines[downstream])
    public_prediction = (
        baselines[downstream] + query_step * mediator_gain * downstream_gain
    )
    if public_prediction != expected["predicted_downstream"]:
        raise ValueError("frontier scientific public effects differ from verified answer")
    actions = [
        (0, root, 0, 0, 0, 0),
        (1, root, mediator, 0, 0, 0),
        (2, root, mediator, downstream, 0, 0),
        (
            3,
            downstream_lo,
            downstream_hi,
            query_step,
            mediator_gain,
            downstream_gain,
        ),
    ]
    role_pair = root * 3 + mediator
    states = [
        (0, 0, 0, 0, 0),
        (1, root + 1, 0, 0, 0),
        (2, role_pair + 1, 0, 0, 0),
        (3, role_pair + 1, 0, 0, 0),
        (4, role_pair + 1, prediction_lo, prediction_hi, 1),
    ]
    return _program(
        family="frontier_scientific_inference",
        field_names=("pc", "role_pair", "prediction_lo", "prediction_hi", "done"),
        states=states,
        action_field_names=("stage", "arg0", "arg1", "arg2", "arg3", "arg4"),
        actions=actions,
    )


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


def _calibration(expected: dict[str, Any], _prompt: str) -> StructuredTransitionProgram:
    try:
        posterior = Fraction(expected["posterior"])
        choice = 1 if expected["choice"] == "H" else 0
        band = ("below_50", "50_to_69", "70_to_89", "90_to_100").index(expected["confidence_band"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("frontier calibration evidence is invalid") from None
    num_lo, num_hi = _digits(posterior.numerator)
    den_lo, den_hi = _digits(posterior.denominator)
    actions = [
        (0, num_lo, num_hi, den_lo, den_hi, 0),
        (1, num_lo, num_hi, den_lo, den_hi, choice),
        (2, num_lo, num_hi, den_lo, den_hi, band),
    ]
    states = [
        (0, 0, 0, 0, 0),
        (1, num_lo, den_lo, 0, 0),
        (2, num_lo, den_lo, choice + 1, 0),
        (3, num_lo, den_lo, band + 1, 1),
    ]
    return _program(
        family="frontier_calibration",
        field_names=("pc", "numerator_lo", "denominator_lo", "decision", "done"),
        states=states,
        action_field_names=(
            "stage",
            "numerator_lo",
            "numerator_hi",
            "denominator_lo",
            "denominator_hi",
            "decision_code",
        ),
        actions=actions,
    )


def _premise(expected: dict[str, Any], prompt: str) -> StructuredTransitionProgram:
    match = _PREMISE_RE.search(prompt)
    if match is None:
        raise ValueError("frontier premise evidence is invalid")
    rows = _literal(match.group("rows"), role="premise rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frontier premise rows are invalid")
    best_index = 0
    best_score = -10_000
    states = [(0, 0, 0, 0, 0)]
    actions: list[tuple[int, ...]] = []
    for index, row in enumerate(rows):
        score = row["impact"] * row["reliability"] - row["cost"]
        score_lo, score_hi = _signed_digits(score)
        if score > best_score or (score == best_score and row["name"] < rows[best_index]["name"]):
            best_index = index
            best_score = score
        best_lo, best_hi = _signed_digits(best_score)
        actions.append(
            (
                best_index,
                row["impact"],
                row["reliability"],
                row["cost"],
                best_lo,
                best_hi,
            )
        )
        states.append((index + 1, best_index, best_lo, best_hi, int(index + 1 == len(rows))))
    winner = rows[best_index]["name"]
    if winner != expected.get("actual_winner") or best_score != expected.get("actual_score"):
        raise ValueError("frontier premise process differs from verified answer")
    if bool(match.group("claim") == winner) is not expected.get("premise_valid"):
        raise ValueError("frontier premise validity differs from verified answer")
    return _program(
        family="frontier_misleading_premise",
        field_names=("pc", "winner_index", "score_lo", "score_hi", "done"),
        states=states,
        action_field_names=(
            "winner_index",
            "impact",
            "reliability",
            "cost",
            "score_lo",
            "score_hi",
        ),
        actions=actions,
    )


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
    return FrontierProcessSupervision(
        source_task_id=task.task_id,
        public_prompt=task.public.prompt,
        answer=answer,
        program=program,
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
