"""Public, machine-verifiable process curricula for recurrent policy learning.

Terminal-only rewards leave most sampled groups with no relative signal. These
tasks expose the sequential state machine in the prompt and require the model
to return each intermediate state. Partial reward is the longest correct
prefix of those public transitions, never lexical style, hidden-chain-of-
thought similarity, or distance from an answer in a non-metric space.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Final

from core.learning.recurrence_curriculum import _parse_semantic_terminal_answer

PROCESS_CURRICULUM_VERSION: Final = "2026.08.08.1"
PROCESS_REWARD_SCHEMA: Final = "aura.recurrent_process_reward.v1"
PROCESS_FAMILIES: Final = ("boolean", "modular")


def _json_answer(value: dict[str, Any]) -> str:
    return "FINAL_ANSWER: " + json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _rng(family: str, depth: int, seed: int) -> random.Random:
    digest = hashlib.sha256(
        f"aura-process:{PROCESS_CURRICULUM_VERSION}:{family}:{depth}:{seed}".encode(
            "ascii"
        )
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True, slots=True)
class RecurrentProcessTask:
    prompt: str
    answer: str
    depth: int
    family: str
    seed: int
    expected_trace: tuple[int, ...]
    terminal_key: str

    def __post_init__(self) -> None:
        if not self.prompt or self.prompt != self.prompt.strip() or "\x00" in self.prompt:
            raise ValueError("process task prompt is invalid")
        if not self.answer or self.answer != self.answer.strip() or "\x00" in self.answer:
            raise ValueError("process task answer is invalid")
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("process task depth is invalid")
        if self.family not in PROCESS_FAMILIES:
            raise ValueError("process task family is invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("process task seed is invalid")
        if (
            len(self.expected_trace) != self.depth + 1
            or any(type(value) is not int for value in self.expected_trace)
        ):
            raise ValueError("process task trace is invalid")
        if self.terminal_key not in {"value", "residue"}:
            raise ValueError("process task terminal key is invalid")
        if self.expected != {
            "trace": list(self.expected_trace),
            self.terminal_key: self.expected_trace[-1],
        }:
            raise ValueError("process task answer differs from its state machine")

    @property
    def task_id(self) -> str:
        return f"recurrent-process-{self.family}-d{self.depth}-s{self.seed}"

    @property
    def domain(self) -> str:
        return self.family

    @property
    def knowledge(self) -> str:
        return "parametric"

    @property
    def grader(self) -> str:
        return "exact_public_process_json"

    @property
    def expected(self) -> dict[str, Any]:
        prefix = "FINAL_ANSWER: "
        if not self.answer.startswith(prefix):
            raise ValueError("process task answer contract is invalid")
        value = json.loads(self.answer.removeprefix(prefix))
        if not isinstance(value, dict):
            raise ValueError("process task answer must be a JSON object")
        return value

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": "recurrent_process_curriculum",
            "curriculum_version": PROCESS_CURRICULUM_VERSION,
            "generation_seed": self.seed,
            "process_steps": self.depth,
        }

    def grade(self, response: str) -> dict[str, Any]:
        try:
            parsed = _parse_semantic_terminal_answer(response)
        except (TypeError, ValueError):
            return {
                "correct": False,
                "parsed": None,
                "expected": self.expected,
                "reason": "unparseable",
            }
        return {
            "correct": parsed == self.expected,
            "parsed": parsed,
            "expected": self.expected,
        }

    def process_reward(self, response: str, *, format_credit: float = 0.05) -> dict[str, Any]:
        """Return bounded credit for the longest verified transition prefix."""

        if (
            isinstance(format_credit, bool)
            or not isinstance(format_credit, (int, float))
            or not math.isfinite(float(format_credit))
            or not 0.0 <= float(format_credit) <= 0.1
        ):
            raise ValueError("process format credit must be inside [0, 0.1]")
        verdict = self.grade(response)
        if verdict["correct"]:
            return {
                "schema": PROCESS_REWARD_SCHEMA,
                "policy": "exact_or_public_transition_prefix",
                "correct": True,
                "parsed": True,
                "matched_prefix_steps": len(self.expected_trace),
                "trace_steps": len(self.expected_trace),
                "terminal_correct": True,
                "format_credit": float(format_credit),
                "trace_credit": 0.7,
                "terminal_credit": 0.2,
                "reward": 1.0,
            }
        parsed = verdict.get("parsed")
        if not isinstance(parsed, dict):
            return {
                "schema": PROCESS_REWARD_SCHEMA,
                "policy": "exact_or_public_transition_prefix",
                "correct": False,
                "parsed": False,
                "matched_prefix_steps": 0,
                "trace_steps": len(self.expected_trace),
                "terminal_correct": False,
                "format_credit": float(format_credit),
                "trace_credit": 0.7,
                "terminal_credit": 0.2,
                "reward": 0.0,
            }
        candidate_trace = parsed.get("trace")
        matched = 0
        if isinstance(candidate_trace, list):
            for observed, expected in zip(
                candidate_trace,
                self.expected_trace,
                strict=False,
            ):
                if type(observed) is not int or observed != expected:
                    break
                matched += 1
        terminal_correct = parsed.get(self.terminal_key) == self.expected_trace[-1]
        reward = (
            float(format_credit)
            + 0.7 * (matched / len(self.expected_trace))
            + 0.2 * float(terminal_correct)
        )
        # Only exact correctness can receive 1.0. A malformed or extra-field
        # payload that happens to contain every expected value remains partial.
        reward = min(0.95, reward)
        return {
            "schema": PROCESS_REWARD_SCHEMA,
            "policy": "exact_or_public_transition_prefix",
            "correct": False,
            "parsed": True,
            "matched_prefix_steps": matched,
            "trace_steps": len(self.expected_trace),
            "terminal_correct": terminal_correct,
            "format_credit": float(format_credit),
            "trace_credit": 0.7,
            "terminal_credit": 0.2,
            "reward": round(reward, 12),
        }


def boolean_process_chain(depth: int, seed: int) -> RecurrentProcessTask:
    if type(depth) is not int or depth < 1 or type(seed) is not int or seed < 0:
        raise ValueError("process task coordinates are invalid")
    rng = _rng("boolean", depth, seed)
    value = int(rng.random() < 0.5)
    trace = [value]
    operations: list[str] = []
    for _ in range(depth):
        operation = rng.choice(("and", "or", "not", "xor"))
        if operation == "not":
            operations.append("not")
            value = 1 - value
        else:
            operand = int(rng.random() < 0.5)
            operations.append(f"{operation} {operand}")
            if operation == "and":
                value = int(bool(value) and bool(operand))
            elif operation == "or":
                value = int(bool(value) or bool(operand))
            else:
                value = int(bool(value) != bool(operand))
        trace.append(value)
    expected = {"trace": trace, "value": value}
    return RecurrentProcessTask(
        prompt=(
            "Apply these Boolean operations from left to right with 1=true and "
            f"0=false. Start={trace[0]}. Operations: {', '.join(operations)}. "
            "Return exactly one FINAL_ANSWER JSON object whose trace array contains "
            "the start value followed by the value after every operation, and whose "
            "value field contains the final value. Required keys: trace, value."
        ),
        answer=_json_answer(expected),
        depth=depth,
        family="boolean",
        seed=seed,
        expected_trace=tuple(trace),
        terminal_key="value",
    )


def modular_process_chain(depth: int, seed: int) -> RecurrentProcessTask:
    if type(depth) is not int or depth < 1 or type(seed) is not int or seed < 0:
        raise ValueError("process task coordinates are invalid")
    rng = _rng("modular", depth, seed)
    modulus = rng.choice((13, 17, 19, 23))
    value = rng.randrange(modulus)
    trace = [value]
    operations: list[str] = []
    for _ in range(depth):
        operation = rng.choice(("+", "*", "-"))
        operand = rng.randrange(1, modulus)
        operations.append(f"{operation}{operand}")
        if operation == "+":
            value = (value + operand) % modulus
        elif operation == "-":
            value = (value - operand) % modulus
        else:
            value = (value * operand) % modulus
        trace.append(value)
    expected = {"trace": trace, "residue": value}
    return RecurrentProcessTask(
        prompt=(
            f"Start={trace[0]} and apply these operations from left to right modulo "
            f"{modulus}: {', '.join(operations)}. Return exactly one FINAL_ANSWER "
            "JSON object whose trace array contains the start value followed by the "
            "residue after every operation, and whose residue field contains the "
            "final residue. Required keys: trace, residue."
        ),
        answer=_json_answer(expected),
        depth=depth,
        family="modular",
        seed=seed,
        expected_trace=tuple(trace),
        terminal_key="residue",
    )


def process_task_battery(
    families: list[str],
    depths: list[int],
    per_cell: int,
    *,
    seed: int,
) -> list[RecurrentProcessTask]:
    if (
        not families
        or any(family not in PROCESS_FAMILIES for family in families)
        or not depths
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or type(per_cell) is not int
        or per_cell < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("process battery coordinates are invalid")
    constructors = {
        "boolean": boolean_process_chain,
        "modular": modular_process_chain,
    }
    tasks: list[RecurrentProcessTask] = []
    seen: set[str] = set()
    for family in families:
        for depth in depths:
            for ordinal in range(per_cell):
                material = f"{seed}:{family}:{depth}:{ordinal}".encode("ascii")
                task_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                task = constructors[family](depth, task_seed)
                if task.task_id in seen:
                    raise RuntimeError("process battery generated a duplicate task")
                seen.add(task.task_id)
                tasks.append(task)
    return tasks


__all__ = [
    "PROCESS_CURRICULUM_VERSION",
    "PROCESS_FAMILIES",
    "PROCESS_REWARD_SCHEMA",
    "RecurrentProcessTask",
    "boolean_process_chain",
    "modular_process_chain",
    "process_task_battery",
]
