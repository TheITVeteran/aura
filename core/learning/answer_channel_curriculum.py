"""Training-only answer-channel curriculum for verifier entry.

This is not a reasoning benchmark. It teaches and measures the mechanical
contract Aura's verifier-driven RL needs before reasoning rewards can matter:
emit exactly one terminal ``FINAL_ANSWER: {JSON object}`` line with the
requested keys and parseable JSON. Gains here are answer-channel readiness,
not frontier reasoning evidence.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ANSWER_CHANNEL_SCHEMA = "aura.answer_channel_curriculum.v1"
ANSWER_CHANNEL_VERSION = "2026.07.22.1"
ANSWER_CHANNEL_FAMILIES = ("json_copy", "typed_boolean", "key_selection")
MAX_ANSWER_CHANNEL_DEPTH = 8


@dataclass(frozen=True, slots=True)
class AnswerChannelTask:
    prompt: str
    answer: str
    depth: int
    family: str
    seed: int

    def __post_init__(self) -> None:
        if not self.prompt.strip() or self.prompt != self.prompt.strip():
            raise ValueError("answer-channel prompt is invalid")
        if not self.answer.startswith("FINAL_ANSWER: "):
            raise ValueError("answer-channel answer contract is invalid")
        if type(self.depth) is not int or not 1 <= self.depth <= MAX_ANSWER_CHANNEL_DEPTH:
            raise ValueError("answer-channel depth is invalid")
        if self.family not in ANSWER_CHANNEL_FAMILIES:
            raise ValueError("answer-channel family is invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("answer-channel seed is invalid")

    @property
    def task_id(self) -> str:
        return f"answer-channel-{self.family}-d{self.depth}-s{self.seed}"

    @property
    def domain(self) -> str:
        return self.family

    @property
    def knowledge(self) -> str:
        return "parametric"

    @property
    def grader(self) -> str:
        return "answer_channel_json"

    @property
    def expected(self) -> dict[str, Any]:
        payload = self.answer.removeprefix("FINAL_ANSWER: ")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("answer-channel expected value is not an object")
        return value

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": "answer_channel_curriculum",
            "curriculum_version": ANSWER_CHANNEL_VERSION,
            "claim_boundary": "format_parseability_only_not_reasoning_gain",
        }

    def grade(self, response: str) -> dict[str, Any]:
        try:
            from core.brain.llm.latent_cortex.frontier_tasks import (
                FrontierTaskError,
                parse_final_answer,
            )

            produced = parse_final_answer(response)
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


def _rng(family: str, depth: int, seed: int) -> random.Random:
    material = f"{ANSWER_CHANNEL_VERSION}:{family}:{depth}:{seed}"
    return random.Random(int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()))


def _validate(depth: int, seed: int) -> None:
    if type(depth) is not int or not 1 <= depth <= MAX_ANSWER_CHANNEL_DEPTH:
        raise ValueError("answer-channel depth is invalid")
    if type(seed) is not int or seed < 0:
        raise ValueError("answer-channel seed is invalid")


def json_copy(depth: int, seed: int) -> AnswerChannelTask:
    _validate(depth, seed)
    rng = _rng("json_copy", depth, seed)
    keys = [f"k{index}" for index in range(1, min(4, depth) + 1)]
    payload = {key: rng.randrange(0, 20) for key in keys}
    facts = ", ".join(f"{key}={value}" for key, value in payload.items())
    return AnswerChannelTask(
        prompt=(
            f"Read these key/value facts: {facts}. Return the same values as "
            "one JSON object."
        ),
        answer=_json_answer(payload),
        depth=depth,
        family="json_copy",
        seed=seed,
    )


def typed_boolean(depth: int, seed: int) -> AnswerChannelTask:
    _validate(depth, seed)
    rng = _rng("typed_boolean", depth, seed)
    values = [rng.randrange(1, 30) for _ in range(depth + 2)]
    threshold = sum(values) // len(values)
    flags = {
        f"above_{index}": value > threshold
        for index, value in enumerate(values[: min(4, len(values))])
    }
    facts = ", ".join(f"value {index}={value}" for index, value in enumerate(values))
    return AnswerChannelTask(
        prompt=(
            f"Given {facts}. Threshold is {threshold}. Return booleans for "
            "whether each named value is above the threshold."
        ),
        answer=_json_answer(flags),
        depth=depth,
        family="typed_boolean",
        seed=seed,
    )


def key_selection(depth: int, seed: int) -> AnswerChannelTask:
    _validate(depth, seed)
    rng = _rng("key_selection", depth, seed)
    options = {f"item_{index}": rng.randrange(1, 99) for index in range(depth + 2)}
    selected = max(options, key=options.__getitem__)
    facts = ", ".join(f"{key}={value}" for key, value in sorted(options.items()))
    return AnswerChannelTask(
        prompt=(
            f"From these scored items, choose the highest: {facts}. Return JSON "
            "with keys selected and score."
        ),
        answer=_json_answer({"selected": selected, "score": options[selected]}),
        depth=depth,
        family="key_selection",
        seed=seed,
    )


TASK_GENERATORS: MappingProxyType[str, Callable[[int, int], AnswerChannelTask]]
TASK_GENERATORS = MappingProxyType(
    {
        "json_copy": json_copy,
        "typed_boolean": typed_boolean,
        "key_selection": key_selection,
    }
)


def _sample_seed(
    *,
    root_seed: int,
    family: str,
    depth: int,
    cell: int,
    attempt: int,
) -> int:
    material = f"{ANSWER_CHANNEL_VERSION}:{root_seed}:{family}:{depth}:{cell}:{attempt}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:16])


def task_battery(
    families: Sequence[str],
    depths: Sequence[int],
    per_cell: int,
    *,
    seed: int,
) -> list[AnswerChannelTask]:
    if not families or any(family not in TASK_GENERATORS for family in families):
        raise ValueError("answer-channel families are invalid")
    if not depths or any(type(depth) is not int for depth in depths):
        raise ValueError("answer-channel depths are invalid")
    if type(per_cell) is not int or per_cell <= 0:
        raise ValueError("answer-channel per_cell is invalid")
    tasks: list[AnswerChannelTask] = []
    prompts: set[str] = set()
    for family in families:
        generator = TASK_GENERATORS[family]
        for depth in depths:
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
                    if task.prompt not in prompts:
                        break
                else:  # pragma: no cover
                    raise RuntimeError("answer-channel generator exhausted")
                prompts.add(task.prompt)
                tasks.append(task)
    return tasks


def disjoint_task_split(
    *,
    families: Sequence[str],
    depths: Sequence[int],
    train_per_cell: int,
    holdout_per_cell: int,
    seed: int,
) -> tuple[list[AnswerChannelTask], list[AnswerChannelTask]]:
    train = task_battery(families, depths, train_per_cell, seed=seed)
    holdout = task_battery(families, depths, holdout_per_cell, seed=seed + 17_117)
    if {task.prompt for task in train} & {task.prompt for task in holdout}:
        raise RuntimeError("answer-channel train and holdout prompts overlap")
    if {task.task_id for task in train} & {task.task_id for task in holdout}:
        raise RuntimeError("answer-channel train and holdout identities overlap")
    return train, holdout


__all__ = [
    "ANSWER_CHANNEL_FAMILIES",
    "ANSWER_CHANNEL_SCHEMA",
    "ANSWER_CHANNEL_VERSION",
    "AnswerChannelTask",
    "TASK_GENERATORS",
    "disjoint_task_split",
    "task_battery",
]
