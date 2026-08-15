"""Answer-blind compilation of public frontier operands into recurrent actions.

This module is an input adapter, not a solver.  It parses only literals and
operation order already present in the user-visible objective.  It never
accepts a verifier answer, private state trace, winner, score, count, schedule,
checksum, or derived scientific role.  The recurrent transition tissue remains
responsible for computing state from these operands.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from core.learning.recurrent_action_schema import (
    ACTION_CARDINALITY,
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    OP_PAIR_ADD,
    OP_PAIR_COPY,
    OP_PAIR_DIV,
    OP_PAIR_EUCLID_STEP,
    OP_PAIR_MUL_IMMEDIATE,
    OP_PAIR_PRODUCT,
    OP_PAIR_SET,
    OP_PAIR_SUB_IMMEDIATE,
    OP_PAIR_SIGNED_SUB_IMMEDIATE,
    OP_RANKED_COMMIT,
    OP_RATIO_BAND,
    OP_RATIO_CHOICE,
    OP_SET_SCALAR,
    OP_SIGNED_PAIR_ADD_IMMEDIATE,
    OP_SIGNED_RANKED_GREATER,
    RECURRENT_ACTION_SCHEMA,
    SEMANTIC_MICRO_ACTION_FIELD_NAMES,
    canonical_instruction_from_public_fields,
)

PUBLIC_FRONTIER_ACTION_SCHEMA: Final = "aura.public_frontier_action_program.v2"
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

_FIELD_NAMES = {
    "frontier_mathematics": ("arg0", "arg1", "arg2", "arg3", "arg4", "arg5"),
    "frontier_coding": SEMANTIC_MICRO_ACTION_FIELD_NAMES,
    "frontier_calibration": SEMANTIC_MICRO_ACTION_FIELD_NAMES,
    "frontier_misleading_premise": SEMANTIC_MICRO_ACTION_FIELD_NAMES,
}


def _micro(opcode: int, *arguments: int) -> tuple[int, ...]:
    if (
        type(opcode) is not int
        or not 0 <= opcode < ACTION_NULL
        or len(arguments) > 6
        or any(type(value) is not int or not 0 <= value < ACTION_NULL for value in arguments)
    ):
        raise ValueError("public semantic micro-instruction is invalid")
    return (opcode, *arguments, *(0 for _index in range(6 - len(arguments))))


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


def _literal(text: str, *, role: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise ValueError(f"{role} literal is invalid") from None


def _digits(value: int) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= MAX_PROCESS_INTEGER:
        raise ValueError("public process integer is outside the two-slot vocabulary")
    return value % PROCESS_RADIX, value // PROCESS_RADIX


def _signed_digits(value: int) -> tuple[int, int]:
    if type(value) is not int:
        raise ValueError("public process signed integer is invalid")
    encoded = value * 2 if value >= 0 else (-value * 2) - 1
    return _digits(encoded)


@dataclass(frozen=True, slots=True)
class PublicFrontierActionProgram:
    """Publicly reproducible recurrent operands with no correctness authority."""

    family: str
    public_prompt_sha256: str
    values: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.family not in _FIELD_NAMES
            or len(self.public_prompt_sha256) != 64
            or not self.values
            or any(
                len(row) != len(ACTION_SLOT_NAMES)
                or any(
                    type(value) is not int or not 0 <= value < ACTION_CARDINALITY
                    for value in row
                )
                for row in self.values
            )
        ):
            raise ValueError("public frontier action program is invalid")

    @property
    def program_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema": PUBLIC_FRONTIER_ACTION_SCHEMA,
                "family": self.family,
                "public_prompt_sha256": self.public_prompt_sha256,
                "values": self.values,
            }
        )

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": PUBLIC_FRONTIER_ACTION_SCHEMA,
            "family": self.family,
            "public_prompt_sha256": self.public_prompt_sha256,
            "steps": len(self.values),
            "action_slot_names": list(ACTION_SLOT_NAMES),
            "recurrent_action_schema": RECURRENT_ACTION_SCHEMA,
            "instruction_dialect": "semantic_micro_v2",
            "program_sha256": self.program_sha256,
            "source": "public_objective_literals_and_order_only",
            "verifier_answer_available": False,
            "private_state_trace_available": False,
            "derived_answer_fields_present": False,
            "correctness_authority": False,
        }
        return {**body, "receipt_sha256": _canonical_sha256(body)}

    def values_for_iterations(self, iterations: int) -> tuple[tuple[int, ...], ...]:
        """Return the causal prefix, padding only after program completion."""

        if type(iterations) is not int or iterations < 1:
            raise ValueError("public action iteration budget is invalid")
        if iterations <= len(self.values):
            return self.values[:iterations]
        null = (ACTION_NULL,) * len(ACTION_SLOT_NAMES)
        return self.values + (null,) * (iterations - len(self.values))


def _mathematics(prompt: str) -> list[tuple[int, ...]]:
    match = _MATH_RE.search(prompt)
    if match is None:
        raise ValueError("public mathematics objective is invalid")
    values = _literal(match.group("values"), role="mathematics values")
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise ValueError("public mathematics values are invalid")
    low_lo, low_hi = _signed_digits(int(match.group("low")))
    high_lo, high_hi = _signed_digits(int(match.group("high")))
    actions = [
        (
            int(match.group("choose")),
            int(match.group("gap")),
            low_lo,
            low_hi,
            high_lo,
            high_hi,
        )
    ]
    for index, value in enumerate(values):
        value_lo, value_hi = _signed_digits(value)
        actions.append((index, value_lo, value_hi, 0, 0, 0))
    return actions


def _coding(prompt: str) -> list[tuple[int, ...]]:
    match = _CODING_RE.search(prompt)
    if match is None:
        raise ValueError("public coding objective is invalid")
    cases = _literal(match.group("cases"), role="coding cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("public coding cases are invalid")
    try:
        names = sorted({event[0] for case in cases for event in case})
    except (TypeError, IndexError):
        raise ValueError("public coding events are invalid") from None
    if len(names) > 4:
        raise ValueError("public coding process exceeds four balance registers")
    actions: list[tuple[int, ...]] = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, list) or not case:
            raise ValueError("public coding case is invalid")
        if case_index:
            for name_index in range(4):
                actions.append(_micro(OP_PAIR_SET, 1 + 2 * name_index, 0, 0))
            actions.append(_micro(OP_SET_SCALAR, 0, case_index))
        for event in case:
            if (
                not isinstance(event, tuple)
                or len(event) != 2
                or not isinstance(event[0], str)
                or type(event[1]) is not int
            ):
                raise ValueError("public coding event is invalid")
            name, delta = event
            encoded_delta = delta * 2 if delta >= 0 else (-delta * 2) - 1
            actions.append(
                _micro(
                    OP_SIGNED_PAIR_ADD_IMMEDIATE,
                    1 + 2 * names.index(name),
                    encoded_delta,
                )
            )
    return actions


def _calibration(prompt: str) -> list[tuple[int, ...]]:
    match = _CALIBRATION_RE.search(prompt)
    if match is None:
        raise ValueError("public calibration objective is invalid")
    try:
        prior = Fraction(match.group("prior"))
        likelihood_h = Fraction(match.group("likelihood_h"))
        likelihood_not_h = Fraction(match.group("likelihood_not_h"))
    except (ValueError, ZeroDivisionError):
        raise ValueError("public calibration fractions are invalid") from None
    if any(not 0 <= value <= 1 for value in (prior, likelihood_h, likelihood_not_h)):
        raise ValueError("public calibration fractions are outside probability bounds")
    # value0/1 = posterior numerator, value2/3 = denominator,
    # value4 = choice, value5 = band, value6/7 = scratch pair.
    return [
        _micro(OP_PAIR_SET, 0, prior.numerator, 0),
        _micro(OP_PAIR_MUL_IMMEDIATE, 0, likelihood_h.numerator),
        _micro(OP_PAIR_MUL_IMMEDIATE, 0, likelihood_not_h.denominator),
        _micro(OP_PAIR_SET, 6, prior.denominator, 0),
        _micro(OP_PAIR_SUB_IMMEDIATE, 6, prior.numerator),
        _micro(OP_PAIR_MUL_IMMEDIATE, 6, likelihood_not_h.numerator),
        _micro(OP_PAIR_MUL_IMMEDIATE, 6, likelihood_h.denominator),
        _micro(OP_PAIR_ADD, 2, 0, 6),
        _micro(OP_PAIR_COPY, 4, 0),
        _micro(OP_PAIR_COPY, 6, 2),
        *(_micro(OP_PAIR_EUCLID_STEP, 4, 6) for _step in range(14)),
        _micro(OP_PAIR_DIV, 0, 0, 4),
        _micro(OP_PAIR_DIV, 2, 2, 4),
        _micro(OP_RATIO_CHOICE, 4, 0, 2),
        _micro(OP_RATIO_BAND, 5, 0, 2),
    ]


def _premise(prompt: str) -> list[tuple[int, ...]]:
    match = _PREMISE_RE.search(prompt)
    if match is None:
        raise ValueError("public premise objective is invalid")
    rows = _literal(match.group("rows"), role="premise rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("public premise rows are invalid")
    names = sorted(row.get("name") for row in rows if isinstance(row, dict))
    if len(names) != len(rows) or len(set(names)) != len(names):
        raise ValueError("public premise names are invalid")
    actions: list[tuple[int, ...]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "impact", "reliability", "cost"}:
            raise ValueError("public premise row is invalid")
        operands = (row["impact"], row["reliability"], row["cost"])
        if not isinstance(row["name"], str) or any(type(value) is not int for value in operands):
            raise ValueError("public premise row values are invalid")
        rank = names.index(row["name"])
        # value0 = winner index, value1/2 = signed winner score,
        # value3 = winner rank, value4 = has-winner, value5/6 = candidate,
        # value7 = comparison decision.
        actions.extend(
            (
                _micro(OP_PAIR_PRODUCT, 5, row["impact"], row["reliability"]),
                _micro(OP_PAIR_SIGNED_SUB_IMMEDIATE, 5, row["cost"]),
                _micro(OP_SIGNED_RANKED_GREATER, 7, 5, 1, rank, 3, 4),
                _micro(OP_RANKED_COMMIT, 7, 5, index, rank),
            )
        )
    return actions


_COMPILERS = {
    "frontier_mathematics": _mathematics,
    "frontier_coding": _coding,
    "frontier_calibration": _calibration,
    "frontier_misleading_premise": _premise,
}


def compile_public_frontier_actions(
    public_prompt: str,
    family: str,
) -> PublicFrontierActionProgram:
    """Compile answer-blind public operands for a supported frontier family."""

    field_names, raw_actions = compile_public_frontier_raw_actions(
        public_prompt,
        family,
    )
    values = tuple(
        canonical_instruction_from_public_fields(
            family,
            field_names,
            action,
            step=step,
            terminal=int(step + 1 == len(raw_actions)),
        )
        for step, action in enumerate(raw_actions)
    )
    return PublicFrontierActionProgram(
        family=family,
        public_prompt_sha256=hashlib.sha256(public_prompt.encode("utf-8")).hexdigest(),
        values=values,
    )


def compile_public_frontier_raw_actions(
    public_prompt: str,
    family: str,
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    """Return the answer-blind source actions before canonical projection."""

    if not isinstance(public_prompt, str) or not public_prompt or public_prompt != public_prompt.strip():
        raise ValueError("public frontier prompt is invalid")
    compiler = _COMPILERS.get(family)
    if compiler is None:
        raise ValueError("frontier family has no answer-blind public action compiler")
    raw_actions = tuple(compiler(public_prompt))
    if not raw_actions:
        raise ValueError("public frontier action program is empty")
    field_names = _FIELD_NAMES[family]
    return field_names, raw_actions


__all__ = [
    "PUBLIC_FRONTIER_ACTION_SCHEMA",
    "PublicFrontierActionProgram",
    "compile_public_frontier_actions",
    "compile_public_frontier_raw_actions",
]
