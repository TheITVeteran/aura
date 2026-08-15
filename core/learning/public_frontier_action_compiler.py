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
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    canonical_instruction_from_public_fields,
)

PUBLIC_FRONTIER_ACTION_SCHEMA: Final = "aura.public_frontier_action_program.v1"
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
    "frontier_coding": (
        "case_index",
        "name_index",
        "signed_delta",
        "pressure",
        "active_count",
        "case_terminal",
    ),
    "frontier_calibration": (
        "prior_numerator",
        "prior_denominator",
        "likelihood_h_numerator",
        "likelihood_h_denominator",
        "likelihood_not_h_numerator",
        "likelihood_not_h_denominator",
    ),
    "frontier_misleading_premise": (
        "row_index",
        "impact",
        "reliability",
        "cost",
        "name_rank",
        "reserved",
    ),
}


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
                or any(type(value) is not int or not 0 <= value < 33 for value in row)
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
    actions: list[tuple[int, ...]] = []
    for case_index, case in enumerate(cases):
        if not isinstance(case, list) or not case:
            raise ValueError("public coding case is invalid")
        for event in case:
            if (
                not isinstance(event, tuple)
                or len(event) != 2
                or not isinstance(event[0], str)
                or type(event[1]) is not int
            ):
                raise ValueError("public coding event is invalid")
            name, delta = event
            actions.append((case_index, names.index(name), delta + 3, 0, 0, 0))
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
    return [
        (
            prior.numerator,
            prior.denominator,
            likelihood_h.numerator,
            likelihood_h.denominator,
            likelihood_not_h.numerator,
            likelihood_not_h.denominator,
        ),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
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
        actions.append((index, *operands, names.index(row["name"]), 0))
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

    if not isinstance(public_prompt, str) or not public_prompt or public_prompt != public_prompt.strip():
        raise ValueError("public frontier prompt is invalid")
    compiler = _COMPILERS.get(family)
    if compiler is None:
        raise ValueError("frontier family has no answer-blind public action compiler")
    raw_actions = compiler(public_prompt)
    field_names = _FIELD_NAMES[family]
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


__all__ = [
    "PUBLIC_FRONTIER_ACTION_SCHEMA",
    "PublicFrontierActionProgram",
    "compile_public_frontier_actions",
]
