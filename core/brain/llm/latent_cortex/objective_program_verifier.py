"""Exact verification for public, self-contained objective programs.

The verifier derives an answer only from the public objective. It never
receives a benchmark answer or private grader state. Recognized objective
families are parsed into a bounded program, executed deterministically, and
compared with the candidate's strict terminal JSON object.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

OBJECTIVE_PROGRAM_VERIFIER_SCHEMA = "aura.rlc.objective_program_verifier.v2"
OBJECTIVE_PROGRAM_SOLUTION_SCHEMA = "aura.rlc.objective_program_solution.v1"

_MODULAR_OBJECTIVE_RE = re.compile(
    r"\AStart at the given value and apply each operation modulo "
    r"(?P<modulus>\d+): start=(?P<start>-?\d+)\. Operations: "
    r"(?P<operations>[+*\-]\d+(?:,\s*[+*\-]\d+)*)\.",
)
_BOOLEAN_OBJECTIVE_RE = re.compile(
    r"\AEvaluate this (?P<depth>\d+)-operation expression with 1=true, "
    r"0=false, and xor meaning exactly one operand is true: "
    r"(?P<expression>.+?)\. Return a value of 1 or 0\.",
)
_BOOLEAN_TOKEN_RE = re.compile(r"\s*(?:(?P<bit>[01])|(?P<op>not|and|or|xor)|(?P<paren>[()]))")
_JSON_FENCE_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>\{[\s\S]*\})\r?\n```\Z",
    re.IGNORECASE,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _BooleanParser:
    tokens: list[str]
    index: int = 0
    operations: int = 0

    def parse(self) -> bool:
        value = self._or()
        if self.index != len(self.tokens):
            raise ValueError("boolean_expression_trailing_tokens")
        return value

    def _peek(self) -> str:
        return self.tokens[self.index] if self.index < len(self.tokens) else ""

    def _take(self, expected: str | None = None) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("boolean_expression_truncated")
        value = self.tokens[self.index]
        if expected is not None and value != expected:
            raise ValueError("boolean_expression_token_mismatch")
        self.index += 1
        return value

    def _or(self) -> bool:
        value = self._xor()
        while self._peek() == "or":
            self._take("or")
            self.operations += 1
            right = self._xor()
            value = value or right
        return value

    def _xor(self) -> bool:
        value = self._and()
        while self._peek() == "xor":
            self._take("xor")
            self.operations += 1
            value = value != self._and()
        return value

    def _and(self) -> bool:
        value = self._not()
        while self._peek() == "and":
            self._take("and")
            self.operations += 1
            right = self._not()
            value = value and right
        return value

    def _not(self) -> bool:
        if self._peek() == "not":
            self._take("not")
            self.operations += 1
            return not self._not()
        return self._atom()

    def _atom(self) -> bool:
        token = self._take()
        if token in {"0", "1"}:
            return token == "1"
        if token != "(":
            raise ValueError("boolean_expression_atom_invalid")
        value = self._or()
        self._take(")")
        return value


def _boolean_tokens(expression: str) -> list[str]:
    tokens: list[str] = []
    cursor = 0
    while cursor < len(expression):
        match = _BOOLEAN_TOKEN_RE.match(expression, cursor)
        if match is None:
            raise ValueError("boolean_expression_lex_invalid")
        tokens.append(match.group("bit") or match.group("op") or match.group("paren"))
        cursor = match.end()
    if not tokens or len(tokens) > 4_096:
        raise ValueError("boolean_expression_size_invalid")
    return tokens


def _candidate_payload(candidate: str) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.frontier_tasks import parse_final_answer

    if "FINAL_ANSWER:" in candidate:
        payload = parse_final_answer(candidate)
    else:
        # Some checkpoints return only the requested JSON object, commonly in
        # one markdown fence.  That is a contract-format defect, but the
        # semantic answer remains uniquely identifiable.  Accept only a whole
        # response object or one whole-response JSON fence: prose-wrapped,
        # multiple, or otherwise ambiguous payloads remain unverifiable.
        bounded = candidate.strip()
        fence = _JSON_FENCE_RE.fullmatch(bounded)
        encoded = fence.group("body") if fence is not None else bounded
        if not (encoded.startswith("{") and encoded.endswith("}")):
            raise ValueError("terminal_answer_not_uniquely_bounded")
        payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("terminal_answer_not_object")
    return payload


def _modular_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    modulus = int(match.group("modulus"))
    start = int(match.group("start"))
    if not 2 <= modulus <= 1_000_000 or not -1_000_000_000 <= start <= 1_000_000_000:
        raise ValueError("modular_program_bounds_invalid")
    value = start % modulus
    operations = [item.strip() for item in match.group("operations").split(",")]
    if not 1 <= len(operations) <= 1_024:
        raise ValueError("modular_program_length_invalid")
    trace: list[int] = [value]
    for operation in operations:
        operand = int(operation[1:])
        if not 0 <= operand <= 1_000_000_000:
            raise ValueError("modular_operand_bounds_invalid")
        if operation[0] == "+":
            value = (value + operand) % modulus
        elif operation[0] == "-":
            value = (value - operand) % modulus
        else:
            value = (value * operand) % modulus
        trace.append(value)
    return {"residue": value}, {
        "modulus": modulus,
        "operation_count": len(operations),
        "trace_sha256": _sha(trace),
    }


def _boolean_expected(match: re.Match[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    declared_depth = int(match.group("depth"))
    parser = _BooleanParser(_boolean_tokens(match.group("expression")))
    value = parser.parse()
    if parser.operations != declared_depth:
        raise ValueError("boolean_operation_count_mismatch")
    return {"value": int(value)}, {
        "declared_operations": declared_depth,
        "executed_operations": parser.operations,
        "expression_sha256": _text_sha(match.group("expression")),
    }


def _execute_objective(objective: str) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    if not isinstance(objective, str):
        raise TypeError("objective program must be text")
    family = ""
    expected: dict[str, Any]
    execution: dict[str, Any]
    modular = _MODULAR_OBJECTIVE_RE.match(objective)
    boolean = _BOOLEAN_OBJECTIVE_RE.match(objective)
    if modular is not None:
        family = "modular_chain"
        expected, execution = _modular_expected(modular)
    elif boolean is not None:
        family = "nested_boolean"
        expected, execution = _boolean_expected(boolean)
    else:
        return None
    return family, expected, execution


def solve_objective_program(objective: str) -> tuple[str, dict[str, Any]] | None:
    """Compile a recognized public objective into a canonical proven answer.

    This is a bounded symbolic solver, not benchmark-answer access: the only
    input is the same user-visible objective given to the generator. The public
    receipt commits to the output and execution trace without embedding answer
    text; callers must still run the independent verifier before promotion.
    """

    executed = _execute_objective(objective)
    if executed is None:
        return None
    family, expected, execution = executed
    candidate = "FINAL_ANSWER: " + json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload = {
        "schema": OBJECTIVE_PROGRAM_SOLUTION_SCHEMA,
        "family": family,
        "objective_sha256": _text_sha(objective),
        "candidate_sha256": _text_sha(candidate),
        "expected_payload_sha256": _sha(expected),
        "execution": execution,
        "authority": "public_objective_deterministic_execution",
    }
    return candidate, {**payload, "receipt_sha256": _sha(payload)}


def validate_objective_program_solution(
    value: Any,
    *,
    objective: str,
    candidate: str,
) -> dict[str, Any]:
    rebuilt = solve_objective_program(objective)
    if rebuilt is None:
        raise ValueError("objective program solution is unavailable")
    expected_candidate, expected_receipt = rebuilt
    if candidate != expected_candidate or value != expected_receipt:
        raise ValueError("objective program solution reconstruction differs")
    return expected_receipt


def verify_objective_program(candidate: str, *, objective: str) -> dict[str, Any] | None:
    """Return an exact verdict for a recognized public objective, else ``None``."""

    if not isinstance(candidate, str) or not isinstance(objective, str):
        raise TypeError("objective program verifier inputs must be text")
    executed = _execute_objective(objective)
    if executed is None:
        return None
    if "FINAL_ANSWER:" not in candidate:
        bounded = candidate.strip()
        if not (
            (bounded.startswith("{") and bounded.endswith("}"))
            or _JSON_FENCE_RE.fullmatch(bounded) is not None
        ):
            return None
    family, expected, execution = executed
    failure_codes: list[str] = []
    try:
        produced = _candidate_payload(candidate)
    except (TypeError, ValueError) as exc:
        produced = None
        failure_codes.append(f"candidate_contract:{type(exc).__name__}")
    if produced is not None and produced != expected:
        failure_codes.append("objective_result_mismatch")
    payload = {
        "schema": OBJECTIVE_PROGRAM_VERIFIER_SCHEMA,
        "family": family,
        "outcome": "refuted" if failure_codes else "verified",
        "objective_sha256": _text_sha(objective),
        "candidate_sha256": _text_sha(candidate),
        "expected_payload_sha256": _sha(expected),
        "produced_payload_sha256": _sha(produced),
        "execution": execution,
        "failure_codes": failure_codes,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


__all__ = [
    "OBJECTIVE_PROGRAM_SOLUTION_SCHEMA",
    "OBJECTIVE_PROGRAM_VERIFIER_SCHEMA",
    "solve_objective_program",
    "validate_objective_program_solution",
    "verify_objective_program",
]
