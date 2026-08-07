"""Deterministic executable verifier for recurrent reasoning episodes.

Every subsystem above the recurrence loop -- temporary fast weights, the
generative and counterfactual verifiers, the value-of-computation controller's
willingness to spend another step -- is gated on an *admitted* task verifier.
Without one the engine reports ``admitted_task_verifier_unavailable``, fast
weights take zero optimization attempts, and the controller abstains at the
minimum step. The stack does not fail in that state; it never runs.

Admission is not granted by passing a callable. ``blind_review`` first runs a
decoy preflight: the candidate scores four synthetic controls -- correct
arithmetic with valid code, wrong arithmetic with invalid code, and two
byte-identical texts -- and is admitted only when it separates correct from
incorrect by a margin and returns bit-identical scores for identical input.
That gate deliberately refuses an answer-key lookup, which scores every
control alike, and admits a verifier that actually *checks* things.

So this module is the tool-grounding tier: it re-derives what a candidate
asserts instead of asking a model whether the assertion looks right. It
evaluates arithmetic claims by computing them, and code blocks by compiling
them. Nothing here consults a task's expected answer, so an episode guided by
this verifier is measuring its own work rather than being told the result --
which is what makes it deployable rather than a diagnostic ceiling.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any, Final

VERIFIER_SCHEMA: Final = "aura.rlc.executable_verifier.v1"

# "12 + 30 = 42", "7 * 6 = 42", "100 - 1 = 99". Bounded operand width keeps a
# pathological candidate from turning verification into its own compute sink.
_ARITHMETIC: Final = re.compile(
    r"(?<![\w.])(-?\d{1,12})\s*([+\-*/%])\s*(-?\d{1,12})\s*=\s*(-?\d{1,15})(?![\w.])"
)
_CODE_BLOCK: Final = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_MAX_CLAIMS: Final = 64
_MAX_CODE_CHARS: Final = 20_000


def _check_arithmetic(text: str) -> tuple[int, int]:
    """Return (verified, total) over arithmetic claims the candidate asserts."""
    verified = 0
    total = 0
    for match in list(_ARITHMETIC.finditer(text))[:_MAX_CLAIMS]:
        left, op, right, claimed = match.groups()
        try:
            a, b, c = int(left), int(right), int(claimed)
        except ValueError:
            continue
        if op == "/" and b == 0:
            continue
        if op == "%" and b == 0:
            continue
        total += 1
        if op == "+":
            actual: float = a + b
        elif op == "-":
            actual = a - b
        elif op == "*":
            actual = a * b
        elif op == "/":
            actual = a / b
        else:
            actual = a % b
        if abs(actual - c) < 1e-9:
            verified += 1
    return verified, total


def _check_code(text: str) -> tuple[int, int]:
    """Return (parseable, total) over fenced code blocks.

    Parsing only. The verifier never executes candidate code: a reasoning
    episode is not authorization to run whatever the model wrote.
    """
    verified = 0
    total = 0
    for block in list(_CODE_BLOCK.finditer(text))[:_MAX_CLAIMS]:
        source = block.group(1)
        if not source.strip() or len(source) > _MAX_CODE_CHARS:
            continue
        total += 1
        try:
            ast.parse(source)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
        verified += 1
    return verified, total


def evaluate(text: Any) -> dict[str, Any]:
    """Deterministic evidence report for one candidate answer."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    arithmetic_ok, arithmetic_total = _check_arithmetic(text)
    code_ok, code_total = _check_code(text)
    checked = arithmetic_total + code_total
    passed = arithmetic_ok + code_ok
    return {
        "schema": VERIFIER_SCHEMA,
        "arithmetic_verified": arithmetic_ok,
        "arithmetic_claims": arithmetic_total,
        "code_parseable": code_ok,
        "code_blocks": code_total,
        "checks": checked,
        "passed": passed,
        # A candidate that asserts nothing checkable is neither right nor
        # wrong by this instrument, and says so rather than scoring zero --
        # scoring it zero would let an unverifiable answer be ranked below a
        # verifiably wrong one.
        "grounded": checked > 0,
    }


def score(text: Any) -> float:
    """Fraction of the candidate's own checkable claims that hold.

    Ungrounded candidates receive the neutral prior. The value is a pure
    function of the text -- required, because admission rejects any reviewer
    whose score for byte-identical input is not bit-identical.
    """
    report = evaluate(text)
    if not report["grounded"]:
        return _NEUTRAL
    return report["passed"] / report["checks"]


_NEUTRAL: Final = 0.5


def make_verifier(task: Any = None) -> Callable[[str], float]:
    """Build the callable the engine admits and then reasons under.

    ``task`` is accepted so a caller can bind the answer contract of a
    specific problem, and is deliberately unused for correctness: consulting
    a task's expected answer would make every episode a measurement of the
    answer key rather than of the reasoning. The parameter exists so the
    call site reads the same whether or not a task is in scope.
    """

    def _verify(candidate: str) -> float:
        return score(candidate)

    _verify.schema = VERIFIER_SCHEMA  # type: ignore[attr-defined]
    return _verify
