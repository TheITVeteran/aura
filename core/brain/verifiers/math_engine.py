"""Math truth engine — exact arithmetic / sympy / z3 checks on a candidate.

Wraps :class:`core.reasoning.symbolic_bridge.SymbolicBridge`. It does not ask the
LLM to be right — it re-checks every stated ``expr = value`` claim with exact
arithmetic and flags the wrong ones. This is the cheapest, highest-yield verifier:
a confident calculation error becomes a hard fail.
"""
from __future__ import annotations

import math
import re
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Bounds keep derivation exact and cheap (no giant powers/factorials).
_MAX_POW_EXP = 64
_MAX_FACT_N = 50


def _i(s: str) -> int:
    return int(s.replace(",", ""))


def _derive_exact_answer(question: str) -> tuple[str, str] | None:
    """Derive a canonical, exactly-computable answer from the QUESTION text.

    Returns (label, exact_value_as_str) for operation classes models reliably fumble
    — modulo, power, gcd, factorial, and binary arithmetic — or None when the question
    isn't one of these. This is what makes the verifier *sound* for these classes: a
    wrong final number becomes a hard fail the amplifier can filter out.
    """
    q = str(question or "").lower()
    try:
        m = re.search(r"(\d[\d,]*)\s*(?:mod(?:ulo)?|%)\s*(\d[\d,]*)", q)
        if m and _i(m.group(2)) != 0:
            return f"{_i(m.group(1))} mod {_i(m.group(2))}", str(_i(m.group(1)) % _i(m.group(2)))

        m = re.search(r"(\d[\d,]*)\s*(?:\^|\*\*|to the power of|raised to(?: the power of)?)\s*(\d+)", q)
        if m and _i(m.group(2)) <= _MAX_POW_EXP:
            return f"{_i(m.group(1))}^{_i(m.group(2))}", str(_i(m.group(1)) ** _i(m.group(2)))

        m = re.search(r"(?:gcd|greatest common divisor)\D*(\d[\d,]*)\D+(\d[\d,]*)", q)
        if m:
            return f"gcd({_i(m.group(1))},{_i(m.group(2))})", str(math.gcd(_i(m.group(1)), _i(m.group(2))))

        m = re.search(r"(\d+)\s*(?:!|factorial)", q)
        if m and _i(m.group(1)) <= _MAX_FACT_N and "trailing" not in q and "zero" not in q:
            return f"{_i(m.group(1))}!", str(math.factorial(_i(m.group(1))))

        m = re.search(r"(\d[\d,]*)\s*(?:times|multiplied by|\*)\s*(\d[\d,]*)", q)
        if m:
            return f"{_i(m.group(1))}*{_i(m.group(2))}", str(_i(m.group(1)) * _i(m.group(2)))
    except (ValueError, OverflowError):
        return None
    return None


def _final_answer_matches(text: str, exact: str) -> bool:
    """True if the exact value appears as one of the numbers in the candidate."""
    try:
        target = float(exact)
    except ValueError:
        return exact in text
    for tok in _NUM.findall(text):
        try:
            if abs(float(tok.replace(",", "")) - target) < 1e-6:
                return True
        except ValueError:
            continue
    return False


class MathTruthEngine:
    name = "math"
    domains = ("math", "arithmetic", "calculation", "logic_math")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "")
        try:
            from core.reasoning.symbolic_bridge import SymbolicBridge

            bridge = SymbolicBridge()
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - import guard
            record_degradation("math_truth_engine", exc)
            return VerificationResult(domain="math", ok=True, checked=False, engine=self.name)

        arithmetic_errors = bridge.check_arithmetic_claims(text)
        # An explicit verification target lets us actually solve & compare.
        target = (context or {}).get("verify_expression") if context else None
        target_ok = None
        target_value: Any = None
        target_label: Any = target
        if target:
            res = bridge.evaluate(str(target))
            if res.ok:
                target_value = res.result
                target_ok = str(res.result) in text or str(res.result).rstrip("0").rstrip(".") in text

        # No explicit target — derive a canonical exact answer from the QUESTION so the
        # verifier is SOUND for modulo/power/gcd/factorial/arithmetic (the classes models
        # fumble). A wrong final number is now a hard fail amplification can filter out.
        if target_ok is None:
            derived = _derive_exact_answer(str((context or {}).get("objective", "")))
            if derived is not None:
                target_label, target_value = derived
                target_ok = _final_answer_matches(text, str(target_value))

        # If there were no numeric claims and nothing to compare against, nothing to check.
        if not arithmetic_errors and target_ok is None and "=" not in text:
            return VerificationResult(domain="math", ok=True, checked=False, engine=self.name)

        issues = [f"arithmetic error: {e['claim']} (correct: {e['correct']})" for e in arithmetic_errors]
        if target_ok is False:
            issues.append(f"answer does not match exact value {target_value} of {target_label}")
        ok = not arithmetic_errors and target_ok is not False
        score = 0.95 if ok else max(0.05, 0.5 - 0.2 * len(arithmetic_errors))
        evidence = [f"exact({target_label}) = {target_value}"] if target_value is not None else []
        return VerificationResult(
            domain="math",
            ok=ok,
            checked=True,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            evidence=evidence,
            detail={"arithmetic_errors": len(arithmetic_errors)},
        )
