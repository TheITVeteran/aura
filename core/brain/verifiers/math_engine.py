"""Math truth engine — exact arithmetic / sympy / z3 checks on a candidate.

Wraps :class:`core.reasoning.symbolic_bridge.SymbolicBridge`. It does not ask the
LLM to be right — it re-checks every stated ``expr = value`` claim with exact
arithmetic and flags the wrong ones. This is the cheapest, highest-yield verifier:
a confident calculation error becomes a hard fail.
"""
from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult


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
        if target:
            res = bridge.evaluate(str(target))
            if res.ok:
                target_value = res.result
                target_ok = str(res.result) in text or str(res.result).rstrip("0").rstrip(".") in text

        # If there were no numeric claims and no target, there was nothing to check.
        if not arithmetic_errors and target is None and "=" not in text:
            return VerificationResult(domain="math", ok=True, checked=False, engine=self.name)

        issues = [f"arithmetic error: {e['claim']} (correct: {e['correct']})" for e in arithmetic_errors]
        if target_ok is False:
            issues.append(f"answer does not match exact value {target_value} of {target}")
        ok = not arithmetic_errors and target_ok is not False
        score = 0.95 if ok else max(0.05, 0.5 - 0.2 * len(arithmetic_errors))
        evidence = [f"exact({target}) = {target_value}"] if target_value is not None else []
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
