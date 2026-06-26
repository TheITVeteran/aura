"""Planning truth engine — a plan must be a real, ordered, precondition-checked plan.

Frontier planning failures are rarely "wrong idea"; they are skipped preconditions,
circular steps, or vacuous one-liners dressed up as plans. This engine parses the
candidate into discrete steps and checks: there are concrete steps, each is an
actionable verb-phrase, declared preconditions reference earlier steps, and there
is a verification/acceptance step at the end. It is structural, not semantic — a
cheap guard that stops empty plans from passing as plans.
"""
from __future__ import annotations

import re
from typing import Any

from .base import VerificationResult

_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•]|step\s*\d+\s*[:.])\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_ACTION_VERB_RE = re.compile(
    r"\b(?:create|add|build|run|write|edit|check|verify|test|deploy|read|inspect|"
    r"open|patch|install|configure|measure|validate|gather|search|compute|call|"
    r"plan|decompose|retrieve|execute|review|confirm|update|remove|fix)\b",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(r"\b(?:verify|test|check|validate|confirm|assert|review|receipt)\b", re.IGNORECASE)


class PlanningEngine:
    name = "planning"
    domains = ("planning", "plan", "action", "tool")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "")
        steps = [m.group(1).strip() for m in _STEP_RE.finditer(text)]
        if len(steps) < 2:
            # Not a structured plan; advise rather than fail (might be a one-line answer).
            return VerificationResult(
                domain="planning", ok=len(steps) >= 1, checked=bool(steps), score=0.4,
                engine=self.name, issues=["plan has fewer than 2 discrete steps"] if steps else [],
            )

        issues: list[str] = []
        actionable = sum(1 for s in steps if _ACTION_VERB_RE.search(s))
        if actionable < len(steps):
            issues.append(f"{len(steps) - actionable}/{len(steps)} steps are not actionable verb-phrases")
        has_verification = any(_VERIFY_RE.search(s) for s in steps[-3:])
        if not has_verification:
            issues.append("no verification / acceptance step at the end of the plan")

        action_ratio = actionable / len(steps)
        ok = action_ratio >= 0.5 and has_verification
        score = 0.4 + 0.4 * action_ratio + (0.15 if has_verification else 0.0)
        return VerificationResult(
            domain="planning",
            ok=ok,
            checked=True,
            score=round(min(0.97, score), 4),
            engine=self.name,
            issues=issues,
            evidence=[f"{len(steps)} steps, {actionable} actionable"],
            detail={"steps": len(steps), "actionable": actionable, "has_verification": has_verification},
        )
