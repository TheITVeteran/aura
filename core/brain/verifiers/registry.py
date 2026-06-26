"""Verifier registry — pick the right truth engines for a task and fold their verdicts.

The amplifier asks ``verify_candidate(answer, task_type=..., context=...)`` and gets
back one :class:`VerificationResult`. The registry runs every engine whose
``domains`` match the task type (plus the always-on logic engine), in parallel, and
combines them: any checked engine that fails flips the hard gate; the soft score is
the mean of the engines that actually checked.
"""
from __future__ import annotations

import asyncio
from typing import Any

from core.runtime.errors import record_degradation

from .base import VerificationResult, Verifier, combine_results
from .citation_engine import CitationEngine
from .code_engine import CodeTruthEngine
from .logic_engine import LogicTruthEngine
from .math_engine import MathTruthEngine
from .planning_engine import PlanningEngine
from .repo_engine import RepoEvidenceEngine

# Map a coarse task_type to the engines that should run. The logic engine is
# always added on top (deductive sanity applies to any prose).
_TASK_ALIASES = {
    "code": "code", "coding": "code", "code_audit": "code_audit", "code_patch": "code_patch",
    "debug": "debug", "bugfix": "code_patch",
    "math": "math", "arithmetic": "math", "calculation": "math",
    "repo": "repo", "repo_audit": "repo_audit", "codebase": "repo_audit",
    "architecture": "architecture", "self_claim": "self_claim",
    "factual": "factual", "fact": "factual", "qa": "factual",
    "planning": "planning", "plan": "planning", "action": "action", "tool": "tool",
}


class VerifierRegistry:
    def __init__(self, verifiers: list[Verifier] | None = None) -> None:
        self._verifiers: list[Verifier] = verifiers or [
            CodeTruthEngine(),
            MathTruthEngine(),
            LogicTruthEngine(),
            RepoEvidenceEngine(),
            CitationEngine(),
            PlanningEngine(),
        ]

    @staticmethod
    def normalize_task(task_type: str | None) -> str:
        t = str(task_type or "generic").strip().lower()
        return _TASK_ALIASES.get(t, t)

    def select(self, task_type: str | None) -> list[Verifier]:
        normalized = self.normalize_task(task_type)
        selected = [v for v in self._verifiers if v.handles(normalized)]
        # Ensure the always-on logic engine is present exactly once.
        if not any(getattr(v, "name", "") == "logic" for v in selected):
            selected.extend(v for v in self._verifiers if getattr(v, "name", "") == "logic")
        return selected

    async def verify(
        self,
        candidate: str,
        *,
        task_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        normalized = self.normalize_task(task_type)
        verifiers = self.select(normalized)
        if not verifiers:
            return VerificationResult(domain=normalized, ok=True, checked=False, engine="none")

        async def _run(v: Verifier) -> VerificationResult:
            try:
                return await v.verify(candidate, context=context)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(f"verifier:{getattr(v, 'name', '?')}", exc)
                return VerificationResult(domain=normalized, ok=True, checked=False, engine=getattr(v, "name", "?"))

        results = await asyncio.gather(*[_run(v) for v in verifiers])
        return combine_results(normalized, list(results))


_registry: VerifierRegistry | None = None


def get_verifier_registry() -> VerifierRegistry:
    global _registry
    if _registry is None:
        _registry = VerifierRegistry()
    return _registry


async def verify_candidate(
    candidate: str,
    *,
    task_type: str | None = None,
    context: dict[str, Any] | None = None,
) -> VerificationResult:
    """Module-level convenience: verify one candidate against its task's truth engines."""
    return await get_verifier_registry().verify(candidate, task_type=task_type, context=context)
