"""Uniform verification result + verifier protocol for the truth engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VerificationResult:
    """The verdict of one truth engine over a candidate answer.

    ``ok`` is the hard gate (no *provable* failure was found). ``checked`` records
    whether a real check actually ran — an engine that found nothing to verify
    (e.g. no code in the candidate) returns ``ok=True, checked=False`` so the
    amplifier does not mistake "nothing to check" for "verified correct".
    ``score`` is a soft 0..1 contribution used to rank surviving candidates.
    """

    domain: str
    ok: bool
    checked: bool
    score: float = 0.5
    engine: str = ""
    issues: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "ok": self.ok,
            "checked": self.checked,
            "score": round(float(self.score), 3),
            "engine": self.engine,
            "issues": self.issues[:8],
            "evidence": self.evidence[:8],
        }


@runtime_checkable
class Verifier(Protocol):
    """A domain truth engine.

    ``domains`` lists the task types it claims (e.g. ``("code",)``); ``"*"`` means
    it always runs (logic/non-sequitur checks apply to any prose answer).
    """

    name: str
    domains: tuple[str, ...]

    def handles(self, task_type: str) -> bool: ...

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult: ...


def combine_results(domain: str, results: list[VerificationResult]) -> VerificationResult:
    """Fold many engine verdicts into one for a candidate.

    Hard gate: ``ok`` is False if *any* engine that actually checked failed.
    ``checked`` is True if at least one engine checked. ``score`` is the mean of
    the checked engines (defaulting to neutral 0.5 when nothing was checkable).
    """
    checked = [r for r in results if r.checked]
    issues: list[str] = []
    evidence: list[str] = []
    engines: list[str] = []
    for r in results:
        issues.extend(r.issues)
        evidence.extend(r.evidence)
        if r.engine:
            engines.append(r.engine)
    ok = all(r.ok for r in checked) if checked else True
    score = sum(r.score for r in checked) / len(checked) if checked else 0.5
    return VerificationResult(
        domain=domain,
        ok=ok,
        checked=bool(checked),
        score=round(score, 4),
        engine="+".join(engines),
        issues=issues[:12],
        evidence=evidence[:12],
        detail={"sub": [r.to_dict() for r in results]},
    )
