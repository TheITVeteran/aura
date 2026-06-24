"""Inference-time reasoning amplification — verifier-filtered self-consistency.

Aura already samples multiple reasoning paths (the CONSISTENCY strategy) and is wired
into the cognitive engine. What was missing is the frontier piece that actually lifts a
base model's accuracy: **filter out provably-wrong paths before voting**. Today's
reasoning models work this way — many samples, self-verification, and convergence — not
a bigger network.

``amplify`` is the reusable core: given a set of candidate reasoning paths it (1) asks
Aura's own deduction engine whether each path contains a provable non-sequitur or
arithmetic error, (2) keeps the verifier-clean paths, and (3) takes the answer the most
of them converge on (self-consistency). The confidence is the real agreement fraction,
lifted when the winning cluster is verifier-clean. This is pure over already-generated
candidates, so it is deterministically testable; ``DeliberationEngine`` wraps it with
parallel sampling for standalone use.

It is wired into ``reasoning_strategies._consistency`` so it runs on the hard/factual
turns the classifier already routes there — i.e. it is causal, not a shelf ornament.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningAmplifier")

_ANSWER_RE = re.compile(r"(?:final answer|answer|therefore)\s*[:\-]?\s*(.+)$", re.IGNORECASE)


def default_extract_answer(text: str) -> str:
    """Pull the conclusion out of a reasoning trace (last 'answer:' or last line)."""
    t = str(text or "").strip()
    matches = _ANSWER_RE.findall(t)
    if matches:
        return matches[-1].strip().rstrip(".")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return (lines[-1] if lines else t).rstrip(".")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower()).strip(" .!?\"'")


async def verify_reasoning(text: str) -> tuple[bool, list[str]]:
    """True if the reasoning has no provable non-sequitur or arithmetic error."""
    try:
        from core.reasoning.symbolic_bridge import SymbolicBridge

        audit = SymbolicBridge().audit_reasoning(text)
        issues: list[str] = []
        for ns in audit.get("non_sequiturs", []) or []:
            issues.append(f"non-sequitur: {ns.get('conclusion')}")
        for ae in audit.get("arithmetic_errors", []) or []:
            issues.append(f"arithmetic: {ae.get('claim')}")
        return (not issues), issues
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_amplifier", exc)
        return True, []


@dataclass
class AmplifiedResult:
    answer: str
    confidence: float
    n: int
    valid_n: int
    agreement: float
    verified: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer[:200],
            "confidence": round(self.confidence, 3),
            "n": self.n,
            "valid_n": self.valid_n,
            "agreement": round(self.agreement, 3),
            "verified": self.verified,
        }


async def amplify(
    candidates: list[str],
    *,
    extract_answer: Callable[[str], str] | None = None,
    verify: Callable[[str], Awaitable[tuple[bool, list[str]]]] | None = None,
) -> AmplifiedResult:
    """Verifier-filtered self-consistency over candidate reasoning paths."""
    extract = extract_answer or default_extract_answer
    verifier = verify or verify_reasoning
    cands = [c for c in candidates if c and str(c).strip()]
    if not cands:
        return AmplifiedResult(answer="", confidence=0.0, n=0, valid_n=0, agreement=0.0)

    flags: list[bool] = []
    all_issues: list[str] = []
    for c in cands:
        ok, issues = await verifier(c)
        flags.append(ok)
        all_issues.extend(issues)
    valid = [c for c, ok in zip(cands, flags, strict=True) if ok]
    pool = valid if valid else cands

    # Self-consistency: cluster the pool by extracted answer; the biggest cluster wins.
    clusters: dict[str, list[str]] = {}
    for c in pool:
        clusters.setdefault(_normalize(extract(c)), []).append(c)
    _key, winners = max(clusters.items(), key=lambda kv: len(kv[1]))
    agreement = len(winners) / len(pool)
    verified_winner = bool(valid) and winners[0] in valid
    confidence = min(0.98, agreement * (1.0 if verified_winner else 0.85))

    logger.info(
        "🧠 [Amplify] %d paths, %d verifier-clean → answer agreement %.0f%% (conf %.2f)",
        len(cands), len(valid), agreement * 100, confidence,
    )
    return AmplifiedResult(
        answer=winners[0],
        confidence=round(confidence, 4),
        n=len(cands),
        valid_n=len(valid),
        agreement=round(agreement, 4),
        verified=verified_winner,
        issues=all_issues[:10],
    )


class DeliberationEngine:
    """Standalone reasoning amplifier: parallel-sample then amplify."""

    def __init__(self, *, n_samples: int = 5, temperatures: list[float] | None = None) -> None:
        self.n_samples = max(1, int(n_samples))
        self.temperatures = temperatures or [0.2, 0.5, 0.7, 0.9, 1.0]

    async def deliberate(
        self,
        question: str,
        generate: Callable[[str, float], Awaitable[str]],
        **kw: Any,
    ) -> AmplifiedResult:
        import asyncio

        async def _one(i: int) -> str:
            try:
                return await generate(question, self.temperatures[i % len(self.temperatures)])
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("reasoning_amplifier", exc)
                return ""

        samples = await asyncio.gather(*[_one(i) for i in range(self.n_samples)])
        return await amplify([s for s in samples if s], **kw)


_instance: DeliberationEngine | None = None


def get_deliberation_engine() -> DeliberationEngine:
    global _instance
    if _instance is None:
        _instance = DeliberationEngine()
    return _instance
