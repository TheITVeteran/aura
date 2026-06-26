"""Citation truth engine — factual claims need local grounding, not vibes.

For factual/architecture tasks the amplifier supplies ``required_evidence`` (memory
hits, retrieved snippets, source spans). This engine checks that the candidate's
load-bearing factual sentences are actually supported by that evidence pack, and
flags ungrounded confident assertions. It is deliberately conservative: it lowers
score and surfaces issues rather than hard-failing, because absence of a citation
is weaker than a provable contradiction.
"""
from __future__ import annotations

import re
from typing import Any

from .base import VerificationResult

_HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|might|possibly|i'?m not sure|uncertain|guess|probably|i believe)\b",
    re.IGNORECASE,
)
_CONFIDENT_FACT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|always|never|definitely|certainly|in fact|the answer is)\b",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if len(s.strip()) > 12]


def _grounded(sentence: str, evidence_blob: str) -> bool:
    """A sentence is grounded if it shares enough content words with the evidence."""
    words = {w for w in re.findall(r"[a-zA-Z]{4,}", sentence.lower())}
    if not words:
        return True
    ev = evidence_blob.lower()
    hits = sum(1 for w in words if w in ev)
    return hits >= max(2, int(0.4 * len(words)))


class CitationEngine:
    name = "citation"
    domains = ("factual", "architecture", "self_claim", "repo_audit")

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        ctx = context or {}
        evidence_items = ctx.get("required_evidence") or ctx.get("evidence") or []
        if isinstance(evidence_items, str):
            evidence_items = [evidence_items]
        evidence_blob = "\n".join(str(e) for e in evidence_items)
        if not evidence_blob.strip():
            # No evidence pack provided → nothing to check against; advise only.
            return VerificationResult(domain="citation", ok=True, checked=False, engine=self.name)

        ungrounded: list[str] = []
        confident_total = 0
        for sent in _sentences(candidate):
            if _HEDGE_RE.search(sent):
                continue  # appropriately hedged claims are not penalised
            if _CONFIDENT_FACT_RE.search(sent):
                confident_total += 1
                if not _grounded(sent, evidence_blob):
                    ungrounded.append(sent[:120])

        issues = [f"ungrounded confident claim: {s}" for s in ungrounded[:6]]
        ratio = 1.0 - (len(ungrounded) / max(1, confident_total))
        ok = not ungrounded
        score = max(0.1, 0.5 + 0.45 * ratio)
        return VerificationResult(
            domain="citation",
            ok=ok,
            checked=confident_total > 0,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            evidence=[f"evidence pack: {len(evidence_items)} item(s)"],
            detail={"confident_claims": confident_total, "ungrounded": len(ungrounded)},
        )
