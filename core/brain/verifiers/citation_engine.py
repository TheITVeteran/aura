"""Citation truth engine — factual claims need local grounding, not vibes.

For factual/architecture tasks the amplifier supplies ``required_evidence`` (memory
hits, retrieved snippets, source spans). This engine checks that the candidate's
load-bearing factual sentences are actually supported by that evidence pack, and
flags ungrounded confident assertions. It is deliberately conservative: it lowers
score and surfaces issues rather than hard-failing, because absence of a citation
is weaker than a provable contradiction.

When the caller supplies NO evidence, the engine no longer shrugs — it fetches
its own receipts from the local corpus (BM25 over ingested knowledge) and checks
against those. Self-fetched semantics are asymmetric on purpose: a CONTRADICTION
against locally-known facts is a hard fail (polarity flip on a shared subject is
real signal), but absence-of-mention is NOT — the corpus is partial, so a true
claim it never ingested must not be punished. ``checked`` is claimed only when
at least one confident sentence actually had overlapping evidence to test.
"""
from __future__ import annotations

import asyncio
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


_NEG_RE = re.compile(r"\b(?:not|never|no|none|without|un\w+|isn'?t|aren'?t|won'?t|can'?t|doesn'?t)\b", re.IGNORECASE)
# Absolutes that frequently mark an over-claim contradicting a bounded fact.
_ABSOLUTE_RE = re.compile(r"\b(?:unlimited|infinite|always|never|every|all|none|no limit|forever|guaranteed)\b", re.IGNORECASE)
_BOUNDED_RE = re.compile(r"\b(?:three|two|one|four|five|\d+|limit|bounded|up to|at most|fails? closed|maximum|cap)\b", re.IGNORECASE)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())}


def _grounded(sentence: str, evidence_blob: str) -> bool:
    """A sentence is grounded if it shares enough content words with the evidence."""
    words = _content_words(sentence)
    if not words:
        return True
    ev = evidence_blob.lower()
    hits = sum(1 for w in words if w in ev)
    return hits >= max(2, int(0.4 * len(words)))


def _contradicts(sentence: str, evidence_items: list[str]) -> bool:
    """A confident sentence contradicts evidence if, on a shared subject, it flips the
    polarity (negation mismatch) or asserts an absolute the evidence bounds.

    Word-overlap grounding alone passes 'the budget is unlimited' against evidence
    saying 'three attempts then fails closed' — they share words. This catches the
    contradiction overlap cannot."""
    s_words = _content_words(sentence)
    if not s_words:
        return False
    s_neg = bool(_NEG_RE.search(sentence))
    s_absolute = bool(_ABSOLUTE_RE.search(sentence))
    for ev in evidence_items:
        e_words = _content_words(str(ev))
        shared = s_words & e_words
        if len(shared) < 2:
            continue
        e_neg = bool(_NEG_RE.search(str(ev)))
        # Polarity flip on a shared subject.
        if s_neg != e_neg:
            return True
        # Absolute claim against a bounded fact.
        if s_absolute and _BOUNDED_RE.search(str(ev)):
            return True
    return False


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
        self_fetched = False
        if not "\n".join(str(e) for e in evidence_items).strip():
            evidence_items = await self._fetch_local_evidence(candidate, ctx)
            self_fetched = bool(evidence_items)
        evidence_blob = "\n".join(str(e) for e in evidence_items)
        if not evidence_blob.strip():
            # No evidence anywhere (caller OR local corpus) → advise only.
            return VerificationResult(domain="citation", ok=True, checked=False, engine=self.name)

        ungrounded: list[str] = []
        contradictions: list[str] = []
        confident_total = 0
        examined = 0
        for sent in _sentences(candidate):
            if _HEDGE_RE.search(sent):
                continue  # appropriately hedged claims are not penalised
            if _CONFIDENT_FACT_RE.search(sent):
                confident_total += 1
                overlaps = any(
                    len(_content_words(sent) & _content_words(str(ev))) >= 2
                    for ev in evidence_items
                )
                if overlaps:
                    examined += 1
                if _contradicts(sent, evidence_items):
                    contradictions.append(sent[:120])
                elif not _grounded(sent, evidence_blob):
                    ungrounded.append(sent[:120])

        issues = [f"contradicts evidence: {s}" for s in contradictions[:6]]
        if self_fetched:
            # Partial-corpus semantics: absence of mention is not wrongness.
            # Report ungrounded claims as advisories, never as failures.
            issues += [f"unconfirmed by local corpus: {s}" for s in ungrounded[:6]]
            ok = not contradictions
            checked = examined > 0
        else:
            issues += [f"ungrounded confident claim: {s}" for s in ungrounded[:6]]
            ok = not contradictions and not ungrounded
            checked = confident_total > 0
        # A contradiction is a hard fail; bare ungroundedness lowers score but is softer.
        bad = len(contradictions) + (0 if self_fetched else len(ungrounded))
        ratio = 1.0 - (bad / max(1, confident_total))
        score = max(0.1, 0.5 + 0.45 * ratio)
        if contradictions:
            score = min(score, 0.15)
        return VerificationResult(
            domain="citation",
            ok=ok,
            checked=checked,
            score=round(score, 4),
            engine=self.name,
            issues=issues,
            evidence=[
                f"evidence pack: {len(evidence_items)} item(s)"
                + (" (self-fetched from local corpus)" if self_fetched else "")
            ],
            detail={"confident_claims": confident_total, "ungrounded": len(ungrounded),
                    "contradictions": len(contradictions),
                    "self_fetched_evidence": self_fetched, "examined": examined},
        )

    @staticmethod
    async def _fetch_local_evidence(candidate: str, ctx: dict[str, Any]) -> list[str]:
        """Pull receipts from the local corpus when the caller brought none.

        Query = the objective (what was asked) when available, else the
        candidate's first confident sentence. Bounded, read-only, off-loop
        (SQLite in a thread), and silent on any miss — an evidence fetch
        must never break verification.
        """
        query = str(ctx.get("objective", "") or "").strip()
        if not query:
            confident = [
                s for s in _sentences(candidate) if _CONFIDENT_FACT_RE.search(s)
            ]
            query = confident[0] if confident else ""
        if len(query) < 12:
            return []
        try:
            from core.knowledge.local_corpus import get_local_corpus_store

            store = get_local_corpus_store()
            hits = await asyncio.wait_for(
                asyncio.to_thread(store.search, query, 4), timeout=3.0
            )
            return [
                f"{hit.title}: {hit.snippet}" for hit in (hits or []) if hit.snippet
            ]
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError, asyncio.TimeoutError):
            return []
