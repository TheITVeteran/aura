"""Calibration gate — Aura may only assert what survived a check.

Not glamorous, very powerful: most bad answers are not *illogical*, they are
*overconfident*. This gate classifies every load-bearing sentence of a candidate
answer by epistemic status —

    KNOWN            backed by memory / established fact in context
    TOOL_VERIFIED    a verifier or sandbox actually confirmed it
    SOURCE_BACKED    grounded in a supplied evidence span
    INFERRED         a reasonable deduction, not directly checked
    GUESSED          confident but unsupported
    UNVERIFIED       a claim we had no way to check
    IMPOSSIBLE_LOCALLY  asserts something that needs a capability we lack

— and then *applies* the verdict: confident-but-unsupported sentences get an
honesty hedge, impossible-locally claims are flagged. The result is an answer the
rest of the system is allowed to speak, plus a calibrated confidence the response
path and governance can read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EpistemicStatus(str, Enum):
    KNOWN = "known"
    TOOL_VERIFIED = "tool_verified"
    SOURCE_BACKED = "source_backed"
    INFERRED = "inferred"
    GUESSED = "guessed"
    UNVERIFIED = "unverified"
    IMPOSSIBLE_LOCALLY = "impossible_locally"


_CONFIDENCE_RANK = {
    EpistemicStatus.TOOL_VERIFIED: 1.0,
    EpistemicStatus.KNOWN: 0.92,
    EpistemicStatus.SOURCE_BACKED: 0.85,
    EpistemicStatus.INFERRED: 0.6,
    EpistemicStatus.UNVERIFIED: 0.4,
    EpistemicStatus.GUESSED: 0.3,
    EpistemicStatus.IMPOSSIBLE_LOCALLY: 0.05,
}

_HEDGE_RE = re.compile(
    r"\b(?:i think|maybe|might|possibly|i'?m not (?:sure|certain)|uncertain|"
    r"i'?m guessing|probably|i believe|it seems|appears to|i'?m not aware|"
    r"as far as i know|to my knowledge)\b",
    re.IGNORECASE,
)
_CONFIDENT_RE = re.compile(
    r"\b(?:is|are|was|were|will|always|never|definitely|certainly|guaranteed|"
    r"the answer is|in fact|undoubtedly|must be|exactly)\b",
    re.IGNORECASE,
)
# Phrases that claim a capability a local model generally cannot have done.
_IMPOSSIBLE_RE = re.compile(
    r"\b(?:i (?:just )?(?:browsed|googled|searched the web|accessed the internet|"
    r"checked online|called the api|ran it on your machine|looked it up online)|"
    r"according to (?:today'?s|the latest) (?:news|web))\b",
    re.IGNORECASE,
)


@dataclass
class ClaimLabel:
    text: str
    status: EpistemicStatus
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text[:160], "status": self.status.value, "reason": self.reason}


@dataclass
class CalibrationReport:
    overall: EpistemicStatus
    confidence: float
    labels: list[ClaimLabel] = field(default_factory=list)
    calibrated_answer: str = ""
    downgraded: int = 0
    flagged_impossible: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.value,
            "confidence": round(self.confidence, 3),
            "downgraded": self.downgraded,
            "flagged_impossible": self.flagged_impossible,
            "labels": [c.to_dict() for c in self.labels[:10]],
        }


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "")) if s.strip()]


class CalibrationGate:
    """Classify and honesty-correct an answer's claims by epistemic status."""

    def assess(
        self,
        answer: str,
        *,
        verification: Any | None = None,
        evidence: list[str] | None = None,
        tool_verified: bool = False,
        known_facts: list[str] | None = None,
    ) -> CalibrationReport:
        evidence_blob = "\n".join(str(e) for e in (evidence or [])).lower()
        known_blob = "\n".join(str(k) for k in (known_facts or [])).lower()
        # A verifier that actually checked and passed promotes factual sentences.
        v_checked = bool(getattr(verification, "checked", False))
        v_ok = bool(getattr(verification, "ok", False))

        labels: list[ClaimLabel] = []
        for sent in _sentences(answer):
            labels.append(self._classify_sentence(
                sent, evidence_blob, known_blob,
                v_checked=v_checked, v_ok=v_ok, tool_verified=tool_verified,
            ))

        calibrated, downgraded, flagged = self._apply(answer, labels)
        overall = self._overall(labels)
        confidence = self._confidence(labels, v_checked=v_checked, v_ok=v_ok)
        return CalibrationReport(
            overall=overall,
            confidence=confidence,
            labels=labels,
            calibrated_answer=calibrated,
            downgraded=downgraded,
            flagged_impossible=flagged,
        )

    def _classify_sentence(
        self,
        sent: str,
        evidence_blob: str,
        known_blob: str,
        *,
        v_checked: bool,
        v_ok: bool,
        tool_verified: bool,
    ) -> ClaimLabel:
        if _IMPOSSIBLE_RE.search(sent):
            return ClaimLabel(sent, EpistemicStatus.IMPOSSIBLE_LOCALLY, "claims a capability not available locally")
        hedged = bool(_HEDGE_RE.search(sent))
        confident = bool(_CONFIDENT_RE.search(sent))
        content_words = {w for w in re.findall(r"[a-zA-Z]{4,}", sent.lower())}

        def _overlap(blob: str) -> float:
            if not content_words or not blob:
                return 0.0
            return sum(1 for w in content_words if w in blob) / len(content_words)

        if known_blob and _overlap(known_blob) >= 0.4:
            return ClaimLabel(sent, EpistemicStatus.KNOWN, "matches a known fact in context")
        if evidence_blob and _overlap(evidence_blob) >= 0.4:
            return ClaimLabel(sent, EpistemicStatus.SOURCE_BACKED, "grounded in supplied evidence")
        if tool_verified and v_checked and v_ok and confident:
            return ClaimLabel(sent, EpistemicStatus.TOOL_VERIFIED, "confirmed by a verifier/sandbox")
        if hedged:
            return ClaimLabel(sent, EpistemicStatus.INFERRED, "appropriately hedged")
        if confident:
            return ClaimLabel(sent, EpistemicStatus.GUESSED, "confident assertion without support")
        return ClaimLabel(sent, EpistemicStatus.UNVERIFIED, "no check available")

    def _apply(self, answer: str, labels: list[ClaimLabel]) -> tuple[str, int, int]:
        """Rewrite the answer so confidence matches epistemic status."""
        downgraded = 0
        flagged = 0
        out: list[str] = []
        for label in labels:
            text = label.text
            if label.status is EpistemicStatus.IMPOSSIBLE_LOCALLY:
                flagged += 1
                out.append(f"[unverifiable locally] {text}")
            elif label.status is EpistemicStatus.GUESSED:
                downgraded += 1
                out.append(self._soften(text))
            else:
                out.append(text)
        return " ".join(out).strip() or str(answer or ""), downgraded, flagged

    @staticmethod
    def _soften(sentence: str) -> str:
        """Insert an honesty hedge into a confident-but-unsupported sentence."""
        if _HEDGE_RE.search(sentence):
            return sentence
        # Prepend a measured qualifier rather than mangling the sentence body.
        lead = sentence[0].lower() + sentence[1:] if sentence[:1].isupper() else sentence
        return f"I'm not fully certain, but {lead}"

    @staticmethod
    def _overall(labels: list[ClaimLabel]) -> EpistemicStatus:
        if not labels:
            return EpistemicStatus.UNVERIFIED
        if any(l.status is EpistemicStatus.IMPOSSIBLE_LOCALLY for l in labels):
            return EpistemicStatus.IMPOSSIBLE_LOCALLY
        # The weakest load-bearing status dominates the headline.
        order = [
            EpistemicStatus.GUESSED, EpistemicStatus.UNVERIFIED, EpistemicStatus.INFERRED,
            EpistemicStatus.SOURCE_BACKED, EpistemicStatus.KNOWN, EpistemicStatus.TOOL_VERIFIED,
        ]
        present = {l.status for l in labels}
        for status in order:
            if status in present:
                return status
        return EpistemicStatus.UNVERIFIED

    @staticmethod
    def _confidence(labels: list[ClaimLabel], *, v_checked: bool, v_ok: bool) -> float:
        if not labels:
            return 0.4
        base = sum(_CONFIDENCE_RANK[l.status] for l in labels) / len(labels)
        if v_checked and not v_ok:
            base *= 0.6  # a verifier actively found a problem
        return round(max(0.05, min(0.98, base)), 4)


_instance: CalibrationGate | None = None


def get_calibration_gate() -> CalibrationGate:
    global _instance
    if _instance is None:
        _instance = CalibrationGate()
    return _instance
