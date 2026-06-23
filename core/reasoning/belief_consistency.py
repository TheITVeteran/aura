"""Logical consistency checking over Aura's beliefs.

Beliefs are stored as natural-language claims (``core/belief_revision.py``). This
module is the bridge that lets the natural-deduction prover act on them: it encodes
each claim as a propositional literal (an atom, or its negation when the claim is
phrased negatively) and uses :mod:`core.reasoning.natural_deduction` to detect when
Aura simultaneously, confidently holds a proposition **and** its negation — a
genuine logical inconsistency in her self-model.

The encoding is deliberately conservative: it only pairs a claim with its *explicit*
negation (same core proposition, opposite polarity), so it does not hallucinate
semantic contradictions. What it catches is sound — "X" together with "not X".

Detected contradictions are surfaced to governance via
:mod:`core.reasoning.deduction_governance`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.reasoning.natural_deduction import (
    Atom,
    Formula,
    Not,
    find_contradiction,
    is_consistent,
)

# Negation cues that flip a claim's polarity.
_NEG_PATTERNS = [
    r"\bnot\b", r"\bno longer\b", r"\bnever\b", r"\bisn'?t\b", r"\baren'?t\b",
    r"\bwasn'?t\b", r"\bweren'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b", r"\bdidn'?t\b",
    r"\bcannot\b", r"\bcan'?t\b", r"\bwon'?t\b", r"\bfalse that\b", r"\buntrue\b",
    r"\bno\b",
]
_NEG_RE = re.compile("|".join(_NEG_PATTERNS), re.IGNORECASE)
# Filler words dropped when forming the core proposition key (so polarity is the
# only difference between "I am sovereign" and "I am not sovereign").
_FILLER_RE = re.compile(r"\b(a|an|the|is|are|was|were|am|be|being|been|do|does|did)\b", re.IGNORECASE)


@dataclass(frozen=True)
class EncodedBelief:
    formula: Formula
    core_key: str
    negated: bool
    source: str


@dataclass
class ConsistencyReport:
    consistent: bool
    contradictions: list[tuple[str, str]] = field(default_factory=list)  # (belief, opposing belief)
    minimal_core: list[str] = field(default_factory=list)
    checked: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "consistent": self.consistent,
            "contradictions": self.contradictions,
            "minimal_core": self.minimal_core,
            "checked": self.checked,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).strip(" .!?")


def encode_belief(content: str) -> EncodedBelief:
    """Encode a natural-language claim as a propositional literal.

    A negation cue flips the polarity; the negation cue and grammatical filler are
    stripped to form the *core key*, so an affirmation and its explicit negation
    map to the same atom with opposite polarity.
    """
    norm = _normalize(content)
    negated = bool(_NEG_RE.search(norm))
    core = _NEG_RE.sub(" ", norm)
    core = _FILLER_RE.sub(" ", core)
    core = re.sub(r"\s+", " ", core).strip()
    atom = Atom(core or norm or "∅")
    return EncodedBelief(
        formula=Not(atom) if negated else atom,
        core_key=atom.name,
        negated=negated,
        source=str(content),
    )


def check_beliefs(
    beliefs: list[tuple[str, float]],
    *,
    min_confidence: float = 0.6,
) -> ConsistencyReport:
    """Check a set of (claim, confidence) beliefs for logical inconsistency.

    Only beliefs at or above ``min_confidence`` participate — a low-confidence
    stray claim should not be treated as a firm contradiction.
    """
    encoded: list[EncodedBelief] = [
        encode_belief(content)
        for content, conf in beliefs
        if content and float(conf) >= min_confidence
    ]
    formulas = [e.formula for e in encoded]
    if is_consistent(formulas):
        return ConsistencyReport(consistent=True, checked=len(encoded))

    core = find_contradiction(formulas) or []
    core_keys = {f.f.name if isinstance(f, Not) else f.name for f in core}  # type: ignore[attr-defined]
    # Pair up the opposing source claims that produced the conflict.
    by_key: dict[str, dict[bool, str]] = {}
    for e in encoded:
        if e.core_key in core_keys:
            by_key.setdefault(e.core_key, {})[e.negated] = e.source
    contradictions: list[tuple[str, str]] = []
    for key, sides in by_key.items():
        if True in sides and False in sides:
            contradictions.append((sides[False], sides[True]))
    return ConsistencyReport(
        consistent=False,
        contradictions=contradictions,
        minimal_core=[str(f) for f in core],
        checked=len(encoded),
    )
