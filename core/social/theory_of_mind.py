"""Evidence-scoped perspective correction over canonical Theory of Mind."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.consciousness.theory_of_mind import TheoryOfMindEngine as CanonicalToM

_FACT_SOURCES = {"authorized_operator_correction", "verified_world_state"}


def _normalized(value: Any, *, field: str, limit: int) -> str:
    result = " ".join(str(value or "").strip().split())[:limit]
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _digest(value: Any) -> str:
    result = str(value or "").strip().casefold()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError("evidence_digest must be a SHA-256 hex digest")
    return result


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence must be finite and between zero and one") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and between zero and one")
    return result


@dataclass(frozen=True)
class EvidenceClaim:
    value: Any
    evidence_digest: str
    source: str
    confidence: float


class PerspectiveSimulator:
    """Compares verified task facts with sourced beliefs for one exact agent."""

    def __init__(self, *, person: str, canonical: CanonicalToM) -> None:
        self.person = _normalized(person, field="person", limit=160)
        self._canonical = canonical
        self._actual_facts: dict[str, EvidenceClaim] = {}

    @property
    def aura_beliefs(self) -> dict[str, Any]:
        """Compatibility view of verified task facts, detached from internal state."""
        return {key: claim.value for key, claim in self._actual_facts.items()}

    def aura_knows(
        self,
        key: str,
        value: Any,
        *,
        evidence_digest: str,
        source: str = "verified_world_state",
        confidence: float = 1.0,
    ) -> None:
        normalized_key = _normalized(key, field="fact key", limit=100)
        normalized_source = _normalized(source, field="fact source", limit=80)
        if normalized_source not in _FACT_SOURCES:
            raise ValueError("fact source is not verified")
        self._actual_facts[normalized_key] = EvidenceClaim(
            value=value,
            evidence_digest=_digest(evidence_digest),
            source=normalized_source,
            confidence=_confidence(confidence),
        )

    def user_believes(
        self,
        key: str,
        value: Any,
        *,
        evidence_digest: str,
        source: str = "observed_task_state",
        confidence: float = 0.8,
    ) -> bool:
        return bool(
            self._canonical.record_belief_hypothesis(
                self.person,
                key=key,
                value=value,
                confidence=confidence,
                evidence_digest=evidence_digest,
                source=source,
            )
        )

    def divergence(self, key: str) -> dict[str, Any] | None:
        normalized_key = _normalized(key, field="fact key", limit=100)
        actual = self._actual_facts.get(normalized_key)
        if actual is None:
            return None
        hypotheses = self._canonical.get_belief_hypotheses(self.person)
        believed = hypotheses.get(normalized_key)
        if believed is None:
            return {
                "kind": "knowledge_gap",
                "key": normalized_key,
                "aura_value": actual.value,
                "actual_evidence_digest": actual.evidence_digest,
                "confidence": actual.confidence,
                "hypothesis": True,
            }
        if believed.get("value") == actual.value:
            return None
        return {
            "kind": "false_belief",
            "key": normalized_key,
            "aura_value": actual.value,
            "user_value": believed.get("value"),
            "actual_evidence_digest": actual.evidence_digest,
            "belief_evidence_digest": believed.get("evidence_digest"),
            "confidence": min(
                actual.confidence,
                float(believed.get("confidence") or 0.0),
            ),
            "hypothesis": True,
        }


class PerspectiveCorrectionEngine:
    """Compatibility facade without an independent person or trust model."""

    def __init__(
        self,
        *,
        person: str,
        canonical: CanonicalToM | None = None,
    ) -> None:
        self.person = _normalized(person, field="person", limit=160)
        self.canonical = canonical or CanonicalToM()
        self.simulator = PerspectiveSimulator(
            person=self.person,
            canonical=self.canonical,
        )

    def explanation_strategy(self, key: str) -> str:
        divergence = self.simulator.divergence(key)
        if not divergence:
            return "confirm_shared_context"
        if divergence["kind"] == "false_belief":
            return "respectfully_correct_false_belief"
        if divergence["kind"] == "knowledge_gap":
            return "explain_from_first_principles"
        return "collaborative_clarification"

    def record_correction(
        self,
        *,
        key: str,
        correct_value: Any,
        evidence_digest: str,
        source: str = "authorized_operator_correction",
        confidence: float = 1.0,
    ) -> None:
        """Correct verified task state without mutating relational trust."""
        self.simulator.aura_knows(
            key,
            correct_value,
            evidence_digest=evidence_digest,
            source=source,
            confidence=confidence,
        )


# Compatibility import for the old module path. This is an adapter, not an owner.
TheoryOfMindEngine = PerspectiveCorrectionEngine

__all__ = [
    "EvidenceClaim",
    "PerspectiveCorrectionEngine",
    "PerspectiveSimulator",
    "TheoryOfMindEngine",
]
