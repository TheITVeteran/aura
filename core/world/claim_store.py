"""core/world/claim_store.py — Structured Semantic Claim Repository.

Exposes metadata, freshness indicators, contradiction matching, and confidence thresholds.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Aura.ClaimStore")


@dataclass
class SemanticClaim:
    claim_id: str
    content: str
    source: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.50
    freshness: float = 1.0  # Degrades over time (decay calculation)
    supporting_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    impact_score: float = 0.50


class ClaimStore:
    """Stores and links claims ingested from external perception connectives."""

    def __init__(self) -> None:
        self.claims: Dict[str, SemanticClaim] = {}

    def add_claim(self, claim: SemanticClaim) -> None:
        self.claims[claim.claim_id] = claim
        logger.info("Claim Ingested [%s]: %s (conf: %.2f)", claim.claim_id, claim.content[:80], claim.confidence)
        self._detect_contradictions(claim)

    def get_claim(self, claim_id: str) -> Optional[SemanticClaim]:
        return self.claims.get(claim_id)

    def list_claims(self) -> List[SemanticClaim]:
        return list(self.claims.values())

    def _detect_contradictions(self, new_claim: SemanticClaim) -> None:
        """Simple keyword-based contradiction detector stub. Establishes links."""
        for cid, claim in self.claims.items():
            if cid == new_claim.claim_id:
                continue
            # If claims reference similar objects but assert opposite predicates
            keywords = set(new_claim.content.lower().split())
            match_count = sum(1 for w in claim.content.lower().split() if w in keywords)
            
            # Simple simulation: if highly similar but containing contradictory indicators
            if match_count > 4 and ("not" in new_claim.content.lower() != ("not" in claim.content.lower())):
                new_claim.contradictions.append(cid)
                claim.contradictions.append(new_claim.claim_id)
                logger.warning("⚠️ Contradiction detected between [%s] and [%s]!", new_claim.claim_id, cid)


# Singleton
_claim_store_instance: ClaimStore | None = None


def get_claim_store() -> ClaimStore:
    global _claim_store_instance
    if _claim_store_instance is None:
        _claim_store_instance = ClaimStore()
    return _claim_store_instance
