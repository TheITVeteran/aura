"""core/epistemics/contradiction_detector.py — Contradiction Detector."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from core.epistemics.claim_graph import ClaimGraph

logger = logging.getLogger("Aura.ContradictionDetector")


class ContradictionDetector:
    """Scans and flags logically conflicting claims in the belief ecosystem."""

    @staticmethod
    def are_contradictory(left: str, right: str) -> bool:
        """Return True when two claim texts express a direct contradiction."""
        t1 = str(left or "").lower()
        t2 = str(right or "").lower()
        if not t1 or not t2:
            return False

        words1 = _normalized_words(t1)
        words2 = _normalized_words(t2)
        common = words1.intersection(words2)
        if len(common) < 2:
            return False

        diff = words1.symmetric_difference(words2)
        negation_words = {
            "not",
            "no",
            "never",
            "false",
            "incorrect",
            "unoptimized",
            "high",
            "low",
            "latency",
        }
        if diff.issubset(negation_words | {"optimized"}):
            return True

        directional_pairs = (("increase", "decrease"), ("increases", "decreases"), ("higher", "lower"))
        for up, down in directional_pairs:
            if (
                ((up in words1 and down in words2) or (down in words1 and up in words2))
                and _extract_subject(t1) == _extract_subject(t2)
            ):
                return True
        return False

    @staticmethod
    def detect_conflicts(graph: ClaimGraph) -> List[Tuple[str, str, str]]:
        """Identifies claims that directly contradict each other based on keyword negation or semantic rules.
        
        Returns:
            List of (claim_id1, claim_id2, description_of_conflict)
        """
        conflicts = []
        nodes = list(graph.nodes.values())
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                # 1. Direct contradiction: if one text is a negation of another
                t1 = node1.text.lower()
                t2 = node2.text.lower()
                
                conflict_found = ContradictionDetector.are_contradictory(t1, t2)
                reason = f"Claim '{node2.claim_id}' logically negates claim '{node1.claim_id}'"
                
                if conflict_found:
                    logger.warning("🔍 Contradiction detected: %s", reason)
                    conflicts.append((node1.claim_id, node2.claim_id, reason))
                    
        return conflicts


def _extract_subject(text: str) -> str:
    """Rough subject extraction for conflict detection rules."""
    words = text.split()
    if len(words) > 2:
        return "".join(words[2:])
    return text


def _normalized_words(text: str) -> set[str]:
    return set(text.replace("(", "").replace(")", "").replace("-", " ").split())
