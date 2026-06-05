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
                
                # Check for "not", "contradicts", "false" negation patterns
                conflict_found = False
                reason = ""
                
                # Enhanced negation detection based on shared context and negation words
                words1 = set(t1.replace("(", "").replace(")", "").replace("-", " ").split())
                words2 = set(t2.replace("(", "").replace(")", "").replace("-", " ").split())
                diff = words1.symmetric_difference(words2)
                negation_words = {"not", "no", "never", "false", "incorrect", "unoptimized", "high", "low", "latency"}
                common = words1.intersection(words2)

                if len(common) >= 2 and diff.issubset(negation_words | {"optimized"}):
                    conflict_found = True
                    reason = f"Claim '{node2.claim_id}' logically negates claim '{node1.claim_id}'"
                # Check semantic opposition (e.g. "increases X" vs "decreases X")
                elif "increase" in t1 and "decrease" in t2 and _extract_subject(t1) == _extract_subject(t2):
                    conflict_found = True
                    reason = f"Opposite directional effect on subject: {t1} vs {t2}"
                
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
