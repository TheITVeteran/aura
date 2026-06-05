"""core/epistemics/truth_engine.py — Truth and Epistemics Engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.epistemics.claim_graph import ClaimGraph, ClaimNode
from core.epistemics.contradiction_detector import ContradictionDetector
from core.epistemics.freshness_monitor import FreshnessMonitor
from core.epistemics.confidence_calibrator import ConfidenceCalibrator
from core.epistemics.source_ranker import get_source_ranker

logger = logging.getLogger("Aura.TruthEngine")


class TruthEngine:
    """The central authority for Aura's truth checking and epistemic audits."""

    def __init__(self) -> None:
        self.graph = ClaimGraph()
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Truth and Epistemics Engine fully initialized.")

    def add_claim(
        self,
        claim_id: str,
        text: str,
        sources: List[str],
        supporting_evidence: List[str] = None,
        impact_score: float = 0.5,
    ) -> ClaimNode:
        """Accepts a new claim, links its sources, and runs an immediate calibration pass."""
        node = ClaimNode(
            claim_id=claim_id,
            text=text,
            sources=sources,
            confidence=0.5,  # initial placeholder
            supporting_evidence=supporting_evidence or [],
            impact_score=impact_score,
        )
        self.graph.add_claim(node)
        self.recalibrate()
        return node

    def recalibrate(self) -> None:
        """Executes the full pipeline: decay, conflict scan, and score calibration."""
        # 1. Age decays
        FreshnessMonitor.decay_freshness(self.graph)

        # 2. Scans for semantic logical conflict
        conflicts = ContradictionDetector.detect_conflicts(self.graph)
        for cid1, cid2, _ in conflicts:
            self.graph.link_contradiction(cid1, cid2)

        # 3. Runs math confidence calibrations
        ConfidenceCalibrator.calibrate(self.graph)

    def get_epistemic_state(self, claim_id: str) -> Dict[str, Any]:
        """Provides full cited and grounded metadata for a given claim."""
        node = self.graph.nodes.get(claim_id)
        if not node:
            return {"ok": False, "error": "claim_not_found"}
        
        return {
            "ok": True,
            "claim_id": node.claim_id,
            "text": node.text,
            "confidence": node.confidence,
            "freshness": node.freshness,
            "sources": node.sources,
            "contradiction_links": node.contradiction_links,
            "supporting_evidence": node.supporting_evidence,
            "reliability_score": node.confidence * node.freshness,
        }

    def generate_report(self) -> Dict[str, Any]:
        """Produces a breakdown of total claims, high-confidence beliefs, and active conflicts."""
        self.recalibrate()
        total_claims = len(self.graph.nodes)
        conflicting_groups = len(self.graph.contradictions)
        
        beliefs = []
        for node in self.graph.nodes.values():
            beliefs.append({
                "id": node.claim_id,
                "text": node.text,
                "confidence": node.confidence,
                "freshness": node.freshness,
            })

        return {
            "total_claims": total_claims,
            "conflicting_groups": conflicting_groups,
            "beliefs": beliefs,
        }


_engine_instance: TruthEngine | None = None


def get_truth_engine() -> TruthEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TruthEngine()
    return _engine_instance
