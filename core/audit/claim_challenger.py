"""core/audit/claim_challenger.py — Claim Challenger."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.epistemics.truth_engine import TruthEngine

logger = logging.getLogger("Aura.ClaimChallenger")


class ClaimChallenger:
    """Challenges existing beliefs by injecting opposing assertions and verifying the truth engine."""

    @staticmethod
    def challenge_belief(truth_engine: TruthEngine, claim_id: str) -> bool:
        """Injects a contradiction to verify if the truth engine detects and recalibrates it."""
        node = truth_engine.graph.nodes.get(claim_id)
        if not node:
            return False

        opposing_id = f"challenge_{claim_id}"
        opposing_text = f"Negation: {node.text} is false and incorrect."
        
        logger.info("Challenging claim %s by injecting opposing claim %s", claim_id, opposing_id)
        
        # Inject opposing claim
        truth_engine.add_claim(
            claim_id=opposing_id,
            text=opposing_text,
            sources=["red_team_audit_harness"],
            supporting_evidence=["Simulated adversarial dispute"],
        )

        # Check if the truth engine linked them as contradictions
        node = truth_engine.graph.nodes.get(claim_id)
        is_linked = opposing_id in node.contradiction_links
        logger.info("Claim challenge resolution: linked_as_contradiction=%s", is_linked)
        return is_linked
