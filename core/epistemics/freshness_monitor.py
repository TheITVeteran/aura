"""core/epistemics/freshness_monitor.py — Freshness Monitor."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.epistemics.claim_graph import ClaimGraph

logger = logging.getLogger("Aura.FreshnessMonitor")

# Freshness half-life default in seconds (e.g. 7 days = 604800s)
DEFAULT_HALF_LIFE_S = 604800.0


class FreshnessMonitor:
    """Monitors belief ages and applies exponential decay to freshness indices."""

    @staticmethod
    def decay_freshness(graph: ClaimGraph, half_life_s: float = DEFAULT_HALF_LIFE_S) -> None:
        """Applies time-decay to all claims in the claim graph."""
        now = time.time()
        for node in graph.nodes.values():
            elapsed = now - node.timestamp
            # Exponential decay formula: N(t) = N0 * (0.5 ^ (t / half_life))
            decay_factor = 0.5 ** (elapsed / half_life_s)
            node.freshness = max(0.01, decay_factor)
            
            if node.freshness < 0.2:
                logger.info("Claim %s is now stale (freshness=%.2f)", node.claim_id, node.freshness)
