"""core/lab/hypothesis_engine.py — Research Lab Hypothesis Engine.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import List

from core.container import ServiceContainer

logger = logging.getLogger("Aura.HypothesisEngine")


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    rationale: str
    verification_metric: str  # E.g. "p-value < 0.05", "accuracy > 0.85"
    timestamp: float


class HypothesisEngine:
    """Formulates target hypothesis statements based on literature gaps."""

    @staticmethod
    async def formulate_hypothesis(topic: str) -> Hypothesis:
        logger.info("🔬 Formulating hypothesis for topic: '%s'", topic)
        router = ServiceContainer.get("llm_router", default=None)
        
        statement = f"Optimizing database indices for {topic} decreases query latency by > 30%."
        rationale = "Structured indexing models decrease linear scan search space to logarithmic complexity."
        metric = "latency_delta_ratio >= 0.30"

        if router and hasattr(router, "think"):
            try:
                response = await router.think(
                    prompt=(
                        f"Formulate a falsifiable software engineering hypothesis about {topic}. "
                        f"Provide: 1. Falsifiable statement, 2. Theoretical rationale, 3. Success metric."
                    )
                )
                # Parse lines simple fallback
                lines = [line.strip() for line in response.split("\n") if line.strip()]
                if len(lines) >= 3:
                    statement = lines[0]
                    rationale = lines[1]
                    metric = lines[2]
            except Exception as e:
                logger.error("Failed to generate hypothesis: %s", e)

        h_id = "hyp_" + hashlib.sha256(statement.encode()).hexdigest()[:8]
        return Hypothesis(
            hypothesis_id=h_id,
            statement=statement,
            rationale=rationale,
            verification_metric=metric,
            timestamp=time.time()
        )
