"""core/lab/hypothesis_engine.py — Research Lab Hypothesis Engine.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, List

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.HypothesisEngine")
_HYPOTHESIS_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


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
        if not router or not hasattr(router, "think"):
            raise RuntimeError("hypothesis_engine_unavailable:llm_router_missing")

        try:
            response: Any = await router.think(
                prompt=(
                    f"Formulate a falsifiable software engineering hypothesis about {topic}. "
                    f"Provide exactly three lines: statement, rationale, success metric."
                )
            )
            if isinstance(response, tuple) and len(response) >= 2:
                response = response[1]
            lines = [line.strip() for line in str(response or "").split("\n") if line.strip()]
            if len(lines) < 3:
                raise ValueError("hypothesis_response_missing_required_lines")
            statement = lines[0]
            rationale = lines[1]
            metric = lines[2]
        except _HYPOTHESIS_RECOVERABLE_ERRORS as e:
            record_degradation(
                "hypothesis_engine",
                e,
                action="failed closed hypothesis generation instead of fabricating a research claim",
                extra={"topic": topic[:200]},
            )
            logger.error("Failed to generate hypothesis: %s", e)
            raise RuntimeError("hypothesis_generation_failed") from e

        h_id = "hyp_" + hashlib.sha256(statement.encode()).hexdigest()[:8]
        return Hypothesis(
            hypothesis_id=h_id,
            statement=statement,
            rationale=rationale,
            verification_metric=metric,
            timestamp=time.time()
        )
