"""core/brain/deep_deliberation.py

Deep Deliberation  (lineage: Deep Thought — The Hitchhiker's Guide to the Galaxy)
================================================================================
"42" is the joke that lands a real lesson: the answer was useless because no one
had worked out the actual QUESTION. So for problems flagged as hard, this engine
refines the question first, then spends an extended reasoning budget on the
refined version. The refinement step is the value — most systems answer the
literal question; this one fixes the question before answering. It lives in
brain/ beside deliberation.py and reasoning_amplifier.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

logger = logging.getLogger("Aura.DeepDeliberation")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("deep_deliberation", exc, action=action, severity=severity)


@dataclass
class DeliberationResult:
    original_question: str
    refined_question: str
    answer: str
    passes: int
    used_model: bool
    timestamp: float = field(default_factory=time.time)


class DeepDeliberationEngine:
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._deliberations = 0
        logger.info("🪐 DeepDeliberationEngine initialized (Deep Thought lineage)")

    @staticmethod
    def _heuristic_refine(question: str) -> str:
        q = question.strip()
        vague = ("how do i", "what should i", "can you help", "what is the best", "fix this", "make it better")
        low = q.lower()
        if any(v in low for v in vague) or len(q.split()) < 6:
            return (
                f"{q.rstrip('?')} — specifically: what concrete outcome defines success, "
                "what constraints apply, and what is the single most important sub-question?"
            )
        return q

    async def deliberate(self, question: str, context: dict | None = None, budget: int = 2) -> DeliberationResult:
        self._deliberations += 1
        refined = self._heuristic_refine(question)
        answer = ""
        used_model = False
        passes = 0

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                refine_prompt = (
                    "Restate the user's question as the *real* question they need answered. "
                    "One sentence.\nQUESTION: " + question[:400]
                )
                refine_out = coerce_text(await asyncio.wait_for(
                    brain.think(refine_prompt, mode=ThinkingMode.FAST, origin="deep_thought", is_background=True),
                    timeout=20.0,
                ))
                if refine_out:
                    refined = refine_out.strip()[:400]
                    passes += 1
                for _ in range(max(1, budget)):
                    ans_out = coerce_text(await asyncio.wait_for(
                        brain.think(
                            "Answer thoroughly and precisely:\n" + refined,
                            mode=ThinkingMode.DEEP if hasattr(ThinkingMode, "DEEP") else ThinkingMode.FAST,
                            origin="deep_thought",
                            is_background=True,
                        ),
                        timeout=45.0,
                    ))
                    if ans_out:
                        answer = ans_out.strip()
                        passes += 1
                        used_model = True
                        break
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="returned refined-question with heuristic note after model deliberation failed")

        if not answer:
            answer = (
                "No model was available to answer, but the question has been sharpened. "
                f"Answer the refined question: {refined}"
            )
        return DeliberationResult(
            original_question=question[:300],
            refined_question=refined,
            answer=answer,
            passes=passes,
            used_model=used_model,
        )

    def get_status(self) -> dict[str, Any]:
        return {"deliberations": self._deliberations, "healthy": True}


_INSTANCE: DeepDeliberationEngine | None = None


def get_deep_deliberation(orchestrator: Any = None) -> DeepDeliberationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DeepDeliberationEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_deep_deliberation(orchestrator: Any = None) -> DeepDeliberationEngine:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.DEEP_THOUGHT, default=None) or get_deep_deliberation(orchestrator)
    ServiceContainer.register_instance(ServiceNames.DEEP_THOUGHT, inst, required=False)
    ServiceContainer.register_instance("deep_thought", inst, required=False)
    return inst


__all__ = [
    "DeepDeliberationEngine",
    "DeliberationResult",
    "get_deep_deliberation",
    "register_deep_deliberation",
]
