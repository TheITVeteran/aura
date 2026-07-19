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

from core.runtime.service_registry import get_runtime_service, register_runtime_service
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
    # True when the answer came from a Recursive Latent Cortex episode on the
    # resident model (workspace recurrence), not ordinary token generation.
    used_latent_cortex: bool = False
    timestamp: float = field(default_factory=time.time)


class DeepDeliberationEngine:
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._deliberations = 0
        logger.info("🪐 DeepDeliberationEngine initialized (Deep Thought lineage)")

    @staticmethod
    def _heuristic_refine(question: str) -> str:
        q = question.strip()
        vague = (
            "how do i",
            "what should i",
            "can you help",
            "what is the best",
            "fix this",
            "make it better",
        )
        low = q.lower()
        if any(v in low for v in vague) or len(q.split()) < 6:
            return (
                f"{q.rstrip('?')} — specifically: what concrete outcome defines success, "
                "what constraints apply, and what is the single most important sub-question?"
            )
        return q

    def refine_question(self, question: str) -> str:
        """Synchronous question refinement (no model call) for idle/background callers."""
        return self._heuristic_refine(question)

    async def deliberate(
        self,
        question: str,
        context: dict | None = None,
        budget: int = 2,
        *,
        timeout_s: float = 45.0,
        foreground_request: bool = True,
    ) -> DeliberationResult:
        if type(foreground_request) is not bool:
            raise ValueError("foreground_request must be boolean")
        self._deliberations += 1
        refined = self._heuristic_refine(question)
        answer = ""
        used_model = False
        used_latent_cortex = False
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
                refine_out = coerce_text(
                    await asyncio.wait_for(
                        brain.think(
                            refine_prompt,
                            mode=ThinkingMode.FAST,
                            origin="deep_thought",
                            is_background=not foreground_request,
                        ),
                        timeout=min(20.0, timeout_s),
                    )
                )
                if refine_out:
                    refined = refine_out.strip()[:400]
                    passes += 1
                # DEEP pass, first choice: a Recursive Latent Cortex episode
                # on the resident model — workspace recurrence buys real
                # computational depth before any token is committed. Honest
                # refusals (busy lane, disabled, no worker) fall through to
                # ordinary generation below.
                try:
                    from core.brain.latent_cortex_service import get_latent_cortex_service

                    # Compiled understanding: digest-first conceptual context
                    # for the episode — dense, provenance-carrying concept
                    # digests instead of raw retrieval, sized for the
                    # episode's bounded compaction budget. Absent or failed
                    # ⇒ the episode proceeds on the question alone.
                    episode_messages = None
                    try:
                        from core.knowledge.compiled_understanding import (
                            get_compiled_understanding,
                        )

                        understanding = await asyncio.wait_for(
                            get_compiled_understanding().understand(refined),
                            timeout=min(20.0, timeout_s),
                        )
                        compiled_context = str(
                            understanding.get("context") or ""
                        ).strip()
                        if compiled_context:
                            episode_messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "Compiled understanding (provenance-"
                                        "tracked concept digests):\n"
                                        + compiled_context
                                    ),
                                },
                                {"role": "user", "content": refined},
                            ]
                    except (ImportError, AttributeError, RuntimeError,
                            TypeError, ValueError, TimeoutError) as cu_exc:
                        _degrade(
                            cu_exc,
                            action=(
                                "ran latent episode without compiled "
                                "understanding context"
                            ),
                        )

                    latent = await asyncio.wait_for(
                        get_latent_cortex_service(self.orchestrator).deep_reason(
                            None if episode_messages else refined,
                            messages=episode_messages,
                            stakes=0.6,
                            uncertainty=0.7,
                            domain="deliberation",
                            timeout_s=min(120.0, timeout_s * 2),
                            foreground_request=foreground_request,
                        ),
                        timeout=min(150.0, timeout_s * 3),
                    )
                    if latent.get("ok") and str(latent.get("text") or "").strip():
                        answer = str(latent["text"]).strip()
                        passes += 1
                        used_model = True
                        used_latent_cortex = True
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                    _degrade(exc, action="fell back to ordinary generation after latent cortex episode failed")
                for _ in range(max(1, budget)):
                    if answer:
                        break
                    ans_out = coerce_text(
                        await asyncio.wait_for(
                            brain.think(
                                "Answer thoroughly and precisely:\n" + refined,
                                mode=(
                                    ThinkingMode.DEEP
                                    if hasattr(ThinkingMode, "DEEP")
                                    else ThinkingMode.FAST
                                ),
                                origin="deep_thought",
                                is_background=not foreground_request,
                            ),
                            timeout=timeout_s,
                        )
                    )
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
            used_latent_cortex=used_latent_cortex,
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
    from core.service_names import ServiceNames

    inst = get_runtime_service(
        ServiceNames.DEEP_THOUGHT,
        default=None,
    ) or get_deep_deliberation(orchestrator)
    register_runtime_service(
        ServiceNames.DEEP_THOUGHT,
        inst,
        required=False,
        owner="core/brain/deep_deliberation.py",
        registered_by="register_deep_deliberation",
    )
    register_runtime_service(
        "deep_thought",
        inst,
        required=False,
        owner="core/brain/deep_deliberation.py",
        registered_by="register_deep_deliberation",
    )
    return inst


__all__ = [
    "DeepDeliberationEngine",
    "DeliberationResult",
    "get_deep_deliberation",
    "register_deep_deliberation",
]
