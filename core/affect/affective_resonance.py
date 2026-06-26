"""core/affect/affective_resonance.py

Affective Resonance  (lineage: Samantha — Her)
=============================================
Samantha's defining quality is attunement: she reads the person's emotional state
in real time and meets them there. This reads the affective tenor of an incoming
message and produces a resonance signal and a recommended tone.

Function on both sides: INTERNAL it sets an affect-resonance modifier that
colours how Aura reasons and phrases the turn; EXTERNAL it makes her response
land as attuned to the person rather than flat. It is paired with an honesty note:
attunement is genuine modelling, not performed affection — and the system says so
when asked (transparency), staying on the right side of the Her cautionary tale.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.AffectiveResonance")

_NEG = ("sad", "angry", "upset", "frustrated", "anxious", "scared", "hurt", "tired",
        "alone", "depressed", "stressed", "worried", "hate", "awful", "terrible")
_POS = ("happy", "excited", "great", "love", "grateful", "glad", "amazing",
        "wonderful", "good", "yay", "thanks", "awesome")
_HIGH_AROUSAL = ("!!!", "urgent", "now", "asap", "freaking", "omg", "help")


@dataclass
class Resonance:
    valence: float             # -1 (negative) .. 1 (positive)
    arousal: float             # 0 (calm) .. 1 (activated)
    resonance: float           # 0 .. 1 — strength of attunement signal
    recommended_tone: str
    timestamp: float = field(default_factory=time.time)


class AffectiveResonance:
    def __init__(self):
        self._reads = 0

    def attune(
        self,
        message: str,
        *,
        user_valence: float | None = None,
        user_arousal: float | None = None,
    ) -> Resonance:
        self._reads += 1
        low = (message or "").lower()

        if user_valence is None:
            neg = sum(1 for w in _NEG if w in low)
            pos = sum(1 for w in _POS if w in low)
            total = neg + pos
            valence = 0.0 if total == 0 else (pos - neg) / total
        else:
            valence = max(-1.0, min(1.0, user_valence))

        if user_arousal is None:
            arousal = min(1.0, 0.2 + 0.2 * sum(1 for m in _HIGH_AROUSAL if m in low) + 0.1 * low.count("!"))
        else:
            arousal = max(0.0, min(1.0, user_arousal))

        # Resonance is strongest when there is clear affect to meet.
        resonance = min(1.0, abs(valence) * 0.6 + arousal * 0.4)

        if valence < -0.2 and arousal > 0.5:
            tone = "steady and grounding"
        elif valence < -0.2:
            tone = "warm and supportive"
        elif valence > 0.3:
            tone = "warm and shared-enthusiasm"
        elif arousal > 0.6:
            tone = "calm and focused"
        else:
            tone = "even and present"

        return Resonance(
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            resonance=round(resonance, 3),
            recommended_tone=tone,
        )

    async def deep_attune(self, message: str, *, timeout: float = 8.0) -> Resonance:
        """Model-deepened attunement: reads emotional subtext the keyword pass can't
        (sarcasm, masked distress). Intended for the rare high-distress case where
        getting the read right matters most. Falls back to the heuristic on failure."""
        base = self.attune(message)
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        brain = resolve_brain()
        if brain is None or not hasattr(brain, "think"):
            return base
        try:
            import asyncio

            from core.brain.types import ThinkingMode

            out = coerce_text(await asyncio.wait_for(
                brain.think(
                    "In a few words: the real emotion behind this message and how the "
                    "listener should sound in reply.\nMESSAGE: " + message[:400],
                    mode=ThinkingMode.FAST, origin="samantha", is_background=True,
                ),
                timeout=timeout,
            ))
            if out:
                base.recommended_tone = out.strip()[:80]
                base.resonance = min(1.0, base.resonance + 0.1)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_engine_degradation(
                "affective_resonance", exc,
                action="returned heuristic attunement after model deepening failed",
            )
        return base

    def get_status(self) -> dict[str, Any]:
        return {"reads": self._reads, "healthy": True}


_INSTANCE: AffectiveResonance | None = None


def get_affective_resonance() -> AffectiveResonance:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AffectiveResonance()
    return _INSTANCE


def register_affective_resonance(orchestrator: Any = None) -> AffectiveResonance:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.SAMANTHA, default=None) or get_affective_resonance()
    ServiceContainer.register_instance(ServiceNames.SAMANTHA, inst, required=False)
    ServiceContainer.register_instance("samantha", inst, required=False)
    return inst


__all__ = ["AffectiveResonance", "Resonance", "get_affective_resonance", "register_affective_resonance"]
