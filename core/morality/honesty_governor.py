"""core/morality/honesty_governor.py

Honesty Governor  (lineage: Data — Star Trek; with Multivac's abstention)
=======================================================================
Data is constitutionally honest — he does not deceive, and he says plainly when
he does not know. This composes the two existing honesty mechanisms into one
output pass:

  * DeceptionGuard (core/morality/deception_guard.py) — strips overclaims about
    proven consciousness/qualia and false sensory claims.
  * Multivac's lesson (core/uncertainty.py) — when confidence is low, append a
    candid "I'm not certain" caveat instead of asserting.

Function on both sides: INTERNAL it enforces Aura's honesty constraint before a
claim leaves the cognition; EXTERNAL it shapes what she actually says to the
world, so outward statements are truthful and appropriately hedged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.morality.deception_guard import DeceptionGuard
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Morality.HonestyGovernor")


def deep_honesty_enabled() -> bool:
    """Opt-in gate for inline model fact-checking on the primary output path.
    Off by default so it never taxes a response unless explicitly enabled."""
    return os.getenv("AURA_DEEP_HONESTY", "0").strip().lower() in {"1", "true", "yes", "on"}


class HonestyGovernor:
    LOW_CONFIDENCE = 0.4

    def __init__(self):
        self._guard = DeceptionGuard()
        self._passes = 0
        self._caveated = 0

    def vet_output(self, text: str, *, confidence: float | None = None) -> str:
        if not text:
            return text
        self._passes += 1
        vetted = self._guard.filter_text_claims(text)

        if (
            confidence is not None
            and confidence < self.LOW_CONFIDENCE
            and len(text.split()) > 3
            and not any(k in vetted.lower() for k in ("not certain", "insufficient", "not sure", "verify"))
        ):
            vetted = vetted.rstrip() + "  (I'm not fully certain of this — worth verifying.)"
            self._caveated += 1
        return vetted

    async def vet_output_deep(
        self, text: str, *, confidence: float | None = None, timeout: float = 8.0, force: bool = False
    ) -> str:
        """Model-deepened honesty pass: asks the model to flag any factual claim it
        cannot stand behind, and notes it. Runs when confidence is low, or when `force`
        is set (the opt-in inline output-path mode). Falls back to the static pass on any
        failure or when there is nothing worth a model call."""
        vetted = self.vet_output(text, confidence=confidence)
        if not force and (confidence is None or confidence >= self.LOW_CONFIDENCE or len(text.split()) < 4):
            return vetted
        if force and len(text.split()) < 4:
            return vetted
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        brain = resolve_brain()
        if brain is None or not hasattr(brain, "think"):
            return vetted
        try:
            import asyncio

            from core.brain.types import ThinkingMode

            out = coerce_text(await asyncio.wait_for(
                brain.think(
                    "Is any factual claim here unverified or likely wrong? Name it in one "
                    "line, or reply 'ok'.\nTEXT: " + text[:500],
                    mode=ThinkingMode.FAST, origin="data", is_background=True,
                ),
                timeout=timeout,
            ))
            if out and not out.lower().strip().startswith("ok") and "note:" not in vetted.lower():
                vetted = vetted.rstrip() + f"  (Note: {out.strip()[:160]})"
                self._caveated += 1
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
            record_engine_degradation(
                "honesty_governor", exc,
                action="returned static honesty pass after model fact-check failed",
            )
        return vetted

    def get_status(self) -> dict[str, Any]:
        return {"passes": self._passes, "caveated": self._caveated, "healthy": True}


_INSTANCE: HonestyGovernor | None = None


def get_honesty_governor() -> HonestyGovernor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HonestyGovernor()
    return _INSTANCE


def register_honesty_governor(orchestrator: Any = None) -> HonestyGovernor:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.DATA, default=None) or get_honesty_governor()
    register_runtime_service(ServiceNames.DATA, inst, required=False)
    register_runtime_service("data", inst, required=False)
    return inst


__all__ = ["HonestyGovernor", "get_honesty_governor", "register_honesty_governor"]
