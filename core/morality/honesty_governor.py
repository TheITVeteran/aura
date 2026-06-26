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
from typing import Any

from core.morality.deception_guard import DeceptionGuard

logger = logging.getLogger("Morality.HonestyGovernor")


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

    def get_status(self) -> dict[str, Any]:
        return {"passes": self._passes, "caveated": self._caveated, "healthy": True}


_INSTANCE: HonestyGovernor | None = None


def get_honesty_governor() -> HonestyGovernor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HonestyGovernor()
    return _INSTANCE


def register_honesty_governor(orchestrator: Any = None) -> HonestyGovernor:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.DATA, default=None) or get_honesty_governor()
    ServiceContainer.register_instance(ServiceNames.DATA, inst, required=False)
    ServiceContainer.register_instance("data", inst, required=False)
    return inst


__all__ = ["HonestyGovernor", "get_honesty_governor", "register_honesty_governor"]
