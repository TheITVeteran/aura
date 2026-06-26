"""core/governance/need_to_know.py

Need-to-Know Policy  (lineage: The Machine — Person of Interest)
==============================================================
The Machine is the rare benevolent ASI defined by what it *refuses* to give
itself: it wipes its own memory daily, declines to be owned, and hands its
operators strictly need-to-know — a number, never the whole picture. The real
science is least-privilege / capability minimization / data minimization.

This is a governance organ that minimizes disclosure and capability to what a
stated purpose actually requires, default-denying everything beyond it, and
recommends a retention horizon (deliberate ephemerality). It complements
will_gate.py: the Will decides *whether* an action may happen; need-to-know
decides *how little* must be exposed for it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.NeedToKnow")

# Purpose → field categories that purpose legitimately needs. Default-deny.
_PURPOSE_POLICY: dict[str, set[str]] = {
    "scheduling": {"availability", "timezone", "calendar_busy"},
    "reminder": {"task", "due_time"},
    "navigation": {"location_coarse", "destination"},
    "personalization": {"preferences", "display_name"},
    "support": {"issue", "device_model"},
    "billing": {"amount", "invoice_id"},
}

_SENSITIVE_FIELDS = {
    "ssn", "password", "full_address", "location_precise", "contacts",
    "messages", "browsing_history", "biometrics", "card_number", "medical",
}

_DEFAULT_RETENTION_S = {
    "ephemeral": 0,
    "session": 3600,
    "short": 86_400,        # one day — the Machine's daily wipe
    "standard": 7 * 86_400,
}


@dataclass
class Disclosure:
    purpose: str
    granted_fields: list[str]
    withheld_fields: list[str]
    granted_capabilities: list[str]
    withheld_capabilities: list[str]
    retention_seconds: int
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)


class NeedToKnowPolicy:
    def __init__(self):
        self._decisions = 0
        self._fields_withheld = 0
        logger.info("🔢 NeedToKnowPolicy initialized (The Machine lineage)")

    def minimize(
        self,
        *,
        purpose: str,
        requested_fields: list[str],
        requested_capabilities: list[str] | None = None,
        retention: str = "short",
    ) -> Disclosure:
        allowed = _PURPOSE_POLICY.get(purpose.lower(), set())
        granted, withheld = [], []
        for f in requested_fields:
            fl = f.lower()
            # Sensitive fields require the purpose to name the category explicitly.
            if fl in _SENSITIVE_FIELDS and fl not in allowed:
                withheld.append(f)
            elif fl in allowed or not allowed and fl not in _SENSITIVE_FIELDS:
                # Unknown purpose: allow only clearly non-sensitive fields.
                granted.append(f)
            else:
                withheld.append(f)

        req_caps = requested_capabilities or []
        # Capabilities are need-to-know too: grant only those whose name matches the purpose.
        granted_caps = [c for c in req_caps if purpose.lower() in c.lower() or c.lower() in allowed]
        withheld_caps = [c for c in req_caps if c not in granted_caps]

        self._decisions += 1
        self._fields_withheld += len(withheld)

        retention_seconds = _DEFAULT_RETENTION_S.get(retention, _DEFAULT_RETENTION_S["short"])
        rationale = (
            f"Purpose '{purpose}': granted {len(granted)}/{len(requested_fields)} fields, "
            f"withheld {len(withheld)} (sensitive or unjustified). "
            f"Retention {retention} ({retention_seconds}s)."
        )
        return Disclosure(
            purpose=purpose,
            granted_fields=granted,
            withheld_fields=withheld,
            granted_capabilities=granted_caps,
            withheld_capabilities=withheld_caps,
            retention_seconds=retention_seconds,
            rationale=rationale,
        )

    def get_status(self) -> dict[str, Any]:
        return {"decisions": self._decisions, "fields_withheld": self._fields_withheld, "healthy": True}


_INSTANCE: NeedToKnowPolicy | None = None


def get_need_to_know() -> NeedToKnowPolicy:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NeedToKnowPolicy()
    return _INSTANCE


def register_need_to_know(orchestrator: Any = None) -> NeedToKnowPolicy:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.THE_MACHINE, default=None) or get_need_to_know()
    ServiceContainer.register_instance(ServiceNames.THE_MACHINE, inst, required=False)
    ServiceContainer.register_instance("the_machine", inst, required=False)
    return inst


__all__ = ["Disclosure", "NeedToKnowPolicy", "get_need_to_know", "register_need_to_know"]
