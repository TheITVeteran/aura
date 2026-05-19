"""Route degradation records into resilience pressure and repair dispatch.

``record_degradation`` is the visibility boundary. This module makes it causal:
degraded records now affect the body/resilience model, and repeated or critical
incidents can enter the existing self-modification error pipeline with cooldowns.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Resilience.DegradationRepair")


ServiceGetter = Callable[[str], Any | None]


@dataclass
class DegradationRepairAction:
    subsystem: str
    severity: str
    incident_id: str = ""
    routed_at: float = field(default_factory=time.time)
    resilience_state: str = "unavailable"
    self_modification_status: str = "not_requested"
    self_modification_dispatched: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _service_get(name: str) -> Any | None:
    try:
        from core.container import ServiceContainer

        return ServiceContainer.get(name, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


class DegradationRepairRouter:
    """Connect degradation records to resilience and repair systems."""

    SELF_MODIFICATION_COOLDOWN_S = 300.0
    INCIDENT_REPEAT_THRESHOLD = 5

    def __init__(
        self,
        *,
        service_getter: ServiceGetter | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        self._service_getter = service_getter or _service_get
        self._cooldown_seconds = (
            float(cooldown_seconds)
            if cooldown_seconds is not None
            else self.SELF_MODIFICATION_COOLDOWN_S
        )
        self._last_self_modification_dispatch: dict[str, float] = {}

    def route(
        self,
        *,
        record: Any,
        error: BaseException,
        incident: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> DegradationRepairAction:
        extra = dict(extra or {})
        action = DegradationRepairAction(
            subsystem=str(getattr(record, "subsystem", "unknown")),
            severity=str(getattr(record, "severity", "degraded")),
            incident_id=str(getattr(incident, "incident_id", "") or ""),
        )
        self._route_to_resilience(record, action)
        self._route_to_self_modification(record, error, incident, extra, action)
        return action

    def _get_service(self, name: str) -> Any | None:
        try:
            return self._service_getter(name)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service lookup failed for %s: %s", name, exc)
            return None

    def _route_to_resilience(self, record: Any, action: DegradationRepairAction) -> None:
        resilience = self._get_service("resilience_engine") or self._get_service("soma")
        if resilience is None or not hasattr(resilience, "record_failure"):
            action.notes.append("resilience_engine_unavailable")
            return

        severity = str(getattr(record, "severity", "degraded"))
        signal = 0.95 if severity == "critical" else 0.55
        stakes = 0.9 if severity == "critical" else 0.6
        try:
            state = resilience.record_failure(
                domain=f"degradation:{getattr(record, 'subsystem', 'unknown')}",
                severity=signal,
                stakes=stakes,
            )
            action.resilience_state = str(getattr(state, "value", state))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.notes.append(f"resilience_route_failed:{type(exc).__name__}")
            logger.debug("Resilience routing failed: %s", exc)

    def _route_to_self_modification(
        self,
        record: Any,
        error: BaseException,
        incident: Any | None,
        extra: dict[str, Any],
        action: DegradationRepairAction,
    ) -> None:
        if not self._should_dispatch_self_modification(record, incident, extra):
            action.self_modification_status = "not_eligible"
            return

        key = f"{getattr(record, 'subsystem', 'unknown')}:{getattr(record, 'error_type', '')}"
        now = time.time()
        last = self._last_self_modification_dispatch.get(key, 0.0)
        if now - last < self._cooldown_seconds:
            action.self_modification_status = "cooldown"
            action.notes.append(f"cooldown_remaining_s:{round(self._cooldown_seconds - (now - last), 1)}")
            return

        engine = self._get_service("self_modification_engine")
        if engine is None or not hasattr(engine, "on_error"):
            action.self_modification_status = "engine_unavailable"
            return

        context = {
            "subsystem": getattr(record, "subsystem", "unknown"),
            "severity": getattr(record, "severity", "degraded"),
            "error_type": getattr(record, "error_type", type(error).__qualname__),
            "error_message": getattr(record, "error_message", str(error)),
            "incident_id": getattr(incident, "incident_id", ""),
            "incident_occurrence_count": int(getattr(incident, "occurrence_count", 0) or 0),
            "degradation_action": getattr(record, "action", ""),
            "extra": extra,
        }
        repair_error = error if isinstance(error, Exception) else RuntimeError(context["error_message"])
        try:
            engine.on_error(
                repair_error,
                context,
                skill_name=str(getattr(record, "subsystem", "degradation")),
                goal=f"Repair degradation in {getattr(record, 'subsystem', 'unknown')}",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            action.self_modification_status = f"dispatch_failed:{type(exc).__name__}"
            logger.debug("Self-modification routing failed: %s", exc)
            return

        self._last_self_modification_dispatch[key] = now
        action.self_modification_dispatched = True
        action.self_modification_status = "dispatched"

    def _should_dispatch_self_modification(
        self,
        record: Any,
        incident: Any | None,
        extra: dict[str, Any],
    ) -> bool:
        if bool(extra.get("repair_requested")):
            return True
        if str(getattr(record, "severity", "")) == "critical":
            return True
        occurrence_count = int(getattr(incident, "occurrence_count", 0) or 0)
        return occurrence_count >= self.INCIDENT_REPEAT_THRESHOLD


_router: DegradationRepairRouter | None = None


def get_degradation_repair_router() -> DegradationRepairRouter:
    global _router
    if _router is None:
        _router = DegradationRepairRouter()
    return _router


def set_degradation_repair_router_for_tests(router: DegradationRepairRouter | None) -> None:
    global _router
    _router = router
