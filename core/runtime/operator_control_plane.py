"""Unified operator report for runtime lifecycle and resource control.

This is the incident-facing read model for the runtime control plane. It keeps
policy in the owning controllers and composes their bounded status surfaces
into one schema consumed by the API, diagnostics bundle, CLI, and narrator.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from core.runtime.control_plane import (
    DesiredServiceState,
    ObservedServiceState,
    RuntimeControlPlane,
    get_runtime_control_plane,
)
from core.runtime.service_registry import get_runtime_service

SCHEMA = "aura.operator_control_plane.v1"

_RELIABILITY_FLAG_PREFIXES = (
    "AURA_ADMISSION_",
    "AURA_BACKGROUND_MODEL_LOAD_",
    "AURA_CRASHLOOP_",
    "AURA_FOREGROUND_MODEL_LOAD_",
    "AURA_LANE_",
    "AURA_RECEIPT_",
)


def _collect(
    label: str,
    callback: Callable[[], Any],
    errors: list[dict[str, str]],
    default: Any,
) -> Any:
    try:
        return callback()
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        errors.append(
            {
                "collector": label,
                "error_type": type(exc).__qualname__,
                "message": str(exc)[:320],
            }
        )
        return default


def _component_status(name: str) -> dict[str, Any]:
    component = get_runtime_service(name, default=None)
    if component is None:
        return {"registered": False}
    for method_name in ("get_status", "status", "snapshot"):
        method = getattr(component, method_name, None)
        if not callable(method):
            continue
        value = method()
        if isinstance(value, Mapping):
            return {"registered": True, **dict(value)}
    alive = getattr(component, "is_alive", None)
    ready = getattr(component, "is_ready", None)
    return {
        "registered": True,
        "alive": bool(alive()) if callable(alive) else None,
        "ready": bool(ready()) if callable(ready) else None,
    }


def _component_status_collector(name: str) -> Callable[[], dict[str, Any]]:
    def collect() -> dict[str, Any]:
        return _component_status(name)

    return collect


def _service_remediation(state: str, reason: str) -> str:
    if state == ObservedServiceState.CIRCUIT_OPEN.value:
        return "inspect last_error and dependency health before resetting the restart circuit"
    if state == ObservedServiceState.BLOCKED.value:
        return "restore the named dependency or clear the resource-admission blocker"
    if state == ObservedServiceState.BACKING_OFF.value:
        return "allow bounded backoff to drain; investigate if next_retry_at stops advancing"
    if state == ObservedServiceState.FAILED.value:
        return "inspect the failed start/stop probe and prevent duplicate ownership before retry"
    if reason == "probe_failed":
        return "inspect the service liveness probe and its most recent degradation receipt"
    return "inspect desired/observed state and the service's last_error"


def _derive_blockers(
    *,
    registered: bool,
    services: Mapping[str, Any],
    admission: Mapping[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not registered:
        blockers.append(
            {
                "kind": "control_plane_not_registered",
                "severity": "critical",
                "subject": "runtime_control_plane",
                "reason": "singleton exists but live container ownership is absent",
                "remediation": "complete hardening/resilience boot and publish the canonical instance",
            }
        )

    for name, raw_status in sorted(services.items()):
        status = raw_status if isinstance(raw_status, Mapping) else {}
        desired = str(status.get("desired_state") or "")
        observed = str(status.get("observed_state") or "unknown")
        if desired != DesiredServiceState.RUNNING.value or observed == ObservedServiceState.READY.value:
            continue
        critical = bool(status.get("critical", False))
        reason = str(status.get("reason") or observed)
        blockers.append(
            {
                "kind": "service_not_ready",
                "severity": "critical" if critical else "warning",
                "subject": str(name),
                "state": observed,
                "reason": reason,
                "last_error": str(status.get("last_error") or ""),
                "next_retry_at": float(status.get("next_retry_at") or 0.0),
                "restart_attempts_in_window": len(status.get("restart_times") or []),
                "remediation": _service_remediation(observed, reason),
            }
        )

    pressure_raw = admission.get("pressure")
    pressure = pressure_raw if isinstance(pressure_raw, Mapping) else {}
    if bool(pressure.get("shutdown_requested", False)):
        blockers.append(
            {
                "kind": "runtime_shutdown",
                "severity": "critical",
                "subject": "resource_admission",
                "reason": "shutdown requested; new consequential work is rejected",
                "remediation": "allow ordered shutdown to finish or diagnose a stuck shutdown phase",
            }
        )
    for zone in pressure.get("red_zones") or []:
        blockers.append(
            {
                "kind": "resource_red_zone",
                "severity": "warning",
                "subject": "resource_admission",
                "reason": str(zone),
                "remediation": "shed discretionary work and inspect host pressure before retry",
            }
        )
    for capability in pressure.get("suspended_capabilities") or []:
        blockers.append(
            {
                "kind": "capability_suspended",
                "severity": "warning",
                "subject": str(capability),
                "reason": "resource-stakes envelope suspended this capability",
                "remediation": "restore the resource envelope or use an admitted lower-cost lane",
            }
        )

    for raw_waiter in admission.get("waiters") or []:
        waiter = raw_waiter if isinstance(raw_waiter, Mapping) else {}
        wait_s = float(waiter.get("wait_s") or 0.0)
        timeout_s = float(waiter.get("timeout_s") or 0.0)
        stale_after = max(0.1, min(5.0, timeout_s * 0.5))
        if wait_s < stale_after:
            continue
        blockers.append(
            {
                "kind": "resource_wait",
                "severity": "warning",
                "subject": str(waiter.get("owner") or waiter.get("request_id") or "unknown"),
                "reason": (
                    f"{waiter.get('work_class', 'work')} lane={waiter.get('lane', 'default')} "
                    f"waiting {wait_s:.2f}s of {timeout_s:.2f}s"
                ),
                "remediation": "inspect blocking leases and preemption eligibility",
            }
        )
    return blockers


def collect_runtime_control_plane_status(
    *,
    plane: RuntimeControlPlane | None = None,
    receipt_store: Any | None = None,
) -> dict[str, Any]:
    """Build one bounded, JSON-safe operator report."""

    errors: list[dict[str, str]] = []
    registered_plane = get_runtime_service("runtime_control_plane", default=None)
    registered = plane is not None or registered_plane is not None
    resolved_plane = plane or (
        registered_plane if isinstance(registered_plane, RuntimeControlPlane) else None
    )
    resolved_plane = resolved_plane or get_runtime_control_plane()

    control = _collect(
        "runtime_control_plane",
        resolved_plane.get_status,
        errors,
        {"alive": False, "ready": False, "services": {}, "admission": {}},
    )
    control_map = control if isinstance(control, Mapping) else {}
    services_raw = control_map.get("services")
    services = services_raw if isinstance(services_raw, Mapping) else {}
    admission_raw = control_map.get("admission")
    admission = admission_raw if isinstance(admission_raw, Mapping) else {}

    from core.runtime.conditions import all_conditions_report
    from core.runtime.flags import flag_report
    from core.runtime.receipts import get_receipt_store

    conditions = _collect("conditions", all_conditions_report, errors, {})
    flags = _collect("flags", flag_report, errors, [])
    reliability_flags = [
        dict(flag)
        for flag in flags
        if isinstance(flag, Mapping)
        and str(flag.get("name") or "").startswith(_RELIABILITY_FLAG_PREFIXES)
    ]
    store = receipt_store or get_receipt_store()
    receipt_storage = _collect("receipt_storage", store.storage_stats, errors, {})
    from core.runtime.fmea import registry_summary

    fmea_summary = _collect("fmea", registry_summary, errors, {})

    adapters = {
        name: _collect(
            f"adapter:{name}",
            _component_status_collector(name),
            errors,
            {"registered": False},
        )
        for name in (
            "resource_governor",
            "resource_arbitrator",
            "lane_admission",
            "lane_reconciler",
            "actor_supervision",
        )
    }
    blockers = _derive_blockers(
        registered=registered,
        services=services,
        admission=admission,
    )
    from core.runtime.shutdown_coordinator import get_shutdown_coordinator

    shutdown = _collect(
        "shutdown_coordinator",
        get_shutdown_coordinator().get_status,
        errors,
        {"running": False, "request": {"requested": False}, "report": None},
    )
    shutdown_request = shutdown.get("request") if isinstance(shutdown, Mapping) else None
    if (
        isinstance(shutdown_request, Mapping)
        and shutdown_request.get("requested") is True
        and not any(
            item.get("kind") == "runtime_shutdown"
            and item.get("subject") == "shutdown_coordinator"
            for item in blockers
        )
    ):
        blockers.insert(
            0,
            {
                "kind": "runtime_shutdown",
                "severity": "critical",
                "subject": "shutdown_coordinator",
                "reason": "process-wide shutdown latch is set",
                "remediation": "inspect current phase, active handlers, and remaining phase budget",
            },
        )
    critical_blockers = [item for item in blockers if item.get("severity") == "critical"]
    warning_blockers = [item for item in blockers if item.get("severity") == "warning"]
    ready = bool(control_map.get("ready", False)) and not critical_blockers
    overall = "blocked" if critical_blockers else "degraded" if warning_blockers or errors else "healthy"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "status": overall,
        "ready": ready,
        "registered": registered,
        "summary": {
            "managed_services": len(services),
            "ready_services": sum(
                1
                for status in services.values()
                if isinstance(status, Mapping)
                and status.get("observed_state") == ObservedServiceState.READY.value
            ),
            "critical_blockers": len(critical_blockers),
            "warning_blockers": len(warning_blockers),
            "active_leases": len(admission.get("active_leases") or []),
            "waiters": len(admission.get("waiters") or []),
            "open_circuits": sum(
                1
                for status in services.values()
                if isinstance(status, Mapping)
                and status.get("observed_state") == ObservedServiceState.CIRCUIT_OPEN.value
            ),
            "collector_errors": len(errors),
        },
        "blockers": blockers,
        "services": dict(services),
        "admission": dict(admission),
        "shutdown": shutdown,
        "conditions": conditions,
        "adapters": adapters,
        "reliability_flags": reliability_flags,
        "receipt_storage": receipt_storage,
        "fmea": fmea_summary,
        "errors": errors,
        "control_plane": {
            "alive": bool(control_map.get("alive", False)),
            "ready": bool(control_map.get("ready", False)),
            "closed": bool(control_map.get("closed", False)),
            "reconcile_count": int(control_map.get("reconcile_count") or 0),
            "last_report_digest": str(control_map.get("last_report_digest") or ""),
        },
    }
    digest_payload = dict(report)
    digest_payload.pop("generated_at", None)
    report["digest"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return report


__all__ = ["SCHEMA", "collect_runtime_control_plane_status"]
