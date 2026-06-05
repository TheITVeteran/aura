"""Runtime Health Contract — defines what MUST be alive for Aura to be considered healthy.

This module is the authoritative source of truth for:
1. Which services are CRITICAL (system halts/degrades if missing)
2. Which services are IMPORTANT (system works but impaired)
3. Which services are OPTIONAL (nice-to-have background enrichments)

The contract is enforced at boot (by StartupValidator) and at runtime
(by the health monitor). Any module can call `evaluate_health()` to get
a typed HealthVerdict with clear pass/fail semantics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.container import ServiceContainer

logger = logging.getLogger("Aura.HealthContract")

HEALTH_CONTRACT_VERSION = "runtime-health-v1"

REQUIRED_HEALTH_PROBE_GROUPS: dict[str, tuple[str, ...]] = {
    "kernel": ("kernel_interface",),
    "inference": ("inference_gate", "llm_router"),
    "memory": ("state_repository", "memory_facade"),
    "scheduler": ("scheduler",),
    "tool_governance": ("unified_will", "authority_gateway", "capability_engine"),
}


class ServiceTier(StrEnum):
    """How critical is this service to Aura's operation?"""

    CRITICAL = "critical"  # System CANNOT function without it
    IMPORTANT = "important"  # System works but user experience is degraded
    OPTIONAL = "optional"  # Background enrichment, loss is invisible to user


@dataclass(frozen=True)
class ServiceRequirement:
    """A single service that Aura depends on."""

    name: str
    container_key: str
    tier: ServiceTier
    description: str
    liveness_check: str | None = None  # Method name to call for deep health check


# ═══════════════════════════════════════════════════════════════════════
# THE CONTRACT: What must be alive?
# ═══════════════════════════════════════════════════════════════════════

RUNTIME_CONTRACT: list[ServiceRequirement] = [
    # ── CRITICAL: Without these, Aura cannot think or respond ──
    ServiceRequirement(
        "InferenceGate",
        "inference_gate",
        ServiceTier.CRITICAL,
        "Routes LLM requests to local MLX or cloud. Without it, Aura cannot generate any response.",
        liveness_check="is_inference_ready",
    ),
    ServiceRequirement(
        "LLM Router",
        "llm_router",
        ServiceTier.CRITICAL,
        "Selects model tier and provider. Without it, InferenceGate has no backend.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "State Repository",
        "state_repository",
        ServiceTier.CRITICAL,
        "Persistent state store. Without it, Aura has no memory between turns.",
        liveness_check="is_initialized",
    ),
    ServiceRequirement(
        "Memory Facade",
        "memory_facade",
        ServiceTier.CRITICAL,
        "Canonical memory gateway. Without it, Aura cannot safely read or write long-term memory.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Kernel Interface",
        "kernel_interface",
        ServiceTier.CRITICAL,
        "Bridge between orchestrator and consciousness kernel.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Scheduler",
        "scheduler",
        ServiceTier.CRITICAL,
        "Canonical runtime scheduler. Without it, maintenance, repair, and background work are unsupervised.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Unified Will",
        "unified_will",
        ServiceTier.CRITICAL,
        "Single locus of authority for consequential decisions.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Authority Gateway",
        "authority_gateway",
        ServiceTier.CRITICAL,
        "Governance gateway for tools, external I/O, memory writes, state changes, and self-modification.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Capability Engine",
        "capability_engine",
        ServiceTier.CRITICAL,
        "Capability-token and skill governance layer. Without it, tool execution cannot be considered healthy.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Output Gate",
        "output_gate",
        ServiceTier.CRITICAL,
        "Delivers responses to the user. Without it, Aura thinks but cannot speak.",
    ),
    # ── IMPORTANT: Aura works but is impaired without these ──
    ServiceRequirement(
        "Event Bus",
        "event_bus",
        ServiceTier.IMPORTANT,
        "Canonical runtime event transport. Without it, subsystems cannot reliably coordinate.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Cognitive Engine",
        "cognitive_engine",
        ServiceTier.IMPORTANT,
        "Manages cognitive state transitions and working memory.",
    ),
    ServiceRequirement(
        "Affect Engine",
        "affect_engine",
        ServiceTier.IMPORTANT,
        "Emotional state management. Without it, responses are emotionally flat.",
    ),
    ServiceRequirement(
        "Compute Orchestrator",
        "compute_orchestrator",
        ServiceTier.IMPORTANT,
        "Resource allocation and thermal pressure control. Without it, long-run survival degrades.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Database Coordinator",
        "database_coordinator",
        ServiceTier.IMPORTANT,
        "SQLite connection pool. Without it, persistent storage degrades.",
    ),
    ServiceRequirement(
        "Drive Engine",
        "drive_engine",
        ServiceTier.IMPORTANT,
        "Motivation and goal management. Without it, autonomous behavior stops.",
    ),
    ServiceRequirement(
        "Agency Core",
        "agency_core",
        ServiceTier.IMPORTANT,
        "Canonical autonomous agency pathway loop. Without it, initiative and swarm tool use degrade.",
    ),
    ServiceRequirement(
        "Lymphatic Reaper",
        "reaper",
        ServiceTier.IMPORTANT,
        "Long-run maintenance supervisor. Without it, stale processes and files accumulate.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Hypervisor",
        "hypervisor",
        ServiceTier.IMPORTANT,
        "Event-loop and memory watchdog. Without it, severe stalls can go undetected.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Event Loop Monitor",
        "event_loop_monitor",
        ServiceTier.IMPORTANT,
        "Fine-grained event-loop lag monitor. Without it, blocking regressions are harder to catch.",
        liveness_check="is_alive",
    ),
    # ── OPTIONAL: Background enrichments ──
    ServiceRequirement(
        "Mycelial Network",
        "mycelial_network",
        ServiceTier.OPTIONAL,
        "Infrastructure graph and pathway routing.",
    ),
    ServiceRequirement(
        "Voice Engine",
        "voice_engine",
        ServiceTier.OPTIONAL,
        "Speech-to-text and text-to-speech capabilities.",
    ),
    ServiceRequirement(
        "Liquid Substrate",
        "liquid_substrate",
        ServiceTier.OPTIONAL,
        "Dynamic emotional substrate for consciousness simulation.",
    ),
    ServiceRequirement(
        "Swarm Protocol",
        "swarm_protocol",
        ServiceTier.OPTIONAL,
        "Multi-agent debate and reasoning.",
    ),
    ServiceRequirement(
        "Agent Delegator",
        "agent_delegator",
        ServiceTier.OPTIONAL,
        "Coordinates parallel task execution and specialized agents.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Stability Guardian",
        "stability_guardian",
        ServiceTier.OPTIONAL,
        "Health monitoring and auto-recovery.",
    ),
    ServiceRequirement(
        "Metrics Exporter",
        "metrics_exporter",
        ServiceTier.OPTIONAL,
        "Prometheus metrics endpoint.",
    ),
]


class HealthLevel(StrEnum):
    """Overall system health classification."""

    HEALTHY = "healthy"  # All critical + important services alive
    DEGRADED = "degraded"  # All critical alive, some important missing
    CRITICAL = "critical"  # Some critical services missing
    DEAD = "dead"  # Cannot function at all


@dataclass
class ServiceStatus:
    """Runtime status of a single service."""

    requirement: ServiceRequirement
    present: bool
    liveness_ok: bool | None = None  # None = no liveness check defined
    error: str | None = None


@dataclass
class HealthVerdict:
    """Result of a health evaluation."""

    level: HealthLevel
    services: list[ServiceStatus]
    timestamp: float = field(default_factory=time.time)

    @property
    def is_operational(self) -> bool:
        """Can Aura function at all?"""
        return self.level in (HealthLevel.HEALTHY, HealthLevel.DEGRADED)

    @property
    def critical_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.CRITICAL
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def important_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.IMPORTANT
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def optional_failures(self) -> list[ServiceStatus]:
        return [
            s
            for s in self.services
            if s.requirement.tier == ServiceTier.OPTIONAL
            and (not s.present or s.liveness_ok is False)
        ]

    @property
    def status_code(self) -> int:
        services = {
            status.requirement.container_key: _service_status_payload(status)
            for status in self.services
        }
        required_probes = _required_probe_status_from_services(services)
        return 200 if self.is_operational and required_probe_groups_pass(required_probes) else 503

    def summary(self) -> str:
        lines = [f"Health: {self.level.value.upper()}"]
        for s in self.services:
            icon = "✓" if s.present and s.liveness_ok is not False else "✗"
            tier = s.requirement.tier.value[0].upper()
            lines.append(
                f"  [{icon}] [{tier}] {s.requirement.name}: "
                f"{'alive' if s.present else 'MISSING'}"
                f"{' (liveness FAIL: ' + (s.error or '') + ')' if s.liveness_ok is False else ''}"
            )
        return "\n".join(lines)

    def to_report(self) -> dict[str, Any]:
        """Canonical machine-readable runtime health report."""
        services = [_service_status_payload(status) for status in self.services]
        report_services = {
            payload["container_key"]: payload
            for payload in services
            if isinstance(payload.get("container_key"), str)
        }
        tier_summary = {
            tier.value: _tier_summary(self.services, tier)
            for tier in (ServiceTier.CRITICAL, ServiceTier.IMPORTANT, ServiceTier.OPTIONAL)
        }
        required_probes = _required_probe_status_from_services(report_services)
        required_probe_ok = required_probe_groups_pass(required_probes)
        probe_blockers = required_probe_blockers(required_probes)
        healthy = self.level == HealthLevel.HEALTHY and required_probe_ok
        operational = self.is_operational and required_probe_ok
        return {
            "contract_version": HEALTH_CONTRACT_VERSION,
            "status": self.level.value,
            "healthy": healthy,
            "operational": operational,
            "status_code": 200 if operational else 503,
            "timestamp_unix": self.timestamp,
            "required_probes": required_probes,
            "probe_blockers": probe_blockers,
            "tier_summary": tier_summary,
            "failures": {
                "critical": [_service_status_payload(status) for status in self.critical_failures],
                "important": [
                    _service_status_payload(status) for status in self.important_failures
                ],
                "optional": [_service_status_payload(status) for status in self.optional_failures],
            },
            "services": services,
        }


def _service_probe_ok(service: dict[str, Any] | None) -> bool:
    if not isinstance(service, dict):
        return False
    if not bool(service.get("present", False)):
        return False
    return str(service.get("liveness", "") or "") == "ok"


def _required_probe_status_from_services(
    services_by_key: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items():
        component_status = {
            key: _service_probe_ok(services_by_key.get(key))
            for key in keys
        }
        probes[group] = {
            "ok": all(component_status.values()),
            "components": component_status,
        }
    probes["all_passed"] = all(
        bool(value.get("ok", False))
        for value in probes.values()
        if isinstance(value, dict)
    )
    return probes


def required_probe_groups_pass(required_probes: Any) -> bool:
    """Return True only when every canonical readiness group explicitly passes.

    This is stricter than trusting ``all_passed`` because heartbeat consumers
    must fail closed on malformed, partial, or transport-only payloads.
    """
    if not isinstance(required_probes, dict):
        return False
    if not bool(required_probes.get("all_passed", False)):
        return False
    for group_name, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        group = required_probes.get(group_name)
        if not isinstance(group, dict) or not bool(group.get("ok", False)):
            return False
        components = group.get("components")
        if not isinstance(components, dict):
            return False
        for component in expected_components:
            if components.get(component) is not True:
                return False
    return True


def required_probe_blockers(required_probes: Any) -> list[str]:
    """Return canonical blockers for malformed or failing required probes."""

    if not isinstance(required_probes, dict):
        return ["runtime_required_probes"]

    blockers: list[str] = []
    if not required_probe_groups_pass(required_probes):
        blockers.append("runtime_required_probes")

    for group_name, expected_components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        group = required_probes.get(group_name)
        if not isinstance(group, dict):
            blockers.append(f"probe:{group_name}")
            continue
        if not bool(group.get("ok", False)):
            blockers.append(f"probe:{group_name}")
            continue
        components = group.get("components")
        if not isinstance(components, dict):
            blockers.append(f"probe:{group_name}")
            continue
        if any(components.get(component) is not True for component in expected_components):
            blockers.append(f"probe:{group_name}")

    return list(dict.fromkeys(blockers))


def required_probe_status(report: dict[str, Any]) -> dict[str, Any]:
    """Return canonical high-level readiness probes from a health report.

    A heartbeat is allowed to claim healthy only when every group here passes:
    kernel, inference, memory, scheduler, and tool governance.
    """
    services = report.get("services", []) if isinstance(report, dict) else []
    services_by_key = {
        str(service.get("container_key")): service
        for service in services
        if isinstance(service, dict) and service.get("container_key")
    }
    return _required_probe_status_from_services(services_by_key)


def _service_status_payload(status: ServiceStatus) -> dict[str, Any]:
    requirement = status.requirement
    liveness = "not_configured"
    if status.liveness_ok is True:
        liveness = "ok"
    elif status.liveness_ok is False:
        liveness = "failed"
    return {
        "name": requirement.name,
        "container_key": requirement.container_key,
        "tier": requirement.tier.value,
        "description": requirement.description,
        "present": status.present,
        "liveness": liveness,
        "liveness_check": requirement.liveness_check,
        "error": status.error,
    }


def _tier_summary(services: list[ServiceStatus], tier: ServiceTier) -> dict[str, int]:
    tier_services = [status for status in services if status.requirement.tier == tier]
    failed = [
        status for status in tier_services if not status.present or status.liveness_ok is False
    ]
    liveness_failed = [status for status in tier_services if status.liveness_ok is False]
    missing = [status for status in tier_services if not status.present]
    return {
        "total": len(tier_services),
        "present": len(tier_services) - len(missing),
        "missing": len(missing),
        "liveness_failed": len(liveness_failed),
        "failed": len(failed),
    }


def evaluate_health() -> HealthVerdict:
    """Evaluate the runtime health contract against the live ServiceContainer.

    This is safe to call from any context — it never throws.
    """
    statuses: list[ServiceStatus] = []

    for req in RUNTIME_CONTRACT:
        try:
            svc = ServiceContainer.get(req.container_key, default=None)
            present = svc is not None

            liveness_ok = None
            error = None
            if present and req.liveness_check:
                try:
                    check_fn = getattr(svc, req.liveness_check, None)
                    if callable(check_fn):
                        result = check_fn()
                        liveness_ok = bool(result)
                        if not liveness_ok:
                            error = f"{req.liveness_check}() returned False"
                    else:
                        liveness_ok = False
                        error = f"missing liveness check: {req.liveness_check}()"
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    liveness_ok = False
                    error = str(exc)

            statuses.append(
                ServiceStatus(
                    requirement=req,
                    present=present,
                    liveness_ok=liveness_ok,
                    error=error,
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ServiceStatus(
                    requirement=req,
                    present=False,
                    liveness_ok=False,
                    error=str(exc),
                )
            )

    # Classify
    critical_alive = all(
        s.present and s.liveness_ok is not False
        for s in statuses
        if s.requirement.tier == ServiceTier.CRITICAL
    )
    important_alive = all(
        s.present and s.liveness_ok is not False
        for s in statuses if s.requirement.tier == ServiceTier.IMPORTANT
    )

    if critical_alive and important_alive:
        level = HealthLevel.HEALTHY
    elif critical_alive:
        level = HealthLevel.DEGRADED
    elif any(s.present for s in statuses if s.requirement.tier == ServiceTier.CRITICAL):
        level = HealthLevel.CRITICAL
    else:
        level = HealthLevel.DEAD

    return HealthVerdict(level=level, services=statuses)


def runtime_health_report() -> dict[str, Any]:
    """Return Aura's canonical runtime health contract report."""
    return evaluate_health().to_report()


def log_health_report() -> HealthVerdict:
    """Evaluate and log the health report. Returns the verdict."""
    verdict = evaluate_health()
    for line in verdict.summary().split("\n"):
        if verdict.level == HealthLevel.HEALTHY:
            logger.info(line)
        elif verdict.level == HealthLevel.DEGRADED:
            logger.warning(line)
        else:
            logger.critical(line)
    return verdict
