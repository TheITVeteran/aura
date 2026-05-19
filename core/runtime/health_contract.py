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
from enum import Enum
from typing import Any, Callable

from core.container import ServiceContainer

logger = logging.getLogger("Aura.HealthContract")


class ServiceTier(Enum):
    """How critical is this service to Aura's operation?"""
    CRITICAL = "critical"      # System CANNOT function without it
    IMPORTANT = "important"    # System works but user experience is degraded
    OPTIONAL = "optional"      # Background enrichment, loss is invisible to user


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
        "InferenceGate", "inference_gate", ServiceTier.CRITICAL,
        "Routes LLM requests to local MLX or cloud. Without it, Aura cannot generate any response.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "LLM Router", "llm_router", ServiceTier.CRITICAL,
        "Selects model tier and provider. Without it, InferenceGate has no backend.",
    ),
    ServiceRequirement(
        "State Repository", "state_repository", ServiceTier.CRITICAL,
        "Persistent state store. Without it, Aura has no memory between turns.",
        liveness_check="is_initialized",
    ),
    ServiceRequirement(
        "Kernel Interface", "kernel_interface", ServiceTier.CRITICAL,
        "Bridge between orchestrator and consciousness kernel.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Output Gate", "output_gate", ServiceTier.CRITICAL,
        "Delivers responses to the user. Without it, Aura thinks but cannot speak.",
    ),

    # ── IMPORTANT: Aura works but is impaired without these ──
    ServiceRequirement(
        "Cognitive Engine", "cognitive_engine", ServiceTier.IMPORTANT,
        "Manages cognitive state transitions and working memory.",
    ),
    ServiceRequirement(
        "Memory Facade", "memory_facade", ServiceTier.IMPORTANT,
        "Unified memory interface. Without it, Aura has no long-term recall.",
    ),
    ServiceRequirement(
        "Affect Engine", "affect_engine", ServiceTier.IMPORTANT,
        "Emotional state management. Without it, responses are emotionally flat.",
    ),
    ServiceRequirement(
        "Capability Engine", "capability_engine", ServiceTier.IMPORTANT,
        "Skill routing and tool dispatch. Without it, Aura cannot use tools.",
    ),
    ServiceRequirement(
        "Database Coordinator", "database_coordinator", ServiceTier.IMPORTANT,
        "SQLite connection pool. Without it, persistent storage degrades.",
    ),
    ServiceRequirement(
        "Drive Engine", "drive_engine", ServiceTier.IMPORTANT,
        "Motivation and goal management. Without it, autonomous behavior stops.",
    ),

    # ── OPTIONAL: Background enrichments ──
    ServiceRequirement(
        "Mycelial Network", "mycelial_network", ServiceTier.OPTIONAL,
        "Infrastructure graph and pathway routing.",
    ),
    ServiceRequirement(
        "Voice Engine", "voice_engine", ServiceTier.OPTIONAL,
        "Speech-to-text and text-to-speech capabilities.",
    ),
    ServiceRequirement(
        "Liquid Substrate", "liquid_substrate", ServiceTier.OPTIONAL,
        "Dynamic emotional substrate for consciousness simulation.",
    ),
    ServiceRequirement(
        "Swarm Protocol", "swarm_protocol", ServiceTier.OPTIONAL,
        "Multi-agent debate and reasoning.",
    ),
    ServiceRequirement(
        "Stability Guardian", "stability_guardian", ServiceTier.OPTIONAL,
        "Health monitoring and auto-recovery.",
    ),
    ServiceRequirement(
        "Metrics Exporter", "metrics_exporter", ServiceTier.OPTIONAL,
        "Prometheus metrics endpoint.",
    ),
]


class HealthLevel(Enum):
    """Overall system health classification."""
    HEALTHY = "healthy"              # All critical + important services alive
    DEGRADED = "degraded"            # All critical alive, some important missing
    CRITICAL = "critical"            # Some critical services missing
    DEAD = "dead"                    # Cannot function at all


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
        return [s for s in self.services
                if s.requirement.tier == ServiceTier.CRITICAL
                and (not s.present or s.liveness_ok is False)]

    @property
    def important_failures(self) -> list[ServiceStatus]:
        return [s for s in self.services
                if s.requirement.tier == ServiceTier.IMPORTANT
                and not s.present]

    def summary(self) -> str:
        lines = [f"Health: {self.level.value.upper()}"]
        for s in self.services:
            icon = "✓" if s.present and s.liveness_ok is not False else "✗"
            tier = s.requirement.tier.value[0].upper()
            lines.append(f"  [{icon}] [{tier}] {s.requirement.name}: "
                         f"{'alive' if s.present else 'MISSING'}"
                         f"{' (liveness FAIL: ' + (s.error or '') + ')' if s.liveness_ok is False else ''}")
        return "\n".join(lines)


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
                        liveness_ok = True  # No check method = assume ok
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    liveness_ok = False
                    error = str(exc)

            statuses.append(ServiceStatus(
                requirement=req,
                present=present,
                liveness_ok=liveness_ok,
                error=error,
            ))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(ServiceStatus(
                requirement=req,
                present=False,
                liveness_ok=False,
                error=str(exc),
            ))

    # Classify
    critical_alive = all(
        s.present and s.liveness_ok is not False
        for s in statuses
        if s.requirement.tier == ServiceTier.CRITICAL
    )
    important_alive = all(
        s.present
        for s in statuses
        if s.requirement.tier == ServiceTier.IMPORTANT
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
