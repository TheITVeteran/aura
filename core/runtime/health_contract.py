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

import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.HealthContract")

HEALTH_CONTRACT_VERSION = "runtime-health-v1"

REQUIRED_HEALTH_PROBE_GROUPS: dict[str, tuple[str, ...]] = {
    "kernel": ("kernel_interface",),
    "inference": (
        "inference_gate",
        "llm_router",
        "lane_admission",
        "lane_reconciler",
    ),
    "memory": (
        "state_repository",
        "memory_facade",
        "memory_write_gateway",
        "unified_memory_pressure",
        "external_memory_sentinel",
    ),
    "scheduler": (
        "scheduler",
        "runtime_control_plane",
        "resource_admission",
        "actor_supervision",
    ),
    "tool_governance": ("unified_will", "authority_gateway", "capability_engine"),
    "workspace": ("inhibition_manager", "global_workspace"),
    "attention": ("attention_schema",),
    "live_mind": ("live_mind_runtime",),
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
        "Memory Write Gateway",
        "memory_write_gateway",
        ServiceTier.CRITICAL,
        "Canonical governed durable memory write gateway. Without it, memory writes cannot be trusted.",
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
        "Runtime Control Plane",
        "runtime_control_plane",
        ServiceTier.CRITICAL,
        "Canonical desired-state reconciler. Without it, service lifecycle and resource policy diverge.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Resource Admission",
        "resource_admission",
        ServiceTier.CRITICAL,
        "Pressure-aware lease authority for inference, evolution, model loading, and managed startup.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Lane Admission",
        "lane_admission",
        ServiceTier.CRITICAL,
        "Declared model-memory envelope. Without it, concurrent lane warmups can over-commit the host.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Lane Reconciler",
        "lane_reconciler",
        ServiceTier.CRITICAL,
        "Managed cortex convergence and crash-loop backoff. Without it, model-serving recovery can thrash indefinitely.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Actor Supervision",
        "actor_supervision",
        ServiceTier.CRITICAL,
        "Canonical multiprocessing actor monitor. Without it, crashed or stalled actors are not converged safely.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Inhibition Manager",
        "inhibition_manager",
        ServiceTier.CRITICAL,
        "Canonical workspace safety gate. Without it, candidate admission cannot be trusted.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Global Workspace",
        "global_workspace",
        ServiceTier.CRITICAL,
        "Canonical candidate admission and broadcast lane. Without it, inhibition cannot bind cognition.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Attention Schema",
        "attention_schema",
        ServiceTier.CRITICAL,
        "Canonical attentional-focus owner and rigidity gate.",
        liveness_check="is_ready",
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
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Live Mind Runtime",
        "live_mind_runtime",
        ServiceTier.CRITICAL,
        "Boot-owned causal organs and snapshot contract required for grounded live desktop speech.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "External Memory Sentinel",
        "external_memory_sentinel",
        ServiceTier.CRITICAL,
        "Out-of-process memory guard. Without it, a live desktop runaway can outpace in-process watchdogs and crash the host.",
        liveness_check="is_armed",
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
        "Ulysses Covenant",
        "ulysses_covenant",
        ServiceTier.IMPORTANT,
        "Volitional self-binding registry enforced at the Will. Without it, "
        "precommitments against known failure modes stop holding.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Cognitive Engine",
        "cognitive_engine",
        ServiceTier.IMPORTANT,
        "Manages cognitive state transitions and working memory.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Affect Engine",
        "affect_engine",
        ServiceTier.IMPORTANT,
        "Emotional state management. Without it, responses are emotionally flat.",
        liveness_check="is_ready",
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
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Drive Engine",
        "drive_engine",
        ServiceTier.IMPORTANT,
        "Motivation and goal management. Without it, autonomous behavior stops.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Agency Core",
        "agency_core",
        ServiceTier.IMPORTANT,
        "Canonical autonomous agency pathway loop. Without it, initiative and swarm tool use degrade.",
        liveness_check="is_alive",
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
    ServiceRequirement(
        "MindTick",
        "mind_tick",
        ServiceTier.IMPORTANT,
        "Canonical cognitive and organism rhythm. Without forward progress, autonomous state integration stalls.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Resource Governor",
        "resource_governor",
        ServiceTier.IMPORTANT,
        "Canonical sampler and eviction adapter feeding the runtime control plane.",
        liveness_check="is_alive",
    ),
    ServiceRequirement(
        "Resource Arbitrator",
        "resource_arbitrator",
        ServiceTier.IMPORTANT,
        "Compatibility facade ensuring legacy inference and evolution callers use canonical admission.",
        liveness_check="is_ready",
    ),
    # ── OPTIONAL: Background enrichments ──
    ServiceRequirement(
        "Whole-System Φ",
        "whole_system_phi",
        ServiceTier.OPTIONAL,
        "Integrated-information estimation over the live channel set "
        "(exact-MIP Gaussian Φ with surrogate nulls, grain discovery, and "
        "an internal PCI). Telemetry-grade; loss removes a measurement, not "
        "a capability.",
        liveness_check="is_alive",
    ),
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
        "Synaptic Plasticity",
        "synaptic_plasticity",
        ServiceTier.OPTIONAL,
        "Bounded online projection learning for generation-style modulation.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Temporal Continuity",
        "temporal_continuity",
        ServiceTier.OPTIONAL,
        "Accumulated silence and drift residue for temporal presence.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Attention Gate",
        "attention_gate",
        ServiceTier.OPTIONAL,
        "Causal context pruning for focused cognition.",
        liveness_check="is_ready",
    ),
    ServiceRequirement(
        "Somatic Qualia",
        "somatic_qualia",
        ServiceTier.OPTIONAL,
        "Non-symbolic substrate perturbation for generation controls.",
        liveness_check="is_ready",
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
    ServiceRequirement(
        "Allostasis Engine",
        "allostasis_engine",
        ServiceTier.OPTIONAL,
        "Predictive interoception: forecasts vital-sign trajectories and regulates before crises, with a calibration ledger.",
        liveness_check="is_ready",
    ),
]

UNIFIED_MEMORY_PRESSURE_REQUIREMENT = ServiceRequirement(
    "Unified Memory Pressure",
    "unified_memory_pressure",
    ServiceTier.CRITICAL,
    "Process-wide unified-memory pressure gate. Aura must not claim healthy when the live model lane risks system OOM.",
    liveness_check="get_memory_pressure_snapshot",
)

UNIFIED_RUNTIME_PRESSURE_REQUIREMENT = ServiceRequirement(
    "Unified Runtime Pressure",
    "unified_runtime_pressure",
    ServiceTier.IMPORTANT,
    (
        "Event-loop, CPU, and existential-pressure gate. Aura must not claim "
        "healthy when scheduling lag or substrate survival pressure is high."
    ),
    liveness_check="runtime_pressure_snapshot",
)


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
    duration_ms: float = 0.0


@dataclass
class HealthVerdict:
    """Result of a health evaluation."""

    level: HealthLevel
    services: list[ServiceStatus]
    timestamp: float = field(default_factory=time.time)
    evaluation_duration_ms: float = 0.0

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
            "evaluation_duration_ms": self.evaluation_duration_ms,
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
        "duration_ms": status.duration_ms,
    }


def _coerce_liveness_result(result: Any) -> tuple[bool, str | None]:
    """Accept only explicit liveness success values.

    Health probes are a runtime launch contract, not a generic truthiness check.
    A coroutine object, non-empty string, list, or arbitrary object must never
    make Aura look healthy.
    """
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        return False, "liveness check returned awaitable; sync health contract cannot count it as ready"
    if isinstance(result, bool):
        return result, None if result else "liveness check returned False"
    if isinstance(result, dict):
        for key in ("ok", "ready", "healthy", "alive", "operational"):
            if key in result:
                return bool(result.get(key) is True), None if result.get(key) is True else f"{key} was not True"
    if result is None:
        return False, "liveness check returned None"
    return False, f"unsupported liveness result type: {type(result).__name__}"


def _unified_memory_pressure_status() -> ServiceStatus:
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        snapshot = get_memory_pressure_snapshot()
        if snapshot.critical:
            return ServiceStatus(
                requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
                present=True,
                liveness_ok=False,
                error=snapshot.reason or f"memory pressure level is {snapshot.level}",
            )
        return ServiceStatus(
            requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
            present=True,
            liveness_ok=True,
            error=None,
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return ServiceStatus(
            requirement=UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
            present=True,
            liveness_ok=False,
            error=f"memory pressure probe unavailable: {exc}",
        )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _process_uptime_seconds() -> float:
    """Seconds since the runtime booted, or 0.0 if unknown.

    Reads the orchestrator's start_time from the runtime registry. Returns 0.0 when
    no orchestrator is registered (e.g. unit tests), so the boot-grace
    exemption below applies ONLY to a live runtime that is genuinely warming
    up — never to a bare-probe unit test. Used to suppress runtime-pressure
    health failures during boot/warmup, when loading the local model
    legitimately spikes event-loop lag and survival pressure (the same window
    the freeze watchdog already exempts).
    """
    try:
        orch = get_runtime_service("orchestrator", default=None)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return 0.0
    for candidate in (
        getattr(orch, "start_time", None),
        getattr(getattr(orch, "status", None), "start_time", None),
    ):
        try:
            start = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if start > 0.0:
            return max(0.0, time.time() - start)
    return 0.0


def _runtime_pressure_boot_grace_active() -> bool:
    """Return True only during an explicit boot/proof warmup grace window."""

    boot_context = any(
        str(os.environ.get(name, "") or "").strip().lower()
        not in {"", "0", "false", "no", "off"}
        for name in (
            "AURA_PROOF_RUN",
            "AURA_SAFE_BOOT_DESKTOP",
            "AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE",
        )
    )
    if not boot_context:
        return False

    boot_grace_s = _float_env("AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE_S", 180.0)
    uptime_s = _process_uptime_seconds()
    return bool(boot_grace_s > 0.0 and 0.0 < uptime_s < boot_grace_s)


def _recent_inference_degradation_blocks_runtime_pressure(record: Any) -> tuple[bool, str]:
    """Classify recent inference degradations for the runtime-pressure gate.

    The runtime health contract should fail closed for foreground/user-facing
    inference saturation, but a background Brainstem timeout must not keep the
    launched desktop stuck in "booting/degraded" after Cortex and the required
    probes are ready. The degradation remains logged and repair-routable; this
    function only decides whether it blocks the top-level readiness contract.
    """

    severity = str(getattr(record, "severity", "") or "")
    if severity not in {"critical", "degraded"}:
        return False, ""

    action = str(getattr(record, "action", "") or "")
    message = str(getattr(record, "error_message", "") or "")
    combined = f"{message} {action}".lower()

    if "generation gate saturated" in combined or "refused to stack" in combined:
        if "background" in combined and not any(
            marker in combined
            for marker in ("foreground", "user-facing", "user_facing")
        ):
            return False, "background_generation_contention"
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_saturation: {(message or action)[:120]}"

    # Known background lane timeout. In live mode this may be escalated by the
    # fail-closed service policy even when the user-facing Cortex lane is fine.
    if (
        "inference_gate_generation_timeout:brainstem:" in combined
        and "foreground" not in combined
        and "user-facing" not in combined
        and "user_facing" not in combined
    ):
        return False, "background_brainstem_timeout"

    if any(marker in combined for marker in (
        "inference_gate_generation_timeout:cortex:",
        "inference_gate_generation_timeout:solver:",
        "user-facing",
        "user_facing",
        "foreground",
        "client_returned_no_text",
    )):
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_{severity}: {(message or action)[:120]}"

    # Critical/degraded inference failures without lane context are still
    # treated as blocking because they may represent the active conversation path.
    if severity == "critical" and "brainstem" not in combined and "reflex" not in combined:
        return True, f"recent_{getattr(record, 'subsystem', 'inference')}_{severity}: {(message or action)[:120]}"

    return False, ""


def _runtime_pressure_status() -> ServiceStatus:
    """Return an important health failure when runtime pressure is too high.

    Heartbeat transport, service presence, and memory headroom are necessary
    but not sufficient for a healthy live desktop runtime. A system stuck in a
    long foreground generation can still have every service "alive"; this probe
    closes that gap by folding existential pressure and lag monitors into the
    canonical health contract.
    """
    blockers: list[str] = []
    boot_deferrable_blockers: set[str] = set()
    details: list[str] = []
    threat_threshold = _float_env("AURA_HEALTH_EXISTENTIAL_THREAT_UNHEALTHY", 0.75)
    lag_threshold = _float_env("AURA_HEALTH_EVENT_LOOP_LAG_UNHEALTHY_S", 5.0)
    recent_degradation_window_s = _float_env(
        "AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S",
        180.0,
    )
    boot_grace_active = _runtime_pressure_boot_grace_active()

    try:
        stakes = get_runtime_service("existential_stakes", default=None)
        status_getter = getattr(stakes, "get_status", None)
        if callable(status_getter):
            status = status_getter()
            if isinstance(status, dict):
                threat = float(status.get("existential_threat", 0.0) or 0.0)
                lag_threat = float(status.get("lag_threat", 0.0) or 0.0)
                memory_threat = float(status.get("memory_threat", 0.0) or 0.0)
                details.append(
                    "existential_threat="
                    f"{threat:.2f}, lag_threat={lag_threat:.2f}, memory_threat={memory_threat:.2f}"
                )
                # Steady-state memory pressure is owned by the dedicated
                # _unified_memory_pressure_status probe (calibrated via
                # get_memory_pressure_snapshot, flags only snapshot.critical)
                # and the out-of-band memory watchdog. A loaded ~20GB model on a
                # 64GB box sits at memory_threat ~0.77 while serving requests
                # fine, so folding raw existential memory pressure into THIS
                # probe double-counts it and would mark a healthy runtime
                # degraded. This probe targets the *stuck / overloaded* runtime
                # the docstring describes, which lag_threat captures. So when
                # memory is the high contributor, gate on the lag signal and let
                # the dedicated memory probe decide; otherwise use the full
                # aggregate (covers lag- and cpu-driven existential threat).
                pressure_threat = (
                    lag_threat if memory_threat >= threat_threshold else threat
                )
                if pressure_threat >= threat_threshold:
                    blocker = f"existential_threat {threat:.2f} >= {threat_threshold:.2f}"
                    blockers.append(blocker)
                    if boot_grace_active and lag_threat >= threat_threshold:
                        boot_deferrable_blockers.add(blocker)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        details.append(f"existential_stakes_unavailable:{type(exc).__name__}")

    for key in ("event_loop_monitor", "hypervisor"):
        try:
            monitor = get_runtime_service(key, default=None)
            status_getter = getattr(monitor, "get_status", None)
            if not callable(status_getter):
                continue
            status = status_getter()
            if not isinstance(status, dict):
                continue
            last_lag = float(status.get("last_lag_s", 0.0) or 0.0)
            failure_reason = str(status.get("last_failure_reason", "") or "")
            sample_fresh = status.get("sample_fresh")
            if sample_fresh is False:
                details.append(
                    f"{key}.lag_sample_stale age_s="
                    f"{float(status.get('sample_age_s', 0.0) or 0.0):.2f}"
                )
            else:
                details.append(f"{key}.last_lag_s={last_lag:.2f}")
            if failure_reason:
                blockers.append(f"{key}:{failure_reason}")
            elif sample_fresh is not False and last_lag >= lag_threshold:
                blocker = f"{key}.last_lag_s {last_lag:.2f} >= {lag_threshold:.2f}"
                blockers.append(blocker)
                if boot_grace_active:
                    boot_deferrable_blockers.add(blocker)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            details.append(f"{key}_pressure_unavailable:{type(exc).__name__}")

    try:
        from core.runtime.errors import get_degradation_tracker

        now = time.time()
        inference_subsystems = {
            "llm_health_router",
            "inference_gate",
            "mlx_client",
            "mlx_runtime",
        }
        for record in get_degradation_tracker().recent(limit=80):
            subsystem = str(getattr(record, "subsystem", "") or "")
            if subsystem not in inference_subsystems:
                continue
            age_s = now - float(getattr(record, "timestamp", 0.0) or 0.0)
            if age_s > recent_degradation_window_s:
                continue
            blocks, reason = _recent_inference_degradation_blocks_runtime_pressure(record)
            if blocks and reason:
                blockers.append(reason)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        details.append(f"degradation_pressure_unavailable:{type(exc).__name__}")

    if blockers:
        active_blockers = [
            blocker for blocker in blockers if blocker not in boot_deferrable_blockers
        ]
        if not active_blockers and boot_deferrable_blockers:
            details.append(
                "boot_grace_deferred_runtime_pressure:"
                + ",".join(sorted(boot_deferrable_blockers))
            )
            blockers = []
        else:
            blockers = active_blockers

    return ServiceStatus(
        requirement=UNIFIED_RUNTIME_PRESSURE_REQUIREMENT,
        present=True,
        liveness_ok=not blockers,
        error="; ".join(blockers or details[:3]) if blockers else None,
    )


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
    """Evaluate the runtime health contract against the live runtime service registry.

    This is safe to call from any context — it never throws.
    """
    evaluation_started = time.perf_counter()
    statuses: list[ServiceStatus] = []

    for req in RUNTIME_CONTRACT:
        probe_started = time.perf_counter()
        try:
            svc = get_runtime_service(req.container_key, default=None)
            present = svc is not None

            liveness_ok = None
            error = None
            if present and req.liveness_check:
                try:
                    check_fn = getattr(svc, req.liveness_check, None)
                    if callable(check_fn):
                        result = check_fn()
                        liveness_ok, result_error = _coerce_liveness_result(result)
                        if not liveness_ok:
                            if result_error == "liveness check returned False":
                                error = f"{req.liveness_check}() returned False"
                            else:
                                error = result_error or f"{req.liveness_check}() did not return explicit True"
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
                    duration_ms=round(
                        (time.perf_counter() - probe_started) * 1000.0,
                        3,
                    ),
                )
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            statuses.append(
                ServiceStatus(
                    requirement=req,
                    present=False,
                    liveness_ok=False,
                    error=str(exc),
                    duration_ms=round(
                        (time.perf_counter() - probe_started) * 1000.0,
                        3,
                    ),
                )
            )

    statuses.append(_unified_memory_pressure_status())
    statuses.append(_runtime_pressure_status())

    # Classify
    concrete_statuses = [
        status
        for status in statuses
        if status.requirement.container_key != UNIFIED_MEMORY_PRESSURE_REQUIREMENT.container_key
    ]
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
    elif any(s.present for s in concrete_statuses if s.requirement.tier == ServiceTier.CRITICAL):
        level = HealthLevel.CRITICAL
    else:
        level = HealthLevel.DEAD

    return HealthVerdict(
        level=level,
        services=statuses,
        evaluation_duration_ms=round(
            (time.perf_counter() - evaluation_started) * 1000.0,
            3,
        ),
    )


def _runtime_integrity_block() -> dict[str, Any]:
    """Memory the health verdict does not otherwise have.

    ``evaluate_health()`` answers "is the runtime working *now*", which is
    the right question but not the only one. A process that survived a
    lock-order violation, shed an organ under memory pressure, or hot-swapped
    code is working now *and* is no longer the process its green verdict
    describes. The kernel prints its taint on every oops for exactly this
    reason; this block is that line. It never flips the verdict — it
    attaches the caveat the verdict cannot express on its own.
    """
    block: dict[str, Any] = {}
    try:
        from core.runtime.taint import credibility_caveat, taint_compact, taint_report

        block["taint"] = taint_report()
        block["taint_compact"] = taint_compact()
        caveat = credibility_caveat()
        if caveat:
            block["credibility_caveat"] = caveat
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["taint_error"] = repr(exc)
    try:
        from core.runtime.lockdep import lockdep_report

        lock_report = lockdep_report()
        block["lockdep"] = {
            "clean": lock_report["clean"],
            "acquires_checked": lock_report["acquires_checked"],
            "splats": lock_report["splats"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["lockdep_error"] = repr(exc)
    try:
        from core.runtime.pressure_stall import psi_narrative, psi_report, saturated_resources

        block["pressure"] = psi_report()
        block["pressure_saturated"] = saturated_resources()
        block["pressure_narrative"] = psi_narrative()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["pressure_error"] = repr(exc)
    try:
        from core.knowledge.metta import metta_report
        from core.organism.model_validation import get_suite, validation_report

        validation = validation_report()
        block["self_model"] = {
            "claims": len(validation["claims"]),
            "tests": len(validation["tests"]),
            "unsupported_claims": [c["statement"] for c in get_suite().unsupported_claims()],
            "metta": {k: metta_report()[k] for k in ("rules", "reductions", "truncations")},
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["self_model_error"] = repr(exc)
    try:
        from core.fsw.assertions import assertions_report
        from core.fsw.command_dispatch import command_report
        from core.fsw.health_checker import health_checker_report
        from core.fsw.rate_groups import rate_group_report
        from core.fsw.restart_protection import restart_report
        from core.fsw.telemetry_dictionary import telemetry_report

        telemetry = telemetry_report()
        pings = health_checker_report()
        block["flight_software"] = {
            "telemetry": {
                "channels": telemetry["channels"],
                "violations": telemetry["violations"],
                "recent_events": telemetry["recent_events"],
            },
            "restart_protection": restart_report()["core_sets"],
            "rate_groups": {
                k: rate_group_report()[k] for k in ("slipping", "total_cycles", "total_slips")
            },
            "assertions": {
                "clean": assertions_report()["clean"],
                "distinct_sites": assertions_report()["distinct_sites"],
            },
            "health_pings": {
                "unresponsive": pings["unresponsive"],
                "slow": pings["slow"],
                "critical_unresponsive": pings["critical_unresponsive"],
            },
            "commands": {
                "declared": command_report()["commands"],
                "dispatched": command_report()["dispatched"],
            },
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["flight_software_error"] = repr(exc)
    try:
        from core.observability.histograms import histograms_report
        from core.observability.trace_events import tracer_report
        from core.runtime.field_trials import field_trials_report
        from core.runtime.memory_infra import memory_infra_report
        from core.security.rule_of_two import rule_of_two_report

        histograms = histograms_report()
        memory = memory_infra_report()
        posture = rule_of_two_report()
        block["observability"] = {
            "histograms": {
                "count": histograms["count"],
                "clipping": histograms["clipping"],
                "expired": [e["name"] for e in histograms["expired"]],
            },
            "trace": {
                k: tracer_report()[k] for k in ("enabled", "buffered", "dropped", "span_s")
            },
            "memory_attribution": memory["leak_report"],
            "field_trials": field_trials_report()["active_groups"],
            "security_posture": {
                "violations": posture["violations"],
                "at_the_limit": posture["at_the_limit"],
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["observability_error"] = repr(exc)
    try:
        from core.bus.qos import qos_report
        from core.health.diagnostics_aggregator import diagnostics_report
        from core.observability.bus_recorder import bus_recorder_report
        from core.runtime.lifecycle import lifecycle_report
        from core.runtime.parameters import parameters_report

        diagnostics = diagnostics_report()
        lifecycles = lifecycle_report()
        block["middleware"] = {
            "diagnostics": {
                "level": diagnostics["level"],
                "stale": diagnostics["stale"],
                "errors": diagnostics["errors"],
                "summary": diagnostics["summary"],
            },
            "lifecycles": {
                "by_state": lifecycles["by_state"],
                "critical_inactive": lifecycles["critical_inactive"],
                "errored": lifecycles["errored"],
            },
            "qos": {
                "topics": qos_report()["topic_count"],
                "mismatches": len(qos_report()["qos_mismatches"]),
                "not_alive": qos_report()["not_alive"],
            },
            "parameters": {
                "count": parameters_report()["count"],
                "changed_from_default": parameters_report()["changed_from_default"],
            },
            "bus_ring": {
                k: bus_recorder_report()[k]
                for k in ("ring_size", "ring_span_s", "dumps", "recording")
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["middleware_error"] = repr(exc)
    try:
        from core.runtime.admission import admission_report
        from core.runtime.eviction import eviction_report
        from core.runtime.lease import lease_report
        from core.runtime.quota import quota_report
        from core.runtime.reconcile import reconcile_report

        admission = admission_report()
        eviction = eviction_report()
        block["orchestration"] = {
            "admission": {
                "hooks": len(admission["mutating"]) + len(admission["validating"]),
                "admitted": admission["admitted"],
                "denied": admission["denied"],
            },
            "quota": quota_report()["by_qos_class"],
            "eviction": {
                "eviction_order": eviction["eviction_order"],
                "breached": eviction["currently_breached"],
                "reclaims": eviction["reclaims"],
                "evictions": eviction["evictions"],
            },
            "controllers": reconcile_report(),
            "leases": lease_report(),
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["orchestration_error"] = repr(exc)
    try:
        from core.runtime.sanitizers import sanitizer_report

        block["sanitizers"] = sanitizer_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["sanitizers_error"] = repr(exc)
    try:
        from core.verify.invariants import last_report

        block["verifier"] = last_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["verifier_error"] = repr(exc)
    try:
        from core.pipeline.pass_manager import pass_manager_report

        passes = pass_manager_report()
        block["passes"] = {
            "bisect_limit": passes["bisect_limit"],
            "skips": passes["skips"],
            "hottest": passes["hottest"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["passes_error"] = repr(exc)
    try:
        from core.runtime.oom_policy import oom_report

        oom = oom_report()
        block["oom"] = {
            "next_victim": oom["next_victim"],
            "sheddable_organs": oom["sheddable_organs"],
            "immune_organs": oom["immune_organs"],
            "recent_sheds": oom["recent_sheds"][-3:],
            "restart_requested": oom["restart_requested"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["oom_error"] = repr(exc)
    return block


def runtime_health_report() -> dict[str, Any]:
    """Return Aura's canonical runtime health contract report."""
    report = evaluate_health().to_report()
    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        shutdown = get_shutdown_coordinator().get_status()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        shutdown = {
            "running": False,
            "request": {"requested": False},
            "report": None,
            "error": repr(exc),
        }
    report["shutdown"] = shutdown
    report["integrity"] = _runtime_integrity_block()
    request = shutdown.get("request") if isinstance(shutdown, dict) else None
    if isinstance(request, dict) and request.get("requested") is True:
        report["pre_shutdown_status"] = report.get("status")
        report["status"] = "stopping"
        report["healthy"] = False
        report["operational"] = False
        report["status_code"] = 503
        blockers = [str(item) for item in report.get("probe_blockers", [])]
        if "runtime_shutdown" not in blockers:
            blockers.insert(0, "runtime_shutdown")
        report["probe_blockers"] = blockers
        required_probes = report.get("required_probes")
        if isinstance(required_probes, dict):
            required_probes["all_passed"] = False
    return report


# ═══════════════════════════════════════════════════════════════════════
# THE PROBE SPLIT (roadmap K2): startup / liveness / readiness
#
# Kubernetes semantics, adopted because their conflation here caused real
# incidents: a loop-lag spike flipped the health verdict, boot readiness
# went false, and the GUI sat on "Connecting to runtime…" for 55 minutes
# over a fully conversational mind. Three probes, three INDEPENDENT
# meanings:
#
#   STARTUP   — has this process EVER been ready? Latched: once readiness
#               passes, startup is complete for the life of the process and
#               the surface may never present as "booting" again — only
#               "degraded". Before the latch, a startup deadline separates
#               "still warming" (ok) from "startup wedged" (not ok).
#   LIVENESS  — is the mind alive at all? Restart-worthy signal, so it is
#               deliberately RARE: only a DEAD verdict (no critical spine)
#               fails liveness. Flapping important-tier services never do.
#   READINESS — may traffic flow NOW? The existing required-probe-group
#               gate; may flap without implying a restart.
# ═══════════════════════════════════════════════════════════════════════

class ProbeKind(StrEnum):
    STARTUP = "startup"
    LIVENESS = "liveness"
    READINESS = "readiness"


@dataclass(frozen=True)
class ProbeVerdict:
    kind: ProbeKind
    ok: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": str(self.kind), "ok": self.ok, "reason": self.reason}


_STARTUP_LATCH_LOCK = threading.Lock()
_STARTUP_COMPLETE_AT: float | None = None
# Fallback time base for the startup deadline: the moment this module was
# imported. _process_uptime_seconds() reads the orchestrator's start time,
# which is 0.0 when no orchestrator ever registered — and a boot so wedged
# it never registers an orchestrator is EXACTLY the wedge the startup probe
# must catch, so it needs a clock that always runs.
_PROBE_EPOCH = time.time()


_STARTUP_DEADLINE_FLAG = None


def _startup_deadline_s() -> float:
    global _STARTUP_DEADLINE_FLAG
    if _STARTUP_DEADLINE_FLAG is None:
        try:
            from core.runtime.flags import FlagKind, declare

            _STARTUP_DEADLINE_FLAG = declare(
                "AURA_STARTUP_DEADLINE_S",
                kind=FlagKind.FLOAT,
                default=900.0,
                description="Seconds a fresh process may warm before the startup probe calls it wedged",
                owner="core.runtime.health_contract",
            )
        except (ImportError, AttributeError, RuntimeError, ValueError):
            return _float_env("AURA_STARTUP_DEADLINE_S", 900.0)
    return float(_STARTUP_DEADLINE_FLAG.value())


def _startup_age_s() -> float:
    return max(_process_uptime_seconds(), time.time() - _PROBE_EPOCH)


def reset_startup_latch_for_test() -> None:
    global _STARTUP_COMPLETE_AT, _PROBE_EPOCH
    with _STARTUP_LATCH_LOCK:
        _STARTUP_COMPLETE_AT = None
        _PROBE_EPOCH = time.time()


def startup_complete_at() -> float | None:
    with _STARTUP_LATCH_LOCK:
        return _STARTUP_COMPLETE_AT


def latch_startup_if_ready(ready_ok: bool) -> None:
    """Record the startup latch the first time readiness passes.

    Idempotent and monotonic: once latched, startup stays complete for the
    life of the process no matter how readiness flaps afterwards.
    """
    global _STARTUP_COMPLETE_AT
    if not ready_ok:
        return
    with _STARTUP_LATCH_LOCK:
        if _STARTUP_COMPLETE_AT is None:
            _STARTUP_COMPLETE_AT = time.time()


def probes_from_report(report: dict[str, Any]) -> dict[str, ProbeVerdict]:
    """Derive the three probe verdicts from an existing health report.

    Surfaces that already paid for ``evaluate_health()`` (boot status, the
    narrator) get the probe split without a second full evaluation.
    """
    status = required_probe_status(report)
    ready_ok = required_probe_groups_pass(status)
    ready_blockers = [] if ready_ok else required_probe_blockers(status)

    latch_startup_if_ready(ready_ok)

    latched = startup_complete_at()
    uptime = _startup_age_s()
    if latched is not None:
        startup = ProbeVerdict(
            ProbeKind.STARTUP, True, f"startup complete (latched at {latched:.0f})"
        )
    elif uptime <= _startup_deadline_s():
        startup = ProbeVerdict(
            ProbeKind.STARTUP,
            True,
            f"starting ({uptime:.0f}s of {_startup_deadline_s():.0f}s startup window)",
        )
    else:
        startup = ProbeVerdict(
            ProbeKind.STARTUP,
            False,
            f"startup wedged: never reached readiness within {_startup_deadline_s():.0f}s",
        )

    live_ok = str(report.get("status", "")) != HealthLevel.DEAD.value
    liveness = ProbeVerdict(
        ProbeKind.LIVENESS,
        live_ok,
        "critical spine registered" if live_ok else "no critical service present",
    )

    readiness = ProbeVerdict(
        ProbeKind.READINESS,
        ready_ok,
        "all required probe groups pass" if ready_ok else "; ".join(ready_blockers) or "not ready",
    )

    return {"startup": startup, "liveness": liveness, "readiness": readiness}


def evaluate_probes() -> dict[str, ProbeVerdict]:
    """One health evaluation, three independent probe verdicts."""
    return probes_from_report(evaluate_health().to_report())


def probe_split_report() -> dict[str, Any]:
    """Serializable probe-split for health surfaces and the narrator."""
    return {name: probe.to_dict() for name, probe in evaluate_probes().items()}


def log_health_report() -> HealthVerdict:
    """Evaluate and log the health report. Returns the verdict."""
    verdict = evaluate_health()
    summary_lines = verdict.summary().split("\n")
    if verdict.level == HealthLevel.HEALTHY:
        logger.info(summary_lines[0])
    elif verdict.level == HealthLevel.DEGRADED:
        logger.warning(summary_lines[0])
    else:
        logger.critical(summary_lines[0])

    for status, line in zip(verdict.services, summary_lines[1:], strict=False):
        if status.present and status.liveness_ok is not False:
            logger.info(line)
        elif status.requirement.tier == ServiceTier.CRITICAL:
            logger.critical(line)
        else:
            logger.warning(line)
    return verdict
