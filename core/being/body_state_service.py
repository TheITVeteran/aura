"""core/being/body_state_service.py — Unified Digital Body Model.

Unifies CPU/memory/disk/thermal/battery/latency/tool-failure/model-availability/
context/sensor/permission/network/memory-corruption/queue-backlog/exception
pressures into one live body state that every major subsystem must read.

Every action pays a body cost. Every failure updates the body.
Every recovery produces measurable relief.

Design:
  - Pulls from BodyState (aura_now.py) + resource_stakes + interoception
  - Adds: metabolic cost tracking, fatigue/recovery debt, error memory
  - Every consequential action pays cost via spend()
  - Successful repair calls relieve()
  - Reads from ConsequenceBus for automatic feedback
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.being.aura_now import BodyState
from core.runtime.consequence_bus import ConsequenceBus, ConsequenceEvent
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.BodyStateService")


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class MetabolicBudget:
    """Tracks running costs and recovery debt."""
    compute_spent: float = 0.0       # cumulative compute cost
    memory_spent: float = 0.0        # cumulative memory-write cost
    tool_calls_total: int = 0
    tool_calls_failed: int = 0
    recovery_debt: float = 0.0       # 0-1, how much recovery is owed
    fatigue: float = 0.0             # 0-1, accumulated fatigue
    relief_accumulated: float = 0.0  # how much relief since last reset
    last_spend_time: float = 0.0
    last_relief_time: float = 0.0


@dataclass
class BodyHealthSnapshot:
    """Complete body state at a point in time."""
    # Standard pressures (from BodyState)
    cpu_pressure: float = 0.0
    memory_pressure: float = 0.0
    disk_pressure: float = 0.0
    thermal_pressure: float = 0.0
    battery_pressure: float = 0.0
    latency_pressure: float = 0.0
    permission_pressure: float = 0.0
    network_pressure: float = 0.0
    context_pressure: float = 0.0
    sensor_pressure: float = 0.0
    tool_failure_pressure: float = 0.0

    # Extended body dimensions
    model_availability: float = 1.0     # 0-1, how available the LLM is
    memory_corruption_risk: float = 0.0 # 0-1
    queue_backlog: float = 0.0          # 0-1, normalized backlog
    unresolved_exceptions: int = 0
    error_rate: float = 0.0             # recent error fraction

    # Metabolic
    fatigue: float = 0.0
    recovery_debt: float = 0.0
    relief: float = 0.0

    # Composite
    total_pressure: float = 0.0
    operational_health: float = 1.0     # 1 = perfect, 0 = critical

    timestamp: float = field(default_factory=time.time)

    def is_strained(self) -> bool:
        return self.total_pressure > 0.6 or self.fatigue > 0.5

    def is_critical(self) -> bool:
        return self.total_pressure > 0.85 or self.fatigue > 0.8

    def needs_recovery(self) -> bool:
        return self.recovery_debt > 0.3 or self.fatigue > 0.4

    def pressure_vector(self) -> dict[str, float]:
        return {
            "cpu": round(self.cpu_pressure, 4),
            "memory": round(self.memory_pressure, 4),
            "disk": round(self.disk_pressure, 4),
            "thermal": round(self.thermal_pressure, 4),
            "battery": round(self.battery_pressure, 4),
            "latency": round(self.latency_pressure, 4),
            "permission": round(self.permission_pressure, 4),
            "network": round(self.network_pressure, 4),
            "context": round(self.context_pressure, 4),
            "sensor": round(self.sensor_pressure, 4),
            "tool_failure": round(self.tool_failure_pressure, 4),
            "model_availability": round(self.model_availability, 4),
            "memory_corruption_risk": round(self.memory_corruption_risk, 4),
            "queue_backlog": round(self.queue_backlog, 4),
            "fatigue": round(self.fatigue, 4),
            "recovery_debt": round(self.recovery_debt, 4),
        }


# Cost tables for different action domains
ACTION_COSTS: dict[str, dict[str, float]] = {
    "tool_execution":     {"compute": 0.04, "fatigue": 0.02, "memory": 0.01},
    "memory_write":       {"compute": 0.01, "memory": 0.03, "fatigue": 0.01, "integrity_risk": 0.01},
    "state_mutation":     {"compute": 0.02, "fatigue": 0.02, "integrity_risk": 0.02},
    "initiative":         {"compute": 0.05, "fatigue": 0.03, "memory": 0.02},
    "exploration":        {"compute": 0.06, "fatigue": 0.04},
    "self_modification":  {"compute": 0.08, "fatigue": 0.06, "integrity_risk": 0.10},
    "response":           {"compute": 0.02, "fatigue": 0.01},
    "reflection":         {"compute": 0.01, "fatigue": 0.005},
    "stabilization":      {"compute": 0.01, "fatigue": -0.02, "recovery": 0.03},  # negative = heals
    "cloud_call":         {"compute": 0.05, "fatigue": 0.02, "network": 0.03},
    "file_write":         {"compute": 0.02, "disk": 0.02, "integrity_risk": 0.01},
    "network_call":       {"compute": 0.03, "network": 0.04, "fatigue": 0.02},
}


class BodyStateService:
    """Unified digital body that every subsystem must read before acting.

    Usage:
        body_service = BodyStateService.get()
        snapshot = body_service.snapshot()
        if snapshot.is_strained():
            # reduce optional actions
        body_service.spend("tool_execution", cost_multiplier=1.0)
        body_service.relieve(0.05)  # after successful recovery
    """

    _instance: BodyStateService | None = None

    def __init__(self) -> None:
        self._metabolic = MetabolicBudget()
        self._error_window: deque[bool] = deque(maxlen=100)  # True=error
        self._last_body_state: BodyState | None = None
        self._last_snapshot: BodyHealthSnapshot | None = None
        self._lesioned = False
        self._consequence_subscribed = False

        # Decay constants
        self._fatigue_decay_rate = 0.002    # per second when idle
        self._recovery_decay_rate = 0.001   # recovery debt decays slowly
        self._last_decay_time = time.monotonic()

    @classmethod
    def get(cls) -> BodyStateService:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _subscribe_consequences(self) -> None:
        """Auto-subscribe to consequence bus for feedback."""
        if self._consequence_subscribed:
            return
        try:
            bus = ConsequenceBus.get()
            bus.subscribe("*", self._on_consequence)
            self._consequence_subscribed = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "body_state_service",
                exc,
                action="continued without consequence-bus body feedback subscription",
            )
            logger.warning("BodyStateService consequence subscription failed: %s", exc)

    def _on_consequence(self, event: ConsequenceEvent) -> None:
        """React to action outcomes from the consequence bus."""
        if event.actual_outcome == "failure":
            self._error_window.append(True)
            self._metabolic.tool_calls_failed += 1
            self._metabolic.recovery_debt = _clip(
                self._metabolic.recovery_debt + event.recovery_required * 0.5
            )
            self._metabolic.fatigue = _clip(
                self._metabolic.fatigue + 0.02
            )
        else:
            self._error_window.append(False)
            if event.recovery_required < 0:
                # Negative recovery = this action was healing
                self._metabolic.relief_accumulated += abs(event.recovery_required)

        # Apply actual body costs from the event
        for dim, cost in event.actual_body_cost.items():
            if dim == "compute":
                self._metabolic.compute_spent += cost
            elif dim == "memory":
                self._metabolic.memory_spent += cost

    def update_body(self, body: BodyState) -> None:
        """Feed fresh BodyState from BeingRuntime.sample()."""
        self._last_body_state = body
        self._subscribe_consequences()

    def spend(self, domain: str, *, cost_multiplier: float = 1.0) -> dict[str, float]:
        """Pay the metabolic cost for an action in the given domain.

        Returns the actual costs applied.
        """
        if self._lesioned:
            return {}

        costs = dict(ACTION_COSTS.get(domain, {"compute": 0.02, "fatigue": 0.01}))
        applied: dict[str, float] = {}

        for dim, base_cost in costs.items():
            actual = base_cost * cost_multiplier
            if dim == "compute":
                self._metabolic.compute_spent += actual
                applied["compute"] = actual
            elif dim == "memory":
                self._metabolic.memory_spent += actual
                applied["memory"] = actual
            elif dim == "fatigue":
                self._metabolic.fatigue = _clip(self._metabolic.fatigue + actual)
                applied["fatigue"] = actual
            elif dim == "integrity_risk":
                self._metabolic.recovery_debt = _clip(
                    self._metabolic.recovery_debt + actual
                )
                applied["integrity_risk"] = actual
            elif dim == "recovery":
                self._metabolic.recovery_debt = _clip(
                    self._metabolic.recovery_debt - actual
                )
                self._metabolic.fatigue = _clip(
                    self._metabolic.fatigue - actual
                )
                applied["recovery"] = actual

        self._metabolic.tool_calls_total += 1
        self._metabolic.last_spend_time = time.time()
        return applied

    def relieve(self, amount: float = 0.05) -> None:
        """Reduce recovery debt and fatigue after successful repair."""
        self._metabolic.recovery_debt = _clip(
            self._metabolic.recovery_debt - amount
        )
        self._metabolic.fatigue = _clip(
            self._metabolic.fatigue - amount * 0.5
        )
        self._metabolic.relief_accumulated += amount
        self._metabolic.last_relief_time = time.time()

    def _apply_natural_decay(self) -> None:
        """Fatigue and recovery debt decay over time."""
        now = time.monotonic()
        elapsed = now - self._last_decay_time
        self._last_decay_time = now

        if elapsed > 0 and elapsed < 3600:  # sanity bound
            self._metabolic.fatigue = _clip(
                self._metabolic.fatigue - self._fatigue_decay_rate * elapsed
            )
            self._metabolic.recovery_debt = _clip(
                self._metabolic.recovery_debt - self._recovery_decay_rate * elapsed
            )

    def snapshot(self) -> BodyHealthSnapshot:
        """Produce a complete body health snapshot."""
        self._apply_natural_decay()

        body = self._last_body_state or BodyState()

        # Compute error rate from window
        error_count = sum(1 for e in self._error_window if e)
        error_rate = error_count / max(1, len(self._error_window))

        # Composite pressures
        pressures = [
            body.cpu_pressure, body.memory_pressure, body.disk_pressure,
            body.thermal_pressure, body.battery_pressure, body.latency_pressure,
            body.permission_pressure, body.network_pressure, body.context_pressure,
            body.sensor_pressure, body.tool_failure_pressure,
            self._metabolic.fatigue, self._metabolic.recovery_debt,
        ]
        avg_pressure = sum(pressures) / len(pressures)
        peak_pressure = max(pressures) if pressures else 0.0
        total_pressure = _clip(avg_pressure * 0.45 + peak_pressure * 0.55)

        # Operational health: inverse of total pressure + error rate
        operational_health = _clip(
            1.0 - total_pressure * 0.5 - error_rate * 0.3 - self._metabolic.fatigue * 0.2
        )

        snap = BodyHealthSnapshot(
            cpu_pressure=body.cpu_pressure,
            memory_pressure=body.memory_pressure,
            disk_pressure=body.disk_pressure,
            thermal_pressure=body.thermal_pressure,
            battery_pressure=body.battery_pressure,
            latency_pressure=body.latency_pressure,
            permission_pressure=body.permission_pressure,
            network_pressure=body.network_pressure,
            context_pressure=body.context_pressure,
            sensor_pressure=body.sensor_pressure,
            tool_failure_pressure=body.tool_failure_pressure,
            error_rate=round(error_rate, 4),
            fatigue=round(self._metabolic.fatigue, 4),
            recovery_debt=round(self._metabolic.recovery_debt, 4),
            relief=round(self._metabolic.relief_accumulated, 4),
            total_pressure=round(total_pressure, 4),
            operational_health=round(operational_health, 4),
            unresolved_exceptions=error_count,
        )
        self._last_snapshot = snap
        return snap

    def lesion(self) -> None:
        """Disable body state tracking (for lesion experiments)."""
        self._lesioned = True

    def restore(self) -> None:
        """Re-enable body state tracking."""
        self._lesioned = False

    @property
    def metabolic(self) -> MetabolicBudget:
        return self._metabolic

    @property
    def is_lesioned(self) -> bool:
        return self._lesioned
