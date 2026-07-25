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
import threading
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


# Outcome/error text that means "declined to act", not "acted and failed".
# A refusal spends nothing, so it must not be charged as if it had.
_REFUSAL_OUTCOME_MARKERS = (
    "defer",
    "aura_now",
    "refused",
    "refusal",
    "not_authorized",
    "unauthorized",
    "blocked",
    "welfare_recovery_required",
    "requires stabilization",
    "admission",
    "backpressure",
    "queued",
)


def _is_refusal_outcome(event: Any) -> bool:
    """Whether a 'failure' consequence is a refusal rather than failed work."""
    text = " ".join(
        str(getattr(event, field, "") or "")
        for field in ("error", "action_content", "domain", "source")
    ).lower()
    if not text.strip():
        return False
    return any(marker in text for marker in _REFUSAL_OUTCOME_MARKERS)


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
        self._metabolic_lock = threading.RLock()
        self._spend_receipts: deque[str] = deque(maxlen=2048)
        self._spend_receipt_set: set[str] = set()

        # Decay constants.
        # Fatigue must recover faster than ordinary action costs accrue it,
        # otherwise it saturates near 1.0 and pins welfare.recovery_drive above
        # the Will's 0.6 defer threshold — producing a permanent
        # welfare_recovery_required_before_action storm that never clears
        # (observed live: fatigue ~0.94 and still climbing, 96k+ defers/boot).
        # At 0.01/s idle fatigue fully recovers in ~100s, decisively overcoming
        # the ~0.0025/s gross accrual while still building under genuine
        # sustained load. The old 0.002/s could not overcome normal drift.
        self._fatigue_decay_rate = 0.01     # per second when idle
        self._recovery_decay_rate = 0.001   # recovery debt decays slowly
        # Who is charging fatigue, and how much. Proportional recovery drains
        # a saturated fatigue in about seventy seconds of quiet, which is
        # provable in isolation — and the 2026-07-25 live runtime still sat at
        # 0.99 through a silent idle window with body_cost_applied empty on
        # every Will receipt. Something charges it that the Will never quoted,
        # and no surface in the system could name it. This ledger is that
        # surface: bounded, per-source, and reported in the snapshot, so the
        # question is one grep instead of an afternoon.
        self._fatigue_charges: dict[str, float] = {}
        self._fatigue_saturation_reported = False
        self._debt_charges: dict[str, float] = {}
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
        with self._metabolic_lock:
            if event.actual_outcome == "failure":
                self._error_window.append(True)
                self._metabolic.tool_calls_failed += 1
                # A REFUSAL is not failed work. When the Will defers an action
                # on welfare grounds the transaction still completes as a
                # "failure", and this branch charged fatigue for it — so the
                # defer storm fed the exhaustion that caused the defer storm.
                # The charge-attribution ledger is what exposed this: it showed
                # only 0.175 of fatigue ever charged through the cost path while
                # the live value sat at 0.985, so the charger had to be here,
                # outside every receipt and every ledger.
                #
                # Nothing was spent. Declining to act costs nothing, and
                # charging for it is how a tired system talks itself into
                # staying tired.
                if _is_refusal_outcome(event):
                    return
                charge_id = str(
                    getattr(event, "will_receipt_id", "") or event.event_id or ""
                )
                self._commit_cost_locked(
                    {"fatigue": 0.02, "integrity_risk": event.recovery_required * 0.5},
                    receipt_id=charge_id or f"consequence:{id(event)}",
                )
            else:
                self._error_window.append(False)
                if event.recovery_required < 0:
                    self._metabolic.relief_accumulated += abs(event.recovery_required)

            # A Will receipt may already have committed the authorization cost.
            # Reusing it here makes later execution/consequence publication
            # idempotent instead of charging the same action twice.
            #
            # Without a receipt there is no idempotency at all, and this became
            # a feedback loop that pinned fatigue at 1.0 for entire sessions.
            # WelfareTransaction publishes actual_body_cost as b_delta — the
            # OBSERVED change in body state across the transaction, i.e. the
            # fatigue that was already charged. Falling back to the unique
            # event_id made every such publication a fresh charge, so measured
            # fatigue was re-charged as new fatigue, which enlarged the next
            # transaction's delta, which charged more. That is why proportional
            # recovery (0.99 → 0.82 in twelve seconds, proven in isolation)
            # could not move the live value: the loop tracked the decay.
            #
            # An observation is not an authorization. No receipt, no charge.
            authorization = str(getattr(event, "will_receipt_id", "") or "").strip()
            if event.actual_body_cost and authorization:
                self._commit_cost_locked(
                    event.actual_body_cost,
                    receipt_id=authorization,
                )

    def update_body(self, body: BodyState) -> None:
        """Feed fresh BodyState from BeingRuntime.sample()."""
        self._last_body_state = body
        self._subscribe_consequences()

    def estimate_cost(
        self,
        domain: str,
        *,
        cost_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Return a bounded body-cost quote without mutating body state."""

        if self._lesioned:
            return {}
        multiplier = float(cost_multiplier)
        if not math.isfinite(multiplier) or multiplier < 0.0 or multiplier > 10.0:
            raise ValueError("body cost multiplier must be finite and within [0, 10]")
        costs = ACTION_COSTS.get(domain, {"compute": 0.02, "fatigue": 0.01})
        return {dim: float(base_cost) * multiplier for dim, base_cost in costs.items()}

    def spend(
        self,
        domain: str,
        *,
        cost_multiplier: float = 1.0,
        receipt_id: str = "",
    ) -> dict[str, float]:
        """Pay the metabolic cost for an action in the given domain.

        A receipt-bound spend is idempotent so retries and nested closure paths
        cannot charge the same authorized action twice.
        """
        if self._lesioned:
            return {}
        costs = self.estimate_cost(domain, cost_multiplier=cost_multiplier)
        receipt = str(receipt_id or "").strip() or (
            f"direct:{time.time_ns()}:{threading.get_ident()}"
        )
        return self.commit_cost(costs, receipt_id=receipt)

    def commit_cost(
        self,
        costs: dict[str, float],
        *,
        receipt_id: str,
    ) -> dict[str, float]:
        """Commit a quoted cost once for a stable action receipt."""

        if self._lesioned:
            return {}
        receipt = str(receipt_id or "").strip()
        if not receipt:
            raise ValueError("body cost commit requires a stable receipt_id")
        normalized: dict[str, float] = {}
        for dim, raw_cost in dict(costs or {}).items():
            cost = float(raw_cost)
            if not math.isfinite(cost) or abs(cost) > 10.0:
                raise ValueError(f"invalid body cost for {dim}")
            normalized[str(dim)] = cost
        with self._metabolic_lock:
            return self._commit_cost_locked(normalized, receipt_id=receipt)

    def _commit_cost_locked(
        self,
        costs: dict[str, float],
        *,
        receipt_id: str,
    ) -> dict[str, float]:
        if receipt_id in self._spend_receipt_set:
            return {}
        applied: dict[str, float] = {}
        for dim, actual in costs.items():
            if dim == "compute":
                self._metabolic.compute_spent += actual
                applied["compute"] = actual
            elif dim == "memory":
                self._metabolic.memory_spent += actual
                applied["memory"] = actual
            elif dim == "fatigue":
                self._metabolic.fatigue = _clip(self._metabolic.fatigue + actual)
                applied["fatigue"] = actual
                if actual > 0:
                    self._note_charge(self._fatigue_charges, receipt_id, actual)
            elif dim == "integrity_risk":
                self._metabolic.recovery_debt = _clip(
                    self._metabolic.recovery_debt + actual
                )
                if actual > 0:
                    self._note_charge(self._debt_charges, receipt_id, actual)
                applied["integrity_risk"] = actual
            elif dim == "recovery":
                self._metabolic.recovery_debt = _clip(
                    self._metabolic.recovery_debt - actual
                )
                self._metabolic.fatigue = _clip(self._metabolic.fatigue - actual)
                applied["recovery"] = actual

        self._metabolic.tool_calls_total += 1
        self._metabolic.last_spend_time = time.time()
        if len(self._spend_receipts) == self._spend_receipts.maxlen:
            oldest = self._spend_receipts.popleft()
            self._spend_receipt_set.discard(oldest)
        self._spend_receipts.append(receipt_id)
        self._spend_receipt_set.add(receipt_id)
        return applied

    def relieve(self, amount: float = 0.05) -> None:
        """Reduce recovery debt and fatigue after successful repair."""
        relief = float(amount)
        if not math.isfinite(relief) or relief < 0.0 or relief > 1.0:
            raise ValueError("body relief must be finite and within [0, 1]")
        with self._metabolic_lock:
            self._metabolic.recovery_debt = _clip(
                self._metabolic.recovery_debt - relief
            )
            self._metabolic.fatigue = _clip(
                self._metabolic.fatigue - relief * 0.5
            )
            self._metabolic.relief_accumulated += relief
            self._metabolic.last_relief_time = time.time()

    _CHARGE_LEDGER_CAP = 32

    @staticmethod
    def _charge_source(receipt_id: str) -> str:
        """Reduce a receipt to the caller family that issued it."""
        text = str(receipt_id or "unattributed").strip() or "unattributed"
        head = text.split(":", 1)[0]
        return head[:48] or "unattributed"

    def _note_charge(
        self, ledger: dict[str, float], receipt_id: str, amount: float
    ) -> None:
        source = self._charge_source(receipt_id)
        if source not in ledger and len(ledger) >= self._CHARGE_LEDGER_CAP:
            # Keep the heaviest charges; an attribution surface must not be
            # the thing that grows without bound.
            smallest = min(ledger, key=ledger.get)
            if ledger[smallest] >= amount:
                return
            ledger.pop(smallest, None)
        ledger[source] = ledger.get(source, 0.0) + float(amount)

    def charge_attribution(self) -> dict[str, dict[str, float]]:
        """Cumulative fatigue and recovery-debt charges, by caller family."""
        with self._metabolic_lock:
            return {
                "fatigue": dict(
                    sorted(
                        self._fatigue_charges.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:8]
                ),
                "recovery_debt": dict(
                    sorted(
                        self._debt_charges.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:8]
                ),
            }

    def _apply_natural_decay(self) -> None:
        """Fatigue and recovery debt decay over time."""
        with self._metabolic_lock:
            now = time.monotonic()
            elapsed = now - self._last_decay_time
            self._last_decay_time = now

            if elapsed > 0 and elapsed < 3600:  # sanity bound
                # Recovery scales with how tired she is. A FLAT rate can be
                # exactly cancelled by ordinary idle-loop costs, and then
                # fatigue never leaves saturation: the 2026-07-25 idle window
                # sat at 0.996 for the whole run with nothing failing, which
                # held welfare.recovery_drive above the Will's defer
                # threshold and produced a standing
                # welfare_recovery_required_before_action storm — her own
                # belief updates, memory writes and initiative all deferred
                # during a completely quiet hour. That is the same class the
                # 0.002→0.01 retune addressed; a constant only postpones it.
                # Proportional rest cannot be cancelled by a constant drip:
                # at fatigue 1.0 recovery runs 1.5x, at 0.1 it runs 0.6x, so
                # deep fatigue always escapes while ordinary dynamics are
                # untouched. Rest that never restores is not fatigue, it is a
                # ratchet.
                recovery_gain = 0.5 + self._metabolic.fatigue
                self._metabolic.fatigue = _clip(
                    self._metabolic.fatigue
                    - self._fatigue_decay_rate * elapsed * recovery_gain
                )
                # Recovery debt is the SAME ratchet and the larger term:
                # welfare weights it 0.4 against fatigue's 0.3, and the live
                # 0.881 recovery_drive is only reachable with debt at or near
                # saturation for the entire idle hour. Paying it down
                # proportionally too, or fixing fatigue alone would leave the
                # drive parked against its 0.6 threshold.
                debt_gain = 0.5 + self._metabolic.recovery_debt
                self._metabolic.recovery_debt = _clip(
                    self._metabolic.recovery_debt
                    - self._recovery_decay_rate * elapsed * debt_gain
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

        # Saturated fatigue is only diagnosable with its charge history. Say it
        # once per saturated stretch — enough to name the source in a live log,
        # never enough to become its own storm.
        if self._metabolic.fatigue >= 0.95:
            if not self._fatigue_saturation_reported:
                self._fatigue_saturation_reported = True
                logger.warning(
                    "Body fatigue saturated at %.4f (recovery_debt %.4f). "
                    "Charges so far by source: %s",
                    self._metabolic.fatigue,
                    self._metabolic.recovery_debt,
                    self.charge_attribution(),
                )
        elif self._metabolic.fatigue < 0.80:
            self._fatigue_saturation_reported = False

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
