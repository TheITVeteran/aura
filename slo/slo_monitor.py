"""slo/slo_monitor.py
━━━━━━━━━━━━━━━━━━━
Live SLO burn-rate monitor for reliability-grade operations tracking.

Continuously measures all SLOs, calculates error budget consumption,
and fires alerts when burn rates exceed safe thresholds.

Usage:
    from slo.slo_monitor import SLOMonitor, SLODefinition, get_slo_monitor

    monitor = get_slo_monitor()
    monitor.record("boot_cold_p95_ms", 2800.0)

    status = monitor.status()
    # → {"slos": {...}, "alerts": [...], "budget_status": "OK"}
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.SLOMonitor")


class BudgetStatus(StrEnum):
    OK = "OK"
    SLOW_BURN = "SLOW_BURN"       # Budget draining faster than replenishment
    FAST_BURN = "FAST_BURN"       # Budget draining dangerously fast
    EXHAUSTED = "EXHAUSTED"       # Error budget consumed
    UNKNOWN = "UNKNOWN"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""
    name: str
    description: str
    target: float           # Target value (e.g., p95 < 2000ms → target=2000)
    unit: str               # "ms", "us", "percent", "score", "count", "tok/s"
    direction: str = "max"  # "max" = value should be ≤ target, "min" = value should be ≥ target
    window_seconds: float = 3600  # Measurement window (1 hour default)
    error_budget_pct: float = 5.0  # Allowed violation percentage (5% = 95th percentile SLO)
    # "sample": each recorded value is judged against target on its own.
    # "count_per_window": each record is an EVENT; the SLO is violated when
    # the number of events inside the window exceeds target. Required for
    # occurrence SLOs (error events, substrate resets) — judging a per-event
    # value of 1.0 against a count target can never trip.
    aggregation: str = "sample"


@dataclass
class SLOSample:
    """A single SLO measurement sample."""
    value: float
    timestamp: float = field(default_factory=time.time)
    violated: bool = False


@dataclass
class SLOAlert:
    """An SLO burn-rate alert."""
    slo_name: str
    severity: AlertSeverity
    message: str
    burn_rate: float
    budget_remaining_pct: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slo": self.slo_name,
            "severity": self.severity.value,
            "message": self.message[:200],
            "burn_rate": round(self.burn_rate, 2),
            "budget_remaining_pct": round(self.budget_remaining_pct, 1),
            "timestamp": self.timestamp,
        }


class SLOTracker:
    """Tracks samples and budget for a single SLO.

    Thread-safe: the sample deque is guarded by an internal lock because
    records arrive from worker threads while status/percentile queries run
    on the server thread — iterating a deque during a concurrent append
    raises RuntimeError.
    """

    def __init__(self, definition: SLODefinition, max_samples: int = 5000) -> None:
        self.definition = definition
        self._lock = threading.Lock()
        self._samples: deque[SLOSample] = deque(maxlen=max_samples)
        self._violation_count = 0
        self._total_count = 0

    def record(self, value: float) -> bool:
        """Record a sample. Returns True if within SLO."""
        with self._lock:
            if self.definition.aggregation == "count_per_window":
                cutoff = time.time() - self.definition.window_seconds
                events_in_window = sum(
                    1 for s in self._samples if s.timestamp >= cutoff
                ) + 1  # include this event
                violated = events_in_window > self.definition.target
            elif self.definition.direction == "max":
                violated = value > self.definition.target
            else:
                violated = value < self.definition.target

            sample = SLOSample(value=value, violated=violated)
            self._samples.append(sample)
            self._total_count += 1
            if violated:
                self._violation_count += 1

        return not violated

    def budget_remaining_pct(self) -> float:
        """Calculate remaining error budget percentage."""
        window_samples = self._windowed_samples()
        if not window_samples:
            return 100.0
        violations = sum(1 for s in window_samples if s.violated)
        violation_rate = violations / len(window_samples) * 100
        budget = self.definition.error_budget_pct
        remaining = max(0.0, budget - violation_rate)
        return remaining / max(budget, 0.001) * 100

    def burn_rate(self) -> float:
        """Calculate burn rate (how fast the error budget is being consumed).

        1.0 = consuming at exactly the sustainable rate
        >1.0 = consuming faster than sustainable (slow burn)
        >10.0 = dangerously fast consumption (fast burn)
        """
        window_samples = self._windowed_samples()
        if not window_samples:
            return 0.0
        violations = sum(1 for s in window_samples if s.violated)
        violation_rate = violations / len(window_samples) * 100
        budget = self.definition.error_budget_pct
        if budget <= 0:
            return 0.0
        return violation_rate / budget

    def budget_status(self) -> BudgetStatus:
        rate = self.burn_rate()
        remaining = self.budget_remaining_pct()
        if remaining <= 0:
            return BudgetStatus.EXHAUSTED
        if rate > 10.0:
            return BudgetStatus.FAST_BURN
        if rate > 1.0:
            return BudgetStatus.SLOW_BURN
        return BudgetStatus.OK

    def percentile(self, pct: float) -> float | None:
        """Calculate a percentile from windowed samples."""
        window = self._windowed_samples()
        if not window:
            return None
        values = sorted(s.value for s in window)
        k = (len(values) - 1) * (pct / 100.0)
        f = int(k)
        c = min(f + 1, len(values) - 1)
        return values[f] + (values[c] - values[f]) * (k - f)

    def status(self) -> dict[str, Any]:
        window = self._windowed_samples()
        p50 = self.percentile(50.0)
        p95 = self.percentile(95.0)
        p99 = self.percentile(99.0)
        return {
            "name": self.definition.name,
            "target": self.definition.target,
            "unit": self.definition.unit,
            "direction": self.definition.direction,
            "samples_in_window": len(window),
            "total_samples": self._total_count,
            "violations_total": self._violation_count,
            "budget_remaining_pct": round(self.budget_remaining_pct(), 1),
            "burn_rate": round(self.burn_rate(), 2),
            "budget_status": self.budget_status().value,
            "p50": round(p50, 3) if p50 is not None else None,
            "p95": round(p95, 3) if p95 is not None else None,
            "p99": round(p99, 3) if p99 is not None else None,
        }

    def _windowed_samples(self) -> list[SLOSample]:
        cutoff = time.time() - self.definition.window_seconds
        with self._lock:
            return [s for s in self._samples if s.timestamp >= cutoff]


class SLOMonitor:
    """Central SLO monitor managing all SLO trackers.

    Thread-safe. Alerts are rate-limited per SLO (ALERT_COOLDOWN_S) so a
    persistently violating SLO cannot storm the log or the fault registry —
    record() sits on hot paths (mind tick, Will decisions), so everything
    past the cooldown gate must stay off the common case.
    """

    ALERT_COOLDOWN_S = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trackers: dict[str, SLOTracker] = {}
        self._alerts: deque[SLOAlert] = deque(maxlen=500)
        self._last_alert_at: dict[str, float] = {}
        self._register_default_slos()

    def define(self, defn: SLODefinition) -> None:
        with self._lock:
            self._trackers[defn.name] = SLOTracker(defn)

    def record(self, slo_name: str, value: float) -> bool:
        """Record a measurement for an SLO. Returns True if within SLO."""
        with self._lock:
            tracker = self._trackers.get(slo_name)
        if tracker is None:
            return True  # Unknown SLO — don't fail

        within = tracker.record(value)
        if not within and self._alert_cooldown_expired(slo_name):
            self._check_alert(tracker)
        return within

    def _alert_cooldown_expired(self, slo_name: str) -> bool:
        now = time.time()
        with self._lock:
            last = self._last_alert_at.get(slo_name, 0.0)
            if (now - last) < self.ALERT_COOLDOWN_S:
                return False
            self._last_alert_at[slo_name] = now
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            slo_statuses = {}
            overall = BudgetStatus.OK
            for name, tracker in self._trackers.items():
                st = tracker.status()
                slo_statuses[name] = st
                bs = BudgetStatus(st["budget_status"])
                if bs == BudgetStatus.EXHAUSTED:
                    overall = BudgetStatus.EXHAUSTED
                elif bs == BudgetStatus.FAST_BURN and overall != BudgetStatus.EXHAUSTED:
                    overall = BudgetStatus.FAST_BURN
                elif bs == BudgetStatus.SLOW_BURN and overall == BudgetStatus.OK:
                    overall = BudgetStatus.SLOW_BURN

            return {
                "overall_status": overall.value,
                "slo_count": len(self._trackers),
                "slos": slo_statuses,
                "recent_alerts": [a.to_dict() for a in list(self._alerts)[-10:]],
            }

    def get_tracker(self, slo_name: str) -> SLOTracker | None:
        with self._lock:
            return self._trackers.get(slo_name)

    def _check_alert(self, tracker: SLOTracker) -> None:
        """Check if an SLO violation warrants an alert (cooldown already passed)."""
        bs = tracker.budget_status()
        if bs == BudgetStatus.OK:
            return

        severity = {
            BudgetStatus.SLOW_BURN: AlertSeverity.WARNING,
            BudgetStatus.FAST_BURN: AlertSeverity.CRITICAL,
            BudgetStatus.EXHAUSTED: AlertSeverity.CRITICAL,
        }.get(bs, AlertSeverity.INFO)

        burn = tracker.burn_rate()
        remaining = tracker.budget_remaining_pct()
        alert = SLOAlert(
            slo_name=tracker.definition.name,
            severity=severity,
            message=f"SLO {tracker.definition.name} budget {bs.value}: "
                    f"burn_rate={burn:.1f}x, remaining={remaining:.0f}%",
            burn_rate=burn,
            budget_remaining_pct=remaining,
        )
        with self._lock:
            self._alerts.append(alert)

        log_fn = logger.critical if severity == AlertSeverity.CRITICAL else logger.warning
        log_fn("SLO ALERT: %s", alert.message)

        # Record in fault taxonomy
        try:
            from core.resilience.fault_taxonomy import get_fault_registry
            get_fault_registry().record_fault(
                "F18", subsystem=f"slo.{tracker.definition.name}",
                details=alert.message,
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Fault taxonomy unavailable for SLO alert: %s", exc)

    def _register_default_slos(self) -> None:
        """Register the expanded set of 25+ SLO definitions."""
        defaults = [
            # Boot SLOs
            SLODefinition("boot_cold_p95_ms", "Cold boot time", 30000, "ms"),
            SLODefinition("boot_warm_p95_ms", "Warm resume time", 5000, "ms"),
            # Inference SLOs
            SLODefinition("inference_first_token_p95_ms", "Time to first token", 2000, "ms"),
            SLODefinition("inference_throughput_tok_s", "Generation throughput", 15, "tok/s",
                          direction="min"),
            SLODefinition("inference_total_p95_ms", "Total inference time", 30000, "ms"),
            # Memory SLOs
            SLODefinition("memory_write_p95_ms", "Memory write latency", 10, "ms"),
            SLODefinition("memory_retrieval_p95_ms", "Memory retrieval latency", 50, "ms"),
            SLODefinition("memory_compaction_p95_s", "Memory compaction time", 5, "s"),
            # Health SLOs
            SLODefinition("health_endpoint_p99_ms", "Health endpoint latency", 100, "ms"),
            SLODefinition("watchdog_check_interval_s", "Watchdog check interval", 10, "s"),
            # Shutdown SLOs
            SLODefinition("shutdown_graceful_p95_s", "Graceful shutdown time", 12, "s"),
            # Recovery SLOs
            SLODefinition("recovery_circuit_breaker_s", "Circuit breaker recovery", 30, "s"),
            SLODefinition("recovery_worker_restart_s", "Worker restart time", 15, "s"),
            # Governance SLOs
            # will_decision latency is the FULL end-to-end decide(): felt-state
            # read, policy evaluation, evidence, substrate receipt, and the
            # consequence-bus publish. The old 5ms target was never achievable
            # for that — measured healthy p95 is ~80-100ms — so it burned at
            # ~16-20x continuously and fired a CRITICAL every minute, burying
            # real criticals (a genuine capability_engine failure was lost in the
            # spam, July 2026). 250ms is realistic headroom over the healthy
            # baseline while still tripping on a true 2.5x+ regression.
            SLODefinition("will_decision_p95_ms", "Will decision latency", 250, "ms"),
            SLODefinition("receipt_verification_p95_ms", "Receipt verification latency", 1, "ms"),
            # Audit SLOs (existing, expanded)
            SLODefinition("audit_chain_append_p95_ms", "Audit chain append", 0.55, "ms",
                          error_budget_pct=10),
            SLODefinition("audit_chain_verify_per_entry_us", "Audit verify per entry", 8.75, "us",
                          error_budget_pct=10),
            # Stability SLOs
            SLODefinition("event_loop_lag_p95_ms", "Event loop lag", 100, "ms"),
            SLODefinition("tick_duration_p95_ms", "Mind tick duration", 500, "ms"),
            SLODefinition("substrate_resets_per_hour", "Substrate reset frequency", 5, "count",
                          aggregation="count_per_window"),
            # Reliability SLOs
            SLODefinition("uptime_pct", "System uptime percentage", 99.9, "percent",
                          direction="min", window_seconds=86400),
            SLODefinition("error_events_per_hour",
                          "Critical/degraded degradation events per hour",
                          10, "count", aggregation="count_per_window"),
            # Resource SLOs
            SLODefinition("memory_rss_mb", "Resident memory size", 4096, "mb"),
            SLODefinition("db_size_mb", "Database size", 2048, "mb"),
            SLODefinition("log_volume_mb_per_hour", "Log volume per hour", 100, "mb"),
        ]
        for defn in defaults:
            self._trackers[defn.name] = SLOTracker(defn)


# ── Module singleton ─────────────────────────────────────────────────

_monitor: SLOMonitor | None = None
_monitor_lock = threading.Lock()


def get_slo_monitor() -> SLOMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = SLOMonitor()
    return _monitor
