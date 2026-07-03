"""infrastructure/canary.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Canary deployment controller for reliability-grade safe deployments.

Implements progressive rollout with anomaly detection and automatic
rollback when canary metrics diverge from baseline.

Deploy-time tool: run_canary() blocks its calling thread through the phased
rollout (phases sleep phase_duration_s each) — run it from a CLI/deploy
process, never on the live event loop.

Usage:
    from infrastructure.canary import CanaryController, CanaryConfig

    controller = CanaryController()
    result = controller.run_canary(
        canary_fn=deploy_canary_build,
        collect_metrics=collect_live_metrics,
        smoke_test=run_smoke_suite,
    )
    if result.passed:
        ...  # phase is PROMOTED; proceed with full rollout
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("Infra.Canary")


class CanaryPhase(StrEnum):
    PREFLIGHT = "preflight"
    CANARY_1PCT = "canary_1pct"
    CANARY_5PCT = "canary_5pct"
    CANARY_25PCT = "canary_25pct"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    phase_duration_s: float = 300         # Time per phase (5 min)
    max_error_rate: float = 0.05          # 5% error rate threshold
    max_latency_increase_pct: float = 20  # 20% latency increase threshold
    auto_rollback: bool = True
    smoke_test_timeout_s: float = 30


@dataclass
class CanaryMetrics:
    """Metrics snapshot for canary comparison."""
    error_rate: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def total_requests(self) -> int:
        return self.success_count + self.failure_count


@dataclass
class CanaryResult:
    """Result of a canary deployment."""
    phase_reached: CanaryPhase
    passed: bool
    baseline_metrics: CanaryMetrics | None = None
    canary_metrics: CanaryMetrics | None = None
    rollback_reason: str = ""
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase_reached.value,
            "passed": self.passed,
            "rollback_reason": self.rollback_reason[:200],
            "duration_s": round(self.duration_s, 1),
            "timestamp": self.timestamp,
        }


class CanaryController:
    """Manages canary deployments with progressive rollout.

    Thread-safe. Only one canary can run at a time.
    """

    def __init__(self, config: CanaryConfig | None = None) -> None:
        self._config = config or CanaryConfig()
        self._lock = threading.Lock()
        self._phase = CanaryPhase.PREFLIGHT
        self._results: list[CanaryResult] = []
        self._running = False

    def run_canary(
        self,
        canary_fn: Callable[[], None],
        collect_metrics: Callable[[], CanaryMetrics],
        smoke_test: Callable[[], bool] | None = None,
    ) -> CanaryResult:
        """Run a canary deployment with progressive rollout.

        1. Preflight: run smoke tests
        2. Progressive phases: 1% → 5% → 25% → 100%
        3. At each phase: compare canary metrics vs baseline
        4. Auto-rollback if anomaly detected
        """
        with self._lock:
            if self._running:
                return CanaryResult(
                    phase_reached=CanaryPhase.FAILED, passed=False,
                    rollback_reason="Another canary is already running",
                )
            self._running = True

        t0 = time.time()
        with self._lock:
            self._phase = CanaryPhase.PREFLIGHT

        try:
            # Preflight
            if smoke_test:
                try:
                    passed = smoke_test()
                except Exception as exc:
                    return self._fail(f"Smoke test failed: {exc}", t0)
                if not passed:
                    return self._fail("Smoke test returned False", t0)

            logger.info("CANARY: Preflight passed, collecting baseline")
            baseline = collect_metrics()

            # Deploy canary
            try:
                canary_fn()
            except Exception as exc:
                return self._fail(f"Canary deployment failed: {exc}", t0)

            # Progressive phases
            phases = [
                CanaryPhase.CANARY_1PCT,
                CanaryPhase.CANARY_5PCT,
                CanaryPhase.CANARY_25PCT,
            ]

            for phase in phases:
                with self._lock:
                    self._phase = phase
                logger.info("CANARY: Phase %s — monitoring for %.0fs",
                            phase.value, self._config.phase_duration_s)

                time.sleep(self._config.phase_duration_s)

                canary_metrics = collect_metrics()
                anomaly = self._detect_anomaly(baseline, canary_metrics)
                if anomaly:
                    return self._rollback(
                        phase, baseline, canary_metrics, anomaly, t0,
                    )

            # All phases passed
            with self._lock:
                self._phase = CanaryPhase.PROMOTED
            duration = time.time() - t0
            result = CanaryResult(
                phase_reached=CanaryPhase.PROMOTED, passed=True,
                baseline_metrics=baseline,
                canary_metrics=collect_metrics(),
                duration_s=duration,
            )
            logger.info("CANARY: Promoted! All phases passed in %.0fs", duration)
            with self._lock:
                self._results.append(result)
            return result

        finally:
            with self._lock:
                self._running = False

    def _detect_anomaly(
        self, baseline: CanaryMetrics, canary: CanaryMetrics,
    ) -> str:
        """Compare canary metrics to baseline. Return anomaly description or empty."""
        # Error rate check
        if canary.error_rate > self._config.max_error_rate:
            return (f"Error rate {canary.error_rate:.1%} exceeds "
                    f"threshold {self._config.max_error_rate:.1%}")

        # Latency check (KS-test simplified as percentage comparison)
        if baseline.latency_p95_ms > 0:
            increase = ((canary.latency_p95_ms - baseline.latency_p95_ms)
                        / baseline.latency_p95_ms * 100)
            if increase > self._config.max_latency_increase_pct:
                return (f"P95 latency increased {increase:.0f}% "
                        f"(threshold: {self._config.max_latency_increase_pct:.0f}%)")

        return ""

    def _rollback(
        self,
        phase: CanaryPhase,
        baseline: CanaryMetrics,
        canary: CanaryMetrics,
        reason: str,
        t0: float,
    ) -> CanaryResult:
        """Execute automatic rollback."""
        with self._lock:
            self._phase = CanaryPhase.ROLLED_BACK
        duration = time.time() - t0
        logger.error("CANARY ROLLBACK at %s: %s", phase.value, reason)
        result = CanaryResult(
            phase_reached=phase, passed=False,
            baseline_metrics=baseline, canary_metrics=canary,
            rollback_reason=reason, duration_s=duration,
        )
        with self._lock:
            self._results.append(result)
        return result

    def _fail(self, reason: str, t0: float) -> CanaryResult:
        with self._lock:
            self._phase = CanaryPhase.FAILED
        duration = time.time() - t0
        logger.error("CANARY FAILED: %s", reason)
        result = CanaryResult(
            phase_reached=CanaryPhase.FAILED, passed=False,
            rollback_reason=reason, duration_s=duration,
        )
        with self._lock:
            self._results.append(result)
        return result

    @property
    def current_phase(self) -> CanaryPhase:
        with self._lock:
            return self._phase

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_phase": self._phase.value,
                "running": self._running,
                "history": [r.to_dict() for r in self._results[-5:]],
            }
