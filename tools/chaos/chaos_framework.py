"""tools/chaos/chaos_framework.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Structured chaos engineering framework for reliability-grade fault injection.

Hypothesis-driven chaos experiments with blast radius controls,
automatic rollback, and result tracking.

Usage:
    from tools.chaos.chaos_framework import (
        ChaosFramework, ChaosExperiment, FaultInjector,
    )

    framework = ChaosFramework()
    experiment = ChaosExperiment(
        name="memory_pressure",
        hypothesis="System degrades gracefully under 90% memory pressure",
        fault=MemoryPressureFault(target_pct=90),
        duration_s=60,
        success_criteria=lambda m: m["health_status"] != "FAILED",
    )
    result = framework.run(experiment)
"""
from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

logger = logging.getLogger("Aura.Chaos")


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    experiment_name: str
    status: ExperimentStatus
    hypothesis: str
    duration_s: float
    metrics_before: dict[str, Any] = field(default_factory=dict)
    metrics_after: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "status": self.status.value,
            "hypothesis": self.hypothesis[:200],
            "duration_s": round(self.duration_s, 2),
            "error": self.error[:200] if self.error else "",
            "timestamp": self.timestamp,
        }


class FaultInjector(ABC):
    """Base class for fault injectors."""

    @abstractmethod
    def inject(self) -> None:
        """Inject the fault."""

    @abstractmethod
    def revert(self) -> None:
        """Revert the fault (restore normal operation)."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this fault type."""


class MemoryPressureFault(FaultInjector):
    """Simulate memory pressure by allocating large blocks."""

    def __init__(self, target_mb: int = 512) -> None:
        self._target_mb = target_mb
        self._allocated: list[bytearray] = []

    def inject(self) -> None:
        block_size = 1024 * 1024  # 1 MB blocks
        for _ in range(self._target_mb):
            try:
                self._allocated.append(bytearray(block_size))
            except MemoryError:
                break
        logger.warning("CHAOS: Injected %d MB memory pressure", len(self._allocated))

    def revert(self) -> None:
        self._allocated.clear()
        logger.info("CHAOS: Memory pressure reverted")

    def name(self) -> str:
        return f"memory_pressure_{self._target_mb}mb"


class LatencyFault(FaultInjector):
    """Inject artificial latency into a callable."""

    def __init__(self, delay_ms: float = 500, jitter_ms: float = 100) -> None:
        self._delay_ms = delay_ms
        self._jitter_ms = jitter_ms
        self._active = False
        self._lock = threading.Lock()

    def inject(self) -> None:
        with self._lock:
            self._active = True
        logger.warning("CHAOS: Injected %dms latency (±%dms jitter)",
                        self._delay_ms, self._jitter_ms)

    def revert(self) -> None:
        with self._lock:
            self._active = False
        logger.info("CHAOS: Latency fault reverted")

    def name(self) -> str:
        return f"latency_{self._delay_ms}ms"

    def apply_delay(self) -> None:
        """Call this in the target path to apply the delay."""
        with self._lock:
            if not self._active:
                return
        delay = self._delay_ms + random.uniform(-self._jitter_ms, self._jitter_ms)
        time.sleep(max(0, delay / 1000))


class ErrorInjectionFault(FaultInjector):
    """Inject random errors into a subsystem."""

    def __init__(self, error_rate: float = 0.1,
                 error_type: type[Exception] = RuntimeError) -> None:
        self._error_rate = error_rate
        self._error_type = error_type
        self._active = False
        self._lock = threading.Lock()

    def inject(self) -> None:
        with self._lock:
            self._active = True
        logger.warning("CHAOS: Injected %.0f%% error rate (%s)",
                        self._error_rate * 100, self._error_type.__name__)

    def revert(self) -> None:
        with self._lock:
            self._active = False
        logger.info("CHAOS: Error injection reverted")

    def name(self) -> str:
        return f"error_injection_{self._error_rate}"

    def maybe_raise(self, message: str = "Chaos-injected error") -> None:
        """Call in target path. Raises with configured probability."""
        with self._lock:
            if not self._active:
                return
        if random.random() < self._error_rate:
            raise self._error_type(message)


class ClockSkewFault(FaultInjector):
    """Simulate clock skew by providing skewed timestamps."""

    def __init__(self, skew_seconds: float = 60) -> None:
        self._skew = skew_seconds
        self._active = False
        self._lock = threading.Lock()

    def inject(self) -> None:
        with self._lock:
            self._active = True
        logger.warning("CHAOS: Injected %.0fs clock skew", self._skew)

    def revert(self) -> None:
        with self._lock:
            self._active = False
        logger.info("CHAOS: Clock skew reverted")

    def name(self) -> str:
        return f"clock_skew_{self._skew}s"

    def now(self) -> float:
        """Return a potentially skewed timestamp."""
        with self._lock:
            if self._active:
                return time.time() + self._skew
        return time.time()


@dataclass
class ChaosExperiment:
    """A hypothesis-driven chaos experiment."""
    name: str
    hypothesis: str
    fault: FaultInjector
    duration_s: float = 30.0
    success_criteria: Callable[[dict[str, Any]], bool] | None = None
    collect_metrics: Callable[[], dict[str, Any]] | None = None
    tags: list[str] = field(default_factory=list)


class ChaosFramework:
    """Orchestrates chaos experiments with safety controls.

    Thread-safe. Only one experiment runs at a time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._results: list[ExperimentResult] = []
        self._kill_switch = threading.Event()
        self._max_results = 500

    def run(self, experiment: ChaosExperiment) -> ExperimentResult:
        """Run a chaos experiment with blast radius controls.

        1. Collect baseline metrics
        2. Inject fault
        3. Wait for duration (or kill switch)
        4. Collect post-fault metrics
        5. Revert fault (ALWAYS — even on error)
        6. Evaluate success criteria
        """
        with self._lock:
            if self._running:
                return ExperimentResult(
                    experiment_name=experiment.name,
                    status=ExperimentStatus.ABORTED,
                    hypothesis=experiment.hypothesis,
                    duration_s=0,
                    error="Another experiment is already running",
                )
            self._running = True
            self._kill_switch.clear()

        logger.warning("CHAOS EXPERIMENT START: %s — %s",
                        experiment.name, experiment.hypothesis)

        metrics_before: dict[str, Any] = {}
        metrics_after: dict[str, Any] = {}
        t0 = time.time()

        try:
            # 1. Baseline metrics
            if experiment.collect_metrics:
                try:
                    metrics_before = experiment.collect_metrics()
                except Exception as exc:
                    logger.error("Failed to collect baseline metrics: %s", exc)

            # 2. Inject fault
            experiment.fault.inject()

            # 3. Wait
            self._kill_switch.wait(timeout=experiment.duration_s)

            # 4. Post-fault metrics
            if experiment.collect_metrics:
                try:
                    metrics_after = experiment.collect_metrics()
                except Exception as exc:
                    logger.error("Failed to collect post-fault metrics: %s", exc)

            duration = time.time() - t0

            # 5. Revert (in finally below)

            # 6. Evaluate
            if experiment.success_criteria:
                try:
                    passed = experiment.success_criteria(metrics_after)
                except Exception as exc:
                    logger.error("Success criteria evaluation failed: %s", exc)
                    passed = False
            else:
                passed = True

            status = ExperimentStatus.PASSED if passed else ExperimentStatus.FAILED
            if self._kill_switch.is_set():
                status = ExperimentStatus.ABORTED

            result = ExperimentResult(
                experiment_name=experiment.name,
                status=status,
                hypothesis=experiment.hypothesis,
                duration_s=duration,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )

        except Exception as exc:
            duration = time.time() - t0
            result = ExperimentResult(
                experiment_name=experiment.name,
                status=ExperimentStatus.ERROR,
                hypothesis=experiment.hypothesis,
                duration_s=duration,
                error=str(exc),
            )
        finally:
            # ALWAYS revert the fault
            try:
                experiment.fault.revert()
            except Exception as revert_exc:
                logger.critical("FAILED TO REVERT CHAOS FAULT: %s", revert_exc)

            with self._lock:
                self._running = False
                self._results.append(result)
                if len(self._results) > self._max_results:
                    self._results = self._results[-self._max_results:]

        logger.warning("CHAOS EXPERIMENT END: %s — %s (%.1fs)",
                        experiment.name, result.status.value, result.duration_s)

        return result

    def kill(self) -> None:
        """Emergency kill switch — abort the running experiment."""
        self._kill_switch.set()
        logger.critical("CHAOS KILL SWITCH ACTIVATED")

    def recent_results(self, limit: int = 20) -> list[ExperimentResult]:
        with self._lock:
            return self._results[-limit:]

    def pass_rate(self) -> float:
        with self._lock:
            return self._pass_rate_locked()

    def _pass_rate_locked(self) -> float:
        # Callers must hold self._lock; the lock is non-reentrant, so
        # status() must use this core rather than pass_rate().
        if not self._results:
            return 1.0
        passed = sum(1 for r in self._results
                    if r.status == ExperimentStatus.PASSED)
        return passed / len(self._results)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "total_experiments": len(self._results),
                "pass_rate": round(self._pass_rate_locked(), 3),
                "recent": [r.to_dict() for r in self._results[-5:]],
            }
