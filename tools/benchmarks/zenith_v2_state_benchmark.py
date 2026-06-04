#!/usr/bin/env python3
"""State derivation responsiveness benchmark.

This benchmark lives in tools because it is an operator/proof harness, not a
runtime kernel organ. It measures whether heavy state derivation stays off the
event loop and emits structured JSON so CI or external evaluators can replay
the result without reading console decoration.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.errors import FallbackClassification, Severity, record_degradation  # noqa: E402

logger = logging.getLogger("Aura.ZenithStateBenchmark")

_BENCHMARK_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


@dataclass
class BenchmarkStep:
    name: str
    ok: bool
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class BenchmarkReport:
    ok: bool
    steps: list[BenchmarkStep]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record_benchmark_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "degraded",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "zenith_state_benchmark",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=severity in {"degraded", "critical"},
        extra=extra,
    )


def _default_state_factory() -> Any:
    from core.state.aura_state import AuraState

    return AuraState()


async def measure_state_derivation_overhead(
    *,
    state_factory: Callable[[], Any] | None = None,
) -> BenchmarkStep:
    factory = state_factory or _default_state_factory
    state = factory()
    state.cognition.working_memory = [{"role": "user", "content": "x" * 1000}] * 50
    state.world.known_entities = {f"entity_{index}": {"data": "y" * 100} for index in range(100)}

    start = time.perf_counter()
    new_state = await state.derive_async("benchmark_test", origin="benchmark")
    duration_ms = (time.perf_counter() - start) * 1000.0

    expected_version = state.version + 1
    if new_state.version != expected_version:
        raise RuntimeError(
            f"derive_async returned version {new_state.version}, expected {expected_version}"
        )
    if new_state.transition_origin != "benchmark":
        raise RuntimeError(
            f"derive_async transition origin {new_state.transition_origin!r} did not match benchmark"
        )

    return BenchmarkStep(
        name="state_derivation_overhead",
        ok=True,
        duration_ms=round(duration_ms, 3),
        details={"state_version": state.version, "derived_version": new_state.version},
    )


async def measure_event_loop_lag(
    *,
    state_factory: Callable[[], Any] | None = None,
    iterations: int = 5,
    lag_threshold_ms: float = 20.0,
) -> BenchmarkStep:
    factory = state_factory or _default_state_factory
    state = factory()
    state.cognition.working_memory = [{"role": "user", "content": "x" * 1000}] * 100
    lag_samples: list[float] = []
    max_lag_ms = 0.0
    stop_monitor = asyncio.Event()

    async def monitor_lag() -> None:
        nonlocal max_lag_ms
        while not stop_monitor.is_set():
            before = time.perf_counter()
            await asyncio.sleep(0.01)
            after = time.perf_counter()
            lag_ms = max(0.0, (after - before - 0.01) * 1000.0)
            max_lag_ms = max(max_lag_ms, lag_ms)
            if lag_ms > lag_threshold_ms:
                lag_samples.append(round(lag_ms, 3))
            await asyncio.sleep(0.001)

    monitor = asyncio.create_task(monitor_lag(), name="zenith_state_benchmark_lag_monitor")
    try:
        for index in range(max(1, iterations)):
            await state.derive_async(f"stress_{index}")
        await asyncio.sleep(0.1)
    finally:
        stop_monitor.set()
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor

    return BenchmarkStep(
        name="event_loop_lag",
        ok=not lag_samples,
        duration_ms=0.0,
        details={
            "iterations": max(1, iterations),
            "lag_threshold_ms": lag_threshold_ms,
            "max_lag_ms": round(max_lag_ms, 3),
            "lag_samples": lag_samples[:10],
        },
        error="" if not lag_samples else "event loop lag exceeded threshold",
    )


async def run_zenith_v2_state_benchmark(
    *,
    state_factory: Callable[[], Any] | None = None,
) -> BenchmarkReport:
    steps: list[BenchmarkStep] = []
    try:
        steps.append(await measure_state_derivation_overhead(state_factory=state_factory))
        steps.append(await measure_event_loop_lag(state_factory=state_factory))
    except _BENCHMARK_RECOVERABLE_ERRORS as exc:
        _record_benchmark_degradation(
            exc,
            action="benchmark failed closed and emitted structured failure report",
            extra={"completed_steps": [step.name for step in steps]},
        )
        logger.exception("Zenith state benchmark failed closed: %s", exc)
        steps.append(
            BenchmarkStep(
                name="benchmark_failed_closed",
                ok=False,
                error=str(exc),
                details={"error_type": type(exc).__qualname__},
            )
        )
    return BenchmarkReport(ok=all(step.ok for step in steps), steps=steps)


async def _main() -> int:
    logging.basicConfig(level=logging.INFO)
    report = await run_zenith_v2_state_benchmark()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
