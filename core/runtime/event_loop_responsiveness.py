"""Event-loop responsiveness probes shared by boot and proof gates."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[Any]]


@dataclass(frozen=True)
class LoopLagSample:
    index: int
    lag_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "lag_ms": round(float(self.lag_ms), 3)}


@dataclass(frozen=True)
class LoopQuiescenceReport:
    stable: bool
    threshold_ms: float
    max_lag_ms: float
    consecutive_ok: int
    required_consecutive: int
    samples: tuple[LoopLagSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": bool(self.stable),
            "threshold_ms": round(float(self.threshold_ms), 3),
            "max_lag_ms": round(float(self.max_lag_ms), 3),
            "consecutive_ok": int(self.consecutive_ok),
            "required_consecutive": int(self.required_consecutive),
            "samples": [sample.to_dict() for sample in self.samples],
        }


async def sample_event_loop_lag(
    *,
    samples: int = 1,
    interval_s: float = 0.05,
    clock: Clock = time.perf_counter,
    sleeper: Sleeper = asyncio.sleep,
) -> tuple[LoopLagSample, ...]:
    """Measure scheduler lag without doing any blocking work."""

    measured: list[LoopLagSample] = []
    for index in range(max(1, int(samples))):
        before = clock()
        await sleeper(interval_s)
        elapsed = max(0.0, clock() - before)
        lag_ms = max(0.0, (elapsed - interval_s) * 1000.0)
        measured.append(LoopLagSample(index=index, lag_ms=lag_ms))
    return tuple(measured)


async def wait_for_event_loop_quiescence(
    *,
    threshold_ms: float = 250.0,
    required_consecutive: int = 3,
    timeout_s: float = 10.0,
    interval_s: float = 0.05,
    clock: Clock = time.perf_counter,
    sleeper: Sleeper = asyncio.sleep,
) -> LoopQuiescenceReport:
    """Wait until the event loop proves consecutive below-budget samples.

    This does not mask stalls: every warmup sample is returned for artifacts.
    Callers can fail if the loop never stabilizes before the timeout.
    """

    start = clock()
    samples: list[LoopLagSample] = []
    consecutive_ok = 0
    required = max(1, int(required_consecutive))
    threshold = max(0.0, float(threshold_ms))
    timeout = max(interval_s, float(timeout_s))

    while (clock() - start) <= timeout:
        measured = await sample_event_loop_lag(
            samples=1,
            interval_s=interval_s,
            clock=clock,
            sleeper=sleeper,
        )
        lag = measured[0].lag_ms
        samples.append(LoopLagSample(index=len(samples), lag_ms=lag))
        if lag <= threshold:
            consecutive_ok += 1
            if consecutive_ok >= required:
                return LoopQuiescenceReport(
                    stable=True,
                    threshold_ms=threshold,
                    max_lag_ms=max((sample.lag_ms for sample in samples), default=0.0),
                    consecutive_ok=consecutive_ok,
                    required_consecutive=required,
                    samples=tuple(samples),
                )
        else:
            consecutive_ok = 0

    return LoopQuiescenceReport(
        stable=False,
        threshold_ms=threshold,
        max_lag_ms=max((sample.lag_ms for sample in samples), default=0.0),
        consecutive_ok=consecutive_ok,
        required_consecutive=required,
        samples=tuple(samples),
    )
