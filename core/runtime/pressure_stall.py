"""core/runtime/pressure_stall.py — PSI-style pressure stall information.

Clean-room adoption of the Linux kernel's Pressure Stall Information
(`/proc/pressure/{cpu,memory,io}`).

Utilization is the wrong number and always has been. "Memory is at 82%"
does not tell you whether anything is *waiting*; a box at 99% memory with
nothing stalling is fine, and a box at 60% that thrashes is dying. PSI
measures the thing that actually matters — **lost work time** — and splits
it two ways:

* ``some``: at least one worker is stalled on this resource. This is the
  latency signal. Some pressure means somebody waited.
* ``full``: *every* non-idle worker is stalled on this resource, so the
  runtime produced no useful work at all during that window. This is the
  throughput-collapse signal, and it is the one worth acting on: full
  pressure is the difference between "slow" and "stopped".

Aura's incident history is full of things PSI names precisely. The
242MB/h leak soak, the model-lane contention that drops a resident 32B
after ~15 turns, the admission stat-walk stall — each shows up as rising
``full`` pressure on a named resource *before* the failure, which is
exactly the lead time the allostasis engine needs to defer work rather
than crash.

Resources tracked here are Aura's real contended ones, not just the
kernel's three:

    cpu        — compute contention among cognition lanes
    memory     — allocation pressure / reclaim waits
    io         — disk and persistence waits
    inference  — waiting for a model lane / GPU slot
    bus        — event-bus publish backpressure
    lock       — waiting on runtime locks

Averages use the kernel's decay: a fixed 2-second accounting period and
exponential decay over 10s / 60s / 300s windows, so the numbers read the
same way `/proc/pressure` does and the same intuitions transfer.

Usage::

    from core.runtime.pressure_stall import stall, Resource

    with stall(Resource.INFERENCE):
        handle = await lane.acquire()

Cost per stall is two monotonic clock reads and a counter update.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.PSI")

#: Kernel-matching accounting period.
PERIOD_S = 2.0
#: Averaging windows, in seconds.
WINDOWS: tuple[int, ...] = (10, 60, 300)
#: Never advance more than this many periods in one catch-up, so a process
#: that was suspended (laptop lid, SIGSTOP) does not spin.
MAX_CATCHUP_PERIODS = 512


class Resource(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    INFERENCE = "inference"
    BUS = "bus"
    LOCK = "lock"


@dataclass
class _Averages:
    """Exponentially decayed percentages, one per window."""

    values: dict[int, float] = field(default_factory=lambda: {w: 0.0 for w in WINDOWS})

    def advance(self, period_pct: float) -> None:
        for window in WINDOWS:
            decay = math.exp(-PERIOD_S / window)
            self.values[window] = self.values[window] * decay + period_pct * (1.0 - decay)

    def to_dict(self) -> dict[str, float]:
        return {f"avg{w}": round(self.values[w], 3) for w in WINDOWS}


@dataclass
class _ResourceState:
    capacity: int = 1
    stalled: int = 0
    #: monotonic time at which the current SOME/FULL episode began
    some_since: float | None = None
    full_since: float | None = None
    #: total stall-seconds accumulated, all time
    some_total: float = 0.0
    full_total: float = 0.0
    #: stall-seconds accumulated within the current accounting period
    some_period: float = 0.0
    full_period: float = 0.0
    some_avg: _Averages = field(default_factory=_Averages)
    full_avg: _Averages = field(default_factory=_Averages)
    #: peak concurrent waiters seen, for capacity sizing
    peak_stalled: int = 0
    episodes: int = 0


class PressureMonitor:
    """Process-wide PSI accounting. Lock-guarded, lazily advanced."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _ResourceState] = {}
        self._period_start = time.monotonic()

    # ── capacity ──────────────────────────────────────────────────────
    def declare_capacity(self, resource: Resource | str, workers: int) -> None:
        """Declare how many workers can contend for a resource.

        ``full`` pressure requires every worker stalled, so capacity must
        reflect the real parallelism (one model lane, N executor threads).
        Defaults to 1, which makes ``full`` == ``some`` — correct for a
        genuinely serial resource.
        """
        if workers < 1:
            raise ValueError(f"capacity must be >= 1, got {workers}")
        with self._lock:
            self._state(str(resource)).capacity = workers

    def _state(self, key: str) -> _ResourceState:
        state = self._states.get(key)
        if state is None:
            state = _ResourceState()
            self._states[key] = state
        return state

    # ── accounting ────────────────────────────────────────────────────
    def _advance_locked(self, now: float) -> None:
        """Roll completed 2s periods into the decayed averages."""
        elapsed = now - self._period_start
        if elapsed < PERIOD_S:
            return
        periods = int(elapsed // PERIOD_S)
        if periods > MAX_CATCHUP_PERIODS:
            # A long suspension: reset the baseline rather than replay it.
            self._period_start = now
            for state in self._states.values():
                state.some_period = 0.0
                state.full_period = 0.0
            return

        for index in range(periods):
            boundary = self._period_start + (index + 1) * PERIOD_S
            for state in self._states.values():
                some = state.some_period
                full = state.full_period
                # Close open episodes at the period boundary so a long
                # stall is attributed to every period it spans.
                if state.some_since is not None and state.some_since < boundary:
                    some += boundary - max(state.some_since, boundary - PERIOD_S)
                if state.full_since is not None and state.full_since < boundary:
                    full += boundary - max(state.full_since, boundary - PERIOD_S)
                state.some_avg.advance(min(100.0, 100.0 * some / PERIOD_S))
                state.full_avg.advance(min(100.0, 100.0 * full / PERIOD_S))
                state.some_period = 0.0
                state.full_period = 0.0
        self._period_start += periods * PERIOD_S

    def begin_stall(self, resource: Resource | str) -> None:
        key = str(resource)
        now = time.monotonic()
        with self._lock:
            self._advance_locked(now)
            state = self._state(key)
            state.stalled += 1
            state.episodes += 1
            state.peak_stalled = max(state.peak_stalled, state.stalled)
            if state.stalled == 1:
                state.some_since = now
            if state.stalled >= state.capacity and state.full_since is None:
                state.full_since = now

    def end_stall(self, resource: Resource | str) -> None:
        key = str(resource)
        now = time.monotonic()
        with self._lock:
            state = self._state(key)
            if state.stalled <= 0:
                return
            was_full = state.stalled >= state.capacity
            state.stalled -= 1
            if was_full and state.stalled < state.capacity and state.full_since is not None:
                delta = now - state.full_since
                state.full_total += delta
                state.full_period += min(delta, PERIOD_S)
                state.full_since = None
            if state.stalled == 0 and state.some_since is not None:
                delta = now - state.some_since
                state.some_total += delta
                state.some_period += min(delta, PERIOD_S)
                state.some_since = None
            self._advance_locked(now)

    # ── reading ───────────────────────────────────────────────────────
    def pressure(self, resource: Resource | str, *, kind: str = "full", window: int = 10) -> float:
        """Current decayed pressure as a 0..1 fraction."""
        key = str(resource)
        with self._lock:
            self._advance_locked(time.monotonic())
            state = self._states.get(key)
            if state is None:
                return 0.0
            averages = state.full_avg if kind == "full" else state.some_avg
            return averages.values.get(window, 0.0) / 100.0

    def report(self) -> dict[str, Any]:
        with self._lock:
            self._advance_locked(time.monotonic())
            out: dict[str, Any] = {}
            for key, state in sorted(self._states.items()):
                out[key] = {
                    "some": state.some_avg.to_dict() | {"total_s": round(state.some_total, 3)},
                    "full": state.full_avg.to_dict() | {"total_s": round(state.full_total, 3)},
                    "capacity": state.capacity,
                    "stalled_now": state.stalled,
                    "peak_stalled": state.peak_stalled,
                    "episodes": state.episodes,
                }
            return out

    def saturated(self, *, window: int = 10, threshold: float = 0.20) -> list[str]:
        """Resources whose ``full`` pressure exceeds the threshold.

        20% full over 10s means one fifth of recent wall time produced no
        work at all on that resource — well past "busy", into "stalling".
        """
        report = self.report()
        return [
            key
            for key, entry in report.items()
            if entry["full"].get(f"avg{window}", 0.0) / 100.0 >= threshold
        ]

    def narrative(self) -> str:
        report = self.report()
        hot = [
            (key, entry["full"]["avg10"], entry["some"]["avg10"])
            for key, entry in report.items()
            if entry["some"]["avg10"] >= 1.0
        ]
        if not hot:
            return "no resource pressure in the last 10s"
        hot.sort(key=lambda item: item[1], reverse=True)
        return "; ".join(
            f"{key}: {full:.0f}% of wall time fully stalled, {some:.0f}% partially"
            for key, full, some in hot
        )

    def reset_for_test(self) -> None:
        with self._lock:
            self._states.clear()
            self._period_start = time.monotonic()


_MONITOR = PressureMonitor()


def get_pressure_monitor() -> PressureMonitor:
    return _MONITOR


@contextmanager
def stall(resource: Resource | str) -> Iterator[None]:
    """Mark the enclosed wait as a stall on ``resource``."""
    _MONITOR.begin_stall(resource)
    try:
        yield
    finally:
        _MONITOR.end_stall(resource)


def declare_capacity(resource: Resource | str, workers: int) -> None:
    _MONITOR.declare_capacity(resource, workers)


def pressure(resource: Resource | str, *, kind: str = "full", window: int = 10) -> float:
    return _MONITOR.pressure(resource, kind=kind, window=window)


def psi_report() -> dict[str, Any]:
    return _MONITOR.report()


def saturated_resources(*, window: int = 10, threshold: float = 0.20) -> list[str]:
    return _MONITOR.saturated(window=window, threshold=threshold)


def psi_narrative() -> str:
    return _MONITOR.narrative()


def reset_pressure_for_test() -> None:
    _MONITOR.reset_for_test()


__all__ = [
    "PERIOD_S",
    "WINDOWS",
    "PressureMonitor",
    "Resource",
    "declare_capacity",
    "get_pressure_monitor",
    "pressure",
    "psi_narrative",
    "psi_report",
    "reset_pressure_for_test",
    "saturated_resources",
    "stall",
]
