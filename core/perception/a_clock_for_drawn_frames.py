"""Keep a drawing's clock separate from the time its message arrives.

Browser frames can queue during a busy event loop and arrive together.
Their capture cadence belongs to the browser. A bracketed clock read maps
that cadence into the caller's monotonic clock without assuming equal origins.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.verify import invariant

logger = logging.getLogger(__name__)


class MeasuredFrameTime(float):
    """A compatible time value with immutable capture bounds and provenance.

    Source-clock values use the earliest possible capture time. A receipt
    without source metadata has no measured lower capture bound. Existing
    float readers can consume both; interval ownership needs these bounds.
    """

    def __new__(cls, at: float, *, lower: float | None, upper: float,
                arrived: float, source: str, source_at: float | None, epoch: int):
        value = super().__new__(cls, at)
        object.__setattr__(value, "lower", lower)
        object.__setattr__(value, "upper", upper)
        object.__setattr__(value, "arrived", arrived)
        object.__setattr__(value, "source", source)
        object.__setattr__(value, "source_at", source_at)
        object.__setattr__(value, "epoch", epoch)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a frame's capture receipt is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a frame's capture receipt is immutable")

    def __reduce__(self):
        """The observation clock keeps its bounds across a private worker boundary."""
        return _restored_frame_time, (float(self), self.lower, self.upper, self.arrived,
                                     self.source, self.source_at, self.epoch)


def _restored_frame_time(at, lower, upper, arrived, source, source_at, epoch):
    return MeasuredFrameTime(at, lower=lower, upper=upper, arrived=arrived,
                             source=source, source_at=source_at, epoch=epoch)


@dataclass(frozen=True)
class _ClockOffset:
    lower: float
    upper: float


def _number(value: Any) -> float | None:
    """A protocol clock needs a finite number; a missing clock stays missing."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


class DrawingClock:
    """One bounded clock correlation for a stream's lifetime."""

    def __init__(self) -> None:
        self.epoch = 0
        self._offsets: dict[str, _ClockOffset] = {}
        self._last_source: dict[str, float] = {}
        self._last_capture = -math.inf
        self._domain: str | None = None

    def correlate(self, domain: str, source: Any, began: float, finished: float, *, resolution: float = 0.) -> bool:
        """The remote read happened somewhere inside its real request interval."""
        value = _number(source)
        if (domain not in ("monotonic", "wall") or value is None
                or not all(math.isfinite(v) for v in (began, finished, resolution))
                or resolution < 0 or not 0 <= began <= finished):
            return False
        offset = _ClockOffset(began - value - resolution, finished - value)
        previous = self._offsets.get(domain)
        if previous is not None:
            # A later query cannot move already emitted frame times.
            return previous == offset
        self._offsets[domain] = offset
        return True

    def reset(self) -> None:
        """A new stream clock cannot continue paths from the old epoch."""
        self.epoch += 1
        self._offsets.clear()
        self._last_source.clear()
        self._last_capture = -math.inf
        self._domain = None

    async def calibrate(self, session: Any, now: Callable[[], float]) -> None:
        """Read supported browser clocks; older adapters keep receipt semantics."""
        await self._read_monotonic(session, now)
        await self._read_wall(session, now)

    async def _read_monotonic(self, session: Any, now: Callable[[], float]) -> None:
        try:
            await asyncio.wait_for(session.send("Performance.enable", {"timeDomain": "timeTicks"}), .35)
            began = now()
            reply = await asyncio.wait_for(session.send("Performance.getMetrics"), .35)
            finished = now()
        except Exception as why:  # noqa: BLE001 - an optional CDP domain may not exist
            logger.debug("the browser monotonic clock is unavailable: %s", why)
            return
        metrics = reply.get("metrics") if isinstance(reply, dict) else None
        if not isinstance(metrics, list):
            return
        value = next((m.get("value") for m in metrics if isinstance(m, dict) and m.get("name") == "Timestamp"), None)
        self.correlate("monotonic", value, began, finished)

    async def _read_wall(self, session: Any, now: Callable[[], float]) -> None:
        try:
            began = now()
            reply = await asyncio.wait_for(session.send("Runtime.evaluate", {
                "expression": "Date.now() / 1000", "returnByValue": True,
            }), .35)
            finished = now()
        except Exception as why:  # noqa: BLE001 - an optional CDP clock read may not exist
            logger.debug("the browser wall clock is unavailable: %s", why)
            return
        result = reply.get("result") if isinstance(reply, dict) else None
        value = result.get("value") if isinstance(result, dict) else None
        # Date.now returns integer milliseconds. The discarded fraction is
        # part of the correlation uncertainty, rather than a newer frame.
        self.correlate("wall", value, began, finished, resolution=1 / 1000)

    def frame(self, metadata: dict[str, Any], arrived: float) -> MeasuredFrameTime | None:
        """Freeze source capture bounds; replay and invalid clocks cannot renew them."""
        if not math.isfinite(arrived) or arrived < 0:
            return None
        present = [(domain, metadata[key]) for domain, key in
                   (("monotonic", "monotonicTimestamp"), ("wall", "timestamp")) if key in metadata]
        if self._domain is not None:
            present = [(domain, raw) for domain, raw in present if domain == self._domain]
            if not present:
                return None
        if not present:
            if self._last_source:
                return None  # a missing clock cannot silently switch a measured stream's time base
            return MeasuredFrameTime(arrived, lower=None, upper=arrived, arrived=arrived,
                                     source="stream-receipt", source_at=None, epoch=self.epoch)
        for domain, raw in present:
            source = _number(raw)
            if source is None or source <= self._last_source.get(domain, -math.inf):
                return None
            if domain in self._offsets:
                return self._mapped(domain, source, arrived)
        return None  # provided source time without a clock correlation is unmeasured

    def _mapped(self, domain: str, source: float, arrived: float) -> MeasuredFrameTime | None:
        offset = self._offsets[domain]
        lower, upper = source + offset.lower, source + offset.upper
        if not 0 <= lower <= arrived or lower <= self._last_capture:
            return None
        # Receipt supplies another upper bound, without changing source cadence.
        upper = min(upper, arrived)
        if upper < lower:
            return None
        self._last_source[domain], self._last_capture = source, lower
        self._domain = domain
        return MeasuredFrameTime(lower, lower=lower, upper=upper, arrived=arrived,
                                 source="browser-" + domain, source_at=source, epoch=self.epoch)


@invariant("perception.queued_frames_keep_the_drawing_clock", scope="perception",
           owner="core/perception/a_clock_for_drawn_frames.py", observational=False)
def _drawing_clock_invariant() -> tuple:
    clock = DrawingClock()
    assert clock.correlate("wall", 1000., 10., 10.02)
    first = clock.frame({"timestamp": 1000.1}, 11.)
    second = clock.frame({"timestamp": 1000.2}, 11.001)
    assert first is not None and second is not None
    assert math.isclose(second - first, .1)
    assert second.upper <= second.arrived and second.lower == float(second)
    assert clock.frame({"timestamp": 1000.2}, 12.) is None
    assert clock.frame({"timestamp": 1010.}, 12.) is None
    return ()
