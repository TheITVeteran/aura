"""Thin a drawing stream only as far as its measured pace allows.

Offline 2026-10-13, taking every fourth compositor frame gave a picture
every 225 ms. The same renderer gave one every 56 ms without thinning.
A frame count is not a time budget: a slower drawing needs fewer skipped
frames. Arrival bursts cannot establish the pace of the drawing.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

from core.verify import invariant


class DrawingCadence:
    """Reduce a sampling stride from consecutive measured source intervals."""

    def __init__(self, stride: int, interval: float) -> None:
        if type(stride) is not int or stride < 1 or not math.isfinite(interval) or interval <= 0:
            raise ValueError('a drawing needs a positive stride and interval')
        self.stride, self.interval = stride, interval
        self._previous: Any = None
        self._intervals: deque[float] = deque(maxlen=3)

    def observe(self, captured: Any) -> None:
        """Only capture bounds from one source clock can teach a source cadence."""
        lower, upper = getattr(captured, 'lower', None), getattr(captured, 'upper', None)
        if (not str(getattr(captured, 'source', '')).startswith('browser-')
                or lower is None or upper is None
                or not all(type(v) in (float, int) and math.isfinite(v) for v in (lower, upper))
                or not 0 <= lower <= upper):
            self._previous = None
            self._intervals.clear()
            return
        previous = self._previous
        if previous is not None:
            same = (captured.source == previous.source and captured.epoch == previous.epoch)
            # Even the latest possible earlier capture must precede the
            # earliest possible new one by more than the sampling budget.
            dt = lower - previous.upper
            if not same or float(captured) <= float(previous) or dt <= 0:
                self._intervals.clear()
            else:
                self._intervals.append(dt)
        self._previous = captured

    def next_stride(self) -> int:
        """One slow interval cannot resample a stream; three must agree."""
        if self.stride <= 1 or len(self._intervals) < 3:
            return self.stride
        fastest = min(self._intervals)
        if fastest <= self.interval:
            return self.stride
        self.stride = max(1, min(self.stride - 1, int(self.stride * self.interval / fastest)))
        self._intervals.clear()
        self._previous = None
        return self.stride


@invariant('perception.drawing_sampling_uses_capture_cadence', scope='perception',
           owner='core/perception/sampling_the_drawings_pace.py', observational=False)
def _drawing_cadence_invariant() -> tuple:
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime

    cadence = DrawingCadence(4, .125)
    for at in (1., 1.225, 1.45, 1.675):
        cadence.observe(MeasuredFrameTime(at, lower=at, upper=at + .001, arrived=3.,
                                         source='browser-wall', source_at=at + 1000., epoch=1))
    assert cadence.next_stride() == 2
    for at in (4., 4.225, 4.45, 4.675):
        cadence.observe(at)
    assert cadence.next_stride() == 2
    return ()
