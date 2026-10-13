"""Keep picture age separate from measured delivery and physical response.

A slow reading makes this picture older. It does not make every later input
slow, and an earlier fast reading cannot make this picture current again.
All times here belong to the caller's monotonic clock.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class ControlDelivery:
    picture_at: float
    began: float
    finished: float


@dataclass
class ControlTiming:
    deliveries: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    effects: deque[float] = field(default_factory=lambda: deque(maxlen=12))

    def delivered(self, picture_at: float, began: float, finished: float) -> ControlDelivery | None:
        """An acknowledgement bounds delivery; processing before dispatch is separate."""
        if not all(math.isfinite(t) for t in (picture_at, began, finished)) or not picture_at <= began <= finished:
            return None
        self.deliveries.append(finished - began)
        return ControlDelivery(picture_at, began, finished)

    def took_effect(self, receipt: ControlDelivery, delay_from_picture: float) -> None:
        """Keep only observed response after acknowledgement, without counting delivery twice.

        A visible effect during delivery supplies no extra delay. This method
        needs a transition fitted under the applicable physical model.
        """
        if math.isfinite(delay_from_picture) and delay_from_picture >= 0.:
            self.effects.append(max(0., delay_from_picture - (receipt.finished - receipt.picture_at)))

    def ahead(self, picture_at: float, now: float) -> float | None:
        """Predict from this picture to a new input's effect; unknown delays add no estimate."""
        if not all(math.isfinite(t) for t in (picture_at, now)) or now < picture_at:
            return None
        delivery = statistics.median(self.deliveries) if self.deliveries else 0.
        effect = statistics.median(self.effects) if self.effects else 0.
        return now - picture_at + delivery + effect


def allow_for_picture_age(run: Any, choosing: Any, picture_at: float, now: float) -> None:
    """Refresh immediately before choosing, after work on this frame has finished."""
    timing = getattr(run, "control_timing", None)
    if timing is not None:
        ahead = timing.ahead(picture_at, now)
        if ahead is not None:
            choosing.response_s = ahead


def measure_velocity_response(run: Any, mine: Any, at: float) -> None:
    """Fit a velocity command transition; acceleration controls need a different model."""
    previous, run.previous_control = run.previous_control, None
    if previous is None or mine is None:
        return
    number, x, y, then, before, after = previous[:6]
    dt = at - then
    if mine.number != number or not 0. < dt <= .25:
        return
    change = (before[0] - after[0], before[1] - after[1])
    norm = change[0] ** 2 + change[1] ** 2
    if norm < 20. ** 2:
        return
    error = (mine.x - x - after[0] * dt, mine.y - y - after[1] * dt)
    delay = (error[0] * change[0] + error[1] * change[1]) / norm
    if -.1 * dt <= delay <= 1.1 * dt:
        delay = min(dt, max(0., delay))
        run.motion_responses.append(delay)
        if len(previous) == 7 and previous[6] is not None:
            run.control_timing.took_effect(previous[6], delay)


@invariant("agency.control_delay_keeps_this_pictures_age", scope="agency",
           owner="core/agency/time_from_picture_to_control.py", observational=False)
def _picture_age_invariant() -> tuple:
    timing = ControlTiming()
    receipt = timing.delivered(1., 1.1, 1.12)
    assert receipt is not None
    timing.took_effect(receipt, .15)
    assert math.isclose(timing.ahead(2., 2.2), .25)
    assert math.isclose(timing.ahead(3., 3.01), .06)
    assert timing.delivered(4., 4.1, 4.) is None
    assert timing.ahead(5., 4.) is None
    return ()
