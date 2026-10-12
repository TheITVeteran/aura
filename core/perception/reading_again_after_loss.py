"""Refresh an asynchronous reading after its measured subject is lost.

A completed request is not a permanent answer. Keep one request in flight,
allow brief occlusion, and pace another look by the previous request's cost.
The caller supplies observation validity and whether the reading is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.verify import invariant


@dataclass
class ReadingAgain:
    """A bounded request rate and a measured absence interval, in the caller's clock."""

    least_interval: float = 2.0
    longest_interval: float = 15.0
    loss_after: float = .35
    started: float | None = None
    completed: float | None = None
    missing_since: float | None = None
    next_at: float = -math.inf
    last_at: float = -math.inf

    def due(self, at: float, *, pending: bool, measured: bool, usable: bool, needed: bool) -> bool:
        if not math.isfinite(at) or at < self.last_at:
            return False
        self.last_at = at
        if self.started is not None and self.completed is None and not pending:
            self.completed = at
            self.next_at = at + max(self.least_interval, min(self.longest_interval, at - self.started))
        if usable or not measured or not needed:
            self.missing_since = None
            return False
        if self.missing_since is None:
            self.missing_since = at
        return (not pending and at >= self.next_at
                and (self.started is None or at - self.missing_since >= self.loss_after))

    def began(self, at: float) -> None:
        self.started, self.completed, self.missing_since = at, None, None


@invariant("perception.lost_readings_are_refreshed_without_duplicate_requests", scope="perception",
           owner="core/perception/reading_again_after_loss.py", observational=False)
def _reading_renewal_invariant() -> tuple:
    reading = ReadingAgain()
    absent = dict(measured=True, usable=False, needed=True)
    assert reading.due(1., pending=False, **absent)
    reading.began(1.)
    assert not reading.due(2., pending=True, **absent)
    assert not reading.due(3., pending=False, **absent)
    assert not reading.due(4., pending=False, **absent)
    assert reading.due(5., pending=False, **absent)
    assert not reading.due(6., pending=False, measured=False, usable=False, needed=True)
    return ()
