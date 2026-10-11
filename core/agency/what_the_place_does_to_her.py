"""What the place does to her whatever she does: the pressures on a player, learned in play.

Four things press on a player that are not anything they did:

- A pull. Where she does nothing, she still goes somewhere: down, along a belt, with a current, aside in a wind. It
  is how her own thing moved while no key was held, kept by where on the screen it was, since a belt or a current is
  in one part of a place and not another. It is what doing nothing is, in every prediction of the next moment.
- A clock. A counter of time that falls by about a second a second is the time left, and the rate it falls says so,
  whatever its words: "TIME 59", "0:59", a number alone. One that rises is time spent, and presses nothing.
- What she has, draining. A bar of hers that empties with time is a clock too: what is left over how fast it goes.
- More and faster. Things coming faster, and more of them, as a run goes on.

From the clock and the drain, one urgency, nought to one: how much of the time she has had is gone. From the speeds,
how much harder it has got. The real-time player reads both (core/agency/playing_as_it_happens.py): urgency makes a
detour to look at something not yet known cost more and a thing gone for harder to let go of; harder gives what
costs her a wider berth. Nothing here names a game, a counter or a key.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Pressures", "is_a_clock"]

#: Words a counter of time is labelled by.
_CLOCK = re.compile(r"\b(?:time|timer|clock|sec|secs|seconds|countdown|left)\b", re.I)

#: The fewest samples, and how slow, in working pixels a second, a pull must be to be no pull.
LEAST_SAMPLES = 8
NO_PULL_BELOW = 12.0

#: How far apart, in seconds, the readings a rate is taken from must be, and how near a second a second a falling
#: counter must fall to be a clock of seconds.
RATE_OVER_S = 3.0
SECONDS_A_SECOND = (0.5, 2.0)

#: The time a drain is judged against: a bar that will be empty in this long, or less, is pressing.
DRAIN_HORIZON_S = 30.0

#: How long, at the start of a run and at its latest, speeds are compared over, and the most "harder" is taken to be.
RAMP_WINDOW_S = 15.0
HARDEST = 3.0


def is_a_clock(label: str) -> bool:
    """Whether a counter's words say it counts time."""
    return bool(_CLOCK.search(label or ""))


@dataclass
class Pressures:
    """What one game has been seen to do to her, kept across the stretches of a run."""

    #: Her own going with no key held, by cell of the screen, and everywhere.
    by_cell: dict[tuple[int, int], deque] = field(default_factory=dict)
    everywhere: deque = field(default_factory=lambda: deque(maxlen=400))
    cell: float = 0.0
    #: Each counter's readings, (when, value), and the most time left any clock has shown.
    counted: dict[str, deque] = field(default_factory=dict)
    most_left: float = 0.0
    #: How much she has had left, (when, share).
    left: deque = field(default_factory=lambda: deque(maxlen=200))
    #: Others' speeds as seen, (when, speed), and when the run began.
    speeds: deque = field(default_factory=lambda: deque(maxlen=4000))
    began: float | None = None

    def begin_run(self) -> None:
        """A run begun again: its clock and its speeds start afresh; what the place pulls is kept."""
        self.counted.clear()
        self.most_left = 0.0
        self.left.clear()
        self.speeds.clear()
        self.began = None

    def saw(self, *, at: float, mine: Any, held: bool, others: Any = (), counters: dict[str, int] | None = None,
            vitals: float | None = None) -> None:
        """One picture: her thing and whether a key was held, the others, the counters read, how much she has left."""
        self.began = at if self.began is None else self.began
        if mine is not None and not held:
            self.cell = self.cell or max(16.0, 2.0 * max(float(mine.w), float(mine.h)))
            going = (float(mine.vx), float(mine.vy))
            self.everywhere.append(going)
            place = (int(mine.x // self.cell), int(mine.y // self.cell))
            self.by_cell.setdefault(place, deque(maxlen=60)).append(going)
        for thing in others:
            if getattr(thing, "moved", False):
                self.speeds.append((at, math.hypot(float(thing.vx), float(thing.vy))))
        for label, value in (counters or {}).items():
            kept = self.counted.setdefault(label, deque(maxlen=120))
            if not kept or kept[-1][1] != value:
                kept.append((at, float(value)))
        if vitals is not None and (not self.left or self.left[-1][1] != vitals):
            self.left.append((at, float(vitals)))

    # -- a pull ----------------------------------------------------------------

    def pull_at(self, x: float, y: float) -> tuple[float, float]:
        """How she goes at (x, y) when she does nothing, in working pixels a second; nought where nothing pulls."""
        kept = None
        if self.cell:
            kept = self.by_cell.get((int(x // self.cell), int(y // self.cell)))
        if not kept or len(kept) < LEAST_SAMPLES:
            kept = self.everywhere
        if len(kept) < LEAST_SAMPLES:
            return 0.0, 0.0
        pull = statistics.median(v[0] for v in kept), statistics.median(v[1] for v in kept)
        return pull if math.hypot(*pull) >= NO_PULL_BELOW else (0.0, 0.0)

    # -- time ------------------------------------------------------------------

    def seconds_left(self) -> float | None:
        """The fewest seconds left on any clock falling at a second a second, or by a drain of what she has."""
        found = []
        for label, kept in self.counted.items():
            rate = _rate(kept)
            # A counter with words is a clock where its words say time; a number alone, where it falls as seconds do, a
            # second a second and one at a time: a score losing ten every ten seconds falls as fast and is none.
            if rate is None or not (is_a_clock(label) if not label.startswith("number at") else _like_seconds(rate, kept)):
                continue
            if rate < 0 and kept[-1][1] >= 0:
                found.append(kept[-1][1] / -rate)
        drain = _rate(self.left)
        if drain is not None and drain < 0 and self.left[-1][1] > 0:
            found.append(self.left[-1][1] / -drain)
        if not found:
            return None
        left = min(found)
        self.most_left = max(self.most_left, left)
        return left

    def urgency(self) -> float:
        """How pressing time is, nought to one: the share of the time she has had that is gone, or a drain near empty."""
        left = self.seconds_left()
        if left is None:
            return 0.0
        clock = 1.0 - left / self.most_left if self.most_left > 0 else 0.0
        drain = _rate(self.left)
        draining = 0.0
        if drain is not None and drain < 0 and self.left:
            draining = max(0.0, 1.0 - (self.left[-1][1] / -drain) / DRAIN_HORIZON_S)
        return min(1.0, max(0.0, clock, draining))

    # -- more and faster ---------------------------------------------------------

    def harder(self, at: float) -> float:
        """How much faster things go now than at the start of the run, one where no faster; at most HARDEST."""
        if self.began is None or at - self.began < 2 * RAMP_WINDOW_S:
            return 1.0
        early = [s for when, s in self.speeds if when - self.began <= RAMP_WINDOW_S]
        late = [s for when, s in self.speeds if at - when <= RAMP_WINDOW_S]
        if len(early) < LEAST_SAMPLES or len(late) < LEAST_SAMPLES:
            return 1.0
        before = statistics.median(early)
        return min(HARDEST, max(1.0, statistics.median(late) / before)) if before > 0 else 1.0


def _rate(kept: Any) -> float | None:
    """How fast a counter changes, a second, from its readings over at least RATE_OVER_S; None before then."""
    if len(kept) < 3 or kept[-1][0] - kept[0][0] < RATE_OVER_S:
        return None
    recent = list(kept)[-12:]
    times = [t for t, _v in recent]
    middle = statistics.fmean(times)
    spread = sum((t - middle) ** 2 for t in times)
    if spread <= 0:
        return None
    values = [v for _t, v in recent]
    mean = statistics.fmean(values)
    return sum((t - middle) * (v - mean) for t, v in recent) / spread


def _like_seconds(rate: float, kept: Any) -> bool:
    """Whether a counter falls as seconds pass, about one a second and one at a time: a clock whose words were not read."""
    steps = [b[1] - a[1] for a, b in zip(list(kept), list(kept)[1:], strict=False)]
    return SECONDS_A_SECOND[0] <= -rate <= SECONDS_A_SECOND[1] and bool(steps) and statistics.median(steps) == -1
