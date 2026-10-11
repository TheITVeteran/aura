"""Spending a supply: a meter or a count used up by a key, spent where it pays most and saved where it pays least.

Shots that run out, fuel that burns, a special move with a few uses. Spent on whatever comes first, a supply is gone
before the best chances come; never spent, it is wasted. The answer is old (online allocation of a budget, as in the
secretary and prophet problems): spend on a chance only where it is among the best that can be expected in the time
left, as many of them as there is supply for. The bar is set from the chances seen so far: of those still to come in
the time left (as often as they have come), the share there is supply for, counting what refills in that time; the
bar is the value only that share of chances have reached. More supply than chances, and every chance is taken.

Which key spends which counter is found the way a gain is put to a meeting: the counter falls just after the key is
pressed, press after press. Nothing here knows a game, a key or a supply.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

__all__ = ["SpendingASupply"]

#: How soon after a press a counter must fall to have been spent by it, and how many such falls make it the key's.
SPENT_WITHIN_S = 0.4
SPENT_TIMES = 3

#: The time a supply is to last where there is no clock: a run's length, as believed.
LAST_S = 60.0

#: The fewest chances seen before the bar is set from them; until then every chance is taken.
LEAST_CHANCES = 5


@dataclass
class SpendingASupply:
    """Which counter each key spends, how it refills, and the chances it has been spent on or let go."""

    presses: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=50)))
    falls: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=50)))
    rises: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=50)))
    last: dict[str, float] = field(default_factory=dict)
    chances: deque = field(default_factory=lambda: deque(maxlen=200))
    began: float | None = None

    # -- which key spends which counter ----------------------------------------

    def pressed(self, key: str, at: float) -> None:
        self.presses[key].append(at)

    def counted(self, at: float, counters: dict[str, float]) -> None:
        """One reading of the counters: each one's falls and rises."""
        self.began = at if self.began is None else self.began
        for label, value in counters.items():
            before = self.last.get(label)
            if before is not None and value < before:
                self.falls[label].append(at)
            elif before is not None and value > before:
                self.rises[label].append((at, value - before))
            self.last[label] = float(value)

    def spends(self, key: str) -> str | None:
        """The counter this key spends: the one that has fallen just after it was pressed, time after time."""
        found = None
        for label, falls in self.falls.items():
            times = sum(1 for press in self.presses.get(key, ()) if any(0.0 <= fall - press <= SPENT_WITHIN_S for fall in falls))
            if times >= SPENT_TIMES and (found is None or times > found[1]):
                found = (label, times)
        return found[0] if found else None

    def refills_a_second(self, label: str, at: float) -> float:
        """How much the counter has come back by, a second, of itself."""
        span = max(1.0, at - (self.began if self.began is not None else at))
        return sum(by for _when, by in self.rises.get(label, ())) / span

    # -- whether to spend ------------------------------------------------------

    def chance(self, at: float, value: float) -> None:
        """A chance to spend came, worth ``value``, spent or not."""
        self.began = at if self.began is None else self.began
        self.chances.append((at, float(value)))

    def bar(self, at: float, left: float, *, seconds_left: float | None = None, refill: float = 0.0) -> float:
        """The least a chance must be worth to spend on it now, with ``left`` to spend."""
        if len(self.chances) < LEAST_CHANCES:
            return float("-inf")
        span = max(1.0, at - (self.began if self.began is not None else at))
        remaining = seconds_left if seconds_left is not None else LAST_S
        coming = len(self.chances) / span * remaining
        budget = left + refill * remaining
        if coming <= budget:
            return float("-inf")
        share = max(0.0, budget / coming)
        values = sorted(value for _when, value in self.chances)
        return values[min(len(values) - 1, int((1.0 - share) * len(values)))]

    def spend(self, at: float, value: float, left: float, *, seconds_left: float | None = None,
              refill: float = 0.0) -> bool:
        """Whether to spend on a chance worth ``value`` now; the chance is counted either way."""
        bar = self.bar(at, left, seconds_left=seconds_left, refill=refill)
        self.chance(at, value)
        return left > 0 and value >= bar
