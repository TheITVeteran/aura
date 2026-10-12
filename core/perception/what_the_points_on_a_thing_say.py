"""What the points followed on a thing say of it that its outline cannot: how soon it will arrive, and that a thing
found again under another name is the one lost.

The points are the marks the eye holds on to (core/perception/points_that_keep_their_motion.py): corners followed from
picture to picture, each keeping its name. Two things a person reads off them at a glance:

- whether a thing is coming at her, and how soon it will be here. As a thing nears, the marks on it spread apart; at
  the rate they spread, it arrives in the time its size would take to grow without end (the time to contact, tau:
  Lee, 1976). This is what a person ducking reads, and what a locust's looming neuron reads, and it needs no idea of
  how big the thing is or how far away. Its outline says the same less surely: offline 2026-10-12 the outlines of a
  food fight's flying food and turning opponent grew and shrank picture by picture, and something was taken to be
  coming at her nearly all the time.
- that a figure is the same one when its outline changes. A person watching someone crouch does not see a new
  person: the marks on them went on. Offline 2026-10-12 the same opponent was found afresh every few looks as he
  turned and crouched, and was not thrown at as the one who had been there all along.

Nothing here knows a game. Measured in the working pixels of the motion eyes.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from core.verify import invariant

__all__ = ["WhatThePointsSay"]

#: The fewest points followed on a thing from one picture to the next for their spreading to be measured.
FEWEST = 4
#: Over how long a thing's spreading is measured.
SPREADING_OVER_S = 0.25
#: How long the points a lost thing carried are remembered, for it to be found under another name; and how many of
#: them, and what share, a new thing must carry to be it.
KEPT_AFTER_LOST_S = 0.6
CARRIED_FEWEST = 3
CARRIED_SHARE = 0.5


class WhatThePointsSay:
    """For each thing of the motion eyes: how its points have spread, and which points it carried."""

    def __init__(self) -> None:
        #: thing number -> (when, the log of how far its points had spread since they were first measured).
        self.scale: dict[int, deque] = {}
        #: thing number -> (the numbers of the points on it, when it was born, when they were last on it).
        self.carried: dict[int, tuple[frozenset[int], float, float]] = {}
        self._at = -math.inf

    def forget(self) -> None:
        self.scale.clear()
        self.carried.clear()
        self._at = -math.inf

    def see(self, points: Sequence[Any], things: dict[int, Any], at: float) -> dict[int, tuple[int, float]]:
        """Take in this picture's points: each thing's spreading, and the things new to the motion eyes that carry the
        points of one lost or unmatched, as {new number: (the one it is, when that one was born)}."""
        self._at = at
        here = [point for point in points if getattr(point, "at", None) == at]
        found_again: dict[int, tuple[int, float]] = {}
        for number, thing in things.items():
            if getattr(thing, "seen", None) != at:
                self.scale.pop(number, None)
                continue
            on = _on_measured_support(here, thing)
            self._spread(number, [point for point in on if point.pictures >= 2], at)
            names = frozenset(point.number for point in on)
            if number not in self.carried and names:
                lost = self._the_one_it_carries(number, names, things, at)
                if lost is not None:
                    born = min(self.carried[lost][1], float(getattr(thing, "born", at)))
                    found_again[number] = (lost, born)
                    self.scale[number] = self.scale.pop(lost, deque(maxlen=30))
                    del self.carried[lost]
            if names or number not in self.carried:
                born = found_again[number][1] if number in found_again else float(getattr(thing, "born", at))
                self.carried[number] = (names, born, at)
        for number, (_names, _born, last) in list(self.carried.items()):
            if number not in things and at - last > KEPT_AFTER_LOST_S:
                del self.carried[number]
                self.scale.pop(number, None)
        return found_again

    def _spread(self, number: int, followed: list[Any], at: float) -> None:
        if len(followed) < FEWEST:
            self.scale.pop(number, None)
            return
        previous = {getattr(point, "before", None) for point in followed}
        if len(previous) != 1 or None in previous or not 0 < at - next(iter(previous)) <= SPREADING_OVER_S:
            self.scale.pop(number, None)
            return
        now = _spread([(point.x, point.y) for point in followed])
        then = _spread([(point.old_x, point.old_y) for point in followed])
        if now <= 0.0 or then <= 0.0:
            self.scale.pop(number, None)
            return
        history = self.scale.setdefault(number, deque(maxlen=30))
        if history and history[-1][0] != next(iter(previous)):
            history.clear()
        last = history[-1][1] if history else 0.0
        history.append((at, last + math.log(now / then)))

    def _the_one_it_carries(self, number: int, names: frozenset[int], things: dict[int, Any], at: float) -> int | None:
        """The thing, gone or not matched in this picture, whose points a new thing now carries."""
        candidates = []
        for other, (theirs, _born, last) in self.carried.items():
            if other == number or not theirs or not 0 < at - last <= KEPT_AFTER_LOST_S:
                continue
            if other in things and float(getattr(things[other], "seen", at)) >= at:
                continue  # still found as itself
            shared = len(names & theirs)
            if shared >= CARRIED_FEWEST and shared >= CARRIED_SHARE * len(theirs):
                candidates.append(other)
        return candidates[0] if len(candidates) == 1 else None

    def closing_s(self, number: int) -> float | None:
        """How soon the thing will arrive, at the rate its points are spreading: inf where they are not spreading,
        None where too few were followed on it long enough to say."""
        history = self.scale.get(number)
        if not history or len(history) < 3:
            return None
        latest_at, latest = history[-1]
        if latest_at != self._at:
            return None
        then = next((point for point in history if latest_at - point[0] <= SPREADING_OVER_S), None)
        if then is None or latest_at - then[0] <= 0.0:
            return None
        rate = (latest - then[1]) / (latest_at - then[0])
        return 1.0 / rate if rate > 0.0 else math.inf


def _spread(places: list[tuple[float, float]]) -> float:
    mx = sum(x for x, _ in places) / len(places)
    my = sum(y for _, y in places) / len(places)
    return math.sqrt(sum((x - mx) ** 2 + (y - my) ** 2 for x, y in places) / len(places))


def _on_measured_support(points: Sequence[Any], thing: Any) -> list[Any]:
    """Only the object's observed pixels own points; scenery inside its box does not."""
    support = getattr(thing, "support", None)
    if not isinstance(support, np.ndarray) or support.ndim != 2:
        return []
    left, top, _right, _bottom = thing.box()
    return [point for point in points if 0 <= (x := int(round(point.x - left))) < support.shape[1]
            and 0 <= (y := int(round(point.y - top))) < support.shape[0] and support[y, x]]


@invariant("perception.point_meaning_requires_current_owned_support", scope="perception",
           owner="core/perception/what_the_points_on_a_thing_say.py", observational=False)
def _point_meaning_invariant() -> tuple:
    from types import SimpleNamespace

    said = WhatThePointsSay()
    body = SimpleNamespace(born=0., seen=.3, box=lambda: (0., 0., 20., 20.), support=np.zeros((20, 20), bool))
    points = [SimpleNamespace(number=i, x=x, y=y, old_x=x / 2, old_y=y / 2,
                              before=.2, at=.3, pictures=2) for i, (x, y) in enumerate(((2, 2), (8, 2), (2, 8), (8, 8)))]
    said.see(points, {1: body}, .3)
    assert said.closing_s(1) is None and not said.carried[1][0]
    said.carried[2] = (frozenset(range(4)), 0., 0.)
    assert said._the_one_it_carries(3, frozenset(range(4)), {}, 1.) is None
    return ()
