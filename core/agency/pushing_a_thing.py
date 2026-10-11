"""Pushing a thing to a place: a thing that moves when she runs into it, got where it goes by pushing from its far side.

A crate onto a pad, a ball into a goal, a car into a space. Run at, it goes the way she was going; met from the wrong
side, it goes the wrong way and may end in a corner it cannot be got out of. A player lines up behind it, on the side
away from where it is to go, and pushes it along one way at a time, going round it to get behind it again.

What can be pushed is learned: a thing she was touching, holding a way, that went that way with her, press after
press. Where it goes is learned too: the still thing it was on when a count went up, or the words name it. Nothing
here knows a game, a crate or a pad.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Pushing"]

#: How well a thing's going must line up with hers, how fast it must go, and how many times, to be pushed by her.
ALONG = 0.7
MOVING = 10.0
PUSHES = 3

#: How near, in her sizes, she must be lined up behind a thing to push it; how near it must come to its place; and how
#: far clear of it she keeps while getting behind it, more than a step of hers.
LINED_UP = 0.35
THERE = 0.3
CLEAR = 0.4


def _touching(a: Any, b: Any, slack: float = 2.0) -> bool:
    return (abs(float(a.x) - float(b.x)) <= (float(a.w) + float(b.w)) / 2 + slack
            and abs(float(a.y) - float(b.y)) <= (float(a.h) + float(b.h)) / 2 + slack)


@dataclass
class Pushing:
    """Which kinds she pushes, where each kind goes, and how to get one there."""

    pushed: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    places: dict[int, set] = field(default_factory=lambda: defaultdict(set))

    def saw(self, mine: Any, things: Any, way: tuple[float, float] | None) -> None:
        """One picture: her thing, the others, and the way the key she holds takes her."""
        if mine is None or not way or math.hypot(*way) < 1e-6:
            return
        far = math.hypot(*way)
        for thing in things:
            if thing.number == mine.number or not _touching(thing, mine):
                continue
            speed = math.hypot(float(thing.vx), float(thing.vy))
            if speed >= MOVING and (float(thing.vx) * way[0] + float(thing.vy) * way[1]) / (speed * far) >= ALONG:
                self.pushed[thing.kind] += 1

    def pushable(self, kind: int) -> bool:
        return self.pushed.get(kind, 0) >= PUSHES

    def counted(self, things: Any) -> None:
        """A count went up: each pushable thing on a still thing has found its place."""
        things = list(things)
        for thing in things:
            if not self.pushable(thing.kind):
                continue
            for under in things:
                if under is not thing and not getattr(under, "moved", True) and _touching(thing, under, -1.0):
                    self.places[thing.kind].add(under.kind)

    def told(self, kind: int, place_kind: int) -> None:
        self.places[kind].add(place_kind)

    def where_to_go(self, mine: Any, thing: Any, place: Any) -> tuple[float, float] | None:
        """Where she should make for to get ``thing`` onto ``place``; None where it is there.

        Lined up behind it along a way it still has to go, she pushes on toward where it goes. Else she gets behind it
        along the way it has furthest to go: straight back from it where she is already on that side, else out to a
        lane beside it and along the lane, keeping clear of it all the while so as not to push it the wrong way."""
        size = max(float(mine.w), float(mine.h), 1.0)
        here, it, there = (float(mine.x), float(mine.y)), (float(thing.x), float(thing.y)), (float(place.x), float(place.y))
        ours, its = (float(mine.w), float(mine.h)), (float(thing.w), float(thing.h))
        apart = [abs(there[a] - it[a]) for a in (0, 1)]
        if max(apart) <= THERE * size:
            return None
        clear = CLEAR * size

        def behind(along: int) -> tuple[float, float, float]:
            sign = 1.0 if there[along] > it[along] else -1.0
            line = it[along] - sign * (its[along] / 2 + ours[along] / 2 + clear)
            return sign, line, it[1 - along]

        def lined_behind(along: int) -> bool:
            sign, line, middle = behind(along)
            return abs(here[1 - along] - middle) <= LINED_UP * size and sign * (here[along] - line) <= clear

        ways = [a for a in (0, 1) if apart[a] > THERE * size]
        along = next((a for a in ways if lined_behind(a)), max(ways, key=lambda a: apart[a]))
        sign, line, middle = behind(along)
        goal = [0.0, 0.0]
        if lined_behind(along):
            goal[along] = there[along] - sign * (its[along] / 2 + ours[along] / 2)
            goal[1 - along] = middle
            return goal[0], goal[1]
        if sign * (here[along] - line) <= 0:
            # Behind it already, beside its line: across to the line.
            goal[along], goal[1 - along] = line, middle
            return goal[0], goal[1]
        # In front of it or beside it: out to a lane clear of it, then along the lane to behind it.
        side = 1.0 if here[1 - along] >= it[1 - along] else -1.0
        lane = it[1 - along] + side * (its[1 - along] / 2 + ours[1 - along] / 2 + clear)
        if abs(here[1 - along] - lane) <= LINED_UP * size or side * (here[1 - along] - lane) > 0:
            goal[along], goal[1 - along] = line, here[1 - along]
        else:
            goal[along], goal[1 - along] = here[along], lane
        return goal[0], goal[1]
