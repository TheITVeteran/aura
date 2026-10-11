"""What comes for her: things that turn toward her and follow her, told from things that go their own way.

Everything else on the screen is predicted to go on as it is going (core/perception/what_moves_in_the_picture.py),
which is right for a falling crate and wrong for a guard that has seen her. A thing that comes for her keeps its going
pointed at her as she moves, and is where she will be, not where its line was taking it. Gone round as if it went
straight on, it is walked into.

Told by how well a thing's going lines up with the way to her, picture after picture, while she moves about: a thing
on its own way lines up now and then, by chance; one that comes for her lines up nearly always. Kept by kind too, so
the next of a kind that has come for her is known before it has shown it. Where a thing comes for her, it is predicted
to go at its own pace toward where she will be. Nothing here knows a game or a kind.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatComesForHer"]

#: How fast, in working pixels a second, a thing must go for its going to point anywhere.
GOING = 12.0

#: How many pictures of a thing are read before it is judged, and how well, on average, its going must line up with
#: the way to her to come for her (the cosine of the angle between them).
LEAST_PICTURES = 12
LINES_UP = 0.8

#: How far she must have moved about, in her own sizes, for lining up to say anything: a thing that goes straight at
#: where she stands still lines up whether it follows her or not.
SHE_MOVED = 3.0


@dataclass
class WhatComesForHer:
    """For each thing and each kind, how its going has lined up with the way to her."""

    by_thing: dict[int, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))
    by_kind: dict[int, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=200)))
    #: Where she has been lately, to tell whether she has moved about.
    hers: deque = field(default_factory=lambda: deque(maxlen=90))

    def saw(self, things: Any, mine: Any) -> None:
        """One picture: the things on the screen and hers."""
        if mine is None:
            return
        self.hers.append((float(mine.x), float(mine.y)))
        for thing in things:
            if thing.number == mine.number:
                continue
            vx, vy = float(thing.vx), float(thing.vy)
            dx, dy = float(mine.x) - float(thing.x), float(mine.y) - float(thing.y)
            speed, far = math.hypot(vx, vy), math.hypot(dx, dy)
            if speed < GOING or far < 1.0:
                continue
            lines_up = (vx * dx + vy * dy) / (speed * far)
            self.by_thing[thing.number].append(lines_up)
            self.by_kind[thing.kind].append(lines_up)

    def _she_moved(self, size: float) -> bool:
        if len(self.hers) < 2:
            return False
        xs, ys = [p[0] for p in self.hers], [p[1] for p in self.hers]
        return max(max(xs) - min(xs), max(ys) - min(ys)) >= SHE_MOVED * size

    def comes_for_her(self, thing: Any, size: float = 16.0) -> bool:
        """Whether this thing, or its kind, keeps its going pointed at her as she moves about."""
        if not self._she_moved(size):
            return False
        for kept in (self.by_thing.get(thing.number), self.by_kind.get(thing.kind)):
            if kept is not None and len(kept) >= LEAST_PICTURES and sum(kept) / len(kept) >= LINES_UP:
                return True
        return False

    @staticmethod
    def where_at(thing: Any, after: float, her_x: float, her_y: float) -> tuple[float, float]:
        """Where a thing that comes for her will be ``after`` seconds on, she being at (her_x, her_y) then: gone at its
        own pace toward her, and no further than her."""
        speed = math.hypot(float(thing.vx), float(thing.vy))
        dx, dy = her_x - float(thing.x), her_y - float(thing.y)
        far = math.hypot(dx, dy)
        if far < 1e-6:
            return float(thing.x), float(thing.y)
        step = min(far, speed * max(0.0, after))
        return float(thing.x) + dx / far * step, float(thing.y) + dy / far * step
