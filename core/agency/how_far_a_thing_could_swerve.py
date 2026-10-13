"""How far a thing could swerve before she can answer it: what to keep clear of it by, beyond where its going takes it.

A thing that goes straight is where its going says, and a berth of a few pixels is enough to pass it. A thing that
wanders is not: offline 2026-10-13, put to the paint world's own code at eight pictures a second with an eighth of a
second before her keys reached it, she passed a bug her straight-line reckoning had going by 1.5 pixels clear, and it
turned into her. The world's code turns a bug by as much as 1.2 radians at a time; in the quarter second she needed to
answer, that is up to 18 pixels off its line.

What a thing could do in the time before she can do anything about it is the part of its going she cannot plan
around, so that is the berth: how much its going has been seen to change at once, for its kind (or, where the place's
code says how fast a kind wanders, that), times the time she needs to answer, a picture and the delay of her hands.
Nothing here names a game.
"""
from __future__ import annotations

import math
from typing import Any

__all__ = ["how_much_its_going_changes", "swerves_of"]

#: The least time between two places a going is reckoned over, and how far apart two goings are compared, in seconds.
RECKONED_OVER_S = 0.1
COMPARED_APART_S = (0.15, 0.6)
#: How far from where it was a thing may be seen, along each axis, in working pixels: what the change between two goings
#: may owe to that and not to its going is four such errors, each along both axes, over the shortest time reckoned.
SEEN_WITHIN_PX = 1.0


def _goings(path: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """A path's goings: (when, across, down) over steps at least ``RECKONED_OVER_S`` long."""
    out: list[tuple[float, float, float]] = []
    start = 0
    for end in range(1, len(path)):
        span = path[end][0] - path[start][0]
        if span >= RECKONED_OVER_S:
            out.append(((path[end][0] + path[start][0]) / 2, (path[end][1] - path[start][1]) / span,
                        (path[end][2] - path[start][2]) / span))
            start = end
    return out


def how_much_its_going_changes(path: list[tuple[float, float, float]]) -> float:
    """The most a thing's going has changed between two of its goings a fraction of a second apart, in pixels a second,
    less what where it was seen can account for; 0 where its path is too short to say."""
    goings = _goings(list(path))
    most = 0.0
    for i, (at, vx, vy) in enumerate(goings):
        for later, wx, wy in goings[i + 1:]:
            apart = later - at
            if apart < COMPARED_APART_S[0]:
                continue
            if apart > COMPARED_APART_S[1]:
                break
            noise = 4.0 * math.sqrt(2.0) * SEEN_WITHIN_PX / RECKONED_OVER_S
            most = max(most, math.hypot(wx - vx, wy - vy) - noise)
    return max(0.0, most)


def swerves_of(things: Any, told: dict[int, float] | None = None) -> dict[int, float]:
    """How much the going of each kind of thing has been seen to change at once, the most of any of its kind, in
    pixels a second; with what the place's code says of a kind (``told``) where that is more."""
    out: dict[int, float] = dict(told or {})
    for thing in getattr(things, "values", lambda: [])():
        path = list(getattr(thing, "path", []) or [])
        if len(path) < 4:
            continue
        kind = int(getattr(thing, "kind", -1))
        out[kind] = max(out.get(kind, 0.0), how_much_its_going_changes(path))
    return out
