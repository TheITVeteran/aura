"""Keep a visual body guess with the pixels and track that were shown to the eyes.

Offline 2026-10-12 an old visual box named new pieces of scenery each time a
body was lost. A box belongs to one picture. Only an observed continuing
track carries that picture's guess into another one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.agency.causal_identification import within_observed_reach
from core.perception.parts_when_things_touch import same_colour_mix
from core.verify import invariant


@dataclass(frozen=True)
class _SeenTrack:
    number: int
    place: tuple[float, float]
    share: tuple[float, float]
    extent: tuple[float, float]
    velocity: tuple[float, float]
    look: tuple[float, ...]
    at: float


def _track_seen(thing: Any, moves: Any, at: float) -> _SeenTrack | None:
    """Copy one track's measurements; later changes cannot rewrite the earlier picture."""
    needed = ("number", "x", "y", "w", "h", "vx", "vy", "seen", "look")
    if not all(hasattr(thing, name) for name in needed) or not hasattr(moves, "share"):
        return None
    if thing.seen != at:
        return None
    place = float(thing.x), float(thing.y)
    extent = float(thing.w), float(thing.h)
    velocity = float(thing.vx), float(thing.vy)
    look = tuple(float(value) for value in thing.look)
    share = tuple(float(value) for value in moves.share(*place))
    if (len(share) != 2 or min(extent) <= 0 or not all(math.isfinite(value) for value in
            (*place, *extent, *velocity, *share, at)) or same_colour_mix(look, look) is not True):
        return None
    return _SeenTrack(thing.number, place, share, extent, velocity, look, at)


class ABodySeenThen:
    """A guess may follow its observed track, but cannot name a replacement after absence."""

    def __init__(self, moves: Any, at: float) -> None:
        self.at = at
        self.observed = {number: track for number, thing in getattr(moves, "things", {}).items()
                         if (track := _track_seen(thing, moves, at)) is not None}
        self.followed = dict(self.observed)
        self.retired: set[int] = set()
        self.number: int | None = None
        self.bound = False
        self.confirmed_at: float | None = at

    def follow(self, moves: Any, at: float, *, measured: bool) -> None:
        """Absence in a read picture retires a guess; an unread picture leaves it unknown."""
        self.confirmed_at = None
        if not measured:
            return
        for number, before in list(self.followed.items()):
            if number in self.retired:
                continue
            thing = getattr(moves, "things", {}).get(number)
            if thing is None:
                self.retired.add(number)
                continue
            now = _track_seen(thing, moves, at)
            if now is None:
                continue
            original = self.observed[number]
            continuing = (same_colour_mix(original.look, now.look) is True
                          and within_observed_reach(before.place, now.place, extent=before.extent,
                                                    velocity=before.velocity, gap=at - before.at))
            if not continuing:
                self.retired.add(number)
                continue
            self.followed[number] = now
        self.confirmed_at = at

    def bind(self, avatar: Any) -> None:
        """Match the returned box to its captured picture once, even if the answer arrives late."""
        if self.bound:
            return
        self.bound = True
        box = getattr(avatar, "box", None)
        if box is None or len(box) != 4 or not all(math.isfinite(value) for value in box):
            return
        left, top, right, bottom = box
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            return
        middle = (left + right) / 2, (top + bottom) / 2
        margin_x, margin_y = .25 * (right - left), .25 * (bottom - top)
        inside = [track for track in self.observed.values()
                  if left - margin_x <= track.share[0] <= right + margin_x
                  and top - margin_y <= track.share[1] <= bottom + margin_y]
        if inside:
            self.number = min(inside, key=lambda track: math.dist(track.share, middle)).number

    def thing(self, moves: Any, at: float) -> Any | None:
        """The same measured track now, or None when its observation supplies no current identity."""
        if self.number is None or self.number in self.retired or self.confirmed_at != at:
            return None
        followed = self.followed.get(self.number)
        thing = getattr(moves, "things", {}).get(self.number)
        return thing if followed is not None and followed.at == at else None


@invariant("agency.visual_body_guess_belongs_to_its_observed_track", scope="agency",
           owner="core/agency/a_body_seen_then.py", observational=False)
def _body_seen_then_invariant() -> tuple:
    from types import SimpleNamespace

    body = SimpleNamespace(number=1, x=10, y=10, w=8, h=8, vx=0, vy=0, seen=0, look=(1, 0))
    moves = SimpleNamespace(things={1: body}, share=lambda x, y: (x / 100, y / 100))
    guess = ABodySeenThen(moves, 0)
    guess.bind(SimpleNamespace(box=(.05, .05, .15, .15)))
    assert guess.thing(moves, 0) is body
    moves.things = {}
    guess.follow(moves, .1, measured=False)
    assert not guess.retired and guess.thing(moves, .1) is None
    guess.follow(moves, .2, measured=True)
    body.seen = .3
    moves.things = {1: body}
    guess.follow(moves, .3, measured=True)
    assert guess.thing(moves, .3) is None, "a replacement inherited an absent track's visual guess"
    return ()
