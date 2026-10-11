"""What each of her controls lights on the screen, learned from her own presses: which control does again a thing shown.

A screen that shows something to be done again (a book's arrows lit one at a time, buttons that flash in turn, a
dancer's steps) shows it as places changing. To do it again she needs, for each place, the control that does it.
That is the correspondence problem of imitation: what was seen is not yet an act of her own. People and robots solve
it the same way, by learning what their own acts do and turning it round, an inverse model got by trying things.
Here the screen teaches it without asking: a game that shows a sequence lights the place a press is for when it is
pressed. So each press she makes, while doing a showing again or to have one shown again, is a lesson, kept by where
it lit.

Before a place has been lit by one of her presses, how it looks is the guess: an arrow shape is the key it points to.
Where the place is worked by the pointer, the place itself is clicked. Where neither says, a control not yet seen to
light anything is tried there, and its echo is learned.

Nothing here knows a game, a key or a colour.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.WhatAPressLights")

__all__ = ["CLICK", "KEY", "WhatAPressLights", "echo_of", "pressed_and_watched", "watched_for"]

#: How long after a press its echo is looked for: a game lights the place pressed at once and for about a quarter of
#: a second (the test worlds' book of arrows lights a pressed arrow for 0.25 seconds).
ECHO_WITHIN_S = 0.35

#: How unlike, on its most different channel, a pixel must be to have lit.
LIT_BY = 40

#: The share of a place that must have lit for the echo to be that place's.
ECHO_COVERS = 0.3

#: The two kinds of act that do a shown thing again.
KEY, CLICK = "key", "click"

Box = tuple[float, float, float, float]


@dataclass
class WhatAPressLights:
    """Which place, as shares of the picture (left, top, right, bottom), each control has lit when pressed."""

    lights: dict[str, Box] = field(default_factory=dict)
    #: Controls pressed whose echo lit none of the places watched: kept so that they are not tried there again.
    lit_nothing: set[str] = field(default_factory=set)
    #: The places the screen has lit in its showings, where an echo is looked for.
    places: list[Box] = field(default_factory=list)

    def saw_place(self, place: Box) -> None:
        """A place the screen lit in a showing, kept once."""
        if not any(_share_of(place, known) >= 0.5 for known in self.places):
            self.places.append(place)

    def learned(self, control: str, place: Box | None) -> None:
        """What a press of ``control`` lit; None where it lit none of the places in question."""
        if place is None:
            self.lit_nothing.add(control)
            return
        # A place lit by another control before belongs to this one now: the latest echo is the truth of it.
        for other in [c for c, box in self.lights.items() if c != control and _share_of(box, place) >= 0.5]:
            del self.lights[other]
        self.lights[control] = place
        self.lit_nothing.discard(control)
        logger.info("pressing %s lights the place at %s", control, tuple(round(v, 3) for v in place))

    def control_for(self, place: Box) -> str | None:
        """The control known to light this place, where one is."""
        best, most = None, 0.0
        for control, box in self.lights.items():
            share = _share_of(box, place)
            if share > most:
                best, most = control, share
        return best if most >= 0.5 else None

    def untried(self, controls: Sequence[str]) -> list[str]:
        """Controls not yet known to light a place, nor seen to light none."""
        return [c for c in controls if c not in self.lights and c not in self.lit_nothing]

    def act_for(self, place: Box, shape_says: str, *, keys: Sequence[str], clicks: bool) -> tuple[str, Any] | None:
        """The act that does again what lit at ``place``: the control seen to light it; else the key its shape names,
        where that key is one she may press; else a click on it, where the place is worked by the pointer; else a
        control not yet seen to light anything. None where there is nothing to try."""
        known = self.control_for(place)
        if known is not None:
            return KEY, known
        if shape_says and (not keys or shape_says in keys):
            return KEY, shape_says
        if clicks:
            return CLICK, ((place[0] + place[2]) / 2, (place[1] + place[3]) / 2)
        # Untried first; then one seen to light nothing, which may have lit too late to be seen: not one known to light
        # another place.
        untried = self.untried(keys) or [c for c in keys if c not in self.lights]
        return (KEY, untried[0]) if untried else None


def _share_of(a: Box, b: Box) -> float:
    """How much of the smaller of two boxes the other covers."""
    wide = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    tall = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return wide * tall / smaller if smaller > 0 else 0.0


def echo_of(before: Any, after: Sequence[Any], places: Sequence[Box]) -> Box | None:
    """Where a press lit, in the pictures ``after`` it against the picture ``before`` it: the one of ``places`` most of
    whose area changed; else a patch that changed, of about a place's size where places are known, which is a place not
    yet shown. None where nothing did. A press can light a place no showing has lit yet, and that is as much a lesson:
    held to the places already seen, three keys in four were taken to light nothing and were given up on (offline
    2026-10-11)."""
    from core.perception.picture_arithmetic import pieces

    first = np.asarray(before)
    if first.ndim != 3:
        return None
    tall, wide = first.shape[:2]
    changed = np.zeros((tall, wide), bool)
    for picture in after:
        now = np.asarray(picture)
        if now.shape == first.shape:
            changed |= np.abs(now[..., :3].astype(np.int16) - first[..., :3].astype(np.int16)).max(axis=2) > LIT_BY
    if not changed.any():
        return None
    best, most = None, 0.0
    for place in places:
        left, top = int(place[0] * wide), int(place[1] * tall)
        right, bottom = max(left + 1, int(place[2] * wide)), max(top + 1, int(place[3] * tall))
        share = float(changed[top:bottom, left:right].mean())
        if share > most:
            best, most = place, share
    if best is not None and most >= ECHO_COVERS:
        return best
    count, _labels, stats, _centres = pieces(changed)
    usual = float(np.median([(p[2] - p[0]) * wide * (p[3] - p[1]) * tall for p in places])) if places else 0.0
    found = None
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 12 or (usual and not 0.3 * usual <= w * h <= 3.0 * usual):
            continue
        if found is None or area > found[0]:
            found = (area, (x / wide, y / tall, (x + w) / wide, (y + h) / tall))
    return found[1] if found is not None else None


async def watched_for(look: Callable[[], Awaitable[Any]], seconds: float) -> list[Any]:
    """The pictures of the next ``seconds``, by the screen's own clock."""
    after = []
    began = None
    while True:
        seen = await look()
        if seen is None:
            break
        picture, at = seen
        began = at if began is None else began
        after.append(picture)
        if at - began >= seconds:
            break
    return after


async def pressed_and_watched(control: str, hands: Any, look: Callable[[], Awaitable[Any]], places: Sequence[Box],
                              lights: WhatAPressLights, *, watched_for_s: float = ECHO_WITHIN_S) -> Box | None:
    """Press ``control`` (tap it), watch the screen answer for a moment, and learn which of ``places`` it lit."""
    seen = await look()
    before = seen[0] if seen is not None else None
    await hands.tap(control)
    after = await watched_for(look, watched_for_s)
    lit = echo_of(before, after, places) if before is not None else None
    if lit is not None or places:
        # Lit nothing is learned only where there were places to light: a press made before any showing was seen, that
        # lit nothing, says nothing of what it lights.
        lights.learned(control, lit)
    if lit is not None:
        lights.saw_place(lit)
    return lit
