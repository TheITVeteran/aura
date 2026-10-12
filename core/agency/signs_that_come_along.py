"""Signs that come along a line to a place, each the sign of a key: its key is held as each gets there.

A screen that sends arrows along a strip toward a mark is asking for each one
as it arrives. Nobody is told which part of the strip counts, or how early is
too early. They see marks coming one behind another at one pace, see that each
is drawn as a key's sign, and press what the front one shows as it comes in:
a dance mat, a spell read off a page, notes down a fretboard, parcels on a
belt each stamped with where it goes.

What tells such a stream from anything else that moves is its form. Several
small things go the same way at the same pace, one behind another on one line;
a flock, a volley or a field of falling things is not on one line at one
pace. And each of them is drawn as the sign of a key
(core/perception/keys_drawn_on_screen.py reads an arrow by its taper).

The one furthest along is the one asked for. Its key is held, and no other,
until it is gone; then the next one's. Held, not tapped: a place that looks at
what is down when the sign reaches its mark sees it down, and a place that
wants the press itself to fall at the mark is given one each time the key is
let go for a frame and taken up again while the sign is still in front.

Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.SignsThatComeAlong")

__all__ = ["SignsAreComing", "SignsComing", "the_sign_in", "watching_for_signs"]

#: The widest a sign is, as a share of the picture's width, and the slowest one comes, in picture-widths a second.
SMALL = 0.08
SLOWEST = 0.04

#: How alike two things' going must be to be going together: their speeds within this share of each other, and
#: their ways within this many degrees.
SAME_PACE = 0.25
SAME_WAY_DEG = 15.0

#: How far off the line of the others, in widths of the larger of two, a thing may be and still be on it.
ONE_LINE = 1.5

#: The fewest things going together that are a stream when none was seen before, and how long a stream seen is
#: remembered while nothing of it is in sight, in seconds: one sign alone on the line of the last ones is the next.
STREAM_OF = 2
REMEMBERED_S = 4.0

#: How far a sign's mark stands from its ground (of 255), how much room is left about a thing where its sign is read
#: (in pixels of the watch's own picture), how many readings of one sign must agree, and how many readings of signs in
#: all it takes for a stream to be one of signs.
MARKED = 36.0
ROOM_ABOUT_IT = 2.0
READINGS_AGREE = 2
SIGNS_READ_AT_LEAST = 3

#: How near along the line, as a share of the picture, a thing is to where a sign has got to for it to be that sign;
#: how long a sign out of sight is still taken to be on the line, in seconds; and how many readings of one sign, what
#: share of all its readings, settle what it is, after which it is read no more.
THE_SAME_SIGN_WITHIN = 0.05
GONE_AFTER_S = 0.35
READINGS_SETTLE = 3
SETTLED_SHARE = 0.7

#: How long the key held is let go for, and how often, so that a place wanting the press itself gets one while the
#: sign is in front: one frame of a slow game, a few times a second.
LET_GO_FOR_S = 0.04
PRESSED_AGAIN_EVERY_S = 0.3


class SignsComing:
    """The stream of signs coming, where there is one; which key the front one asks for; and holding it."""

    def __init__(self, known: Any = None) -> None:
        #: The stream's going, picture-widths a second across and down, and a place on its line (shares); when last seen.
        self.going: tuple[float, float] | None = None
        self.on_line: tuple[float, float] | None = None
        self.seen_at = -math.inf
        #: Whether the line was known before this stretch of play: the first sign of the next showing comes alone, a
        #: third of the way to its mark before a second is behind it, and is one of them from the moment it moves.
        self.known_before = bool(known)
        if known:
            self.going, self.on_line = (tuple(known[0]), tuple(known[1]))
        #: The signs on the line now, each where it is along it (shares of the picture, along the stream's going), what
        #: it has been read as, what it is taken to be once its readings agree, and when it was last in sight. Held
        #: here and not by the watch's numbers: a sign swells as it reaches its mark, the watch takes it for a new
        #: thing, and read afresh at that moment it reads as anything.
        self.marks: list[dict[str, Any]] = []
        self.looked_at: float | None = None
        #: The key asked for now, and whether there is a sign at the front.
        self.asked: str = ""
        self.front: bool | None = None
        self.held: str = ""
        self.pressed_at = -math.inf
        self.answered = 0
        self.said = False
        #: How many times a thing of the stream has been read as a key's sign, and when last: a line of things going
        #: together that are drawn as no key's sign (a rank of marchers, a train of carts) asks for nothing.
        self.signs_read = SIGNS_READ_AT_LEAST if known else 0
        self.sign_read_at = -math.inf

    # -- seeing ---------------------------------------------------------------------------------------------------

    def see(self, moves: Any, picture: Any, at: float) -> None:
        """Take in a look: the stream, the front sign, and the key it shows."""
        tall, wide = moves.shape
        if not tall or not wide:
            return
        coming = self._the_stream(moves, wide, tall, at)
        since, self.looked_at = (at - self.looked_at if self.looked_at is not None else 0.0), at
        if self.going is not None:
            pace = math.hypot(*self.going)
            for mark in self.marks:
                mark["along"] += pace * since      # each goes on at the stream's pace whether or not it is in sight
        if coming:
            self.seen_at = at
            for thing in coming:
                self._took_in(thing, picture, moves, at)
        elif at - self.seen_at > REMEMBERED_S and not self.known_before:
            self.going = self.on_line = None
        self.marks = [mark for mark in self.marks if at - mark["seen_at"] <= GONE_AFTER_S]
        front = max(self.marks, key=lambda mark: mark["along"], default=None)
        self.front = True if front is not None else None
        self.asked = (front["is"] or _mostly(front["read"])) if front is not None else ""

    def _took_in(self, thing: Any, picture: Any, moves: Any, at: float) -> None:
        """A thing of the stream, matched to the sign it is by where that sign has got to, or taken as a new one; and
        read, until its readings agree."""
        tall, wide = moves.shape
        across, down = self.going
        pace = math.hypot(across, down)
        along = (thing.x / wide * across + thing.y / tall * down) / pace
        mark = min(self.marks, key=lambda one: abs(one["along"] - along), default=None)
        # As near as a sign can have got since it was last in sight, and never nearer than half the gap between two.
        within = max(THE_SAME_SIGN_WITHIN, 1.5 * pace * (at - mark["seen_at"])) if mark is not None else 0.0
        if mark is None or abs(mark["along"] - along) > within:
            mark = {"along": along, "read": Counter(), "is": "", "seen_at": at}
            self.marks.append(mark)
        mark["along"], mark["seen_at"] = along, at
        if mark["is"]:
            return
        sign = _the_sign_of(thing, picture, moves)
        if not sign:
            return
        mark["read"][sign] += 1
        self.signs_read, self.sign_read_at = self.signs_read + 1, at
        most, times = mark["read"].most_common(1)[0]
        if times >= READINGS_SETTLE and times >= SETTLED_SHARE * sum(mark["read"].values()):
            mark["is"] = most

    def _the_stream(self, moves: Any, wide: int, tall: int, at: float) -> list[Any]:
        """The things coming along one line at one pace: the largest such set, or the one thing on the line of a stream
        seen a moment ago."""
        small = [thing for thing in moves.things.values()
                 if max(thing.w, thing.h) <= SMALL * wide and math.hypot(thing.vx, thing.vy) >= SLOWEST * wide]
        best: list[Any] = []
        for thing in small:
            together = [other for other in small if _go_together(thing, other)]
            if len(together) > len(best):
                best = together
        if len(best) >= STREAM_OF:
            pace = (float(np.median([t.vx for t in best])) / wide, float(np.median([t.vy for t in best])) / tall)
            if self.going is None or self.signs_read < SIGNS_READ_AT_LEAST or _the_same_way(pace, self.going):
                self.going, self.on_line = pace, (best[0].x / wide, best[0].y / tall)
                return best
            # A stream of signs is known, and these go another way: something else on the screen, going together.
        if self.going is None or (at - self.seen_at > REMEMBERED_S and not self.known_before):
            return []
        # One alone, going as the stream went, on its line: the next of them.
        like = SimpleGoing(self.going[0] * wide, self.going[1] * tall, self.on_line[0] * wide, self.on_line[1] * tall)
        return [thing for thing in small if _go_together(like, thing)]

    def known(self) -> Any:
        """The line the signs come along and how they go, for the next stretch of the same game; None where no stream
        of signs has been seen."""
        if self.going is None or self.on_line is None or self.signs_read < SIGNS_READ_AT_LEAST:
            return None
        return (tuple(self.going), tuple(self.on_line))

    def asks(self, at: float) -> bool:
        """Whether a stream of signs is coming now: while it is, nothing else is pressed. A stream, in sight, and its
        things read as signs more than once or twice and lately."""
        return (self.going is not None and self.front is not None and at - self.seen_at <= 2 * PRESSED_AGAIN_EVERY_S
                and self.signs_read >= SIGNS_READ_AT_LEAST and at - self.sign_read_at <= REMEMBERED_S)

    # -- answering ------------------------------------------------------------------------------------------------

    async def hold(self, hands: Any, at: float, say: Callable[[str], Any] | None = None) -> None:
        """Hold the key the front sign shows, and no other; let it go for a frame now and then so the press is new."""
        if self.asked != self.held:
            if self.held:
                await hands.up(self.held)
            self.held = ""
            if self.asked:
                await hands.down(self.asked)
                self.held, self.pressed_at = self.asked, at
                self.answered += 1
                logger.info("a sign for %s is at the front of those coming: holding it", self.asked)
                if say is not None and not self.said and self.answered >= 2:
                    self.said = True
                    say("Signs are coming along a line, each one a key: I'm holding each one's key as it comes to the front.")
        elif self.held and at - self.pressed_at >= PRESSED_AGAIN_EVERY_S:
            await hands.up(self.held)
            await asyncio.sleep(LET_GO_FOR_S)
            await hands.down(self.held)
            self.pressed_at = at

    async def let_go(self, hands: Any) -> None:
        if self.held:
            try:
                await hands.up(self.held)
            except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
                logger.debug("letting go of %s failed: %s", self.held, why)
            self.held = ""

    async def answered_now(self, hands: Any, run: Any, at: float, say: Callable[[str], Any] | None = None) -> bool:
        """For the loop that plays as it happens: where signs are coming, whatever else she held is let go and the
        front one's key is held, and that is all of this moment's acting; where none are, her hold on one is let go.
        Whether signs are coming."""
        if not self.asks(at):
            await self.let_go(hands)
            return False
        if getattr(run, "held", ""):
            await hands.up(run.held)
            run.held = ""
        await self.hold(hands, at, say=say)
        return True


def _the_same_way(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two goings are one way, near enough."""
    pace_a, pace_b = math.hypot(*a), math.hypot(*b)
    if pace_a <= 0 or pace_b <= 0:
        return False
    return (a[0] * b[0] + a[1] * b[1]) / (pace_a * pace_b) >= math.cos(math.radians(2 * SAME_WAY_DEG))


def _mostly(read: Counter[str]) -> str:
    """What a sign not yet settled has mostly been read as, where it has been read the same way twice or only one way."""
    if not read:
        return ""
    most, times = read.most_common(1)[0]
    return most if times >= READINGS_AGREE or len(read) == 1 else ""


class SimpleGoing:
    """A going and a place on its line, standing in for a thing of a stream no longer in sight."""

    def __init__(self, vx: float, vy: float, x: float, y: float) -> None:
        self.vx, self.vy, self.x, self.y, self.w, self.h = vx, vy, x, y, 0.0, 0.0


def _go_together(a: Any, b: Any) -> bool:
    """Whether two things go the same way at the same pace, one on the other's line."""
    pace_a, pace_b = math.hypot(a.vx, a.vy), math.hypot(b.vx, b.vy)
    if pace_a <= 0 or pace_b <= 0 or abs(pace_a - pace_b) > SAME_PACE * max(pace_a, pace_b):
        return False
    turn = math.degrees(math.acos(max(-1.0, min(1.0, (a.vx * b.vx + a.vy * b.vy) / (pace_a * pace_b)))))
    if turn > SAME_WAY_DEG:
        return False
    # How far b is off the line through a along a's going.
    off = abs((b.x - a.x) * a.vy - (b.y - a.y) * a.vx) / pace_a
    return off <= ONE_LINE * max(a.w, a.h, b.w, b.h, 1.0)


def _the_sign_of(thing: Any, picture: Any, moves: Any) -> str:
    """The key a thing is drawn as the sign of, from the picture as it is and not as the watch shrank it: the mark it
    makes against the ground about it, read by its taper; '' where it is no arrow, or cannot be made out."""
    pixels = np.asarray(picture)
    tall, wide = pixels.shape[:2]
    high, across = moves.shape
    scale = wide / max(1, across)
    half_w = max(3, round((thing.w / 2 + ROOM_ABOUT_IT) * scale))
    half_h = max(3, round((thing.h / 2 + ROOM_ABOUT_IT) * scale))
    column, row = round(thing.x * scale), round(thing.y * tall / max(1, high))
    left, top = max(0, column - half_w), max(0, row - half_h)
    return the_sign_in(pixels[top:min(tall, row + half_h), left:min(wide, column + half_w), :3])


def the_sign_in(patch: Any) -> str:
    """The key a patch of picture shows the sign of, or ''. The ground is what most of the patch is painted; the mark
    is the largest single piece of what is unlike it that does not run off the patch. Read against the patch's rim
    instead, a dark arrow on a light strip with a dark edge was taken for ground, and the strip for the mark (offline
    2026-10-12)."""
    from scipy import ndimage

    from core.perception.keys_drawn_on_screen import which_way_it_points

    patch = np.asarray(patch).astype(np.int16)
    if patch.ndim != 3 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return ""
    ground = np.median(patch.reshape(-1, patch.shape[2]), axis=0).astype(np.int16)
    unlike = np.abs(patch - ground).max(axis=2) > MARKED
    labels, count = ndimage.label(unlike)
    best, most = None, 0
    for number, place in enumerate(ndimage.find_objects(labels), 1):
        if place is None:
            continue
        runs_off = (place[0].start == 0 and place[0].stop == patch.shape[0]) or (
            place[1].start == 0 and place[1].stop == patch.shape[1])
        size = int((labels[place] == number).sum())
        if not runs_off and size > most:
            best, most = labels[place] == number, size
    return which_way_it_points(best) if best is not None else ""


class SignsAreComing(Exception):  # noqa: N818 - a thing seen, not a fault
    """Raised out of a watch of the screen made for something else, when signs are seen coming along a line."""


def watching_for_signs(look: Callable[[], Any]) -> Callable[[], Any]:
    """``look``, with every picture it takes also watched for signs coming along a line: raises SignsAreComing, with
    what was learned of the line, the moment they are. A watch for places lighting up in turn is the same looking, and
    a stream of arrows crossing the screen during it is not a showing that has not begun."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    moves, signs = WhatMoves(), SignsComing()

    async def looked() -> Any:
        seen = await look()
        if seen is not None:
            moves.see(seen[0], seen[1])
            signs.see(moves, seen[0], seen[1])
            if signs.asks(seen[1]):
                raise SignsAreComing(signs.known())
        return seen

    return looked
