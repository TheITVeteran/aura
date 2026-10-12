"""Where a click is a shot and the pointer aims it, and nothing in the place is her own body: what the shot goes at,
and when she gets out of the way.

Some places put a player nowhere in the picture. She stands this side of it, and what she does goes into it: a
food fight seen over a table, a gallery of targets, a snowball thrown at whoever shows. A person told "take aim with
your mouse and click to throw" does not look for herself there. She throws at whoever has just shown, since they
will not be up for long; failing that, at the biggest thing going across. And where the words give her a key for
getting out of the way, she presses it when something comes at her, which in such a place means it grows.

Nothing here knows a game. A thing is whatever her eyes follow in the picture (core/perception/what_moves_in_the_picture.py).
"""
from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from typing import Any

__all__ = ["A_FIGURE", "JUST_SHOWN_S", "comes_at_her", "what_the_shot_goes_at"]

#: How long after it comes into sight a thing is one that has just shown, in seconds: about as long as one stays up.
JUST_SHOWN_S = 1.5
#: How fast a thing must go to be going somewhere and not standing where it showed, in pixels a second.
GOING = 8.0
#: How much bigger a thing must have got, over the pictures her eyes keep of it, to be coming at her.
GROWN = 1.3
#: The fewest pictures of a thing its growing is judged over.
OVER = 4


#: The least a thing that has just shown is, as a share of the biggest thing that has moved, to be someone showing and
#: not a bit of the picture redrawn (a bar going down, a sparkle).
A_FIGURE = 0.25


def what_the_shot_goes_at(things: Iterable[Any], at: float, *, not_numbers: Collection[int] = (),
                          not_kinds: Collection[int] = (), against_someone: bool = False) -> Any | None:
    """The thing to send the shot at: the biggest of those that have just shown and stand where they showed; where
    none has, the biggest of those going somewhere; and where nothing goes, the biggest that has moved, standing for a
    moment; None where there is none of these. What has just shown is someone only where it is near the size of what
    moves: offline 2026-10-11 a food fight's opponent ran to and fro the whole round, and each throw went at a speck of
    the counter redrawn as her bar went down. ``not_numbers`` and ``not_kinds`` are left out: her own thing, and the
    kinds her own presses bring out."""
    # What grows is coming at her, to be got out of the way of, not thrown at: offline 2026-10-11 her throws went at the
    # food flying at her face.
    mine = [thing for thing in things if thing.number not in not_numbers and thing.kind not in not_kinds
            and not _growing(thing)]
    # Someone gets about; a thing that only sways where it hangs (a lamp, a flag, a sign) is part of the room: offline
    # 2026-10-11 throws went at a swinging lamp while the opponent ran to and fro below it.
    about = [thing for thing in mine if _gets_about(thing)]
    mine = about or mine
    figures = [thing for thing in mine if getattr(thing, "moved", False) or math.hypot(thing.vx, thing.vy) > GOING]
    least = A_FIGURE * max((float(thing.size) for thing in figures), default=0.0)
    shown = [thing for thing in mine if at - float(getattr(thing, "born", at)) <= JUST_SHOWN_S
             and math.hypot(thing.vx, thing.vy) <= GOING and float(thing.size) >= least]
    going = [thing for thing in mine if math.hypot(thing.vx, thing.vy) > GOING]
    # Where the place is played against someone, the one who has been there all along and gets about is who is thrown
    # at, before whatever has just shown (a splat on her own screen, her own throw in flight): offline 2026-10-11 a food
    # fight's opponent ran to and fro the whole round and was hardly thrown at.
    there = [thing for thing in figures if at - float(getattr(thing, "born", at)) >= THERE_ALL_ALONG_S
             and at - float(getattr(thing, "seen", at)) <= SEEN_JUST_NOW_S] if against_someone else []
    for lot in (there, shown, going, figures):
        if lot:
            return max(lot, key=lambda thing: float(thing.size))
    return None


#: How long a thing has been in sight to have been there all along, and how lately it must have been seen to be there now.
THERE_ALL_ALONG_S = 2.0
SEEN_JUST_NOW_S = 0.3


def _growing(thing: Any) -> bool:
    return comes_at_her([thing])


#: How far, in its own sizes, a thing's path must reach for it to get about and not sway where it is.
GETS_ABOUT = 0.5


def _gets_about(thing: Any) -> bool:
    """Whether a thing's path, over the pictures her eyes keep of it, reaches half its own size or more; a thing not yet
    followed long enough to tell is taken to."""
    path = [point for point in (getattr(thing, "path", ()) or ()) if len(point) >= 3]
    if len(path) < OVER:
        return True
    xs, ys = [point[1] for point in path], [point[2] for point in path]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) >= GETS_ABOUT * math.sqrt(max(1.0, float(thing.size)))


def comes_at_her(things: Iterable[Any]) -> bool:
    """Whether something is coming at her out of the picture: it has grown by a third over the last pictures of it,
    and went on growing to the last."""
    for thing in things:
        sizes = [float(size) for size in getattr(thing, "sizes", ())]
        if len(sizes) >= OVER and sizes[-OVER] > 0.0 and sizes[-1] >= GROWN * sizes[-OVER] and sizes[-1] >= sizes[-2]:
            return True
    return False
