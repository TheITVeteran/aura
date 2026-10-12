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

__all__ = ["JUST_SHOWN_S", "comes_at_her", "what_the_shot_goes_at"]

#: How long after it comes into sight a thing is one that has just shown, in seconds: about as long as one stays up.
JUST_SHOWN_S = 1.5
#: How fast a thing must go to be going somewhere and not standing where it showed, in pixels a second.
GOING = 8.0
#: How much bigger a thing must have got, over the pictures her eyes keep of it, to be coming at her.
GROWN = 1.3
#: The fewest pictures of a thing its growing is judged over.
OVER = 4


def what_the_shot_goes_at(things: Iterable[Any], at: float, *, not_numbers: Collection[int] = (),
                          not_kinds: Collection[int] = ()) -> Any | None:
    """The thing to send the shot at: the biggest of those that have just shown and stand where they showed; where
    none has, the biggest of those going somewhere; None where there is neither. ``not_numbers`` and ``not_kinds``
    are left out: her own thing, and the kinds her own presses bring out."""
    mine = [thing for thing in things if thing.number not in not_numbers and thing.kind not in not_kinds]
    shown = [thing for thing in mine if at - float(getattr(thing, "born", at)) <= JUST_SHOWN_S
             and math.hypot(thing.vx, thing.vy) <= GOING]
    going = [thing for thing in mine if math.hypot(thing.vx, thing.vy) > GOING]
    for lot in (shown, going):
        if lot:
            return max(lot, key=lambda thing: float(thing.size))
    return None


def comes_at_her(things: Iterable[Any]) -> bool:
    """Whether something is coming at her out of the picture: it has grown by a third over the last pictures of it,
    and went on growing to the last."""
    for thing in things:
        sizes = [float(size) for size in getattr(thing, "sizes", ())]
        if len(sizes) >= OVER and sizes[-OVER] > 0.0 and sizes[-1] >= GROWN * sizes[-OVER] and sizes[-1] >= sizes[-2]:
            return True
    return False
