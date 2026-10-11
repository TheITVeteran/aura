"""The key that points against the way things are going: the one worth trying first where a place pulls what is hers.

Dropped into a place she knows nothing of, a person does not try her keys in the order they are written. Where the
thing in front of her is falling, she presses up; where it is being carried left, right. She does not yet know the
thing is hers or that the key will answer. But it is the one key that, if it does answer, keeps her alive to find out
the rest, and holding it twice running shows in two seconds whether anything answers to it.

Nothing here knows a game: the biggest moving thing in the picture, the way it goes with nothing held, and the arrow
that points the other way, where that arrow is one of her keys.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = ["FIRST", "against_its_going"]

#: How many holds and rests go first to the key that points against the going, before her keys have their turns.
FIRST = 4
#: The arrow that points against a going: across, then down.
_AGAINST = (("right", "left"), ("down", "up"))


def against_its_going(keys: Sequence[str], things: Iterable[Any], faster_than: float) -> str:
    """The arrow among ``keys`` that points against the way the biggest of ``things`` that is moving, and going faster
    than ``faster_than`` (pixels a second), is going; "" where nothing is going or no key of hers points that way."""
    going = [thing for thing in things if thing.moved and math.hypot(thing.vx, thing.vy) >= faster_than]
    if not going:
        return ""
    biggest = max(going, key=lambda thing: float(thing.w) * float(thing.h))
    axis = 0 if abs(biggest.vx) > abs(biggest.vy) else 1
    key = _AGAINST[axis][0 if (biggest.vx, biggest.vy)[axis] < 0 else 1]
    return key if key in keys else ""
