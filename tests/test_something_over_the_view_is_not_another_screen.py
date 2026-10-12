"""Something thrown over the view that clears again is not another screen: the room behind it, and the things in it,
are still there.

Offline 2026-10-12 every hit in a food fight covered the view with a splat and shook it, was taken for a new screen,
ended her play and had every figure in the room found afresh. Nothing here is that game: a striped room with a square
going across it, a blot over most of it for a moment, and then a different room.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def _room(square_x: int, *, blot: bool = False, other: bool = False) -> np.ndarray:
    picture = np.zeros((120, 200, 3), np.uint8)
    picture[:, :] = (40, 90, 160) if not other else (200, 180, 40)
    picture[::10] = (220, 220, 220) if not other else (20, 20, 20)
    picture[60:72, square_x:square_x + 12] = (250, 40, 40)
    if blot:
        picture[5:115, 10:190] = (120, 30, 10)
    return picture


def test_a_blot_over_the_view_that_clears_leaves_the_same_screen_and_its_things():
    eyes, at, x = WhatMoves(width=200), 0.0, 20
    for _ in range(20):
        eyes.see(_room(x), at)
        at, x = at + 0.05, x + 2
    square, = [number for number, thing in eyes.things.items()
               if thing.colour[0] > 200 and thing.colour[1] < 80 and thing.seen == eyes._last_at]
    screens = eyes.screens
    happened = eyes.see(_room(x, blot=True), at)
    assert any(h["what"] == "new screen" for h in happened)
    at, x = at + 0.05, x + 2
    happened = eyes.see(_room(x), at)
    assert any(h["what"] == "same screen back" for h in happened)
    assert square in eyes.things and eyes.seeing and eyes.screens == screens + 1
    for _ in range(4):
        at, x = at + 0.05, x + 2
        eyes.see(_room(x), at)
    assert square in eyes.things


def test_another_room_is_another_screen():
    eyes, at, x = WhatMoves(width=200), 0.0, 20
    for _ in range(20):
        eyes.see(_room(x), at)
        at, x = at + 0.05, x + 2
    assert any(h["what"] == "new screen" for h in eyes.see(_room(x, other=True), at))
    at += 0.05
    assert not any(h["what"] == "same screen back" for h in eyes.see(_room(x, other=True), at))
