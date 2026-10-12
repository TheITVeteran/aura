"""A thing whose look changes as it goes is seen going, at the pace it goes; a thing told where it is has been measured.

Offline 2026-10-12, with only an outline carried along unchanged taken for going, 96 of every 100 things seen crossing a
food fight's picture were taken to be standing still: anything drawn in turns (a walk, a flap, a spin) changes its
pixels from one picture to the next. And every thing a world told where it was, not tracked from pixels, was taken for
one whose going was never measured, and nothing it did under any key was learned from.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.translation_from_visible_support import motion_was_measured, seen_going
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _a_flapping_thing_crossing(step: int) -> np.ndarray:
    picture = np.full((120, 200, 3), (40, 90, 160), np.uint8)
    x = 20 + 4 * step
    picture[50:66, x:x + 16] = (230, 210, 60)
    if step % 2:                                                  # its wings up, then down: a look of two poses
        picture[50:54, x:x + 16] = (250, 120, 40)
    else:
        picture[62:66, x:x + 16] = (250, 120, 40)
    return picture


def test_a_thing_changing_its_look_as_it_crosses_the_picture_is_seen_going_at_its_pace():
    moves = WhatMoves()
    for step in range(30):
        moves.see(_a_flapping_thing_crossing(step), step / 25)
    going = [thing for thing in moves.things.values() if thing.seen == 29 / 25]
    assert going, "the crossing thing was lost"
    thing = max(going, key=lambda t: abs(t.vx))
    assert seen_going(thing.motion_evidence) and thing.vx == pytest.approx(4 * 25, rel=0.2)   # four pixels a picture


def test_a_thing_told_where_it_is_has_its_going_measured():
    told = Thing(1, 40., 50., 12., 12., np.zeros(64), (200, 200, 200), np.zeros((1, 1, 3), np.uint8), born=0., seen=1.)
    assert motion_was_measured(told, 1.0) and not motion_was_measured(told, 2.0)
