"""A moving body keeps its pixels and identity as it crosses another shape."""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.parts_when_things_touch import _joined_parts_invariant
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def test_a_body_crossing_a_mark_does_not_hand_it_its_identity():
    eyes = WhatMoves()
    body = None
    for step in range(80):
        picture = np.full((120, 200, 3), (20, 30, 40), np.uint8)
        picture[50:60, 70:80] = (100, 100, 100)
        x = 20 + max(0, step - 20) * 2
        if step >= 20:
            picture[52:60, x:x + 8] = (40, 220, 80)
        eyes.see(picture, step * .05)
        if step == 20:
            body = next(t.number for t in eyes.things.values() if t.colour == (40, 220, 80))
        if step >= 20:
            assert body in eyes.things
            mine = eyes.things[body]
            assert abs(mine.x - (x + 4)) <= 2
            assert mine.colour == (40, 220, 80)


def test_apart_parts_account_for_the_pixels_and_ambiguous_ones_stay_joined():
    assert _joined_parts_invariant() == ()


def test_a_nearby_thing_of_her_size_needs_her_look_to_continue_her_identity():
    from types import SimpleNamespace

    hers = WhichIsHers()
    look = np.zeros(64)
    look[12] = 1
    body = Thing(1, 50, 50, 8, 8, look, (0, 240, 0), np.zeros((8, 8, 3)), born=0, seen=1, kind=0)
    hers.number = body.number
    hers._remember_own_thing(SimpleNamespace(things={1: body}), 1)
    mark = Thing(2, 52, 50, 8, 8, np.roll(look, 8), (100, 100, 100), body.patch, born=0, seen=1.1, kind=1)
    assert hers._one_of_her_kind(SimpleNamespace(things={2: mark}), 1.1) is None
    # A different view can change its filed kind. Measured look and reach
    # can still continue it; size and reach alone cannot.
    mark.look = look.copy()
    assert hers._one_of_her_kind(SimpleNamespace(things={2: mark}), 1.1) == 2
    mark.look = None
    assert hers._one_of_her_kind(SimpleNamespace(things={2: mark}), 1.1) is None
