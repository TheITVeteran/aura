"""What a moving thing leaves behind is the scene, not the thing; and an arrow moves what is hers the way it points.

LIVE 2026-10-07 a game's painter left a trail of paint, the trail and the painter were followed as one thing that
only grew (31 pixels wide at rest, 143 by 126 after a few seconds), its middle barely moved under the keys, and in
thirty seconds no key was found to move anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.agency.which_one_answers_to_her import WhichIsHers, _Speeds, gone_the_ways_of_the_keys
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

FPS = 30


def _frames(draw, seconds):
    for n in range(int(seconds * FPS)):
        at = n / FPS
        picture = np.full((360, 480, 3), (210, 200, 160), np.uint8)
        draw(picture, at)
        yield picture, at


@pytest.mark.parametrize("trail", [(150, 60, 170), (40, 40, 40)])  # a paint and a pen's ink: what is left behind is not the point
def test_a_thing_that_leaves_a_trail_is_followed_at_its_own_size(trail):
    def draw(picture, at):
        if at < 0.8:
            return  # the scene alone, as it is first looked at
        x = 40 + int((at - 1.2) * 120) if at > 1.2 else 40
        if at > 1.2:
            picture[175:185, 40:x] = trail                                      # what it has left behind
        picture[160:200, x:x + 30] = (230, 120, 30)                             # the thing itself
    moves = WhatMoves()
    for picture, at in _frames(draw, 3.5):
        moves.see(picture, at)
    moving = [t for t in moves.moving(faster_than=8.0) if t.h > 10 * moves.scale]
    assert moving, "the thing is not seen to move"
    thing = max(moving, key=lambda t: t.size)
    assert thing.w < 3 * 30 * moves.scale, (thing.w, 30 * moves.scale)  # at its own size, give or take: not the whole trail
    assert thing.vx > 60 * moves.scale  # and going at its own speed: 120 pixels a second


def test_a_thing_at_rest_does_not_fade_into_the_scene():
    def draw(picture, at):
        if at < 0.8:
            return
        x = 40 + int((min(at, 1.6) - 0.8) * 100)
        picture[160:200, x:x + 30] = (230, 120, 30)
    moves = WhatMoves()
    for picture, at in _frames(draw, 4.0):
        moves.see(picture, at)
    assert any(abs(t.w - 30 * moves.scale) < 6 and abs(t.h - 40 * moves.scale) < 6 for t in moves.things.values()), moves.things


def test_a_thing_waiting_where_something_reaches_it_is_still_the_thing():
    """Measured offline 2026-10-10: a basket waiting under a falling egg grew as the egg reached it, its still body went
    into the scene, and she lost her own basket (37 caught a minute on 3 October, 10 since)."""
    def draw(picture, at):
        if at < 0.8:
            return
        x = 200 + int((min(at, 1.4) - 0.8) * 100)                              # the basket goes, then waits
        picture[300:330, x:x + 50] = (120, 70, 30)
        drop = (at - 1.4) % 1.0
        if at > 1.4:
            y = 100 + int(drop * 220)                                         # an egg falls into it, again and again
            picture[y:y + 14, x + 18:x + 30] = (250, 250, 240)
    moves = WhatMoves()
    for picture, at in _frames(draw, 5.0):
        moves.see(picture, at)
    baskets = [t for t in moves.things.values() if abs(t.w - 50 * moves.scale) < 10 and t.h >= 20 * moves.scale]
    assert baskets, [(t.w, t.h) for t in moves.things.values()]


def _moves_with(things):
    class _Thing:
        def __init__(self, number):
            self.number, self.kind, self.moved, self.x, self.y = number, 0, True, 50.0, 50.0

    class _Moves:
        pass
    seen = _Moves()
    seen.things = {n: _Thing(n) for n in things}
    return seen


def _pressed(hers, number, key, vx, vy, n):
    """Three pictures of a press, each where its going took it: a speed is seen as a thing going from place to place."""
    speeds = hers._by_thing.setdefault(number, _Speeds())
    for step in range(3):
        at = float(n) + step / FPS
        speeds.add(key, vx, vy, press=float(n), position=(50.0 + vx * at, 50.0 + vy * at))


def _rested(hers, number, vx=0.0, vy=0.0, n=100):
    """The rests between a trial's presses: what her keys move stands still in them."""
    for at in (n, n + 1):
        _pressed(hers, number, "", vx, vy, float(at))


@pytest.mark.parametrize(("presses", "chosen"), [
    # Hers went right, down and left as pressed; another went its own way.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5)], 2: [("right", -40, 30), ("down", 50, -20), ("left", 30, 60)]}, 1),
    # Two keys only: not enough to say.
    ({1: [("right", 90, 0), ("left", -95, 5)]}, None),
    # One key against: not hers by this.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5), ("up", 0, 70)]}, None),
    # Two that both agree: neither.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5)], 2: [("right", 60, 5), ("down", 3, 70), ("left", -80, -4)]}, None),
    # Held against a wall under one key says nothing either way.
    ({1: [("right", 90, 0), ("down", 0, 80), ("left", -95, 5), ("up", 0, -3)]}, 1),
])
def test_what_goes_the_way_each_arrow_points_is_hers(presses, chosen):
    hers = WhichIsHers()
    n = 0
    for number, keys in presses.items():
        for key, vx, vy in keys:
            n += 1
            _pressed(hers, number, key, vx, vy, n)
        _rested(hers, number, *((40, -30) if number == 2 and chosen == 1 else (0, 0)), n=100 + 10 * number)
    assert gone_the_ways_of_the_keys(hers, _moves_with(presses)) == chosen


def test_what_goes_on_going_while_she_rests_is_not_hers_however_it_went_under_her_keys():
    """Measured offline 2026-10-10: in a Pong of the worked-on worlds she took the ball for herself in two runs of three,
    a bouncing ball having gone three keys' ways by chance."""
    hers = WhichIsHers()
    for n, (key, vx, vy) in enumerate((("right", 90, 10), ("down", 10, 80), ("left", -95, 5)), start=1):
        _pressed(hers, 1, key, vx, vy, n)
    _rested(hers, 1, 70, -60)                                               # it goes on going between her presses
    assert gone_the_ways_of_the_keys(hers, _moves_with([1])) is None


def test_in_a_world_that_scrolls_what_holds_its_place_against_it_is_hers():
    """LIVE 2026-10-07 a platformer's hero stood mid-picture while everything else went by against the key."""
    hers = WhichIsHers()
    n = 0
    for key, world, hero in (("right", (-80, 0), (2, 0)), ("left", (85, 0), (-3, 0)), ("up", (0, 0), (0, -90))):
        n += 1
        for number in (2, 3, 4, 5):  # barrels, crates, trees: the view moving
            _pressed(hers, number, key, world[0] + number, world[1], n)
        _pressed(hers, 1, key, hero[0], hero[1], n)
    for number in (1, 2, 3, 4, 5):
        _rested(hers, number, n=100 + 10 * number)
    assert gone_the_ways_of_the_keys(hers, _moves_with([1, 2, 3, 4, 5])) == 1


def test_a_world_that_does_not_move_as_one_is_not_taken_for_the_view():
    """Things going every which way are things, not the view: each is measured as it went."""
    hers = WhichIsHers()
    n = 0
    for key, mine in (("right", (90, 0)), ("down", (0, 80)), ("left", (-95, 5))):
        n += 1
        for number, own in ((2, (60, 40)), (3, (-50, 30)), (4, (10, -70)), (5, (-30, -40))):
            _pressed(hers, number, key, *own, n)
        _pressed(hers, 1, key, *mine, n)
    _rested(hers, 1)
    for number, own in ((2, (60, 40)), (3, (-50, 30)), (4, (10, -70)), (5, (-30, -40))):
        _rested(hers, number, *own, n=100 + 10 * number)
    assert gone_the_ways_of_the_keys(hers, _moves_with([1, 2, 3, 4, 5])) == 1
