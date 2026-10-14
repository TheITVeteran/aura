"""How long her throw takes to land is learned from her throw: a splat that stays where she aimed, or the throw itself on
its way there. Not from whatever else comes into sight where she aimed.

On 14 October 2026, in a food fight that draws no splat (tests/fixtures/worlds_that_move/foodfight.html), what came into
sight where she aimed was the other side's food leaving his hand. Her throws were taken to land in 0.25 to 0.67 s where
they took 0.83 s, and she threw about 1.2 times a second where she had thrown 2.7.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.thrown_at_what_shows import TAKES_AT_FIRST_S, HowLongAThrowTakes

pytestmark = pytest.mark.unit


def _thing(x, y, born, vx=0.0, vy=0.0):
    return SimpleNamespace(x=x, y=y, born=born, vx=vx, vy=vy)


def test_what_flies_off_from_where_she_aimed_is_not_her_throw_landing():
    takes = HowLongAThrowTakes()
    for shot in (1.0, 3.0, 5.0):
        takes.thrown(shot, 120.0, 50.0)
        # His food leaves his hand, by where she aimed, and goes down the table at her.
        for after in (0.3, 0.45, 0.6):
            takes.saw([_thing(121.0, 52.0 + 75.0 * (after - 0.3), shot + 0.3, vy=75.0)], shot + after)
    assert takes.landings == [] and takes.seconds() == TAKES_AT_FIRST_S


def test_her_throw_on_its_way_says_how_long_it_will_take():
    takes = HowLongAThrowTakes()
    for shot in (1.0, 3.0, 5.0):
        takes.thrown(shot, 120.0, 35.0)
        # Her throw, from beside her at (120, 135), 120 pixels a second straight up the table: 100 pixels in 0.83 s.
        # It is first seen a tenth of a second after it was thrown, 12 pixels on its way.
        born = shot + 0.1
        takes.saw([_thing(120.0, 135.0 - 12.0, born, vy=-120.0)], born)
        takes.saw([_thing(120.0, 135.0 - 30.0, born, vy=-120.0)], born + 0.15)
    assert takes.seconds() == pytest.approx(100.0 / 120.0, abs=0.02)


def test_a_thing_going_elsewhere_after_her_throw_is_not_her_throw():
    takes = HowLongAThrowTakes()
    for shot in (1.0, 3.0):
        takes.thrown(shot, 120.0, 35.0)
        takes.saw([_thing(40.0, 135.0, shot + 0.1, vx=100.0, vy=-20.0)], shot + 0.2)     # off across the table
    assert takes.landings == []


def test_a_throw_seen_after_her_next_one_is_the_next_one():
    takes = HowLongAThrowTakes()
    takes.thrown(1.0, 120.0, 35.0)
    takes.thrown(1.2, 160.0, 35.0)
    # Come into sight after the second throw and going at where the first was aimed: neither throw's.
    takes.saw([_thing(120.0, 120.0, 1.3, vy=-120.0)], 1.35)
    assert takes.landings == []


def test_a_splat_that_stays_where_she_aimed_is_her_throw_landing():
    takes = HowLongAThrowTakes()
    for shot in (1.0, 3.0):
        takes.thrown(shot, 120.0, 35.0)
        takes.saw([_thing(121.0, 36.0, shot + 0.8)], shot + 0.8)                    # just come: not yet judged
        takes.saw([_thing(121.0, 36.0, shot + 0.8)], shot + 0.95)
    assert takes.seconds() == pytest.approx(0.8)
