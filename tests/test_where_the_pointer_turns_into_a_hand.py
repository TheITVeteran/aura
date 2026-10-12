"""Where the pointer turns into a hand is a place that answers a click, found by moving the mouse over the screen."""
from __future__ import annotations

import pytest

from core.perception.where_the_pointer_turns import ACROSS, DOWN, places_the_pointer_turns

pytestmark = pytest.mark.unit


def test_neighbouring_places_where_the_pointer_is_a_hand_are_one_place_at_their_middle():
    doorway = {(0, 3), (0, 4), (0, 5)}                    # the left edge of a room, three cells tall
    drawer = {(7, 6), (8, 6)}
    places = places_the_pointer_turns(doorway | drawer)
    assert len(places) == 2
    assert places[0] == pytest.approx((0.5 / ACROSS, 4.5 / DOWN))
    assert places[1] == pytest.approx((8.0 / ACROSS, 6.5 / DOWN))


def test_a_screen_that_is_a_hand_everywhere_or_nowhere_says_nothing_of_where():
    everywhere = {(c, r) for c in range(ACROSS) for r in range(DOWN)}
    assert places_the_pointer_turns(everywhere) == [] and places_the_pointer_turns(set()) == []
