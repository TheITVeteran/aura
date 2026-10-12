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


def test_a_screen_of_play_in_a_place_played_in_time_is_neither_swept_nor_waited_on_to_rest():
    """Offline 2026-10-11 a food fight's ten-second round was over before play began, sixteen seconds after it showed,
    spent sweeping the pointer over it and waiting for it to stand still after a click."""
    from types import SimpleNamespace

    from core.skills import screen_pursuit_looking as looking
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, a_screen_of_play_in_time

    fight = SimpleNamespace(words=["TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW. DODGE FOOD BY PRESSING THE SPACE BAR."],
                            keep={}, way_on_shown=False, under_her_hand=False)
    assert a_screen_of_play_in_time(fight, ["MENU", "YOU", "JOHNNY"])
    assert not a_screen_of_play_in_time(fight, ["HOW TO PLAY", "START"])
    calm = SimpleNamespace(words=["Choose a page and drag stickers onto it."], keep={}, way_on_shown=False)
    assert not a_screen_of_play_in_time(calm, ["MENU"])
    token = AS_IT_HAPPENS.set(fight)
    try:
        assert looking._how_long_to_wait() == looking.GOING_ON_ANSWERS_S
    finally:
        AS_IT_HAPPENS.reset(token)
