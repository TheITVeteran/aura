"""The way a place said it is played holds on its later screens, and play that keeps ending on a still screen with
nothing to show for it hands the screen over, though the place goes on in time.

Offline 2026-10-12 "Click on your ball and hold down the mouse button" was read six screens before the course, was no
longer among the words she went by there, and the course was played as a world going by itself, three seconds at a time,
a hundred times over, with not one putt.
"""
from __future__ import annotations

import time

import pytest

from core.agency.the_way_it_is_played import SEND
from core.skills.screen_pursuit_as_it_happens import (
    STILL_STRETCHES_AT_MOST,
    PlayingAsItHappens,
    _left_alone_after,
    _stuck_still,
    _ways_asked_in_this_game,
)

pytestmark = pytest.mark.unit


def _playing() -> PlayingAsItHappens:
    return PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 60)


def test_the_way_a_place_said_holds_on_a_screen_that_says_none():
    playing = _playing()
    assert _ways_asked_in_this_game(playing, []) == []
    assert _ways_asked_in_this_game(playing, [SEND]) == [SEND]          # the putter's lesson
    assert _ways_asked_in_this_game(playing, []) == [SEND]              # the course, its words only a name and a score


def test_stretches_ending_still_with_nothing_to_show_hand_the_screen_over():
    playing = _playing()
    still = {"hers": "", "gains": 0, "losses": 0, "ended": "nothing on the screen has moved for a while"}
    for _ in range(STILL_STRETCHES_AT_MOST - 1):
        _left_alone_after(playing, still)
    assert not _stuck_still(playing)
    _left_alone_after(playing, still)
    assert _stuck_still(playing)
    _left_alone_after(playing, {**still, "gains": 1, "ended": "the screen says the round is over"})
    assert not _stuck_still(playing), "a stretch that got somewhere starts the count again"
