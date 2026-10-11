"""A moving screen that answers to nothing is left alone twice as long each time the same screen does it again.

LIVE 2026-10-11 a building game's characters idled on its board, and every twenty seconds she spent nineteen finding
again that nothing there answered to her.
"""
from __future__ import annotations

import time

import pytest

from core.skills.screen_pursuit_as_it_happens import (
    LEAVE_A_MOVING_MENU_S,
    LEFT_ALONE_AT_MOST_S,
    PlayingAsItHappens,
    _left_alone_for,
)

pytestmark = pytest.mark.unit


def test_the_same_screen_is_left_longer_each_time_and_a_new_one_starts_afresh():
    reflexes = PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="win it", ends_at=time.monotonic() + 600)
    reflexes.said_now = "drop add a part score"
    waits = [_left_alone_for(reflexes) for _ in range(6)]
    assert waits[:3] == [LEAVE_A_MOVING_MENU_S, 2 * LEAVE_A_MOVING_MENU_S, 4 * LEAVE_A_MOVING_MENU_S]
    assert waits[-1] == LEFT_ALONE_AT_MOST_S
    reflexes.said_now = "level two pick your parts"
    assert _left_alone_for(reflexes) == LEAVE_A_MOVING_MENU_S
