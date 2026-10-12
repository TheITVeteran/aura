"""A game picked as one of several has its share of the time, however often its play is handed back to it.

LIVE 2026-10-12 a name machine, the first of three games picked, was handed back four times for "I'll keep playing it",
was given four fresh minutes each time, and held the asking for eighteen minutes before the second game was begun.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_a_game_handed_back_again_has_what_is_left_of_its_share_not_a_fresh_one(monkeypatch):
    from core.skills import sovereign_browser_drawing as drawing

    clock = [1000.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr(drawing, "_SHARE_BEGAN", {})
    url = "https://archive.org/details/a-name-machine"
    assert drawing._the_share_ends(url, 400.0, 240.0) == 1240.0           # first hand-over: one run's time
    clock[0] = 1240.0
    assert drawing._the_share_ends(url, 400.0, 240.0) == 1400.0           # handed back: what is left of the share
    clock[0] = 1400.0
    assert drawing._the_share_ends(url, 400.0, 240.0) == 1400.0           # spent: nothing more
    assert drawing._the_share_ends("https://archive.org/details/another", 400.0, 240.0) == 1640.0   # another game's own
    clock[0] = 1400.0 + 2 * drawing.PLAY_UNTIL_WON_S + 1.0
    assert drawing._the_share_ends(url, 400.0, 240.0) == clock[0] + 240.0  # asked again long after: a share afresh


def test_a_run_that_stopped_just_short_of_its_deadline_ran_out_of_time():
    from core.skills import sovereign_browser_drawing as drawing

    assert 0.0 < drawing.RAN_OUT_WITHIN_S <= drawing.ROUND_S / 4
