"""Her eyes for play in a process of their own see what they see here, and stand in for them in every way that is read.

What moves in a picture of play is most of what a picture costs, and in Aura's own process it waits on everything else
she is doing (measured live 2026-10-13: a fifth of the time). Seen in a process of its own it waits on nothing; and
nothing may be seen differently for it. Invented pictures: squares going across a backdrop.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception import seeing_play_elsewhere as elsewhere
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def _pictures(count: int = 40) -> list[tuple[np.ndarray, float]]:
    out = []
    for i in range(count):
        picture = np.full((240, 320, 3), 40, np.uint8)
        picture[20:40, :] = 90                                              # a strip that never moves
        x = 10 + 6 * i
        picture[100:124, x:x + 24] = (220, 60, 60)                          # a red square going right
        y = 200 - 4 * i if i < 30 else 80
        picture[max(0, y):max(0, y) + 16, 250:266] = (60, 200, 90)          # a green one going up, then still
        out.append((picture, i / 20))
    return out


def test_what_is_seen_over_there_is_what_is_seen_here():
    here, there = WhatMoves(), elsewhere.SeenElsewhere()
    try:
        if not there.elsewhere:
            pytest.skip("no process of their own could be started here")
        kept: dict[int, object] = {}
        for picture, at in _pictures():
            assert repr(here.see(picture, at)) == repr(there.see(picture, at))
            assert list(here.things) == list(there.things)
            for number, thing in here.things.items():
                assert repr(vars(thing)) == repr(vars(there.things[number]))
                # A thing seen in two pictures is one object, as it is to the eyes themselves.
                assert kept.setdefault(number, there.things[number]) is there.things[number]
            assert here.shape == there.shape and here.has_looked == there.has_looked
            assert here.share(10, 20) == there.share(10, 20) and len(here.moving()) == len(there.moving())
        assert there.elsewhere
    finally:
        elsewhere.stop_them()


def test_where_the_other_process_goes_away_the_eyes_are_opened_here_and_play_goes_on():
    there = elsewhere.SeenElsewhere()
    try:
        if not there.elsewhere:
            pytest.skip("no process of their own could be started here")
        pictures = _pictures(20)
        for picture, at in pictures[:10]:
            there.see(picture, at)
        elsewhere.stop_them()
        elsewhere._GAVE_UP = True                                           # and it cannot be started again
        for picture, at in pictures[10:]:
            there.see(picture, at)
        assert not there.elsewhere and there.shape[0] > 0            # seeing on, here, from a first look of its own
    finally:
        elsewhere._GAVE_UP = False
        elsewhere.stop_them()


def test_the_eyes_can_be_kept_in_this_process(monkeypatch):
    monkeypatch.setenv("AURA_PLAY_EYES_IN_THIS_PROCESS", "1")
    assert type(elsewhere.eyes_for_play()) is WhatMoves


def test_the_play_loop_opens_its_eyes_this_way():
    import inspect

    from core.agency import playing_as_it_happens

    assert "eyes_for_play(kinds=keep.get(\"kinds\"))" in inspect.getsource(playing_as_it_happens._play_as_it_happens)
