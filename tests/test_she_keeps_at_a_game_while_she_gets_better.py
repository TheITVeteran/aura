"""She keeps at a game while she is getting better at it, and says so and leaves it when she is not.

LIVE 2026-10-07 she played one game for twelve minutes and another for thirteen, each round lost as fast as the last.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser_drawing import not_getting_better

pytestmark = pytest.mark.unit


def _r(gains, took, ended="lost"):
    return {"gains": gains, "took_s": took, "ended": ended}


@pytest.mark.parametrize(("runs", "leave"), [
    ([_r(0, 20), _r(0, 21)], False),                        # too few rounds to say
    ([_r(0, 20), _r(0, 19), _r(0, 21)], True),              # three rounds, the last two no better
    ([_r(0, 20), _r(0, 19), _r(0, 30)], False),             # lasting much longer is better
    ([_r(1, 20), _r(0, 40), _r(2, 15)], False),             # gaining more is better
    ([_r(0, 20), _r(0, 19), _r(0, 18, "won")], False),      # a round won is better than any
    ([_r(3, 60), _r(0, 30), _r(1, 50), _r(2, 55)], True),   # the best was early, and not reached again
])
def test_whether_she_is_still_getting_better(runs, leave):
    assert not_getting_better(runs) is leave


@pytest.mark.parametrize(("goal", "many"), [
    ("Play this game and win it. (It is “Tunnel Rush”, the first of the 3 picked.)", 3),
    ("Play this game and win it.", 1),
    ("Cook this recipe. (It is “Soup”, the second of the 2 picked.)", 2),
])
def test_one_of_several_things_asked_for_has_its_share_of_the_time(goal, many):
    from core.skills.sovereign_browser_drawing import (
        LEAST_SHARE_S,
        PLAY_UNTIL_WON_S,
        _how_many_asked,
    )

    assert _how_many_asked(goal) == many
    assert max(LEAST_SHARE_S, PLAY_UNTIL_WON_S / many) <= PLAY_UNTIL_WON_S


def test_when_her_time_runs_out_she_says_how_far_she_got():
    """Offline 2026-10-12 games that only keep a score were left with "its share of the time" and nothing of the 2400
    points on the screen; a person leaving a game says how far they got."""
    from core.skills.sovereign_browser_drawing import _where_it_stood

    assert _where_it_stood([{"reached_end": False, "score": 2400}]) == " My run was still going, at 2400."
    assert _where_it_stood([{"reached_end": False, "score": None, "gains": 3}]) == " My run was still going, at 3 by my own count."
    said = _where_it_stood([{"reached_end": True, "score": 120}, {"reached_end": False, "score": 300}])
    assert "My best was run 1, with 120" in said and said.endswith("The run in play stood at 300 when the time ran out.")
    assert _where_it_stood([{"reached_end": False, "score": None}]) == ""
    assert _where_it_stood([]) == ""
