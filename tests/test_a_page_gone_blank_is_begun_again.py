"""A page whose thing has shown nothing for a while has gone, and is loaded again; a few times at most.

Offline 2026-10-13 a game whose "submit score" went to a server that was not there turned white, and she played the
white page as it happened for four minutes. Invented readings and an invented page here.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import a_page_gone_blank as blank

pytestmark = pytest.mark.unit

_NOTHING = {"ok": True, "text": "", "layout": [], "shapes": [], "objects": []}


def test_a_still_screen_with_nothing_on_it_shows_nothing():
    assert blank.shows_nothing(_NOTHING)
    assert not blank.shows_nothing({**_NOTHING, "text": "TAP TO START"})
    assert not blank.shows_nothing({**_NOTHING, "objects": [{"x": 10, "y": 10, "width": 4, "height": 4}]})
    assert not blank.shows_nothing({**_NOTHING, "shapes": [{"kind": "triangle"}]})
    # A reading that did not come back, or of a screen that was moving, says nothing either way.
    assert not blank.shows_nothing({**_NOTHING, "ok": False})
    assert not blank.shows_nothing({"ok": True, "text": "", "layout": []})


def test_it_has_gone_only_where_every_reading_for_a_while_showed_nothing():
    keep: dict = {}
    assert not blank.gone_blank(keep, _NOTHING, 100.0)
    assert not blank.gone_blank(keep, _NOTHING, 100.0 + blank.GONE_BLANK_S / 2)
    assert not blank.gone_blank(keep, {"ok": False}, 100.0 + blank.GONE_BLANK_S - 1)     # no answer: still blank since
    assert blank.gone_blank(keep, _NOTHING, 100.0 + blank.GONE_BLANK_S)
    # Anything shown starts the while again.
    assert not blank.gone_blank(keep, {**_NOTHING, "text": "LEVEL 2"}, 200.0)
    assert not blank.gone_blank(keep, _NOTHING, 201.0)
    assert not blank.gone_blank(keep, _NOTHING, 201.0 + blank.GONE_BLANK_S - 1)


class _Page:
    url = "http://127.0.0.1/a-game"

    def __init__(self) -> None:
        self.reloads = 0

    async def reload(self, **_kw):
        self.reloads += 1

    async def evaluate(self, *_a, **_kw):
        return None


def test_a_game_gone_blank_is_loaded_again_and_left_once_it_keeps_going_blank(monkeypatch):
    import time

    import core.skills.sovereign_browser_drawing as drawing

    said: list[str] = []
    monkeypatch.setattr(drawing, "_tell", said.append)

    async def started(page, band):
        return band

    monkeypatch.setattr(drawing, "_start_what_is_covered", started)
    page, band, keep, runs = _Page(), (0.0, 0.1, 1.0, 0.9), {"blank_since": 5.0}, []
    for _ in range(blank.BEGUN_AGAIN_AT_MOST):
        run = {"ended": "", "reached_end": True}
        runs.append(run)
        band, again = asyncio.run(drawing._begun_again_where_it_went_blank(page, band, run, runs, keep,
                                                                           time.monotonic() + 60))
        assert again and run["reached_end"] is False and "blank_since" not in keep
    run = {"ended": "", "reached_end": False}
    band, again = asyncio.run(drawing._begun_again_where_it_went_blank(page, band, run, [run], keep, time.monotonic() + 60))
    assert not again and page.reloads == blank.BEGUN_AGAIN_AT_MOST
    assert said[0] == "The game has gone blank, so I'm loading it again." and said[-1].startswith("The game has gone blank again")


def test_the_run_handed_over_is_over_once_its_page_has_gone_blank(monkeypatch):
    from types import SimpleNamespace

    import core.skills.screen_pursuit as pursuit
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS

    clock = [1000.0]
    monkeypatch.setattr(pursuit.time, "monotonic", lambda: clock[0])
    reflexes = SimpleNamespace(keep={}, over_because="", acts_that_changed=0,
                               run_is_over=lambda observation, can_do=None: bool(reflexes.over_because))
    can_do = SimpleNamespace(did_something={})
    token = AS_IT_HAPPENS.set(reflexes)
    try:
        assert not pursuit._a_handed_over_run_is_over(_NOTHING, can_do)
        clock[0] += blank.GONE_BLANK_S
        assert pursuit._a_handed_over_run_is_over(_NOTHING, can_do) and reflexes.over_because == blank.GONE_BLANK
    finally:
        AS_IT_HAPPENS.reset(token)
