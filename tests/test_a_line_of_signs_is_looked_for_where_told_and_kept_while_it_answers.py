"""Where a place's words ask for signs to be followed: when the loop that only answers signs has the screen, when it
gives it back, and what is kept of the line between one stretch and the next.

Offline 2026-10-12, a book of arrows: a line's ink was taken from another screen, nothing was seen on it afterwards,
and it was read that way for the rest of the game; and in a place whose signs are lit in turn, not sent along a line,
a line must not be looked for for ever.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import core.agency.signs_that_come_along as signs_module
import core.skills.screen_pursuit_as_it_happens as reflex_module
from core.skills.screen_pursuit_as_it_happens import (
    A_LINE_LOOKED_FOR,
    _answered_the_signs,
    _signs_may_come,
)

pytestmark = pytest.mark.unit

LINE = {"going": (0.3, 0.0), "on_line": (0.5, 0.8), "ink": (50, 120, 50), "size": 40.0, "shapes": {}}


def _reflexes(**kept) -> SimpleNamespace:
    judged = []
    return SimpleNamespace(keep=dict(kept), look=None, page=None, ends_at=time.monotonic() + 600.0, stretches=[],
                           played_at=0.0, judged=judged, _judge=lambda way, stretch: judged.append(stretch))


def _stretches(monkeypatch, *outcomes: dict) -> list[dict]:
    """Each stretch of the loop comes out as the next of ``outcomes``; what each was given to go by is returned."""
    given: list[dict] = []
    coming = iter(outcomes)

    async def answered(look, hands, *, seconds, known=None, say=None):
        given.append({"known": known})
        return {"seconds": 3.5, "ended": "no signs are coming", "pictures": 100, "answered": 0, "judged": 0,
                "known": None, **next(coming)}

    monkeypatch.setattr(signs_module, "answer_signs_as_they_come", answered)
    monkeypatch.setattr(reflex_module, "_keep_what_she_learned", lambda page, keep: None)
    return given


def test_signs_judged_are_a_stretch_of_play_and_what_was_learned_of_their_line_is_kept(monkeypatch):
    _stretches(monkeypatch, {"judged": 12, "answered": 12, "known": LINE})
    reflexes = _reflexes(signs_told=True)
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is True
    assert reflexes.keep["signs_going"] == LINE and reflexes.keep["signs_seen"] and not reflexes.keep["signs_none"]
    assert reflexes.stretches[0]["gains"] == 12 and reflexes.judged


def test_a_line_is_looked_for_twice_where_told_and_then_the_place_s_signs_are_taken_to_be_shown_another_way(monkeypatch):
    given = _stretches(monkeypatch, {}, {})
    reflexes = _reflexes(signs_told=True)
    assert _signs_may_come(reflexes)
    for _look in range(A_LINE_LOOKED_FOR):
        assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is False
    assert not _signs_may_come(reflexes) and not reflexes.stretches
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is False
    assert len(given) == A_LINE_LOOKED_FOR                       # the third time it is not even looked for


def test_a_known_line_is_waited_on_once_and_forgotten_when_nothing_is_judged_on_it_again(monkeypatch):
    given = _stretches(monkeypatch, {}, {}, {"judged": 3, "answered": 3, "known": LINE})
    reflexes = _reflexes(signs_told=True, signs_seen=True, signs_going=LINE)
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is True       # between two showings: waited
    assert reflexes.keep["signs_going"] == LINE
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is False      # read wrongly, or not there
    assert reflexes.keep["signs_going"] is None
    assert _signs_may_come(reflexes)                             # seen here before: still looked for
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is True
    assert given[2]["known"] is None and reflexes.keep["signs_going"] == LINE


def test_a_sign_held_for_and_never_judged_is_not_a_line_seen(monkeypatch):
    _stretches(monkeypatch, {"answered": 1})
    reflexes = _reflexes(signs_told=True)
    assert asyncio.run(_answered_the_signs(reflexes, time.monotonic())) is False
    assert not reflexes.keep.get("signs_seen") and reflexes.keep["signs_none"] == 1
