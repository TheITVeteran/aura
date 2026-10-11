"""A page whose player puts its drawing up late is asked again, not given up on.

Offline 2026-10-10 a game's page was asked where it draws before its player had made its drawing, and she gave it up as
a page that no longer draws.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.skills import sovereign_browser_drawing as drawing

pytestmark = pytest.mark.unit


class _Page:
    def __init__(self, ready_after: int):
        self.asked, self.ready_after = 0, ready_after

    async def evaluate(self, _script):
        self.asked += 1
        if self.asked < self.ready_after:
            return ""
        return json.dumps({"left": 0.2, "top": 0.1, "right": 0.8, "bottom": 0.7})


def test_the_drawing_is_found_once_the_player_has_put_it_up(monkeypatch):
    monkeypatch.setattr(drawing, "ASKED_EVERY_S", 0.0)
    page = _Page(ready_after=4)
    assert asyncio.run(drawing._where_it_draws(page)) == (0.2, 0.1, 0.8, 0.7)
    assert page.asked == 4


def test_a_page_that_never_draws_is_given_up_in_time(monkeypatch):
    monkeypatch.setattr(drawing, "ASKED_EVERY_S", 0.0)
    monkeypatch.setattr(drawing, "DRAWING_COMES_UP_S", 0.05)
    assert asyncio.run(drawing._where_it_draws(_Page(ready_after=10**9))) is None
