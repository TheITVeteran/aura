"""A place found by the pointer turning into a hand is clicked where it answers now, not where it was when the screen
was swept: a start button that moves (a top spinning about a title screen) is followed.

Offline 2026-10-11 (the 56-game campaign, pass 3): a title's "Click to Start" pointed at a spinning top that slid about
the screen; every click went to where the top had been when the screen was swept, and the game was never started.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.skills.screen_pursuit_on_a_page import OnAPage

pytestmark = pytest.mark.unit


class _Page:
    """A page whose one button is now at ``button`` (pixels), the pointer a hand within ``reach`` of it."""

    def __init__(self, button, reach=40.0):
        self.button, self.reach, self.at, self.clicked = button, reach, (0.0, 0.0), []
        self.mouse = SimpleNamespace(move=self._move, click=self._click)

    async def _move(self, x, y):
        self.at = (x, y)

    async def _click(self, x, y):
        self.clicked.append((x, y))

    async def evaluate(self, _script, point):
        x, y = point
        near = abs(x - self.button[0]) <= self.reach and abs(y - self.button[1]) <= self.reach
        return "pointer" if near else "auto"


@pytest.mark.asyncio
async def test_a_swept_place_whose_button_moved_is_clicked_where_the_hand_is_now():
    page = _Page(button=(1100.0, 600.0))
    surface = OnAPage(page, "game")
    surface._swept = (frozenset({"click", "start"}), [(0.79, 0.78)])            # where the sweep found it
    assert await surface.click(0.79, 0.78, [0, 0, 1280, 768])
    x, y = page.clicked[-1]
    assert abs(x - 1100.0) <= 40.0 and abs(y - 600.0) <= 40.0, page.clicked


@pytest.mark.asyncio
async def test_a_swept_place_still_there_and_any_other_click_are_clicked_where_they_are():
    page = _Page(button=(1011.2, 599.04))
    surface = OnAPage(page, "game")
    surface._swept = (frozenset({"start"}), [(0.79, 0.78)])
    assert await surface.click(0.79, 0.78, [0, 0, 1280, 768])
    assert page.clicked[-1] == pytest.approx((0.79 * 1280, 0.78 * 768))
    assert await surface.click(0.30, 0.30, [0, 0, 1280, 768])                    # not a swept place: no looking about
    assert page.clicked[-1] == pytest.approx((0.30 * 1280, 0.30 * 768))
