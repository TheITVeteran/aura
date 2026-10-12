"""Places alike to open against a clock are opened each once, quickly, in reading order, what each held kept, and
those that held the same thing opened together.

Offline 2026-10-11 a corridor of eighteen doors was opened one every five to fourteen seconds, each click waiting on a
whole look and a decision, and the clock ran out with a third of them unopened. The boxes here are invented.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.agency.searching_among_places import alike_pairs, search_the_places, signature

pytestmark = pytest.mark.unit

CLOSED = (200, 170, 90)
BEHIND = {0: (220, 30, 30), 1: (30, 30, 220), 2: (30, 200, 40), 3: (220, 30, 30), 4: (250, 250, 250), 5: (30, 200, 40)}


class _Boxes:
    """Six boxes in two rows; a click opens one, and two that held alike stay open together."""

    def __init__(self):
        self.places = [(0.2 + 0.3 * (i % 3), 0.3 + 0.4 * (i // 3)) for i in range(6)]
        self.open: set[int] = set()
        self.clicks: list[tuple[float, float]] = []

    def picture(self):
        pixels = np.full((200, 300, 3), 60, dtype=np.uint8)
        for i, (x, y) in enumerate(self.places):
            colour = BEHIND[i] if i in self.open else CLOSED
            cx, cy = int(x * 300), int(y * 200)
            pixels[cy - 15:cy + 15, cx - 15:cx + 15] = colour
        return pixels

    async def look(self):
        return (self.picture(), 0.0)

    async def click(self, x, y):
        self.clicks.append((round(x, 2), round(y, 2)))
        i = min(range(6), key=lambda k: abs(self.places[k][0] - x) + abs(self.places[k][1] - y))
        self.open.add(i)


def test_each_place_is_opened_once_in_reading_order_and_what_each_held_is_kept(monkeypatch):
    monkeypatch.setattr("core.agency.searching_among_places.SHOWS_S", 0.0)
    boxes = _Boxes()
    held: dict = {}
    done = asyncio.run(search_the_places(boxes.look, boxes, boxes.places, seconds=30, held=held))
    assert done["opened"] == 6 and len(held) == 6
    assert boxes.clicks[:6] == [(round(x, 2), round(y, 2)) for x, y in boxes.places]
    assert done["pairs"] == 2, "the two reds and the two greens are opened together"


def test_what_two_places_held_alike_is_a_pair_and_what_looks_closed_is_nothing_found():
    boxes = _Boxes()
    boxes.open = set(range(6))
    picture = boxes.picture()
    held = {place: signature(picture, place) for place in boxes.places}
    pairs = alike_pairs(held, closed=signature(_Boxes().picture(), boxes.places[0]))
    assert {frozenset(pair) for pair in pairs} == {frozenset((boxes.places[0], boxes.places[3])),
                                                   frozenset((boxes.places[2], boxes.places[5]))}
