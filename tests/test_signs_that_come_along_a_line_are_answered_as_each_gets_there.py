"""Signs that come along a line to a mark, each the sign of a key, have their keys held as each gets there.

Nothing here is a spell book or a dance mat. A strip with arrows drawn on it going one way at one pace; a place near
its end where the front arrow is judged by what is held down at that moment, and by nothing else. Which keys, where
the place is and how early is too early, she is not told.
"""
from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from core.agency.signs_that_come_along import SignsComing, _go_together
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit

WIDE, TALL = 480, 320
STRIP = (70, 170, 90)
INK = (20, 70, 30)


def _stamp(picture: np.ndarray, way: str | None, x: float, y: int) -> None:
    """An arrow pointing ``way`` with its middle at (x, y); or, with no way, a plain block of the same size."""
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (40, 40), 0)
    draw, m = ImageDraw.Draw(canvas), 20
    if way is None:
        draw.rectangle([m - 8, m - 6, m + 8, m + 6], fill=255)
    else:
        draw.rectangle([m - 10, m - 2, m + 3, m + 2], fill=255)
        draw.polygon([(m + 3, m - 8), (m + 11, m), (m + 3, m + 8)], fill=255)
        canvas = canvas.rotate({"right": 0, "up": 90, "left": 180, "down": 270}[way])
    mark = np.asarray(canvas) > 128
    left, top = round(x) - 20, y - 20
    if left < 0 or left + 40 > WIDE:
        return
    picture[top:top + 40, left:left + 40][mark] = INK


class _Strip:
    """Arrows come along a strip; the front one is judged while it is in the place near the end."""

    def __init__(self, signs: bool = True, seed: int = 4, pace: float = 150.0) -> None:
        self.random = random.Random(seed)
        self.signs, self.pace = signs, pace
        self.coming: list[list] = []          # [way, x]
        self.now, self.next_at = 0.0, 0.3
        self.down: set[str] = set()
        self.good = self.wrong = 0
        self.pressed: list[str] = []

    async def look(self):
        self.now += 1 / 30
        if self.now >= self.next_at:
            self.next_at = self.now + 0.6
            self.coming.append([self.random.choice(["left", "right", "up", "down"]), 60.0])
        for sign in self.coming:
            sign[1] += self.pace / 30
        if self.coming:
            way, x = self.coming[0]
            if 400 <= x < 440 and self.down == {way}:
                self.good += 1
                self.coming.pop(0)
            elif x >= 440:
                self.wrong += 1
                self.coming.pop(0)
        picture = np.full((TALL, WIDE, 3), (200, 225, 235), np.uint8)
        picture[270:310] = STRIP
        picture[262:318, 405:455] = (120, 200, 130)                              # the place they come to
        for way, x in self.coming:
            _stamp(picture, way if self.signs else None, x, 290)
        return picture, self.now


class _Hands:
    def __init__(self, strip: _Strip) -> None:
        self.strip = strip

    async def down(self, key):
        self.strip.down.add(key)
        self.strip.pressed.append(key)

    async def up(self, key):
        self.strip.down.discard(key)


async def _played(strip: _Strip, seconds: float) -> tuple[SignsComing, list[str]]:
    strip.down = set()
    moves, signs, hands, said = WhatMoves(), SignsComing(), _Hands(strip), []
    while strip.now < seconds:
        picture, at = await strip.look()
        moves.see(picture, at)
        signs.see(moves, picture, at)
        if signs.asks(at):
            await signs.hold(hands, at, say=said.append)
        else:
            await signs.let_go(hands)
    return signs, said


def test_each_sign_has_its_key_held_as_it_comes_to_the_front():
    strip = _Strip()
    signs, said = asyncio.run(_played(strip, 14.0))
    judged = strip.good + strip.wrong
    assert judged >= 15, (strip.good, strip.wrong)
    # The first ones go by while she is still finding that it is a stream of signs; after them, nearly all.
    assert strip.good >= judged - 5, (strip.good, strip.wrong)
    assert said == ["Signs are coming along a line, each one a key: I'm holding each one's key as it comes to the front."]


def test_things_going_together_that_are_no_key_s_sign_ask_for_nothing():
    strip = _Strip(signs=False)
    signs, said = asyncio.run(_played(strip, 8.0))
    assert strip.pressed == [] and said == [] and not signs.asks(strip.now)


def test_going_together_is_one_way_one_pace_one_line():
    from types import SimpleNamespace

    front = SimpleNamespace(x=300.0, y=290.0, vx=75.0, vy=0.0, w=8.0, h=6.0)
    assert _go_together(front, SimpleNamespace(x=220.0, y=291.0, vx=72.0, vy=1.0, w=8.0, h=6.0))
    assert not _go_together(front, SimpleNamespace(x=220.0, y=250.0, vx=75.0, vy=0.0, w=8.0, h=6.0))       # off its line
    assert not _go_together(front, SimpleNamespace(x=220.0, y=290.0, vx=30.0, vy=0.0, w=8.0, h=6.0))       # another pace
    assert not _go_together(front, SimpleNamespace(x=220.0, y=290.0, vx=0.0, vy=75.0, w=8.0, h=6.0))       # another way


def test_a_line_known_from_before_has_its_first_sign_answered_though_it_comes_alone():
    """Offline 2026-10-12: the first arrow of each showing comes a third of the way alone, and with three lives the
    first three unanswered were the game."""
    async def twice() -> tuple[_Strip, _Strip]:
        first = _Strip(seed=4)
        learned, _said = await _played(first, 6.0)
        second = _Strip(seed=9)
        second.down = set()
        moves, signs, hands = WhatMoves(), SignsComing(learned.known()), _Hands(second)
        while second.now < 6.0:
            picture, at = await second.look()
            moves.see(picture, at)
            signs.see(moves, picture, at)
            if signs.asks(at):
                await signs.hold(hands, at)
            else:
                await signs.let_go(hands)
        return first, second

    first, second = asyncio.run(twice())
    assert second.wrong <= 1 and second.good >= 6, (second.good, second.wrong)
    assert first.wrong >= second.wrong


def test_a_watch_made_for_something_else_notices_signs_coming():
    from core.agency.signs_that_come_along import SignsAreComing, watching_for_signs

    strip = _Strip()

    async def watch() -> object:
        look = watching_for_signs(strip.look)
        try:
            while strip.now < 6.0:
                await look()
        except SignsAreComing as seen:
            return seen.args[0]
        return None

    known = asyncio.run(watch())
    going = known["going"]
    assert going[0] > 0.2 and abs(going[1]) < 0.05, known                                   # going right, on a level line


def test_only_a_place_that_asks_for_keys_shown_to_be_followed_has_them_followed():
    """Offline 2026-10-12 a tunnel's arrows to be flown through for points came at her in a line, and she held the keys
    they pointed for instead of steering."""
    from core.agency.signs_that_come_along import told_to_follow_signs

    assert told_to_follow_signs("Follow the arrow keys in Grim's Magic Book to cast the correct clean-up spell.")
    assert told_to_follow_signs("Press the arrows when they reach the line!")
    assert not told_to_follow_signs("You have two minutes to collect as many points as possible. Zip Arrows 1000 pts")
    assert not told_to_follow_signs("Use the arrow keys to move")
