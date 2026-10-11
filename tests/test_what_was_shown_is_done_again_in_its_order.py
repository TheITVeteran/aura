"""What a screen shows to be followed is done again in its order: arrows read off their shapes are pressed as keys,
places that light up one after another are clicked in the order they lit. Nothing here is a dance or a spell."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from core.agency.doing_again_what_was_shown import (
    Shown,
    arrows_in,
    asks_to_follow,
    do_again,
    lit_in_turn,
)

pytestmark = pytest.mark.unit


def _arrow(picture, x, y, way, size=24):
    """A block arrow: a shaft and a wider head, pointing ``way``."""
    tile = np.zeros((size, size), bool)
    third = size // 3
    tile[size // 2 - 3:size // 2 + 3, 0:size - third] = True               # shaft
    for i in range(third):                                                  # head, narrowing to the tip
        tile[size // 2 - (third - i) - 2:size // 2 + (third - i) + 2, size - third + i] = True
    turned = {"right": tile, "left": tile[:, ::-1], "down": tile.T, "up": tile.T[::-1, :]}[way]
    picture[y:y + size, x:x + size][turned] = (240, 230, 40)


def test_arrows_drawn_in_a_row_are_read_as_the_keys_they_point_to():
    picture = np.full((120, 300, 3), (40, 30, 90), np.uint8)
    ways = ["right", "up", "left", "down", "down"]
    for n, way in enumerate(ways):
        _arrow(picture, 20 + 55 * n, 48, way)
    assert arrows_in(picture) == ways


def test_arrow_characters_in_the_writing_are_read_first():
    assert arrows_in(np.zeros((10, 10, 3), np.uint8), [{"text": "← ↑ → ↓", "center_x": 0.5, "center_y": 0.5}]) == ["left", "up", "right", "down"]


def test_places_that_light_up_in_turn_are_found_in_their_order():
    order = [4, 0, 8, 2]
    clock = {"t": 0.0}

    async def look():
        clock["t"] += 1 / 20
        t = clock["t"]
        picture = np.full((90, 90, 3), 60, np.uint8)
        step = int(t / 0.5)
        if step < len(order) and (t % 0.5) < 0.3:
            cell = order[step]
            r, c = divmod(cell, 3)
            picture[r * 30:(r + 1) * 30, c * 30:(c + 1) * 30] = 230
        return picture, t

    lit = asyncio.run(lit_in_turn(look, 2.4, places=9))
    expected = [((c + 0.5) / 3, (r + 0.5) / 3) for r, c in (divmod(cell, 3) for cell in order)]
    assert lit == pytest.approx(expected), lit


def test_what_was_shown_is_done_in_its_order(monkeypatch):
    import core.agency.doing_again_what_was_shown as shown_module

    monkeypatch.setattr(shown_module, "BETWEEN_S", 0.0)
    done = []

    class _Hands:
        async def tap(self, key):
            done.append(key)

        async def click(self, x, y):
            done.append((x, y))

    assert asyncio.run(do_again(Shown(keys=["up", "up", "left"]), _Hands())) == 3
    assert done == ["up", "up", "left"]


def test_words_that_ask_to_follow_are_heard():
    assert asks_to_follow("Follow the arrow keys in Grim's Magic Book to cast the correct clean-up spell.")
    assert asks_to_follow("Watch the lights, then repeat them in the same order")
    assert not asks_to_follow("Catch the falling fruit")


def test_a_screen_that_asks_to_follow_its_arrows_has_them_pressed_on_her_page(monkeypatch):
    import time

    import core.agency.doing_again_what_was_shown as shown_module
    import core.skills.screen_pursuit_as_it_happens as reflexes

    monkeypatch.setattr(shown_module, "BETWEEN_S", 0.0)
    picture = np.full((120, 300, 3), (40, 30, 90), np.uint8)
    for n, way in enumerate(["left", "left", "up", "right"]):
        _arrow(picture, 20 + 55 * n, 48, way)
    pressed = []

    class _OnHerPage(reflexes.PlayingAsItHappens):
        async def look(self):
            return picture, time.monotonic()

        async def tap(self, key):
            pressed.append(key)

    monkeypatch.setattr(reflexes, "_said_while_playing", lambda line: None)
    monkeypatch.setattr("core.perception.what_the_pixels_show.recognize_text", lambda _p: [])
    playing = _OnHerPage(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 30)
    playing.words = ["Follow the arrow keys in the book to cast the spell"]
    assert asyncio.run(playing._did_again_what_it_showed())
    assert pressed == ["left", "left", "up", "right"]


def test_arrows_always_drawn_and_lit_in_turn_are_the_keys_in_the_order_they_lit():
    """A book of four arrows, all drawn all the time, one going a darker red at a time: the order is the lighting, read
    as the keys the lit arrows point to, and the words changing at the end (WATCH to YOUR TURN) are no place in it."""
    from core.agency.doing_again_what_was_shown import shown_in_turn

    spots = {"up": (88, 10), "right": (150, 60), "down": (88, 110), "left": (26, 60)}
    order = ["left", "up", "up", "down", "right"]
    clock = {"t": 0.0}

    def drawn(lit: str | None, ended: bool):
        picture = np.full((160, 200, 3), (192, 184, 160), np.uint8)
        for way, (x, y) in spots.items():
            tile = np.zeros((40, 200, 3), np.uint8)[:40, :40]
            _arrow(tile, 4, 4, way, size=30)
            mask = tile.any(axis=2)
            picture[y:y + 40, x:x + 40][mask] = (230, 50, 50) if way == lit else (150, 140, 120)
        picture[150:158, 4:40 if ended else 20] = 255
        return picture

    async def look():
        clock["t"] += 1 / 25
        t = clock["t"]
        # The showing begins a moment after the watching does, as a screen saying WATCH gives one.
        step = int((t - 0.5) / 0.85) if t >= 0.5 else -1
        lit = order[step] if 0 <= step < len(order) and ((t - 0.5) % 0.85) < 0.6 else None
        return drawn(lit, ended=step >= len(order)), t

    shown = asyncio.run(shown_in_turn(look, 10.0))
    assert shown.keys == order, shown


def test_letters_of_the_writing_are_not_read_as_arrows():
    """A screen saying YOUR TURN under a pad of arrows: the Y and the T taper as arrow heads do, and are no keys."""
    import shutil

    if shutil.which("tesseract") is None:
        pytest.skip("no tesseract program here")
    from PIL import Image, ImageDraw, ImageFont

    from core.perception.what_the_pixels_show import recognize_text

    picture = Image.new("RGB", (480, 320), (43, 29, 58))
    ImageDraw.Draw(picture).text((180, 280), "YOUR TURN", fill=(255, 216, 74), font=ImageFont.load_default(size=18))
    pixels = np.asarray(picture).copy()
    assert arrows_in(pixels, recognize_text(np.ascontiguousarray(pixels[:, :, ::-1]))) == []
