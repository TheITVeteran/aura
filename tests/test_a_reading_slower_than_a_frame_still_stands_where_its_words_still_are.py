"""Reading a frame's words can take longer than the frame stays current. What was read still stands where the words'
own pixels are still on the screen, and the menu she just left is retired by its pixels before any reading is done.

Offline 2026-10-12, a book of arrows: the NEXT of the instructions she had left was still what her reflexes went by
when its first round was lost, and a screen that asked her to choose was not known for one, so an arrow sliding
beside its choices was taken for a sign to be followed.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.skills.current_words_before_play import read_current_words
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

pytestmark = pytest.mark.unit

SLOW_S = 0.4          # longer than a frame stays current


def _a_word_on_a_screen(shade: int) -> tuple[np.ndarray, dict]:
    pixels = np.full((64, 64, 3), shade, np.uint8)
    pattern = ((np.indices((32, 32)).sum(axis=0) // 4) % 2 * 240).astype(np.uint8)
    pixels[16:48, 16:48] = pattern[:, :, None]
    return pixels, {"x": .25, "y": .25, "width": .5, "height": .5, "center_x": .5, "center_y": .5}


def _on_a_menu() -> tuple[PlayingAsItHappens, np.ndarray]:
    pixels, place = _a_word_on_a_screen(0)
    reflex = PlayingAsItHappens(page=SimpleNamespace(url="canvas://a-place"), band=(0, 0, 1, 1),
                               goal="Follow the signs", ends_at=time.monotonic() + 10)
    reflex.read(ScreenReading({"text": "NEXT", "layout": [{"text": "NEXT", **place}]}, picture_snapshot=make_snapshot(pixels)))
    assert reflex._a_menu_first()
    return reflex, pixels


def _read_slowly(monkeypatch, regions: list[dict]) -> None:
    from core.perception import what_the_pixels_show as reader

    def slow(_pixels):
        time.sleep(SLOW_S)
        return [dict(region) for region in regions]

    monkeypatch.setattr(reader, "recognize_text", slow)


@pytest.mark.asyncio
async def test_a_menu_she_has_left_is_retired_by_its_pixels_however_long_reading_the_new_screen_takes(monkeypatch):
    reflex, before = _on_a_menu()
    after = np.full_like(before, 60)

    async def frame():
        return after, time.monotonic()

    monkeypatch.setattr(reflex, "look", frame)
    _read_slowly(monkeypatch, [])
    result = await read_current_words(reflex, goes_on=True)
    assert result is not None and result["prior_words_changed"]
    assert not reflex.way_on_shown and reflex.words[-1] == "" and not reflex._a_menu_first()


@pytest.mark.asyncio
async def test_words_read_slowly_stand_where_their_pixels_are_still_on_the_screen(monkeypatch):
    reflex, _menu = _on_a_menu()
    now_showing, place = _a_word_on_a_screen(90)
    now_showing[16:48, 16:48] = now_showing[16:48, 16:48][::-1]          # another word where the menu's was

    async def frame():
        return now_showing[:, :, ::-1], time.monotonic()                 # the stream is RGB

    monkeypatch.setattr(reflex, "look", frame)
    _read_slowly(monkeypatch, [{"text": "CHOOSE YOUR CHARACTER", **place}])
    result = await read_current_words(reflex, goes_on=True)
    assert result is not None and reflex.said_now == "CHOOSE YOUR CHARACTER"
    assert reflex.words[-1] == "CHOOSE YOUR CHARACTER"


@pytest.mark.asyncio
async def test_words_read_slowly_do_not_stand_where_the_screen_has_changed_under_them(monkeypatch):
    reflex, _menu = _on_a_menu()
    first, place = _a_word_on_a_screen(90)
    first[16:48, 16:48] = first[16:48, 16:48][::-1]
    frames = [first, np.full_like(first, 200)]

    async def frame():
        return frames[min(1, frame.looks)][:, :, ::-1], time.monotonic()

    frame.looks = 0

    def looked(_pixels):
        frame.looks += 1
        time.sleep(SLOW_S)
        return [{"text": "CHOOSE YOUR CHARACTER", **place}]

    from core.perception import what_the_pixels_show as reader

    monkeypatch.setattr(reflex, "look", frame)
    monkeypatch.setattr(reader, "recognize_text", looked)
    await read_current_words(reflex, goes_on=True)
    assert reflex.said_now != "CHOOSE YOUR CHARACTER"
