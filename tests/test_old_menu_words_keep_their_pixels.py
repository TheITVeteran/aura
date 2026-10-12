"""A missed OCR pass does not remove a menu, but changed measured word pixels retire its old reading."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.perception.words_kept_with_their_pixels import words_with_their_pixels
from core.skills.current_words_before_play import read_current_words
from core.skills.screen_pursuit_as_it_happens import PlayingAsItHappens

pytestmark = pytest.mark.unit


def menu_read():
    pixels = np.zeros((64, 64, 3), np.uint8)
    pattern = ((np.indices((32, 32)).sum(axis=0) // 4) % 2 * 240).astype(np.uint8)
    pixels[16:48, 16:48] = pattern[:, :, None]
    region = {"text": "PLAY", "x": .25, "y": .25, "width": .5, "height": .5, "center_x": .5, "center_y": .5}
    reading = ScreenReading({"text": "PLAY", "layout": [region]}, picture_snapshot=make_snapshot(pixels))
    assert words_with_their_pixels(reading) is not None
    return reading, pixels


def test_missing_pixels_are_unknown_and_uniform_fading_keeps_the_word():
    reading, pixels = menu_read()
    words = words_with_their_pixels(reading)
    assert words.still_in(None) is None
    assert words.still_in((pixels.astype(float) * .7 + 20).astype(np.uint8)) is True
    assert words.still_in(np.zeros((64, 32, 3), np.uint8)) is None


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.asyncio
async def test_current_pixel_evidence_decides_whether_the_old_menu_still_holds_control(monkeypatch, changed):
    from core.perception import what_the_pixels_show as pixels_reader

    reading, before = menu_read()
    reflex = PlayingAsItHappens(page=SimpleNamespace(url="canvas://new-place"), band=(0, 0, 1, 1),
                               goal="Guide a body", ends_at=time.monotonic() + 10)
    reflex.read(reading)
    assert reflex._a_menu_first()
    after = np.full_like(before, 60) if changed else before

    async def frame():
        return after, time.monotonic()

    monkeypatch.setattr(reflex, "look", frame)
    monkeypatch.setattr(pixels_reader, "recognize_text", lambda _pixels: [])
    result = await read_current_words(reflex, goes_on=True)
    if changed:
        assert result["prior_words_changed"]
        assert not reflex.way_on_shown and reflex.words[-1] == ""
        assert not reflex._a_menu_first()
    else:
        assert result is None and reflex.way_on_shown
        assert reflex.words[-1] == "PLAY"


def test_the_registered_kept_word_pixels_invariant():
    from core.perception.words_kept_with_their_pixels import _kept_word_pixels_invariant

    assert _kept_word_pixels_invariant() == ()
