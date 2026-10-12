"""The new screen's measured words precede a slower scene interpretation."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.skills.current_words_before_play import read_current_words, words_of_this_frame

pytestmark = pytest.mark.unit


def test_empty_stale_and_future_readings_do_not_erase_a_menu():
    from core.skills.current_words_before_play import _current_play_words_invariant

    assert _current_play_words_invariant() == ()


def test_words_of_a_clipped_view_keep_their_place_in_the_whole_view():
    region = {"text": "READY", "x": .1, "y": .2, "center_x": .2, "center_y": .3, "width": .2, "height": .1}
    read = words_of_this_frame([region], at=1, now=1.05, band=(.2, .3, .8, .9))
    assert read["layout"][0]["center_x"] == pytest.approx(.32)
    assert read["layout"][0]["center_y"] == pytest.approx(.48)
    assert region["center_x"] == .2


@pytest.mark.asyncio
async def test_an_unmeasured_word_pass_leaves_the_old_screen_alone(monkeypatch):
    from core.perception import what_the_pixels_show as pixels

    monkeypatch.setattr(pixels, "recognize_text", lambda _picture: [])
    reads = []

    async def look():
        return np.zeros((8, 8, 3), np.uint8), time.monotonic()

    reflex = SimpleNamespace(look=look, read=reads.append, band=(0, 0, 1, 1))
    assert await read_current_words(reflex, goes_on=True) is None
    assert not reads


@pytest.mark.asyncio
async def test_current_play_is_controlled_before_the_slow_read_of_its_scene(monkeypatch):
    from core.perception import what_the_pixels_show as pixels
    from core.skills import screen_pursuit_as_it_happens as play

    events = []
    reflex = play.PlayingAsItHappens(page=None, band=(0, 0, 1, 1), goal="Guide a body", ends_at=time.monotonic() + 30)
    reflex.read({"text": "PLAY", "layout": [{"text": "PLAY"}]})
    assert reflex.way_on_shown

    async def frame():
        events.append("pixels")
        return np.zeros((8, 8, 3), np.uint8), time.monotonic()

    async def control():
        assert reflex.said_now == "FUEL 73" and not reflex.way_on_shown
        events.append("control")

    async def slow_read():
        events.append("scene")
        return {"text": "FUEL 73"}

    async def legend():
        return None

    monkeypatch.setattr(reflex, "look", frame)
    monkeypatch.setattr(reflex, "while_it_moves", control)
    monkeypatch.setattr(reflex, "run_is_over", lambda _seen: False)
    monkeypatch.setattr(reflex, "read_a_legend", legend)
    monkeypatch.setattr(reflex, "goes_on_playing", lambda *_args: False)
    monkeypatch.setattr(pixels, "recognize_text", lambda _picture: [{"text": "FUEL 73"}])
    monkeypatch.setattr(play, "_it_goes_on_without_her", lambda _reflex: True)
    monkeypatch.setattr(play, "a_screen_of_play_in_time", lambda _reflex: False)
    token = play.AS_IT_HAPPENS.set(reflex)
    try:
        await play.looked_at_as_it_happens(slow_read)
    finally:
        play.AS_IT_HAPPENS.reset(token)
    assert events[:3] == ["pixels", "control", "scene"]
