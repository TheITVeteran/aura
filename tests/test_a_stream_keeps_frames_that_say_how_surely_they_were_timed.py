"""Pictures stamped with a capture receipt (a time that also says how surely it is known) are pictures the stream sees.

Offline 2026-10-13, after frames were given their drawing's own clock, the stream that watches what her acts do kept
none of them: it took a time only as a plain float. Nothing her acts did could be told, every act was left unjudged
(1,276 of 1,280 in a pass), and nothing she made was ever taken to be made.
"""
from __future__ import annotations

import asyncio

import numpy as np

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.watching_it_happen import Watching


def test_frames_with_measured_times_are_watched_and_acts_on_them_judged():
    shown = {"n": 0}

    async def take():
        shown["n"] += 1
        picture = np.zeros((60, 80, 3), np.uint8)
        if shown["n"] > 6:
            picture[10:40, 10:50] = 200                                   # her act answered: something appeared
        at = 1000.0 + shown["n"] / 8
        return picture, MeasuredFrameTime(at, lower=at, upper=at + 0.01, arrived=at + 0.01, source="browser-wall",
                                         source_at=at, epoch=0)

    async def watch():
        watching = Watching(take=take, per_second=40.0)
        watching.start()
        await asyncio.sleep(0.15)
        watching.acted("click", 1000.0 + 6.5 / 8)
        await asyncio.sleep(0.3)
        await watching.stop()
        return watching

    watching = asyncio.run(watch())
    assert len(watching.moments) >= 5
    window = watching.answer_window()
    assert type(window["captured_through"]) is float and window["captured_through"] >= window["finished"]


def test_what_the_eyes_keep_of_when_things_were_seen_is_a_plain_time():
    """A thing born in a picture stamped with a capture receipt is born at a time every check takes for a number."""
    from core.perception.what_moves_in_the_picture import WhatMoves

    eyes = WhatMoves()
    for i in range(40):
        picture = np.full((120, 160, 3), 30, np.uint8)
        picture[50:66, 10 + 3 * i:26 + 3 * i] = (230, 60, 60)
        at = 10.0 + i / 20
        eyes.see(picture, MeasuredFrameTime(at, lower=at, upper=at + 0.01, arrived=at + 0.01, source="browser-wall",
                                            source_at=at, epoch=0))
    assert eyes.things and all(type(thing.born) is float for thing in eyes.things.values())


def test_whose_texture_could_go_on_is_what_a_part_would_ask_of_the_thing():
    """Asked once a picture instead of once for every part of it: a thing it turns away, every part would."""
    from types import SimpleNamespace

    from core.perception.texture_that_continues_when_outlines_join import could_continue

    motion = SimpleNamespace(before=1.0, at=1.1, support=3, points=(1, 2, 3), locations=((1, 1), (2, 2), (3, 1)),
                             dx=1.0, dy=0.0, error=0.5)
    patch, support = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool)
    thing = SimpleNamespace(number=7, born=0.5, seen=1.0, patch=patch, support=support)
    assert could_continue(thing, {7: motion}, 1.1)
    assert not could_continue(thing, {}, 1.1)                                  # no measured motion
    assert not could_continue(thing, {7: motion}, 1.5)                         # not this picture's
    assert not could_continue(SimpleNamespace(**{**vars(thing), "support": np.zeros((4, 4), bool)}), {7: motion}, 1.1)
