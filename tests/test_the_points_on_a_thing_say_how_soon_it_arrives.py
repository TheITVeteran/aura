"""The points followed on a thing say how soon it will arrive, and that a thing found again under another name is the
one lost.

Offline 2026-10-12 a food fight's opponent was found afresh every few looks as he turned and crouched, and the outlines
of the food flying at her grew and shrank picture by picture. Nothing here is that game: a checked tile that comes at
the eye, one that slides past at one size, and one found again under another name.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import FollowingPoints
from core.perception.what_the_points_on_a_thing_say import WhatThePointsSay

pytestmark = pytest.mark.unit


def _tile(x: float, y: float, half: float) -> np.ndarray:
    picture = np.full((144, 240, 3), 200, np.uint8)
    ys, xs = np.mgrid[0:144, 0:240]
    inside = (np.abs(xs - x) <= half) & (np.abs(ys - y) <= half)
    checks = ((((xs - x + half) * 4 // (2 * half)).astype(int) + ((ys - y + half) * 4 // (2 * half)).astype(int)) % 2) == 0
    picture[inside & checks] = (20, 20, 20)
    picture[inside & ~checks] = (240, 60, 60)
    return picture


def _thing(x: float, y: float, half: float, born: float, seen: float) -> SimpleNamespace:
    return SimpleNamespace(born=born, seen=seen, box=lambda: (x - half - 1, y - half - 1, x + half + 1, y + half + 1))


def _watched(frames):
    eyes, said = FollowingPoints(limit=96), WhatThePointsSay()
    for at, picture, things in frames:
        said.see(eyes.see(picture, at), things, at)
    return said


def test_a_thing_coming_at_the_eye_arrives_soon_and_one_going_by_never_does():
    frames, half, at = [], 14.0, 0.0
    for _ in range(14):                                       # its size doubles every half second
        frames.append((at, _tile(120, 72, half), {1: _thing(120, 72, half, 0.0, at)}))
        at += 0.05
        half *= 2 ** (0.05 / 0.5)
    closing = _watched(frames).closing_s(1)
    assert closing is not None and 0.3 <= closing <= 1.0

    frames, x, at = [], 60.0, 0.0
    for _ in range(14):                                       # across at one size
        frames.append((at, _tile(x, 72, 16), {1: _thing(x, 72, 16, 0.0, at)}))
        at += 0.05
        x += 2.0
    closing = _watched(frames).closing_s(1)
    assert closing is None or closing > 5.0


def test_a_thing_found_again_under_another_name_is_the_one_there_all_along():
    eyes, said, at, x = FollowingPoints(limit=96), WhatThePointsSay(), 0.0, 80.0
    for _ in range(6):
        said.see(eyes.see(_tile(x, 72, 16), at), {1: _thing(x, 72, 16, 0.0, at)}, at)
        at += 0.05
        x += 1.0
    # Its outline is lost: the old name is kept a moment unmatched, and a new one is born where it is.
    found = said.see(eyes.see(_tile(x, 72, 16), at), {1: _thing(x - 1, 72, 16, 0.0, at - 0.05), 2: _thing(x, 72, 17, at, at)}, at)
    assert found == {2: (1, 0.0)}
    said.forget()
    assert said.carried == {} and said.scale == {}


def test_what_comes_at_her_is_judged_by_its_points_where_they_were_followed():
    from core.agency.thrown_at_what_shows import comes_at_her

    growing_outline = [10.0, 14.0, 19.0, 26.0]
    assert comes_at_her([SimpleNamespace(closing_s=0.4, sizes=[10.0] * 4)])
    assert not comes_at_her([SimpleNamespace(closing_s=float("inf"), sizes=growing_outline)])   # its outline only swells
    assert comes_at_her([SimpleNamespace(closing_s=None, sizes=growing_outline)])               # no points: by outline
