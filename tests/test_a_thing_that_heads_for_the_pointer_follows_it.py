"""A thing that heads for the pointer at its own pace follows it: a boat eased toward the mouse, never under it.

LIVE 2026-10-10 "Nothing here follows the mouse" of a boat the mouse steered, and its game was lost to GAME OVER.
"""
from __future__ import annotations

import math

import pytest

from core.agency.which_one_answers_to_her import _chases_along, _measured_along

pytestmark = pytest.mark.unit


def _sweep(n: int = 60) -> list[float]:
    """The pointer swept back and forth across the picture."""
    return [120 + 90 * math.sin(k / 6.0) for k in range(n)]


def test_a_boat_eased_toward_the_pointer_heads_for_it_though_it_is_never_under_it():
    pointer, boat, pairs = _sweep(), 120.0, []
    for x in pointer:
        boat += 0.12 * (x - boat)                       # eases a little of the way each picture
        pairs.append((x, 80.0, boat, 150.0))           # and keeps to its lane, low down
    assert _measured_along(pairs, (240.0, 160.0))[0] is not True    # not under the pointer: the old test misses it
    assert _chases_along(pairs)[0] is True


def test_a_thing_going_its_own_way_while_the_pointer_is_swept_does_not_head_for_it():
    pointer = _sweep()
    pairs = [(x, 80.0, 120 + 80 * math.sin(k / 2.3 + 1.0), 40.0) for k, x in enumerate(pointer)]
    assert _chases_along(pairs)[0] is not True
    still = [(x, 80.0, 60.0, 40.0) for x in pointer]
    assert _chases_along(still) == (None, None)


def test_where_the_words_say_the_mouse_moves_her_one_sighting_over_fewer_pictures_is_a_finding():
    """Offline 2026-10-11 an egg-catching game's rounds were lost in under ten seconds each while she found her body."""
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import THE_GUIDE, TOLD, Guide

    guide = Guide(place="a place")
    guide.take_in(TOLD, ["Use the mouse to move Mac. Click to throw the ball."])
    token = THE_GUIDE.set(guide)
    try:
        hers = WhichIsHers()
        mac = SimpleNamespace(number=7, kind=3, x=120.0, y=150.0, w=12.0, h=20.0, seen=0.0, moved=True)
        moves = SimpleNamespace(things={7: mac}, shape=(160, 240))
        for k in range(16):
            at = k * 0.06
            px = 120 + 90 * math.sin(k / 2.5)
            hers.pointed(px, 80.0, at)
            mac.x, mac.seen = px, at
            hers._what_follows_the_pointer(moves, at)
        assert hers.follows_pointer and hers.number == 7
    finally:
        THE_GUIDE.reset(token)


def test_of_the_parts_that_follow_the_pointer_the_one_her_eyes_took_for_her_comes_before_the_biggest():
    """Offline 2026-10-11 the ball a boy held was bigger than any one part of him her eyes followed, was taken for her,
    and was thrown."""
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.cognition.a_guide_to_a_place import THE_GUIDE, TOLD, Guide

    guide = Guide(place="a place")
    guide.take_in(TOLD, ["Use the mouse to move Mac. Click to throw the ball."])
    guide.seen.mine = (3, "a boy")
    token = THE_GUIDE.set(guide)
    try:
        hers = WhichIsHers()
        boy = SimpleNamespace(number=7, kind=3, x=120.0, y=150.0, w=8.0, h=12.0, seen=0.0, moved=True)
        ball = SimpleNamespace(number=8, kind=5, x=126.0, y=140.0, w=14.0, h=14.0, seen=0.0, moved=True)
        moves = SimpleNamespace(things={7: boy, 8: ball}, shape=(160, 240))
        for k in range(16):
            at = k * 0.06
            px = 120 + 90 * math.sin(k / 2.5)
            hers.pointed(px, 80.0, at)
            boy.x, boy.seen, ball.x, ball.seen = px, at, px + 6, at
            hers._what_follows_the_pointer(moves, at)
        assert hers.follows_pointer and hers.number == 7
    finally:
        THE_GUIDE.reset(token)
