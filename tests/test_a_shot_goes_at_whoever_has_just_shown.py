"""Told to aim with the mouse and click to throw, with no body of her own in the place, she throws at whoever shows.

LIVE 2026-10-11 a food fight read "TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW ... THE FIRST TO SPLAT THE OTHER FIVE
TIMES WINS DODGE FOOD BY PRESSING THE SPACE BAR", the last two lines run together on the screen. Space was taken for
a key to hit with as well as to get out of the way; she tapped it three times a second as her shot, the tap came
before the throw and ended the turn, and she lost in fourteen seconds having thrown nothing. Those who throw back in
such a game show for a moment and stand still, and only things going somewhere were thrown at.
"""
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.thrown_at_what_shows import comes_at_her, what_the_shot_goes_at
from core.agency.what_the_rules_said import WhatTheRulesSaid

pytestmark = pytest.mark.unit

FOOD_FIGHT = ("GET READY FOR A FOOD FIGHT YOU AGAINST YOUR FAVORITE CARTOON STARS! TAKE AIM WITH YOUR MOUSE AND CLICK TO "
              "THROW. HOW TO PLAY THE FIRST TO SPLAT THE OTHER FIVE TIMES WINS DODGE FOOD BY PRESSING THE SPACE BAR.")


def _thing(number: int, x: float, y: float, *, vx: float = 0.0, vy: float = 0.0, size: float = 900.0, born: float = 0.0,
           sizes: tuple[float, ...] = (), kind: int = 1) -> SimpleNamespace:
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, vx=vx, vy=vy, size=size, born=born,
                           sizes=deque(sizes or (size,) * 5), where_at=lambda after: (x + vx * after, y + vy * after))


def test_whoever_has_just_shown_is_thrown_at_before_what_goes_across() -> None:
    stood_long = _thing(1, 100, 100, born=0.0)
    just_shown = _thing(2, 300, 120, born=9.2)
    crossing = _thing(3, 200, 200, vx=60.0, born=3.0, size=2500.0)
    assert what_the_shot_goes_at([stood_long, just_shown, crossing], 10.0).number == 2
    assert what_the_shot_goes_at([stood_long, crossing], 10.0).number == 3
    assert what_the_shot_goes_at([stood_long], 10.0) is None
    assert what_the_shot_goes_at([just_shown], 10.0, not_numbers=(2,)) is None


def test_a_speck_redrawn_is_no_one_showing_and_someone_standing_after_moving_is_thrown_at() -> None:
    """Offline 2026-10-11 a food fight's opponent ran to and fro, and each throw went at a speck of the counter redrawn
    as her bar went down: what has just shown is someone only near the size of what moves."""
    runner = _thing(1, 200, 150, vx=80.0, born=1.0, size=2700.0)
    speck = _thing(2, 40, 10, born=9.8, size=20.0)
    assert what_the_shot_goes_at([runner, speck], 10.0).number == 1
    standing = SimpleNamespace(**{**vars(_thing(3, 220, 150, born=1.0, size=2700.0)), "moved": True})
    assert what_the_shot_goes_at([standing, speck], 10.0).number == 3


def test_a_thing_that_grows_is_coming_at_her() -> None:
    assert comes_at_her([_thing(1, 0, 0, sizes=(100, 110, 125, 150, 180))])
    assert not comes_at_her([_thing(1, 0, 0, sizes=(100, 100, 101, 100, 100))])
    assert not comes_at_her([_thing(1, 0, 0, sizes=(180, 150, 125, 110, 100))])


class _Hands:
    def __init__(self) -> None:
        self.taps: list[str] = []
        self.clicks: list[tuple[float, float]] = []

    async def tap(self, key: str) -> None:
        self.taps.append(key)

    async def click(self, x: float, y: float) -> None:
        self.clicks.append((round(x, 2), round(y, 2)))


def _a_turn(things: list[SimpleNamespace], at: float = 10.0) -> _Hands:
    rules = WhatTheRulesSaid.read(FOOD_FIGHT)
    assert rules.a_click_is_a_shot and "space" in rules.dodge_keys
    hands = _Hands()
    run = SimpleNamespace(held="", last_trigger_began=-10.0, taps=0, last_click=-10.0)
    hers = SimpleNamespace(number=None, makes={}, tapped=lambda *a, **k: None)
    moves = SimpleNamespace(things={t.number: t for t in things}, shape=(400, 600))
    asyncio.run(play._fire_while_finding_herself(hands, run, hers, moves, rules, at))
    return hands


def test_the_key_for_getting_out_of_the_way_is_not_fired_and_the_throw_is_made() -> None:
    hands = _a_turn([_thing(2, 300, 120, born=9.2)])
    assert hands.taps == [], "the key given to dodging was pressed as a shot"
    assert hands.clicks == [(0.5, 0.3)]


def test_it_is_pressed_when_something_comes_at_her_and_the_throw_still_made() -> None:
    hands = _a_turn([_thing(2, 300, 120, born=9.2), _thing(5, 320, 300, born=8.0, sizes=(100, 120, 150, 190, 240))])
    assert hands.taps == ["space"]
    assert len(hands.clicks) == 1


def test_where_a_click_is_her_shot_and_nothing_moves_her_there_is_no_body_of_hers_to_look_for() -> None:
    """Offline 2026-10-11 a food fight's opponent was said to be her: "That looks like me: the muscular man"."""
    from core.agency.naming_what_she_sees import _no_body_of_hers

    assert _no_body_of_hers(SimpleNamespace(pointer_trigger=True, keys=["space"]), SimpleNamespace(pointer={}))
    assert not _no_body_of_hers(SimpleNamespace(pointer_trigger=True, keys=["w", "a", "s", "d"]), SimpleNamespace(pointer={}))
    assert not _no_body_of_hers(SimpleNamespace(pointer_trigger=True, keys=[]), SimpleNamespace(pointer={"moves": "mac"}))
    assert not _no_body_of_hers(SimpleNamespace(pointer_trigger=False, keys=["space"]), SimpleNamespace(pointer={}))


def test_a_thing_that_only_sways_where_it_hangs_is_part_of_the_room() -> None:
    """Offline 2026-10-11 throws went at a swinging lamp while the opponent ran to and fro below it."""
    lamp = SimpleNamespace(**{**vars(_thing(1, 120, 10, vx=30.0, born=0.0, size=2900.0)), "moved": True,
                              "path": deque([(t, 120 + (-1) ** t * 3, 10) for t in range(10)])})
    runner = SimpleNamespace(**{**vars(_thing(2, 200, 80, vx=20.0, born=0.0, size=2500.0)), "moved": True,
                                "path": deque([(t, 100 + 15 * t, 80) for t in range(10)])})
    assert what_the_shot_goes_at([lamp, runner], 10.0).number == 2
    assert what_the_shot_goes_at([lamp], 10.0).number == 1, "where nothing else is there, what moves is still thrown at"
