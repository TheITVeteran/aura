"""Where the biggest thing in the picture is going one way by itself, the key that points against it is tried first.

LIVE 2026-10-11 a lander that is down in five seconds was tried with up, down, left and right in turn, four seconds
a round. She was never found to be the lander in ten goes, so nothing she learned of it could be used.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.the_key_against_its_going import FIRST, against_its_going

pytestmark = pytest.mark.unit

ARROWS = ["up", "down", "left", "right"]


def _thing(vx: float, vy: float, size: float = 30.0, moved: bool = True) -> SimpleNamespace:
    return SimpleNamespace(vx=vx, vy=vy, w=size, h=size, moved=moved)


def test_the_arrow_against_the_way_the_biggest_moving_thing_goes() -> None:
    assert against_its_going(ARROWS, [_thing(0.0, 80.0)], 10.0) == "up"            # falling
    assert against_its_going(ARROWS, [_thing(0.0, -80.0)], 10.0) == "down"         # rising
    assert against_its_going(ARROWS, [_thing(-90.0, 5.0)], 10.0) == "right"        # carried left
    assert against_its_going(ARROWS, [_thing(90.0, 5.0)], 10.0) == "left"
    # The biggest decides, not the fastest; a thing standing still or barely moving decides nothing.
    assert against_its_going(ARROWS, [_thing(200.0, 0.0, size=4.0), _thing(0.0, 60.0, size=40.0)], 10.0) == "up"
    assert against_its_going(ARROWS, [_thing(0.0, 3.0)], 10.0) == ""
    assert against_its_going(ARROWS, [_thing(0.0, 80.0, moved=False)], 10.0) == ""
    # Where no key of hers points that way, none is put first.
    assert against_its_going(["left", "right", "space"], [_thing(0.0, 80.0)], 10.0) == ""


def test_it_is_held_twice_running_before_the_others_have_their_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    held: list[str] = []

    async def hold(_hands, run, _hers, key, at, *, trying=False):
        held.append(key)
        run.held_since = at

    monkeypatch.setattr(play, "_hold", hold)
    run = SimpleNamespace(keys=list(ARROWS), trying=0, tried_at=-10.0, held_since=0.0, countered=0, countering="")
    falling = SimpleNamespace(things={1: _thing(0.0, 120.0)})

    async def tried(times: int) -> None:
        for step in range(times):
            await play._try_the_keys(None, run, None, float(step), falling)

    asyncio.run(tried(FIRST + 4))
    assert held[:FIRST] == ["up", "", "up", ""]
    assert held[FIRST:] == ["up", "", "down", ""]     # then her keys in their order, each with its rest


def test_with_nothing_going_her_keys_are_tried_in_their_order(monkeypatch: pytest.MonkeyPatch) -> None:
    held: list[str] = []

    async def hold(_hands, run, _hers, key, at, *, trying=False):
        held.append(key)
        run.held_since = at

    monkeypatch.setattr(play, "_hold", hold)
    run = SimpleNamespace(keys=list(ARROWS), trying=0, tried_at=-10.0, held_since=0.0, countered=0, countering="")
    still = SimpleNamespace(things={})

    async def tried() -> None:
        for step in range(4):
            await play._try_the_keys(None, run, None, float(step), still)

    asyncio.run(tried())
    assert held == ["up", "", "down", ""]
