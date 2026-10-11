"""A thing that moves when she runs into it is got to its place by pushing from its far side
(core/agency/pushing_a_thing.py).

A world of its own: a floor, a crate that goes the way she walks into it, a pad it is to be pushed onto, and a new
crate and pad each time one is got there. Making straight for the crate, she shoves it about and seldom gets it home;
lining up behind it, away from the pad, and pushing one way at a time, she gets it there again and again. What can be
pushed, and where it goes, are learned from what happens.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.pushing_a_thing import Pushing

pytestmark = pytest.mark.unit

STEP, SPEED = 1 / 30, 100.0
KEYS = {"left": (-SPEED, 0.0), "right": (SPEED, 0.0), "up": (0.0, -SPEED), "down": (0.0, SPEED)}


def _over(a, b) -> bool:
    return abs(a.x - b.x) < (a.w + b.w) / 2 and abs(a.y - b.y) < (a.h + b.h) / 2


def _somewhere(rand, w):
    return SimpleNamespace(x=rand.uniform(70, 410), y=rand.uniform(70, 250), w=w, h=w)


def _a_minute(planning: bool, seed: int, pushing: Pushing | None = None) -> int:
    rand = random.Random(seed)
    pushing = pushing or Pushing()
    her = SimpleNamespace(number=0, kind=0, x=40.0, y=40.0, w=16.0, h=16.0, vx=0.0, vy=0.0, moved=True)
    crate = SimpleNamespace(number=1, kind=1, vx=0.0, vy=0.0, moved=True, **vars(_somewhere(rand, 24.0)))
    pad = SimpleNamespace(number=2, kind=2, vx=0.0, vy=0.0, moved=False, **vars(_somewhere(rand, 30.0)))
    got, at = 0, 0.0
    while at < 60.0:
        goal = pushing.where_to_go(her, crate, pad) if planning else (crate.x, crate.y)
        goal = goal or (crate.x, crate.y)
        key = min(KEYS, key=lambda k: abs(goal[0] - her.x - KEYS[k][0] * STEP) + abs(goal[1] - her.y - KEYS[k][1] * STEP))
        way = KEYS[key]
        before = (her.x, her.y)
        her.x = min(470.0, max(10.0, her.x + way[0] * STEP))
        her.y = min(310.0, max(10.0, her.y + way[1] * STEP))
        crate.vx = crate.vy = 0.0
        if _over(her, crate):
            moved = (her.x - before[0], her.y - before[1])
            nx, ny = crate.x + moved[0], crate.y + moved[1]
            if 12.0 <= nx <= 468.0 and 12.0 <= ny <= 308.0:
                crate.x, crate.y, crate.vx, crate.vy = nx, ny, moved[0] / STEP, moved[1] / STEP
            else:
                her.x, her.y = before
        pushing.saw(her, [crate, pad], way)
        if math.dist((crate.x, crate.y), (pad.x, pad.y)) <= 6.0:
            got += 1
            pushing.counted([her, crate, pad])
            spot, place = _somewhere(rand, 24.0), _somewhere(rand, 30.0)
            crate.x, crate.y, pad.x, pad.y = spot.x, spot.y, place.x, place.y
        at += STEP
    return got


def test_lined_up_behind_it_she_gets_it_home_again_and_again():
    straight = sum(_a_minute(False, seed) for seed in range(6))
    behind = sum(_a_minute(True, seed) for seed in range(6))
    assert behind >= 3 * max(1, straight) and behind >= 18, (straight, behind)


def test_what_is_pushed_and_where_it_goes_are_learned():
    pushing = Pushing()
    _a_minute(True, 3, pushing)
    assert pushing.pushable(1) and not pushing.pushable(2)
    assert 2 in pushing.places[1]
