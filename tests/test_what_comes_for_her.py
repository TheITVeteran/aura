"""Things that come for her are told from things that go their own way, and predicted to be where she will be
(core/agency/what_comes_for_her.py). In a world of its own: she walks about at random; one thing follows her, one goes
back and forth on its own line, one goes straight across."""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.what_comes_for_her import WhatComesForHer

pytestmark = pytest.mark.unit

STEP = 1 / 30


def _world(seconds: float, seed: int = 1):
    rand = random.Random(seed)
    her = SimpleNamespace(number=0, kind=0, x=240.0, y=160.0, vx=0.0, vy=0.0)
    chaser = SimpleNamespace(number=1, kind=5, x=40.0, y=40.0, vx=0.0, vy=0.0)
    pacer = SimpleNamespace(number=2, kind=6, x=100.0, y=280.0, vx=90.0, vy=0.0)
    crosser = SimpleNamespace(number=3, kind=7, x=0.0, y=200.0, vx=70.0, vy=0.0)
    comes = WhatComesForHer()
    way, until, at = (0.0, 0.0), 0.0, 0.0
    history = []
    while at < seconds:
        if at >= until:
            angle = rand.uniform(0, 2 * math.pi)
            way, until = (110 * math.cos(angle), 110 * math.sin(angle)), at + 0.6
        her.vx, her.vy = way
        her.x = min(460.0, max(20.0, her.x + her.vx * STEP))
        her.y = min(300.0, max(20.0, her.y + her.vy * STEP))
        dx, dy = her.x - chaser.x, her.y - chaser.y
        far = math.hypot(dx, dy) or 1.0
        chaser.vx, chaser.vy = 60 * dx / far, 60 * dy / far
        chaser.x, chaser.y = chaser.x + chaser.vx * STEP, chaser.y + chaser.vy * STEP
        pacer.x += pacer.vx * STEP
        if pacer.x < 60 or pacer.x > 420:
            pacer.vx = -pacer.vx
        crosser.x = (crosser.x + crosser.vx * STEP) % 480
        comes.saw([chaser, pacer, crosser], her)
        history.append(((her.x, her.y), (chaser.x, chaser.y), (chaser.vx, chaser.vy)))
        at += STEP
    return comes, (chaser, pacer, crosser), history


def test_a_thing_that_follows_her_is_told_from_things_that_go_their_own_way():
    comes, (chaser, pacer, crosser), _ = _world(8.0)
    assert comes.comes_for_her(chaser)
    assert not comes.comes_for_her(pacer)
    assert not comes.comes_for_her(crosser)


def test_the_next_of_a_kind_that_came_for_her_is_known_before_it_shows_it():
    comes, (chaser, _pacer, _crosser), _ = _world(8.0)
    newcomer = SimpleNamespace(number=99, kind=chaser.kind, x=0.0, y=0.0, vx=0.0, vy=0.0)
    assert comes.comes_for_her(newcomer)


def test_one_that_comes_for_her_is_predicted_nearer_where_it_goes_than_its_straight_line_says():
    _comes, _things, history = _world(10.0)
    ahead = int(0.6 / STEP)
    followed, straight = [], []
    for now in range(30, len(history) - ahead):
        (_her_now, chaser_now, going) = history[now]
        her_then, chaser_then, _ = history[now + ahead]
        thing = SimpleNamespace(x=chaser_now[0], y=chaser_now[1], vx=going[0], vy=going[1])
        guess = WhatComesForHer.where_at(thing, 0.6, *her_then)
        line = (chaser_now[0] + going[0] * 0.6, chaser_now[1] + going[1] * 0.6)
        followed.append(math.dist(guess, chaser_then))
        straight.append(math.dist(line, chaser_then))
    assert sum(followed) < 0.7 * sum(straight), (sum(followed), sum(straight))


def test_lining_up_while_she_stands_still_says_nothing():
    comes = WhatComesForHer()
    her = SimpleNamespace(number=0, kind=0, x=240.0, y=160.0, vx=0.0, vy=0.0)
    falling = SimpleNamespace(number=1, kind=3, x=240.0, y=0.0, vx=0.0, vy=100.0)
    for _ in range(40):
        falling.y += 100.0 * STEP
        comes.saw([falling], her)
    assert not comes.comes_for_her(falling)
