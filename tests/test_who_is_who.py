"""What each thing is to her is told from what it does, with nothing in the game to say so (core/agency/who_is_who.py).

A world of its own, a minute long. She goes for coins. A chaser comes for her and hurts; a rival races her for the
coins and hurts nobody; a companion trails her, and when the chaser gets it she loses a life; wanderers go their own
way; a block stands in the middle; sparks flash and go; a cloud drifts along the top, far from her; a tile of the
backdrop sits in a corner she never goes to; a faller drops from the top now and then, and costs her a life when it
lands; and near the end two more things turn up. Every thing is told apart by the
end, the chaser is suspected before it has ever hurt her, and nothing she is told of is said twice.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.what_comes_for_her import WhatComesForHer
from core.agency.who_is_who import (
    FRIENDLY,
    IN_HER_WAY,
    NEGLIGIBLE,
    NEUTRAL,
    PART_OF_THE_PLACE,
    PENDING,
    PRIZE,
    RIVAL,
    THREAT,
    UNKNOWN,
    WhoIsWho,
)

pytestmark = pytest.mark.unit

STEP = 1 / 30
COIN, CHASER, RIVAL_KIND, COMPANION, WANDERER, BLOCK, SPARK, CLOUD, TILE, NEWCOMER, LATE, FALLER = range(1, 13)


def _thing(number, kind, x, y, w=16.0, h=16.0, vx=0.0, vy=0.0, moved=True):
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, w=w, h=h, vx=vx, vy=vy, moved=moved)


def _touching(a, b) -> bool:
    return abs(a.x - b.x) < (a.w + b.w) / 2 and abs(a.y - b.y) < (a.h + b.h) / 2


def _toward(thing, x, y, speed, keep_off=0.0):
    dx, dy = x - thing.x, y - thing.y
    far = math.hypot(dx, dy) or 1.0
    pace = speed if far > keep_off else 0.0
    thing.vx, thing.vy = pace * dx / far, pace * dy / far
    thing.x, thing.y = thing.x + thing.vx * STEP, thing.y + thing.vy * STEP


def _a_minute(seed: int = 1):
    rand = random.Random(seed)
    numbers = iter(range(100, 100000))
    her = _thing(0, 0, 100.0, 200.0)
    coins = [_thing(next(numbers), COIN, rand.uniform(30, 450), rand.uniform(160, 290), 12, 12, moved=False)
             for _ in range(3)]
    chaser = _thing(next(numbers), CHASER, 440.0, 290.0)
    rival = _thing(next(numbers), RIVAL_KIND, 400.0, 170.0)
    companion = _thing(next(numbers), COMPANION, 70.0, 200.0)
    wanderers = [_thing(next(numbers), WANDERER, rand.uniform(50, 430), rand.uniform(160, 290), vx=70, vy=50)
                 for _ in range(2)]
    block = _thing(next(numbers), BLOCK, 240.0, 225.0, 40, 40, moved=False)
    cloud = _thing(next(numbers), CLOUD, 0.0, 30.0, 60, 20, vx=20)
    tile = _thing(next(numbers), TILE, 450.0, 20.0, 40, 40, moved=False)
    faller = None
    who, comes = WhoIsWho(), WhatComesForHer()
    touching: set[int] = set()
    sparks: list[tuple[float, SimpleNamespace]] = []
    late: list[SimpleNamespace] = []
    first_blow, suspected_at, at, back_at = None, None, 0.0, 0.0
    last_faller_kind = None
    while at < 60.0:
        # She makes for the nearest coin, and is stopped by the block.
        goal = min(coins, key=lambda c: math.hypot(c.x - her.x, c.y - her.y))
        before = (her.x, her.y)
        _toward(her, goal.x, goal.y, 100.0)
        her.y = min(300.0, max(150.0, her.y))
        way = (her.x - before[0], her.y - before[1])
        if _touching(her, block):
            her.x, her.y = before
            her.vx = her.vy = 0.0
            who.stopped(her, way, [block])
            her.x += rand.choice((-1, 1)) * 20.0
            her.y = min(300.0, max(150.0, her.y + rand.choice((-1, 1)) * 20.0))
        _toward(chaser, her.x, her.y, 75.0)
        rival_goal = min(coins, key=lambda c: math.hypot(c.x - rival.x, c.y - rival.y))
        _toward(rival, rival_goal.x, rival_goal.y, 60.0)
        if companion is not None:
            _toward(companion, her.x, her.y, 120.0, keep_off=30.0)
        for wanderer in wanderers:
            wanderer.x += wanderer.vx * STEP
            wanderer.y += wanderer.vy * STEP
            if not 20 < wanderer.x < 460:
                wanderer.vx = -wanderer.vx
            if not 150 < wanderer.y < 300:
                wanderer.vy = -wanderer.vy
        cloud.x = (cloud.x + cloud.vx * STEP) % 480
        landed = False
        if faller is None and rand.random() < 0.01:
            faller = _thing(next(numbers), FALLER, rand.uniform(20, 460), 0.0, 10, 10, vy=60.0)
        if faller is not None:
            faller.y += faller.vy * STEP
            if faller.y > 315.0:
                faller, landed = None, True
        if rand.random() < 0.2:
            sparks.append((at, _thing(next(numbers), SPARK, rand.uniform(0, 480), rand.uniform(150, 300), 3, 3)))
        sparks = [(when, spark) for when, spark in sparks if at - when < 0.2]
        if at >= 50.0 and not late:
            late.append(_thing(next(numbers), LATE, 30.0, 40.0, 20, 20, moved=False))
        if at >= 59.9 and len(late) == 1:
            late.append(_thing(next(numbers), NEWCOMER, 300.0, 40.0))
        # What meeting each thing does to her, and to the others.
        taken = [c for c in coins if _touching(c, rival)]
        hurt = False
        for thing in [*coins, chaser, rival, *([companion] if companion else []), *wanderers]:
            if not _touching(her, thing):
                touching.discard(thing.number)
                continue
            if thing.number in touching:
                continue
            touching.add(thing.number)
            if thing.kind == COIN:
                who.met(COIN, "gain")
                taken.append(thing)
            elif thing.kind == CHASER:
                hurt = True
            else:
                who.met(thing.kind, "nothing")
        companion_lost = companion is not None and _touching(chaser, companion)
        everything = [her, *coins, chaser, rival, *([companion] if companion else []), *wanderers, block, cloud, tile,
                      *(s for _w, s in sparks), *late, *([faller] if faller else [])]
        if hurt:
            who.blow(at, her, everything)
            first_blow = at if first_blow is None else first_blow
            chaser.x, chaser.y = rand.choice(((20.0, 290.0), (460.0, 290.0)))
        for coin in taken:
            if coin in coins:
                coins.remove(coin)
                coins.append(_thing(next(numbers), COIN, rand.uniform(30, 450), rand.uniform(160, 290), 12, 12,
                                    moved=False))
        if companion_lost:
            companion = None
        everything = [her, *coins, chaser, rival, *([companion] if companion else []), *wanderers, block, cloud, tile,
                      *(s for _w, s in sparks), *late, *([faller] if faller else [])]
        comes.saw([t for t in everything if t is not her], her)
        prizes = {kind for kind, kept in who.kinds.items() if kept.gains}
        who.saw(at, everything, her, prizes=prizes, comes_for_her=lambda t: comes.comes_for_her(t, 16.0),
                shape=(320.0, 480.0))
        if landed:
            who.blow(at, her, everything)
            last_faller_kind = FALLER
        if companion_lost:
            who.blow(at, her, everything)
            chaser.x, chaser.y = rand.choice(((20.0, 290.0), (460.0, 290.0)))
            back_at = at + 1.0
        if companion is None and at >= back_at:
            companion = _thing(next(numbers), COMPANION, her.x - 30.0, her.y)
        if suspected_at is None and who.of(chaser).name == THREAT and not who.of(chaser).sure:
            suspected_at = at
        at += STEP
    a_faller = _thing(-1, FALLER, 100.0, 100.0, 10, 10, vy=60.0) if last_faller_kind else None
    return who, {"faller": a_faller, "coin": coins[0], "chaser": chaser, "rival": rival, "companion": companion, "wanderer": wanderers[0],
                 "block": block, "spark": sparks[-1][1] if sparks else None, "cloud": cloud, "tile": tile,
                 "late": late[0], "newcomer": late[1]}, first_blow, suspected_at


@pytest.mark.parametrize("seed", range(1, 9))
def test_every_thing_is_told_apart_by_what_it_does(seed):
    who, things, _first_blow, _suspected = _a_minute(seed)
    found = {name: who.of(thing) for name, thing in things.items() if thing is not None}
    expected = {"faller": (THREAT, True), "coin": (PRIZE, True), "chaser": (THREAT, True), "rival": (RIVAL, True), "companion": (FRIENDLY, True),
                "wanderer": (NEUTRAL, True), "block": (IN_HER_WAY, True), "spark": (NEGLIGIBLE, True),
                "cloud": (NEGLIGIBLE, True), "tile": (PART_OF_THE_PLACE, True), "late": (UNKNOWN, False),
                "newcomer": (PENDING, False)}
    for name, (designation, sure) in expected.items():
        if name not in found:
            continue
        # Lost once, the companion is assumed friendly; twice, friendly for sure. Either is right.
        if name == "companion":
            assert found[name].name == designation, (name, found[name])
            continue
        assert (found[name].name, found[name].sure) == (designation, sure), (name, found[name])


@pytest.mark.parametrize("seed", range(1, 9))
def test_what_comes_for_her_is_suspected_before_it_has_ever_hurt_her(seed):
    _who, _things, first_blow, suspected_at = _a_minute(seed)
    assert suspected_at is not None and (first_blow is None or suspected_at < first_blow), (suspected_at, first_blow)


def test_what_she_has_found_out_is_said_once_each_and_one_at_a_time():
    """LIVE 2026-10-11 one loss had seven kinds said to be threats in a fifth of a second."""
    who, things, _first_blow, _suspected = _a_minute()
    present = [thing for thing in things.values() if thing is not None]
    said: list[str] = []
    while True:
        one = who.to_say(present, lambda thing: f"kind {thing.kind}")
        if not one:
            break
        assert len(one) == 1
        said += one
    assert any("friendly competition" in line for line in said), said
    assert any("a threat" in line for line in said), said
    assert len(said) == len(set(said)) and not who.to_say(present, lambda thing: f"kind {thing.kind}")


def test_once_is_a_sign_and_twice_is_shown():
    """LIVE 2026-10-11 a scrolling game had twenty kinds each said for sure to be a threat, a prize or a rival, every one
    from a single meeting that came by chance. Once, a kind is played as what it seemed and not said; twice, it is."""
    from core.agency.who_is_who import PRIZE, THREAT, WhoIsWho

    who = WhoIsWho()
    coin, spike = _thing(1, 7, 50.0, 50.0), _thing(2, 8, 90.0, 50.0)
    who.met(7, "gain")
    who.met(8, "loss")
    assert (who.of(coin).name, who.of(coin).sure) == (PRIZE, False)
    assert (who.of(spike).name, who.of(spike).sure) == (THREAT, False)
    assert who.to_say([coin, spike], lambda thing: f"kind {thing.kind}") == []
    who.met(7, "gain")
    assert who.to_say([coin, spike], lambda thing: f"kind {thing.kind}") == ["The kind 7 is worth getting: getting it counted."]


def test_what_a_kind_did_is_weighed_against_what_it_did_not():
    """A kind met a dozen times for nothing and once as a count went up is not a prize: it is played and said as what
    it mostly did."""
    from core.agency.who_is_who import PRIZE, THREAT, WhoIsWho

    who = WhoIsWho()
    bush, bat = _thing(1, 3, 50.0, 50.0), _thing(2, 4, 90.0, 50.0)
    who.met(3, "gain")
    who.met(4, "loss")
    for _ in range(12):
        who.met(3, "nothing")
        who.met(4, "nothing")
    assert who.of(bush).name not in (PRIZE, THREAT) and who.of(bat).name not in (PRIZE, THREAT)
    coin = _thing(3, 5, 10.0, 10.0)
    for what in ("gain", "nothing", "gain"):
        who.met(5, what)
    assert (who.of(coin).name, who.of(coin).sure) == (PRIZE, True)
    spike = _thing(4, 6, 30.0, 10.0)
    who.met(6, "loss")
    assert who.of(spike).name == THREAT
