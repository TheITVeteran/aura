"""A thing gone for is kept to until there is a reason to let it go (core/agency/a_thing_gone_for.py).

Measured in a small world of its own: a body that goes at what it chooses, answering a change of way a moment late as
a key does, coins that come and stay, and each coin seen some pixels off where it is in every picture, as play sees
things. There, choosing afresh at every picture turns near ties into a dither; keeping to the one chosen changes its
mind less than half as often and gets as many or more (304 coins to 295 over ten runs, 2026-10-11). Where a body turns at once and things are seen exactly, choosing afresh does as well or better (measured
2026-10-11: 359 coins to 339 over ten runs): that is why the margin is learned in each game and not fixed.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from core.agency.a_thing_gone_for import GoneFor

pytestmark = pytest.mark.unit

SPEED, STEP, LAG_S, SEEN_OFF = 120.0, 1 / 30, 0.15, 15.0


def _played(choose, *, seed: int, seconds: float = 20.0):
    """Coins got, and changes of mind while the thing gone for was still there, in ``seconds``."""
    rand = random.Random(seed)
    me, going = [240.0, 160.0], [0.0, 0.0]
    coins = [SimpleNamespace(number=n, x=me[0] + rand.uniform(-150, 150), y=me[1] + rand.uniform(-100, 100))
             for n in range(6)]
    following = None
    got = changes = 0
    at = 0.0
    while at < seconds:
        at += STEP
        seen = [(math.hypot(c.x + rand.gauss(0, SEEN_OFF) - me[0], c.y + rand.gauss(0, SEEN_OFF) - me[1]) / SPEED,
                 (c.x, c.y), c) for c in coins]
        chosen = choose(seen, at)
        if chosen is None:
            continue
        if following is not None and chosen[2].number != following and any(c.number == following for c in coins):
            changes += 1
        following = chosen[2].number
        tx, ty = chosen[1]
        far = max(1e-6, math.hypot(tx - me[0], ty - me[1]))
        # The way she is going answers the way she wants a moment late, as a key held does.
        share = min(1.0, STEP / LAG_S)
        going = [going[0] + ((tx - me[0]) / far * SPEED - going[0]) * share,
                 going[1] + ((ty - me[1]) / far * SPEED - going[1]) * share]
        me = [me[0] + going[0] * STEP, me[1] + going[1] * STEP]
        for coin in [c for c in coins if math.hypot(c.x - me[0], c.y - me[1]) < 10]:
            coins.remove(coin)
            got += 1
            coins.append(SimpleNamespace(number=1000 + got, x=me[0] + rand.uniform(-150, 150),
                                         y=me[1] + rand.uniform(-100, 100)))
    return got, changes


def test_keeping_to_a_thing_gets_as_much_with_far_fewer_changes_of_mind():
    afresh = [_played(lambda seen, at: min(seen, key=lambda c: c[0]), seed=seed) for seed in range(10)]
    kept = []
    for seed in range(10):
        held = GoneFor()
        kept.append(_played(lambda seen, at, held=held: held.choose(seen, at), seed=seed))
    got_afresh, got_kept = sum(g for g, _ in afresh), sum(g for g, _ in kept)
    changes_afresh, changes_kept = sum(c for _, c in afresh), sum(c for _, c in kept)
    assert got_kept >= got_afresh, (afresh, kept)
    assert changes_kept < changes_afresh / 2, (afresh, kept)


def test_a_thing_much_nearer_is_still_turned_for():
    held = GoneFor()
    far = SimpleNamespace(number=1)
    near = SimpleNamespace(number=2)
    assert held.choose([(3.0, (0, 0), far)], 0.0)[2] is far
    assert held.choose([(3.0, (0, 0), far), (1.0, (5, 5), near)], 0.1)[2] is near


def test_a_thing_not_got_nearer_is_let_go_and_passed_over_for_a_while():
    held = GoneFor()
    stuck = SimpleNamespace(number=1)
    other = SimpleNamespace(number=2)
    at = 0.0
    while at < 2.5:
        chosen = held.choose([(1.0, (0, 0), stuck), (1.2, (9, 9), other)], at)
        at += 0.1
    assert chosen[2] is other
    assert held.choose([(1.0, (0, 0), stuck), (1.2, (9, 9), other)], at + 0.1)[2] is other


def test_a_thing_gone_is_let_go_and_the_next_held():
    held = GoneFor()
    one, two = SimpleNamespace(number=1), SimpleNamespace(number=2)
    assert held.choose([(1.0, (0, 0), one), (1.1, (1, 1), two)], 0.0)[2] is one
    assert held.choose([(1.1, (1, 1), two)], 0.1)[2] is two
    assert held.number == 2


def test_changes_that_got_nothing_make_changing_harder_and_urgency_easier():
    held = GoneFor()
    before = held.margin
    held.changed_at, held.changed_to = 0.0, 7
    held.judge_changes(5.0)
    assert held.margin > before
    one, two = SimpleNamespace(number=1), SimpleNamespace(number=2)
    held.choose([(2.0, (0, 0), one)], 6.0)
    calm = held.choose([(2.0, (0, 0), one), (1.5, (1, 1), two)], 6.1)[2]
    pressed = held.choose([(2.0, (0, 0), one), (1.5, (1, 1), two)], 6.2, urgency=1.0)[2]
    assert calm is one and pressed is two
