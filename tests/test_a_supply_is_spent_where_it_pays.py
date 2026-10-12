"""A supply is spent where it pays most (core/agency/spending_a_supply.py).

A world of its own. Chances come at random, each worth something different; spending one unit of a supply takes a
chance's worth. Ten units for a minute of chances: spent on whatever comes first, they are gone in seconds; spent on
the chances that clear the bar, the same ten take far more. Where the supply refills faster than chances come, every
chance is taken. And which key spends which counter is found from the counter falling as the key is pressed.
"""
from __future__ import annotations

import random

import pytest

from core.agency.spending_a_supply import SpendingASupply

pytestmark = pytest.mark.unit


def _a_minute(choosing: bool, seed: int, *, supply: float = 10.0, refill: float = 0.0, seconds: float = 60.0) -> float:
    rand = random.Random(seed)
    spending, left, took, at = SpendingASupply(), supply, 0.0, 0.0
    while at < seconds:
        at += rand.expovariate(1.0)
        left = min(supply, left + refill * 1.0) if refill else left
        value = rand.lognormvariate(0.0, 1.0)
        spend = (spending.spend(at, value, left, seconds_left=seconds - at, refill=refill) if choosing
                 else left >= 1.0)
        if spend and left >= 1.0:
            left -= 1.0
            took += value
    return took


def test_a_short_supply_spent_on_what_clears_the_bar_takes_far_more():
    first_come = sum(_a_minute(False, seed) for seed in range(40))
    chosen = sum(_a_minute(True, seed) for seed in range(40))
    assert chosen > 1.6 * first_come, (first_come, chosen)


def test_a_supply_that_refills_faster_than_chances_come_is_spent_on_each():
    first_come = sum(_a_minute(False, seed, supply=3.0, refill=1.5) for seed in range(20))
    chosen = sum(_a_minute(True, seed, supply=3.0, refill=1.5) for seed in range(20))
    assert chosen >= 0.97 * first_come, (first_come, chosen)


def test_the_counter_a_key_spends_is_the_one_that_falls_as_it_is_pressed():
    spending, ammo, score, at = SpendingASupply(), 20, 0, 0.0
    for i in range(12):
        spending.pressed("space", at)
        spending.counted(at + 0.1, {"ammo": ammo, "score": score})
        ammo -= 1
        spending.counted(at + 0.2, {"ammo": ammo, "score": score})
        if i % 3 == 0:
            score += 1
        spending.pressed("left", at + 0.5)
        at += 1.0
    assert spending.spends("space") == "ammo"
    assert spending.spends("left") is None


def test_a_key_pressed_often_does_not_spend_what_falls_now_and_then():
    # Fire pressed four times a second; a life lost now and then, always just after a press.
    spending, lives, at = SpendingASupply(), 9, 0.0
    for i in range(80):
        spending.pressed("space", at)
        spending.counted(at + 0.05, {"lives": lives})
        if i % 10 == 9:
            lives -= 1
            spending.counted(at + 0.1, {"lives": lives})
        at += 0.25
    assert spending.spends("space") is None
