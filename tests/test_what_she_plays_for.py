"""What she plays for, and what every other pull is worth in its terms (core/agency/what_she_plays_for.py).

Two worlds of their own. In a race to a target, each gain is taken by a quick way with a chance of a blow or a slow
safe way, against a clock: care that follows the stakes does as well as care held fixed, and far better where the
fixed care plays too safe. In a stream of threats,
some coming for her and some going their own way, time spent taking one out is time not spent gaining: never taking
them out and always taking them out each fail badly somewhere, and taking out what is worth it, priced from her record
of the runs before, does nearly as well as the better of the two everywhere.
"""
from __future__ import annotations

import random

import pytest

from core.agency.what_she_plays_for import (
    FIND_OUT,
    LAST,
    MOST,
    TARGET,
    WhatSheGoesFor,
    chance_of_reaching,
    count_expected,
)

pytestmark = pytest.mark.unit


# -- the end that leads ----------------------------------------------------------


def test_the_end_that_leads_is_the_ask_then_the_words_then_what_the_screen_counts():
    goes = WhatSheGoesFor()
    goes.saw(0.0, {"lives": 3})
    assert goes.lead() == LAST
    goes.saw(0.1, {"score": 0, "lives": 3})
    assert goes.lead() == MOST
    goes.told(words="Arrow keys to walk. Collect 60 stars to win.")
    assert goes.lead() == TARGET and goes.target == 60
    goes.told(asked="Have a look round this game and see what's in it.")
    assert goes.lead() == FIND_OUT
    counted_out_of = WhatSheGoesFor()
    counted_out_of.saw(0.0, {"fish": 2}, out_of={"fish": 40})
    assert counted_out_of.lead() == TARGET and counted_out_of.target == 40


def test_the_chances_are_those_of_blows_coming_at_random():
    # No blows to spare and a minute of play needed with a blow due every thirty seconds: e^-2.
    assert chance_of_reaching(60, 1, 1.0, 30.0, None) == pytest.approx(0.1353, abs=1e-3)
    assert chance_of_reaching(60, 1, 1.0, 30.0, 59.0) == 0.0
    assert count_expected(3, 1.0, 30.0, None) == pytest.approx(90.0)
    # With ten seconds left a third life is all but never reached: it adds almost nothing to what the first gives.
    third = count_expected(3, 1.0, 30.0, 10.0) - count_expected(2, 1.0, 30.0, 10.0)
    assert third < 0.05 * count_expected(1, 1.0, 30.0, 10.0)


# -- self-preservation -----------------------------------------------------------


def _at(goes: WhatSheGoesFor, at: float, count: int, lives: int, vitals: float | None = None) -> None:
    goes.saw(at, {"stars": count, "lives": lives}, vitals=vitals)


def test_with_the_target_all_but_sure_a_blow_is_worth_many_gains():
    # Ten of twenty got in twenty seconds, three lives, a minute left: one gain more barely moves the chance of getting
    # there, and a blow does. She plays safe.
    goes = WhatSheGoesFor()
    goes.told(words="Collect 20 stars.")
    _at(goes, 0.0, 0, 3)
    for i in range(1, 11):
        _at(goes, 2.0 * i, i, 3)
    in_gains = goes.worth_of_a_blow_s(20.0, 60.0) * goes.rate(20.0)
    assert in_gains > 10, in_gains


def test_with_the_clock_nearly_out_and_lives_in_hand_a_blow_is_cheap():
    goes = WhatSheGoesFor()
    _at(goes, 0.0, 0, 3)
    for i in range(1, 11):
        _at(goes, 3.0 * i, i, 3)
    assert goes.care(30.0, seconds_left=5.0) < 0.5 * goes.care(30.0, seconds_left=120.0)


def test_she_can_take_the_hits_left_in_her_bar_and_a_bar_in_each_life_to_come():
    goes = WhatSheGoesFor()
    goes.saw(0.0, {"score": 0, "lives": 2}, vitals=1.0)
    goes.saw(1.0, {"score": 1, "lives": 2}, vitals=0.8)
    goes.saw(2.0, {"score": 2, "lives": 2}, vitals=0.6)
    assert goes.blows_left() == 3 + 5


def test_as_her_bar_empties_against_a_clock_each_hit_is_worth_more():
    def hit_worth(vitals: float) -> float:
        goes = WhatSheGoesFor()
        goes.saw(0.0, {"score": 0}, vitals=1.0)
        goes.saw(1.0, {"score": 1}, vitals=0.8)
        for i in range(2, 20):
            goes.saw(float(i), {"score": i}, vitals=0.8)
        goes.saw(20.0, {"score": 20}, vitals=vitals)
        return goes.worth_of_a_blow_s(20.0, 40.0)

    assert hit_worth(0.2) > 2 * hit_worth(0.8)


def test_care_that_follows_the_stakes_wins_a_race_more_often_than_care_held_fixed():
    target, lives, clock = 20, 3, 70.0
    safe_s, quick_s, risk, set_back = 4.0, 2.0, 0.12, 1.5

    def race(follows: bool, seed: int, berth: float) -> bool:
        rand = random.Random(seed)
        goes = WhatSheGoesFor()
        goes.told(words=f"Collect {target} stars.")
        at, count, left = 0.0, 0, lives
        _at(goes, at, count, left)
        while at < clock and left > 0 and count < target:
            care = goes.care(at, clock - at) if follows else 1.0
            if quick_s + berth * care * risk < safe_s:
                at += quick_s
                if rand.random() < risk:
                    left, at = left - 1, at + set_back
                else:
                    count += 1
            else:
                at, count = at + safe_s, count + 1
            _at(goes, at, count, left)
        return count >= target and at <= clock

    # However wide the berth her play gives was set, care that follows the stakes does as well as care held at it, and
    # far better where the berth held fixed plays too safe to get there in time.
    gaps = []
    for berth in (5.0, 10.0, 14.0, 20.0):
        fixed = sum(race(False, seed, berth) for seed in range(400)) / 400
        following = sum(race(True, seed, berth) for seed in range(400)) / 400
        assert following >= fixed - 0.02, (berth, fixed, following)
        gaps.append(following - fixed)
    assert max(gaps) > 0.5, gaps


# -- threat elimination ----------------------------------------------------------

STEP = 0.1
HAZARD = {1: 0.05, 2: 0.004}
COMES = {1: True, 2: False}


def _a_run(policy: str, seed: int, goes: WhatSheGoesFor, *, shoot_s: float = 1.5, seconds: float = 10000.0,
           clock: bool = False, mix: tuple[int, ...] = (1, 2, 2), every_s: float = 4.0) -> int:
    """Gains come at one every two seconds while she is not taking a threat out; a threat comes every ``every_s``
    seconds and stays until it lands a blow, which sets her back and clears them all, or is taken out."""
    rand = random.Random(seed)
    goes.begin_run(0.0)
    at, count, lives, threats, busy, next_one, number = 0.0, 0, 3, [], 0.0, 0.0, 0
    goes.saw(at, {"score": count, "lives": lives})
    while at < seconds and lives > 0:
        if at >= next_one:
            number += 1
            threats.append((number, rand.choice(mix)))
            next_one = at + every_s
        goes.saw_threats(at, threats)
        if busy > 0:
            busy -= STEP
            if busy <= 0 and threats:
                goes.took_out(threats.pop(0)[0])
        else:
            if threats and policy != "never":
                pick = 0
                if policy == "priced":
                    left = seconds - at if clock else None
                    worth = [goes.worth_taking_out_s(kind, comes_for_her=COMES[kind], at=at, seconds_left=left)
                             for _number, kind in threats]
                    pick = max(range(len(worth)), key=worth.__getitem__)
                    pick = pick if worth[pick] > shoot_s else None
                if pick is not None:
                    threats.insert(0, threats.pop(pick))
                    busy = shoot_s
            if busy <= 0 and rand.random() < 0.5 * STEP:
                count += 1
        for _number, kind in list(threats):
            if rand.random() < HAZARD[kind] * STEP:
                lives, at = lives - 1, at + 2.0
                goes.blow_from(kind)
                threats.clear()
                break
        at += STEP
        goes.saw(at, {"score": count, "lives": lives})
    return count


PLACES = {
    "threats quick to take out": {},
    "a short clock": {"seconds": 20.0, "clock": True},
    "a long clock": {"seconds": 120.0, "clock": True},
    "slow to take out, harmless": {"shoot_s": 6.0, "mix": (2,)},
    "slow to take out, coming for her": {"shoot_s": 6.0, "mix": (1,)},
    "slow to take out, few": {"shoot_s": 6.0, "mix": (1,), "every_s": 12.0},
}


def test_threats_are_taken_out_where_it_is_worth_it_and_no_fixed_rule_does_as_well_everywhere():
    seeds = 60
    best_ratio = {"never": 1.0, "always": 1.0}
    for name, place in PLACES.items():
        fixed = {policy: sum(_a_run(policy, seed, WhatSheGoesFor(), **place) for seed in range(seeds)) / seeds
                 for policy in ("never", "always")}
        priced = 0.0
        for seed in range(seeds):
            goes = WhatSheGoesFor()
            for run in range(3):
                score = _a_run("priced", 1000 * seed + run, goes, **place)
            priced += score / seeds
        best = max(fixed.values())
        # The third run of a place, priced from the two before, does nearly as well as the better fixed rule.
        assert priced >= 0.85 * best, (name, fixed, priced)
        for policy, score in fixed.items():
            best_ratio[policy] = min(best_ratio[policy], score / max(best, priced))
    # Each fixed rule does less than half as well as the best somewhere.
    assert max(best_ratio.values()) < 0.5, best_ratio


# -- finding out, and when it stops paying -----------------------------------------


def test_a_thing_not_yet_known_is_worth_a_detour_by_what_the_kinds_met_have_been():
    kind_world = WhatSheGoesFor()
    kind_world.saw(0.0, {"score": 0, "lives": 3})
    assert kind_world.detour_for_the_unknown(1.0) == 2.0, "no record: the player's own measure stands"
    for kind in range(4):
        kind_world.met(kind, "gain")
    harsh = WhatSheGoesFor()
    harsh.saw(0.0, {"score": 0, "lives": 3})
    for kind in range(4):
        harsh.met(kind, "loss")
    assert kind_world.detour_for_the_unknown(1.0, alike_on_screen=2) < harsh.detour_for_the_unknown(1.0)
    looking = WhatSheGoesFor(asked="explore the level and find out what everything does")
    assert looking.detour_for_the_unknown(1.0) == 1.0


def test_a_count_that_has_stopped_moving_is_spent_and_said_once():
    goes = WhatSheGoesFor()
    for i in range(10):
        goes.saw(2.0 * i, {"score": 5 * i})
    assert not goes.spent(25.0)
    assert goes.spent(60.0)
    assert goes.what_to_say(60.0)
    assert not goes.what_to_say(61.0)
