"""She imagines the next moment under each plan she could follow, from what she has learned of the world, and does the
first part of the best: under what is worth catching, out from under what costs her, over what runs at her.

One-step rules, one per kind of play, were what she had before; a plan found by imagining needs none for its kind.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.agency.imagining_ahead import best_plan, plans_to_imagine
from core.agency.what_meeting_things_does import AVOID, MEET

pytestmark = pytest.mark.unit


class _Thing(SimpleNamespace):
    def where_at(self, after):
        return self.x + self.vx * after, self.y + self.vy * after


def _choosing(things, *, ways=None, lifts=(), paths=None):
    mine = _Thing(number=0, kind=0, x=200.0, y=280.0, w=30.0, h=20.0, vx=0.0, vy=0.0, moved=False)
    return SimpleNamespace(
        mine=mine, ways=ways if ways is not None else {"left": (-200.0, 0.0), "right": (200.0, 0.0)},
        moves=SimpleNamespace(shape=(300, 400)), others=lambda: [t for t, _s in things],
        stance=lambda thing: next(s for t, s in things if t is thing), physics=None, caution=1.0, response_s=0.0,
        lifts=list(lifts), press_paths=(lambda key: (paths or {}).get(key)), rest=(0.0, 0.0), barred=None)


def test_she_gets_under_what_falls_and_is_worth_catching():
    egg = _Thing(number=1, kind=1, x=330.0, y=40.0, w=16.0, h=16.0, vx=0.0, vy=260.0, moved=True)
    imagined = best_plan(_choosing([(egg, MEET)]), "", (None, None))
    assert imagined.decides and imagined.plan.first == "right" and imagined.gets == 1


def test_she_steps_out_from_under_what_falls_and_costs_her():
    rock = _Thing(number=2, kind=2, x=205.0, y=60.0, w=24.0, h=24.0, vx=0.0, vy=300.0, moved=True)
    imagined = best_plan(_choosing([(rock, AVOID)]), "", (None, None))
    assert imagined.decides and imagined.plan.first in ("left", "right") and imagined.costs == 0


def test_she_jumps_what_runs_at_her_when_she_cannot_get_out_of_its_way():
    runner = _Thing(number=3, kind=3, x=360.0, y=280.0, w=24.0, h=20.0, vx=-260.0, vy=0.0, moved=True)

    def jump(after):
        return 0.0, -max(0.0, 400.0 * after - 500.0 * after * after) if after < 0.8 else 0.0

    choosing = _choosing([(runner, AVOID)], ways={"left": (-60.0, 0.0), "right": (60.0, 0.0)}, lifts=("up",),
                         paths={"up": jump})
    imagined = best_plan(choosing, "", (None, None))
    assert imagined.decides and imagined.costs == 0 and imagined.plan.presses


def test_where_nothing_that_matters_is_in_reach_imagining_does_not_decide():
    far = _Thing(number=4, kind=4, x=20.0, y=20.0, w=10.0, h=10.0, vx=0.0, vy=0.0, moved=False)
    imagined = best_plan(_choosing([(far, MEET)]), "right", (None, None))
    assert imagined is not None and not imagined.decides


def test_the_plans_are_every_key_held_held_in_turn_and_pressed_partway_and_go_on_as_she_is_first():
    plans = plans_to_imagine(["", "left", "right"], ["up"], "left")
    assert plans[0].first == "left"
    assert any(len(plan.held) == 2 for plan in plans) and any(plan.presses for plan in plans)


def test_a_moment_is_imagined_in_the_time_of_a_picture():
    things = [(_Thing(number=n, kind=n, x=30.0 * n, y=10.0 * n, w=16.0, h=16.0, vx=20.0, vy=150.0, moved=True),
               MEET if n % 2 else AVOID) for n in range(1, 13)]

    def jump(after):
        return 0.0, -100.0 * after

    choosing = _choosing(things, ways={"left": (-200.0, 0.0), "right": (200.0, 0.0), "up": (0.0, -200.0),
                                       "down": (0.0, 200.0)}, lifts=("space",), paths={"space": jump})
    began = time.perf_counter()
    for _ in range(5):
        best_plan(choosing, "", (None, None))
    assert (time.perf_counter() - began) / 5 < 0.06


def test_a_body_carried_on_by_its_own_going_is_brought_onto_its_pad_slowly():
    """A lander falls under a pull and is held up by a key; coming onto the pad fast breaks it there."""
    pad = _Thing(number=5, kind=5, x=200.0, y=290.0, w=60.0, h=8.0, vx=0.0, vy=0.0, moved=False)
    pushes = {"": (0.0, 80.0), "up": (0.0, -160.0), "left": (-120.0, 80.0), "right": (120.0, 80.0)}
    hers = SimpleNamespace(carried=True, push_of=lambda key: pushes.get(key))
    meeting = SimpleNamespace(gentle_below=lambda kind: 40.0)
    choosing = _choosing([(pad, MEET)], ways={"up": (0.0, -60.0), "left": (-60.0, 0.0), "right": (60.0, 0.0)})
    choosing.hers, choosing.meeting = hers, meeting
    choosing.mine.y, choosing.mine.vy = 200.0, 120.0     # falling onto the pad, too fast to land on it
    imagined = best_plan(choosing, "", (None, None))
    assert imagined.decides and imagined.plan.first == "up" and imagined.costs == 0


def test_what_cannot_come_within_her_reach_is_left_out_before_she_plans():
    """Construal (Ho et al. 2022): a thing that cannot reach where she could be in the time imagined cannot change her
    route; a fast one from far off that can is imagined, a near one going away is not."""
    from types import SimpleNamespace

    import numpy as np

    from core.agency.imagining_ahead import STEP_S, _soonest_within_reach

    scene = SimpleNamespace(times=np.arange(1, 37) * STEP_S, start=(100.0, 100.0), size=(10.0, 10.0))
    small = SimpleNamespace(w=6.0, h=6.0)
    coming = _soonest_within_reach(lambda t: (400.0 - 300.0 * t, 100.0), scene, 50.0, small)       # 300 px off, fast
    going = _soonest_within_reach(lambda t: (160.0 + 300.0 * t, 100.0), scene, 50.0, small)        # 60 px off, leaving
    assert coming is not None and going is None
