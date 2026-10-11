"""What the place does to her whatever she does is learned: a pull by where it pulls, a clock by how it falls, a bar by
how it drains, and things coming faster (core/agency/what_the_place_does_to_her.py)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.what_the_place_does_to_her import Pressures, is_a_clock

pytestmark = pytest.mark.unit


def _her(x, y, vx, vy):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy, w=16, h=16)


def test_a_belt_in_one_part_of_the_place_pulls_her_there_and_not_elsewhere():
    pressures = Pressures()
    at = 0.0
    for step in range(60):
        at += 1 / 30
        # The lower half carries her right at 60 a second; the upper half leaves her be.
        pressures.saw(at=at, mine=_her(40 + step, 250, 60.0, 0.0), held=False)
        pressures.saw(at=at, mine=_her(40 + step, 40, 0.0, 0.0), held=False)
    assert pressures.pull_at(60, 250) == pytest.approx((60.0, 0.0))
    assert pressures.pull_at(60, 40) == (0.0, 0.0)


def test_her_going_while_a_key_is_held_is_no_pull():
    pressures = Pressures()
    for step in range(60):
        pressures.saw(at=step / 30, mine=_her(100, 100, 150.0, 0.0), held=True)
    assert pressures.pull_at(100, 100) == (0.0, 0.0)


def test_a_clock_falling_a_second_a_second_says_how_long_is_left_and_presses_more_as_it_goes():
    pressures = Pressures()
    urgency = []
    for tenth in range(0, 600, 5):
        at = tenth / 10
        pressures.saw(at=at, mine=None, held=False, counters={"time": 60 - int(at), "score": int(at) // 7})
        urgency.append(pressures.urgency())
    assert pressures.seconds_left() == pytest.approx(1.0, abs=1.5)
    assert urgency[20] < 0.2 and urgency[-1] > 0.8
    assert all(b >= a - 0.05 for a, b in zip(urgency, urgency[1:], strict=False))


def test_a_number_alone_falling_a_second_a_second_is_a_clock_and_a_rising_one_is_not():
    falling, rising = Pressures(), Pressures()
    for second in range(20):
        falling.saw(at=float(second), mine=None, held=False, counters={"number at 0.50,0.0": 90 - second})
        rising.saw(at=float(second), mine=None, held=False, counters={"number at 0.50,0.0": second})
    assert falling.seconds_left() == pytest.approx(71.0, abs=2.0)
    assert rising.seconds_left() is None and rising.urgency() == 0.0


def test_a_score_going_down_now_and_then_is_no_clock():
    pressures = Pressures()
    for second in range(30):
        pressures.saw(at=float(second), mine=None, held=False, counters={"score": 100 - 10 * (second // 10)})
    assert pressures.seconds_left() is None


def test_what_she_has_draining_presses_as_it_nears_empty():
    pressures = Pressures()
    for second in range(40):
        pressures.saw(at=float(second), mine=None, held=False, vitals=1.0 - 0.02 * second)
    assert pressures.seconds_left() == pytest.approx(11.0, abs=2.0)
    assert pressures.urgency() > 0.5


def test_things_coming_faster_as_the_run_goes_on_make_it_harder():
    pressures = Pressures()
    for tick in range(0, 1200):
        at = tick / 20
        speed = 100.0 if at < 20 else 220.0
        pressures.saw(at=at, mine=None, held=False, others=[SimpleNamespace(moved=True, vx=speed, vy=0.0)])
    assert pressures.harder(60.0) == pytest.approx(2.2, abs=0.05)


def test_a_new_run_starts_its_clock_afresh_and_keeps_what_the_place_pulls():
    pressures = Pressures()
    for second in range(20):
        pressures.saw(at=float(second), mine=_her(10, 10, 0.0, 80.0), held=False, counters={"time": 30 - second})
    pressures.begin_run()
    assert pressures.seconds_left() is None
    assert pressures.pull_at(10, 10) == pytest.approx((0.0, 80.0))


def test_the_words_of_a_clock():
    assert is_a_clock("TIME") and is_a_clock("seconds left") and not is_a_clock("SCORE")
