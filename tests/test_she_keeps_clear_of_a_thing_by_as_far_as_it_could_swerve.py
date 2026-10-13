"""She keeps clear of what costs her by as far as it could swerve before she can answer it.

Offline 2026-10-13, put to the paint world's own code (tools/oracles/her_choices_on_a_world.py) at eight pictures a
second with an eighth of a second before her keys reached the world, she lost lives on four seeds of six and all of
them on one: a bug her straight-line reckoning had passing 1.5 pixels clear turned into her, as the world's code turns
a bug by up to 1.2 radians at a time. Kept clear by how much its kind's going has been seen to change, times the
time she takes to answer (core/agency/how_far_a_thing_could_swerve.py), she won all six and lost one life between them.
"""
from __future__ import annotations

import math

import pytest

from core.agency.how_far_a_thing_could_swerve import how_much_its_going_changes, swerves_of

pytestmark = pytest.mark.unit


def _path(turn_at: float | None, turn: float, speed: float = 40.0, every: float = 0.125, jitter: float = 0.0):
    out, x, y, heading = [], 100.0, 100.0, 0.0
    for i in range(20):
        at = i * every
        if turn_at is not None and abs(at - turn_at) < every / 2:
            heading += turn
        x += math.cos(heading) * speed * every
        y += math.sin(heading) * speed * every
        wobble = jitter * (1 if i % 2 else -1)
        out.append((at, x + wobble, y - wobble))
    return out


def test_a_thing_going_straight_could_swerve_by_nothing_however_it_is_seen():
    assert how_much_its_going_changes(_path(None, 0.0)) == 0.0
    assert how_much_its_going_changes(_path(None, 0.0, jitter=1.0)) == 0.0


def test_a_thing_that_turns_could_swerve_by_the_change_in_its_going():
    turned = how_much_its_going_changes(_path(1.0, 1.2, speed=80.0))
    assert turned >= 2 * 80.0 * math.sin(0.6) - 4 * math.sqrt(2) / 0.1 - 1e-6 and turned > 0.0
    assert how_much_its_going_changes(_path(1.0, 1.2, speed=80.0)) > how_much_its_going_changes(_path(1.0, 0.4, speed=80.0))


def test_a_kind_could_swerve_by_the_most_any_of_it_has_and_by_what_the_code_says():
    class _Thing:
        def __init__(self, kind, path):
            self.kind, self.path = kind, path

    things = {1: _Thing(3, _path(None, 0.0)), 2: _Thing(3, _path(1.0, 1.2, speed=80.0)), 3: _Thing(4, _path(None, 0.0))}
    seen = swerves_of(things)
    assert seen[3] == how_much_its_going_changes(things[2].path) and seen[4] == 0.0
    assert swerves_of(things, told={4: 40.0})[4] == 40.0


@pytest.mark.slow
def test_at_eight_pictures_a_second_she_paints_the_floor_and_bugs_seldom_touch_her():
    from tools.oracles.a_world_run_from_its_code import node_is_here
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld

    if not node_is_here():
        pytest.skip("node is not installed")
    lost = 0
    for seed in (1, 2, 3, 4, 5, 6):
        with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / 8, late_by=1) as play:
            came_to = play.run(150.0)
        assert came_to["ended"][-1]["why"] == "won", came_to
        lost += 3 - came_to["lives"]
    assert lost <= 2, lost


@pytest.mark.slow
def test_kept_clear_only_by_where_its_going_takes_it_she_is_touched(monkeypatch):
    import core.agency.how_far_a_thing_could_swerve as swerving
    from tools.oracles.a_world_run_from_its_code import node_is_here
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld

    if not node_is_here():
        pytest.skip("node is not installed")
    monkeypatch.setattr(swerving, "swerves_of", lambda things, told=None: {})
    lost = 0
    for seed in (1, 3, 6):
        with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / 8, late_by=1) as play:
            lost += 3 - play.run(150.0)["lives"]
    assert lost >= 3, lost
