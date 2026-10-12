"""Continuing owned marks measure a path independently of changing box centres."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.position_from_the_same_points import (
    _point_path_invariant,
    measured_position,
    position_from_points,
)
from core.perception.translation_from_visible_support import motion_was_measured
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _body():
    return Thing(1, 40., 40., 20., 20., np.full(64, 1 / 64), (200, 200, 200),
                 np.full((20, 20, 3), 200, np.uint8), born=0., seen=1.,
                 support=np.ones((20, 20), bool), motion_evidence="unavailable")


def _blob(*, x=40., y=40., w=20., h=20.):
    return dict(x=x, y=y, w=w, h=h, look=np.full(64, 1 / 64), colour=(200, 200, 200),
                patch=np.full((int(h), int(w), 3), 200, np.uint8), support=np.ones((int(h), int(w)), bool))


def _motion(before, at, dx=0., dy=0., *, total_x=0., total_y=0., names=(1, 2, 3, 4)):
    marks = tuple((x + total_x, y + total_y) for x, y in ((34., 34.), (46., 34.), (34., 46.), (46., 46.)))
    return PointMotion(dx, dy, .01, len(names), before, at, names, marks[:len(names)])


def _seed():
    body = _body()
    assert position_from_points(body, _blob(), _motion(1., 1.1), 1.1) is None
    body.seen = 1.1
    return body


def test_owned_marks_keep_a_path_while_the_box_grows_at_one_edge():
    body = _seed()
    for step in range(1, 6):
        at = 1.1 + step * .04
        result = position_from_points(body, _blob(x=40 + step, w=20 + 2 * step),
                                      _motion(body.seen, at, dx=2, total_x=2 * step), at)
        assert result == ((40. + 2 * (step - 1), 40.), (40. + 2 * step, 40.))
        body.seen = at


def test_a_changing_box_cannot_move_stationary_marks():
    body = _seed()
    result = position_from_points(body, _blob(x=44, w=28), _motion(1.1, 1.2), 1.2)
    assert result == ((40., 40.), (40., 40.))


@pytest.mark.parametrize("change", ["replacement", "one_left", "birth", "gap", "backward", "stale",
                                  "outside", "no_mask", "wrong_mask_extent", "duplicate", "nonfinite", "missing"])
def test_an_unobserved_or_replacement_set_of_marks_cannot_extend_a_path(change):
    body, blob, at = _seed(), _blob(), 1.2
    motion = _motion(1.1, at, dx=2., total_x=2.)
    if change == "replacement":
        motion = _motion(1.1, at, dx=2., total_x=2., names=(5, 6, 7, 8))
    elif change == "one_left":
        motion = _motion(1.1, at, dx=2., total_x=2., names=(1, 5, 6, 7))
    elif change == "birth":
        body.born = 1.1
    elif change == "gap":
        at, motion = 2., _motion(1.1, 2., dx=2., total_x=2.)
    elif change == "backward":
        at, motion = 1., _motion(1.1, 1., dx=2., total_x=2.)
    elif change == "stale":
        motion = _motion(1., at, dx=2., total_x=2.)
    elif change == "outside":
        blob = _blob(x=90.)
    elif change == "no_mask":
        blob["support"] = None
    elif change == "wrong_mask_extent":
        blob["support"] = np.ones((40, 40), bool)
    elif change == "duplicate":
        motion = _motion(1.1, at, dx=2., total_x=2., names=(1, 1, 1, 1))
    elif change == "nonfinite":
        motion = _motion(1.1, at, dx=float("nan"), total_x=2.)
    elif change == "missing":
        motion = None
    assert position_from_points(body, blob, motion, at) is None


def test_new_points_cannot_pass_the_path_along_by_changing_one_name_at_a_time():
    body = _seed()
    for step, names in enumerate(((1, 2, 3, 5), (2, 3, 5, 6), (3, 5, 6, 7)), start=1):
        # Each new set has a majority in common with the preceding set, but
        # the original marks must supply every measured displacement.
        at = 1.1 + step * .04
        motion = _motion(body.seen, at, names=names)
        result = position_from_points(body, _blob(), motion, at)
        if step > 1:
            assert result is None
        body.seen = at


def test_collinear_marks_and_deforming_marks_do_not_locate_a_rigid_patch():
    body = _seed()
    line = PointMotion(0., 0., .01, 4, 1.1, 1.2, (1, 2, 3, 4), ((34., 40.), (38., 40.), (42., 40.), (46., 40.)))
    assert position_from_points(body, _blob(), line, 1.2) is None
    body = _seed()
    stretch = PointMotion(0., 0., .01, 4, 1.1, 1.2, (1, 2, 3, 4),
                          ((31., 31.), (49., 31.), (31., 49.), (49., 49.)))
    assert position_from_points(body, _blob(), stretch, 1.2) is None


def _move(moves, body, at, step, *, points=True):
    if points:
        moves._point_motion[body.number] = _motion(body.seen, at, dx=2., total_x=2. * step)
    else:
        moves._point_motion.clear()
    moves._moved(body, _blob(x=40 + step, w=20 + 2 * step), at)


def test_control_curves_use_the_point_path_while_geometry_keeps_the_observed_box():
    moves, body, presses = WhatMoves(), _body(), WhatAPressDoes()
    moves.things[1] = body
    _move(moves, body, 1.1, 0)
    for step in range(1, 5):
        _move(moves, body, 1.1 + step * .04, step)
    assert motion_was_measured(body, body.seen)
    assert measured_position(body) == (48., 40.) and body.x == 44.
    presses.pressed("right", body.seen, [body], epoch=1, independent=True)
    for step in range(5, 11):
        at = 1.1 + step * .04
        _move(moves, body, at, step)
        presses.saw({1: body}, at)
    assert body.vx == pytest.approx(50., abs=.01)
    assert body.x == 50. and measured_position(body) == (60., 40.)
    assert presses.entity_presses
    curve = presses.entity_presses[-1].samples
    assert curve[-1][1] == pytest.approx(0., abs=.01)  # constant drift was removed, using the same path


def test_key_trials_measure_point_displacement_rather_than_box_drift():
    moves, body, hers = WhatMoves(), _body(), WhichIsHers()
    moves.things[1] = body
    _move(moves, body, 1.1, 0)
    hers.holding("right", 1.1, trying=True)
    for step in range(1, 9):
        at = 1.1 + step * .04
        _move(moves, body, at, step)
        hers.saw(moves, [], at)
    speeds = hers._by_thing[1]
    assert speeds.timed[("right", 1.1)]
    assert all(vx == pytest.approx(50., abs=.01) for _at, vx, _vy in speeds.timed[("right", 1.1)])
    assert speeds.highest[0] == 56.


def test_losing_marks_ends_the_path_before_box_positions_can_replace_it():
    moves, body = WhatMoves(), _body()
    moves.things[1] = body
    _move(moves, body, 1.1, 0)
    _move(moves, body, 1.14, 1)
    assert body.motion_position is not None
    _move(moves, body, 1.18, 2, points=False)
    assert body.motion_position is None and not body.motion_path
    assert not motion_was_measured(body, 1.18)


def test_an_unmatched_box_cannot_keep_an_old_point_origin_with_a_new_stamp(monkeypatch):
    moves, body = WhatMoves(), _body()
    moves.things[1] = body
    _move(moves, body, 1.1, 0)
    _move(moves, body, 1.14, 1)
    monkeypatch.setattr(moves, "_still_there", lambda thing, picture: True)
    moves._the_unmatched(np.zeros((80, 80, 3), np.uint8), {}, 1.18)
    assert body.motion_position is None and body.point_pose is None
    assert not motion_was_measured(body, 1.18) and not body.motion_path


def test_a_current_stamp_cannot_make_an_old_point_position_current():
    body = SimpleNamespace(x=4., y=5., seen=2., motion_measured_at=1., motion_position=(30., 40.))
    assert measured_position(body) == (4., 5.)


def test_the_point_path_invariant():
    assert _point_path_invariant() == ()
