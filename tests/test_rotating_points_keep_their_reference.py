"""Named rigid marks locate their retained reference through rotation and changing proposals."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import FollowingPoints, PointMotion, SeenPoint
from core.perception.position_from_the_same_points import position_from_points
from core.perception.rigid_motion_from_named_points import (
    RigidMotion,
    _rigid_motion_invariant,
    rigid_motion,
)

pytestmark = pytest.mark.unit


def _rotate(points, angle, offset=(0., 0.)):
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(points) @ np.array(((c, s), (-s, c))) + offset


@pytest.mark.parametrize('angle', [0., .15, -.5, math.pi / 2, 2.8])
def test_spread_named_correspondences_fit_rotation_and_translation(angle):
    starts = np.array(((0., 0.), (12., 0.), (0., 12.), (12., 12.), (6., 3.)))
    ends = _rotate(starts, angle, (30., -7.))
    fit = rigid_motion(starts, ends)
    assert fit is not None and fit.inliers == tuple(range(len(starts)))
    assert fit.apply((6., 6.)) == pytest.approx(_rotate((6., 6.), angle, (30., -7.)))
    assert fit.inverse(ends) == pytest.approx(starts)


def test_a_minor_outlier_does_not_move_the_retained_origin():
    starts = np.array(((0., 0.), (12., 0.), (0., 12.), (12., 12.), (6., 3.)))
    ends = _rotate(starts, .6, (20., 30.))
    ends[-1] += (25., -10.)
    fit = rigid_motion(starts, ends)
    assert fit is not None and fit.inliers == (0, 1, 2, 3)
    assert fit.apply((6., 6.)) == pytest.approx(_rotate((6., 6.), .6, (20., 30.)))


@pytest.mark.parametrize('bad', ['two', 'line', 'reflection', 'scale', 'deform', 'missing', 'too_many'])
def test_an_unresolved_or_deforming_set_cannot_supply_a_rigid_pose(bad):
    starts = np.array(((0., 0.), (12., 0.), (0., 12.), (12., 12.)))
    ends = _rotate(starts, .5, (20., 30.))
    if bad == 'two':
        starts, ends = starts[:2], ends[:2]
    elif bad == 'line':
        starts = ends = np.array(((0., 0.), (5., 0.), (10., 0.)))
    elif bad == 'reflection':
        ends = starts * (-1., 1.)
    elif bad == 'scale':
        ends = starts * 1.8
    elif bad == 'deform':
        ends[:2] += (7., 3.)
    elif bad == 'missing':
        ends[0, 0] = float('nan')
    else:
        starts = ends = np.zeros((257, 2))
    assert rigid_motion(starts, ends) is None


def test_the_point_observer_measures_reference_motion_during_a_turn():
    starts = np.array(((34., 34.), (46., 34.), (34., 46.), (46., 46.)))
    ends = _rotate(starts - (40., 40.), .45) + (42., 39.)
    following = FollowingPoints()
    following.points = tuple(SeenPoint(i, *end, *start, 0., 1., 1.1, .01, 3)
                             for i, (start, end) in enumerate(zip(starts, ends, strict=True)))
    thing = SimpleNamespace(x=40., y=40., w=20., h=20., seen=1., support=np.ones((20, 20), bool))
    motion = following.motion_of(thing, 1.1)
    assert motion is not None and motion.rigid is not None
    assert (motion.dx, motion.dy) == pytest.approx((2., -1.))
    assert motion.support == 4 and motion.rigid.angle == pytest.approx(.45)


@pytest.mark.parametrize('degrees', [5., 8., 12.])
def test_rotated_pixels_supply_their_own_correspondences(degrees):
    from scipy import ndimage

    image = np.full((120, 160, 3), 25, np.uint8)
    image[35:65, 40:64] = np.random.default_rng(47).integers(35, 240, (30, 24, 3), dtype=np.uint8)
    body = SimpleNamespace(x=52., y=50., w=24., h=30., seen=1., support=np.ones((30, 24), bool))
    following = FollowingPoints()
    following.see(image, 1.)
    following.see(ndimage.rotate(image, -degrees, reshape=False, order=1, mode='constant', cval=25), 1.1)
    motion = following.motion_of(body, 1.1)
    expected = _rotate(np.array((body.x, body.y)) - (79.5, 59.5), math.radians(degrees)) + (79.5, 59.5)
    assert motion is not None and motion.has_rigid_pose()
    assert motion.rigid.apply((body.x, body.y)) == pytest.approx(expected, abs=.6)
    assert motion.rigid.angle == pytest.approx(math.radians(degrees), abs=.02)
    assert motion.support >= 3 and motion.before == 1. and motion.at == 1.1


def test_the_original_reference_survives_twenty_turns_without_following_the_box():
    center = np.array((40., 40.))
    marks = np.array(((34., 34.), (46., 34.), (34., 46.), (46., 46.)))
    names = (1, 2, 3, 4)
    body = SimpleNamespace(born=0., seen=1., point_pose=None, x=40., y=40.)
    blob = dict(x=40., y=40., w=36., h=36., support=np.ones((36, 36), bool))
    seed = PointMotion(0., 0., .01, 4, 1., 1.1, names, tuple(map(tuple, marks)))
    assert position_from_points(body, blob, seed, 1.1) is None
    body.seen = 1.1
    previous_true = center.copy()
    for step in range(1, 21):
        angle, shift = .17 * step, np.array((2. * step, -float(step)))
        actual = center + shift
        ends = _rotate(marks - center, angle) + actual
        offset = actual - _rotate(previous_true, .17)
        increment = RigidMotion(.17, tuple(offset), (0, 1, 2, 3), 0.)
        delta = increment.apply((body.x, body.y)) - (body.x, body.y)
        at = 1.1 + .04 * step
        motion = PointMotion(*delta, .01, 4, body.seen, at, names, tuple(map(tuple, ends)), increment)
        blob.update(x=actual[0] + (3. if step % 2 else -3.), y=actual[1] + 2.)
        before, position = position_from_points(body, blob, motion, at)
        assert before == pytest.approx(previous_true) and position == pytest.approx(actual)
        assert body.point_pose.origin == (40., 40.)
        body.x, body.y, body.seen = blob['x'], blob['y'], at
        previous_true = actual


def test_the_rigid_geometry_invariant_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _rigid_motion_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'rigid_point_motion_requires_spread_consensus')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('error', [float('nan'), float('inf'), -1., 2.])
def test_an_invalid_pose_receipt_cannot_seed_a_named_path(error):
    body = SimpleNamespace(born=0., seen=1., point_pose=None, x=40., y=40.)
    blob = dict(x=40., y=40., w=20., h=20., support=np.ones((20, 20), bool))
    names, locations = (1, 2, 3), ((34., 34.), (46., 34.), (34., 46.))
    pose = RigidMotion(.2, (0., 0.), (0, 1, 2), error)
    motion = PointMotion(0., 0., .01, 3, 1., 1.1, names, locations, pose)
    assert not motion.has_rigid_pose()
    assert position_from_points(body, blob, motion, 1.1) is None and body.point_pose is None


def test_named_rotating_marks_can_keep_a_tall_body_as_it_becomes_wide():
    from core.perception.what_moves_in_the_picture import WhatMoves

    following, moves = FollowingPoints(), WhatMoves()
    starts = np.array(((32., 47.), (48., 47.), (32., 53.), (48., 53.)))
    ends = _rotate(starts - (40., 50.), math.pi / 2) + (40., 50.)
    following.points = tuple(SeenPoint(i, *end, *start, 0., 1., 1.1, .01, 3)
                             for i, (start, end) in enumerate(zip(starts, ends, strict=True)))
    thing = SimpleNamespace(number=1, x=40., y=50., w=18., h=8., size=144., seen=1.,
                            vx=0., vy=0., path=[], look=np.array((.5, .5)), support=np.ones((8, 18), bool))
    blob = dict(x=40., y=50., w=8., h=18., look=thing.look, support=np.ones((18, 8), bool))
    assert moves._cost(thing, blob, 1.1) == math.inf
    moves._point_motion = {1: following.motion_of(thing, 1.1)}
    assert moves._cost(thing, blob, 1.1) < math.inf
    # A transform without its continuing named marks supplies no relaxation.
    motion = moves._point_motion[1]
    moves._point_motion[1] = PointMotion(0., 0., .01, 4, 1., 1.1, rigid=motion.rigid)
    assert moves._cost(thing, blob, 1.1) == math.inf
