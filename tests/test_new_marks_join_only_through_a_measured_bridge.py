"""New landmarks enter an established origin through current independent support."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.position_from_the_same_points import (
    _point_handoff_invariant,
    position_from_points,
)

pytestmark = pytest.mark.unit
PLACES = {1: (34., 34.), 2: (46., 34.), 3: (34., 46.), 4: (46., 46.),
          5: (40., 30.), 6: (30., 40.), 7: (50., 40.), 8: (40., 50.), 9: (35., 39.), 10: (45., 45.)}


def _body():
    return SimpleNamespace(born=0., seen=1., point_pose=None)


def _observe(body, step, names, *, places=PLACES):
    at = 1.1 + step * .04
    shifted = tuple((places[n][0] + 2 * step, places[n][1] - step) for n in names)
    motion = PointMotion(2. if step else 0., -1. if step else 0., .01, len(names), body.seen, at, names, shifted)
    blob = dict(x=40. + step, y=40., w=40., h=40., support=np.ones((40, 40), bool))
    result = position_from_points(body, blob, motion, at)
    body.seen = at
    return result


def test_gradual_landmark_turnover_keeps_the_origin_and_independently_known_displacement():
    body = _body()
    assert _observe(body, 0, (1, 2, 3, 4)) is None
    sets = ((1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8),
            (4, 5, 6, 7, 8, 9), (5, 6, 7, 8, 9, 10))
    for step, names in enumerate(sets, 1):
        before, now = _observe(body, step, names)
        assert before == (40. + 2 * (step - 1), 40. - (step - 1))
        assert now == (40. + 2 * step, 40. - step)
        assert body.point_pose.origin == (40., 40.)
        assert len(body.point_pose.anchors) <= len(names)
    assert not set(body.point_pose.anchors) & {1, 2, 3, 4}


def test_two_prior_marks_cannot_admit_an_arbitrary_new_set():
    body = _body()
    assert _observe(body, 0, (1, 2, 3, 4)) is None
    assert _observe(body, 1, (1, 2, 5, 6, 7, 8)) is None


def test_collinear_prior_marks_cannot_be_rescued_by_new_marks_that_spread_out():
    body = _body()
    places = {1: (32., 34.), 2: (40., 34.), 3: (48., 34.), 4: (40., 46.), 5: (34., 48.)}
    assert _observe(body, 0, (1, 2, 3, 4), places=places) is None
    assert _observe(body, 1, (1, 2, 3, 5), places=places) is None


def test_new_landmarks_are_a_minority_and_cannot_take_over_on_the_next_picture():
    body = _body()
    assert _observe(body, 0, (1, 2, 3, 4)) is None
    places = dict(PLACES)
    places.update({n: (30. + n % 15, 30. + n % 13) for n in range(11, 100)})
    assert _observe(body, 1, (1, 2, 3, 4, *range(11, 100)), places=places) is not None
    anchors = body.point_pose.anchors
    assert len(anchors) == 6 and len(set(anchors) - {1, 2, 3, 4}) == 2
    assert _observe(body, 2, tuple(range(11, 100)), places=places) is None


def test_new_marks_must_own_current_pixels_before_they_can_join_the_origin():
    body = _body()
    assert _observe(body, 0, (1, 2, 3, 4)) is None
    names = (1, 2, 3, 4, 5)
    motion = PointMotion(0., 0., .01, 5, body.seen, 1.2, names, tuple(PLACES[n] for n in names))
    support = np.ones((40, 40), bool)
    support[10, 20] = False  # the fifth landmark has no visible ownership
    blob = dict(x=40., y=40., w=40., h=40., support=support)
    assert position_from_points(body, blob, motion, 1.2) is not None
    assert 5 not in body.point_pose.anchors


def test_landmark_admission_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _point_handoff_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "new_point_marks_require_a_measured_origin_bridge")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
