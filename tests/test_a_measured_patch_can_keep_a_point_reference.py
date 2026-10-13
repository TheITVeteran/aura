"""Independent visible support can carry a calibrated reference across missing marks."""
import numpy as np
import pytest

from core.perception.boundary_support_for_translation import visible_boundary
from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.position_from_the_same_points import measured_position
from core.perception.translation_from_visible_support import motion_was_measured
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _blob(x=40., width=20, colour=200):
    image = np.full((100, 180, 3), 30, np.uint8)
    left, top = round(x - width / 2), 30
    image[top:top + 20, left:left + width] = colour
    support = np.ones((20, width), bool)
    return dict(x=x, y=40., w=float(width), h=20., look=np.full(64, 1 / 64),
                colour=(colour,) * 3, patch=image[top:top + 20, left:left + width].copy(), support=support,
                boundary=visible_boundary(image, left, top, left + width, top + 20, support))


def _motion(before, at, total, *, dx=2., names=(1, 2, 3, 4)):
    marks = tuple((x + total, y) for x, y in ((34., 34.), (46., 34.), (34., 46.), (46., 46.)))
    return PointMotion(dx, 0., .01, 4, before, at, names, marks)


def _established():
    blob, eyes = _blob(), WhatMoves()
    body = Thing(1, 40., 40., 20., 20., blob['look'], blob['colour'], blob['patch'], born=0., seen=1.,
                 support=blob['support'], boundary=blob['boundary'], motion_evidence='unavailable')
    eyes.things[1] = body
    eyes._point_motion[1] = _motion(1., 1.1, 0., dx=0.)
    eyes._moved(body, blob, 1.1)
    eyes._point_motion[1] = _motion(1.1, 1.2, 2.)
    eyes._moved(body, _blob(43., 22), 1.2)
    assert measured_position(body) == (42., 40.) and body.x == 43.
    eyes._point_motion.clear()
    return eyes, body


def test_matching_pixels_and_exterior_edges_keep_the_calibrated_reference():
    eyes, body = _established()
    for at, x in ((1.3, 45.), (1.4, 47.), (1.5, 49.)):
        eyes._moved(body, _blob(x, 22), at)
        assert motion_was_measured(body, at)
        assert measured_position(body) == (x - 1., 40.)
        assert body.motion_path[-1] == (at, x - 1., 40.)
    assert body.vx == pytest.approx(20.) and body.motion_evidence == 'support'


def test_replacement_marks_join_through_the_independent_patch_measurement():
    eyes, body = _established()
    eyes._moved(body, _blob(45., 22), 1.3)
    eyes._point_motion[1] = _motion(1.3, 1.4, 6., names=(5, 6, 7, 8))
    eyes._moved(body, _blob(47., 22), 1.4)
    assert measured_position(body) == (46., 40.)
    assert body.point_pose.origin == (46., 40.)
    eyes._point_motion[1] = _motion(1.4, 1.5, 8., names=(5, 6, 7, 8))
    eyes._moved(body, _blob(49., 22), 1.5)
    assert motion_was_measured(body, 1.5) and body.motion_evidence == 'points'
    assert measured_position(body) == (48., 40.) and body.point_pose.origin == (46., 40.)


def _edge_marks(body, *, dx=2.):
    return PointMotion(dx, 0., .01, 4, body.seen, 1.3, (1, 2, 3, 4),
                       ((33., 34.), (45., 34.), (33., 46.), (45., 46.)))


def test_missing_corner_ownership_cannot_discard_agreeing_whole_patch_motion():
    eyes, body = _established()
    blob, marks = _blob(45., 22), _edge_marks(body)
    assert not marks.inside(blob)
    eyes._point_motion[1] = marks
    assert eyes._cost(body, blob, 1.3) < float('inf')
    eyes._moved(body, blob, 1.3)
    assert motion_was_measured(body, 1.3) and measured_position(body) == (44., 40.)
    assert body.motion_evidence == 'support' and body.point_pose is None


@pytest.mark.parametrize('missing', ['edges', 'pixels', 'agreement'])
def test_off_patch_points_still_block_a_match_without_independent_agreement(missing):
    eyes, body = _established()
    blob, marks = _blob(45., 22), _edge_marks(body, dx=15. if missing == 'agreement' else 2.)
    if missing == 'edges':
        blob['boundary'] = None
    elif missing == 'pixels':
        blob = _blob(45., 22, 70)
    eyes._point_motion[1] = marks
    assert eyes._cost(body, blob, 1.3) == float('inf')
    eyes._moved(body, blob, 1.3)
    assert not motion_was_measured(body, 1.3)


@pytest.mark.parametrize('change', ['edges', 'pixels', 'size', 'birth', 'gap', 'stamp', 'contradicted'])
def test_an_unmeasured_or_changed_patch_cannot_bridge_the_reference(change):
    eyes, body = _established()
    at, blob = 1.3, _blob(47., 22)
    if change == 'edges':
        blob['boundary'] = None
    elif change == 'pixels':
        blob = _blob(47., 22, 70)
    elif change == 'size':
        blob = _blob(47., 28)
    elif change == 'birth':
        body.born = 1.2
    elif change == 'gap':
        at = 1.6
    elif change == 'stamp':
        body.motion_measured_at = 1.1
    else:
        eyes._point_motion[1] = _motion(1.2, at, 4., dx=0., names=(5, 6, 7, 8))
    eyes._moved(body, blob, at)
    assert not motion_was_measured(body, at) and body.motion_position is None
    assert not body.motion_path


def test_the_reference_bridge_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.a_reference_kept_by_visible_support import _reference_bridge_invariant

    assert _reference_bridge_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_reference_bridge_needs_independent_visible_support')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
