"""New moving regions receive observations without inheriting scenery's point identities."""
import math

import numpy as np
import pytest
from scipy import ndimage

from core.perception.picture_arithmetic import shrink
from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction, _colours
from core.perception.points_shared_with_changing_regions import (
    _point_capacity_invariant,
    share_point_capacity,
)
from core.perception.points_that_keep_their_motion import SeenPoint, _grey

pytestmark = pytest.mark.unit


def test_new_changing_regions_get_slots_while_quiet_reference_marks_remain():
    quiet = [SeenPoint(i, 10. + i, 10., 10. + i, 10., 0., 1., 1.1, .01, 20) for i in range(12)]
    fresh = [SeenPoint(i, 10. + i, 30., 10. + i, 30., 1.1, 1.1, 1.1, 0., 1) for i in range(12, 18)]
    activity = np.zeros((48, 64), bool)
    activity[25:35] = True
    kept = share_point_capacity(quiet + fresh, activity, (1., 1.), 9)
    assert len(kept) == 9 and sum(p.number >= 12 for p in kept) == 6
    assert sum(p.number < 12 for p in kept) == 3
    assert all(p.pictures == 1 and p.before == p.at and p.born == 1.1 for p in kept if p.number >= 12)


def test_continuing_motion_keeps_its_marks_when_another_region_appears():
    moving = [SeenPoint(i, 10. + i, 10., 8. + i, 10., 0., 1., 1.1, .01, 5) for i in range(6)]
    fresh = [SeenPoint(i, 10. + i, 30., 10. + i, 30., 1.1, 1.1, 1.1, 0., 1) for i in range(6, 15)]
    activity = np.ones((48, 64), bool)
    kept = share_point_capacity(moving + fresh, activity, (1., 1.), 9)
    assert len(kept) == 9 and {p.number for p in moving} <= {p.number for p in kept}


@pytest.mark.parametrize('which', ['quiet', 'changing'])
def test_unused_reservations_return_to_the_available_regions(which):
    points = [SeenPoint(i, 10. + i, 10., 10. + i, 10., 1., 1., 1., 0., 1) for i in range(12)]
    activity = np.full((48, 64), which == 'changing', bool)
    assert len(share_point_capacity(points, activity, (1., 1.), 9)) == 9


def _scene(index):
    texture = np.random.default_rng(91).integers(30, 230, (48, 64, 3), dtype=np.uint8)
    source = np.full((384, 640, 3), 30, np.uint8)
    for x in range(25, 630, 60):
        for y in range(25, 380, 60):
            source[y:y + 8, x:x + 8] = 200
    if index:
        left = 120 + 2 * index
        source[160:208, left:left + 64] = texture
    return source


def test_the_actual_source_tracker_admits_and_follows_a_new_patch_among_quiet_regions():
    points = PointsBeforeReduction(limit=48, pixel_budget=1_000_000)
    first = _scene(0)
    original = points.see_source(first, shrink(first, 160, 96), 1.)
    old = {p.number for p in original}
    supported = 0
    for index in range(1, 15):
        source = _scene(index)
        seen = points.see_source(source, shrink(source, 160, 96), 1. + index * .05)
        on = [p for p in seen if (120 + 2 * index) / 4 <= p.x < (184 + 2 * index) / 4 and 40 <= p.y < 52]
        if index == 1:
            assert any(p.number not in old and p.pictures == 1 and p.before == p.at for p in on)
        supported += sum(p.pictures >= 2 and math.hypot(p.x - p.old_x, p.y - p.old_y) > .2 for p in on) >= 3
        assert len(seen) <= 48
    assert supported >= 8


def test_change_does_not_keep_marks_when_their_visible_pixels_disappear():
    points = PointsBeforeReduction(limit=48, pixel_budget=1_000_000)
    quiet = _scene(0)
    points.see_source(quiet, shrink(quiet, 160, 96), .9)
    first = _scene(1)
    previous = points.see_source(first, shrink(first, 160, 96), 1.)
    interior = {p.number for p in previous if 34 <= p.x < 44 and 43 <= p.y < 49}
    assert interior
    source = _scene(0)
    seen = points.see_source(source, shrink(source, 160, 96), 1.1)
    assert not interior & {p.number for p in seen}


@pytest.mark.parametrize('shape', [(48, 64), (200, 320), (576, 960)])
def test_shared_color_filter_preserves_planes_and_brightness(shape):
    source = np.random.default_rng(91).integers(0, 256, (*shape, 3), dtype=np.uint8)
    old = np.stack([ndimage.gaussian_filter(source[..., c].astype(np.float32) / 255., .6)
                    for c in range(3)], axis=-1)
    current = _colours(source)
    np.testing.assert_array_equal(current, old)
    np.testing.assert_allclose(current.mean(axis=2), _grey(source), atol=2e-7, rtol=0)


def test_changing_capacity_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _point_capacity_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'changing_regions_receive_point_capacity')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
