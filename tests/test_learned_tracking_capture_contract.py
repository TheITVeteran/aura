"""Independent visual models keep observations apart from hidden or supplied geometry."""
import numpy as np
import pytest

from core.perception.learned_tracking.contracts import (
    MaskResult,
    MaskTrack,
    ObjectSeed,
    PointResult,
    PointSeed,
    PointTrack,
    TrackingFrame,
    _tracking_visibility_invariant,
    following_frame,
    validate_objects,
    validate_points,
)


def test_a_capture_owns_its_pixels_and_visible_points_need_independent_visibility():
    pixels = np.zeros((20, 30, 3), np.uint8)
    frame = TrackingFrame(pixels, 1., 2, 3)
    pixels[:] = 255
    assert not frame.pixels.any() and not frame.pixels.flags.writeable
    assert _tracking_visibility_invariant() == ()
    with pytest.raises(ValueError):
        PointTrack(1, 3., 4., None, True)
    with pytest.raises(ValueError):
        PointResult('invented', 2, 3, 1., (20, 30), (PointTrack(1, -1., 3., .8, True),), .01)
    with pytest.raises(ValueError):
        PointResult('invented', 2, 3, 1., (20, 30), (PointTrack(1, 3., 4., .8, True),), .01, 'warming')


@pytest.mark.parametrize('departure', ['epoch', 'shape', 'number', 'time'])
def test_capture_continuation_cannot_cross_a_surface_generation_geometry_or_clock(departure):
    first = TrackingFrame(np.zeros((20, 30, 3), np.uint8), 1., 2, 3)
    shape = (21, 30) if departure == 'shape' else (20, 30)
    second = TrackingFrame(np.zeros((*shape, 3), np.uint8), 1. if departure == 'time' else 1.1,
                           3 if departure == 'epoch' else 2, 3 if departure == 'number' else 4)
    with pytest.raises(ValueError):
        following_frame(first, second)


def test_query_identity_is_bounded_unique_and_inside_the_current_capture():
    frame = TrackingFrame(np.zeros((20, 30, 3), np.uint8), 1., 2, 3)
    validate_points(frame, (PointSeed(1, 4., 5.),))
    for seeds in ((), (PointSeed(1, 4., 5.), PointSeed(1, 6., 7.)), (PointSeed(1, 35., 5.),)):
        with pytest.raises(ValueError):
            validate_points(frame, seeds)
    validate_objects(frame, (ObjectSeed(1, box=(1., 2., 5., 6.)),))
    with pytest.raises(ValueError):
        validate_objects(frame, (ObjectSeed(1, mask=np.zeros((21, 30), bool)),))


def test_seed_masks_are_immutable_proposals_and_cannot_grant_visible_measurement():
    mask = np.ones((20, 30), bool)
    seed = ObjectSeed(1, mask=mask)
    result = MaskResult('invented', 2, 3, 1., mask.shape, (MaskTrack(1, mask, None, False, True),), .01, 'seeded')
    mask[:] = False
    assert seed.mask.all() and result.masks[0].mask.all()
    with pytest.raises(ValueError):
        MaskTrack(1, np.ones((20, 30), bool), .99, True, True)
    with pytest.raises(ValueError):
        MaskResult('invented', 2, 3, 1., (21, 30), result.masks, .01)


def test_the_tracking_capture_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'occluded_tracking_is_not_a_visible_measurement')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
