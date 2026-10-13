"""A quicker edge calculation cannot change what the current pixels establish."""
import warnings

import numpy as np
import pytest

from core.perception.boundary_support_for_translation import _median_of_three_depths

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('kept', [1, 2, 3, 4, 5, 6, 7])
def test_every_combination_of_missing_depths_keeps_the_observed_colour_median(kept):
    random = np.random.default_rng(19)
    pixels = random.uniform(-100., 300., (4, 9, 3, 3)).astype(np.float32)
    for depth in range(3):
        if not kept & (1 << depth):
            pixels[:, :, depth] = np.nan
    assert np.array_equal(_median_of_three_depths(pixels), np.nanmedian(pixels, axis=2))


def test_empty_depths_remain_missing_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.isnan(_median_of_three_depths(np.full((4, 9, 3, 3), np.nan, np.float32))).all()


def test_a_single_large_finite_sample_is_not_doubled_before_returning_it():
    pixels = np.full((1, 1, 3, 3), np.nan, np.float32)
    pixels[:, :, 0] = np.finfo(np.float32).max
    with np.errstate(over='raise'):
        assert np.array_equal(_median_of_three_depths(pixels), pixels[:, :, 0])


def test_two_sample_surroundings_keep_the_same_colour_and_contrast_boundary():
    from core.perception.background_visible_beside_a_patch import exposed_background

    picture = np.full((20, 20, 3), 80, np.uint8)
    picture[5:13, 6:14] = 240
    picture[:, 4], picture[:, 5] = 70, 90
    picture[:, 14], picture[:, 15] = 90, 70
    inside = np.ones((12, 8), bool)
    measured = exposed_background(picture, inside, left=6, top=5, difference=20.)
    assert int(measured.sum()) == 32
    picture[:, 4] = 40
    picture[:, 15] = 40
    picture[13:17, 6:14] = 150
    assert not exposed_background(picture, inside, left=6, top=5, difference=20.).any()


def test_the_bounded_edge_calculation_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite
    from core.perception.boundary_support_for_translation import _bounded_edge_samples_invariant

    assert _bounded_edge_samples_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'bounded_edge_samples_keep_their_observed_median')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
