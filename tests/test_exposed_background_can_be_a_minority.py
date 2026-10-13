"""The body need not be the minority for exposed scenery to leave its foreground."""
import math

import numpy as np
import pytest

from core.perception.a_small_thing_in_a_joined_patch import keep_pixels_apart_from_the_floor
from core.perception.background_visible_beside_a_patch import (
    _exposed_background_invariant,
    exposed_background,
)
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_local_opposite_sides_separate_a_body_from_its_exposed_wake(axis, direction):
    picture = np.repeat((80 + np.arange(30) * 2)[:, None, None], 40, axis=1)
    picture = np.repeat(picture, 3, axis=2).astype(np.uint8)
    picture[8:20, 16:24] = 240
    top = 5 if direction < 0 else 8
    inside = np.ones((15, 8), bool)
    left = 16
    if axis:
        picture, inside, left, top = picture.transpose(1, 0, 2), inside.T, top, left
    backdrop = np.zeros_like(picture, dtype=np.float32)
    kept = keep_pixels_apart_from_the_floor(picture, inside, left=left, top=top,
                                           backdrop=backdrop, difference=20.)
    here = picture[top:top + inside.shape[0], left:left + inside.shape[1]]
    assert int(kept.sum()) == 96
    assert np.all(here[kept] == 240)
    assert np.all(backdrop[top:top + inside.shape[0], left:left + inside.shape[1]][kept] == 0)
    assert np.array_equal(backdrop[top:top + inside.shape[0], left:left + inside.shape[1]][~kept], here[~kept])


def test_enclosed_matching_body_details_do_not_supply_an_open_background_path():
    picture = np.full((20, 20, 3), 80, np.uint8)
    picture[5:13, 6:14] = 240
    picture[8:10, 9:11] = 80
    floor = exposed_background(picture, np.ones((12, 8), bool), left=6, top=5, difference=20.)
    assert int(floor.sum()) == 32
    assert not floor[3:5, 3:5].any()


def test_padding_around_the_strict_mask_still_connects_exposed_background_to_the_outside():
    picture = np.full((20, 20, 3), 80, np.uint8)
    picture[5:13, 6:14] = 240
    inside = np.zeros((16, 12), bool)
    inside[2:14, 2:10] = True
    floor = exposed_background(picture, inside, left=4, top=3, difference=20.)
    assert int(floor.sum()) == 32
    assert not floor[~inside].any()


def test_texture_cannot_be_called_quiet_background_by_averaging_its_colours():
    y, x = np.indices((20, 20))
    picture = np.repeat((((x + y) % 2) * 255)[:, :, None], 3, axis=2).astype(np.uint8)
    picture[5:15, 5:15] = 128
    assert not exposed_background(picture, np.ones((10, 10), bool), left=5, top=5, difference=20.).any()


def test_disagreeing_opposite_sides_supply_no_background_correction():
    picture = np.full((20, 20, 3), 120, np.uint8)
    picture[5:15, :5] = 0
    picture[5:15, 15:] = 240
    picture[:5, 5:15] = 40
    picture[15:, 5:15] = 200
    assert not exposed_background(picture, np.ones((10, 10), bool), left=5, top=5, difference=20.).any()


@pytest.mark.parametrize("difference", [0., -1., math.inf, math.nan])
def test_an_unavailable_difference_cannot_remove_foreground(difference):
    picture = np.full((20, 20, 3), 80, np.uint8)
    assert not exposed_background(picture, np.ones((10, 10), bool), left=5, top=5, difference=difference).any()


def test_a_full_view_patch_has_no_opposite_side_observation():
    picture = np.full((20, 20, 3), 80, np.uint8)
    assert not exposed_background(picture, np.ones((20, 20), bool), left=0, top=0, difference=20.).any()


def test_the_motion_eyes_keep_the_body_box_instead_of_its_longer_wake():
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros((30, 40, 3), np.float32)
    eyes._now = .1
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[5:19, 16:24] = 240
    mask = np.zeros((30, 40), np.uint8)
    mask[5:25, 16:24] = 1
    blobs = eyes._blobs(picture, (mask, mask))
    assert len(blobs) == 1
    assert (blobs[0]["x"], blobs[0]["y"], blobs[0]["w"], blobs[0]["h"]) == (20, 12, 8, 14)
    assert np.all(eyes._backdrop[19:25, 16:24] == 80)
    assert not eyes._backdrop[5:19, 16:24].any()


def test_exposed_background_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _exposed_background_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "exposed_background_keeps_enclosed_body_detail")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
