"""An initial solid body's corners retain its full connected extent."""
import numpy as np
import pytest

from core.perception.the_whole_patch_around_a_visible_corner import (
    _whole_patch_invariant,
    complete_visible_patches,
)
from core.perception.what_moves_in_the_picture import WhatMoves

pytestmark = pytest.mark.unit


def test_a_still_solid_body_is_seen_whole_instead_of_four_corners():
    image = np.full((180, 240, 3), (40, 30, 60), np.uint8)
    image[60:100, 30:60] = (70, 140, 230)
    moves = WhatMoves()
    for n in range(15):
        moves.see(image, n * .05)
    bodies = [t for t in moves.things.values() if t.colour == (70, 140, 230)]
    assert len(bodies) == 1
    assert bodies[0].box() == (30., 60., 60., 100.)


def test_connected_background_cannot_expand_a_corner_seed_over_the_scene():
    image = np.full((60, 80, 3), 80, np.uint8)
    backdrop = image.astype(np.float32)
    backdrop[10:13, 10:13] = 0
    before = backdrop.copy()
    result = complete_visible_patches(image, backdrop, difference=40, largest=1200, longest=40)
    assert np.array_equal(result, before)


def test_a_different_patch_of_the_same_colour_needs_its_own_visible_seed():
    image = np.full((60, 80, 3), 30, np.uint8)
    image[12:32, 10:30] = image[12:32, 50:70] = (60, 150, 230)
    backdrop = image.astype(np.float32)
    backdrop[12:15, 10:13] = 30
    result = complete_visible_patches(image, backdrop, difference=40, largest=1200, longest=40)
    assert np.all(result[12:32, 10:30] == 30)
    assert np.array_equal(result[12:32, 50:70], image[12:32, 50:70])


def test_the_whole_patch_invariant():
    assert _whole_patch_invariant() == ()
