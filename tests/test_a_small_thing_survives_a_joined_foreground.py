"""Joined scenery cannot discard a distinct visible body by majority colour or area."""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.a_small_thing_in_a_joined_patch import (
    _small_joined_thing_invariant,
    a_recent_thing_inside,
    connected_parts_before_rejecting,
    keep_pixels_apart_from_the_floor,
)
from core.perception.what_moves_in_the_picture import Thing, WhatMoves

pytestmark = pytest.mark.unit


def _body() -> Thing:
    look = np.zeros(64)
    look[63] = 1
    return Thing(1, 20, 14, 4, 4, look, (240, 240, 240), np.full((4, 4, 3), 240, np.uint8),
                 born=0, seen=0, kind=0)


def test_a_small_thing_survives_a_whole_view_foreground_and_keeps_its_match():
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros((30, 40, 3), np.float32)
    eyes._now = .1
    eyes.things = {1: _body()}
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 18:22] = 240
    masks = (np.ones((30, 40), np.uint8), np.ones((30, 40), np.uint8))
    blobs = eyes._blobs(picture, masks)
    assert len(blobs) == 1
    assert (blobs[0]["x"], blobs[0]["y"], blobs[0]["w"], blobs[0]["h"]) == (20, 14, 4, 4)
    matched, fresh = eyes._match(blobs, .1)
    assert matched == {1: 0} and fresh == []


def test_majority_floor_pixels_do_not_erase_the_other_visible_pixels():
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros((30, 40, 3), np.float32)
    eyes._now = .1
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 18:22] = 240
    strict = np.zeros((30, 40), np.uint8)
    strict[5:25, 5:35] = 1
    blobs = eyes._blobs(picture, (strict, strict))
    assert len(blobs) == 1 and blobs[0]["colour"] == (240, 240, 240)
    assert np.all(eyes._backdrop[12:16, 18:22] == 0)
    assert np.all(eyes._backdrop[5:10, 5:35] == 80)


def test_a_previously_unseen_body_can_appear_inside_an_oversized_patch():
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros((30, 40, 3), np.float32)
    eyes._now = .1
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 18:22] = 240
    mask = np.ones((30, 40), np.uint8)
    blobs = eyes._blobs(picture, (mask, mask))
    assert len(blobs) == 1 and blobs[0]["colour"] == (240, 240, 240)


def test_floor_cleanup_does_not_join_two_disconnected_bodies():
    eyes = WhatMoves(width=40)
    eyes._backdrop = np.zeros((30, 40, 3), np.float32)
    eyes._now = .1
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 8:12] = picture[12:16, 28:32] = 240
    mask = np.zeros((30, 40), np.uint8)
    mask[5:25, 5:35] = 1
    blobs = eyes._blobs(picture, (mask, mask))
    assert sorted(blob["x"] for blob in blobs) == [10, 30]
    assert all(blob["w"] == blob["h"] == 4 for blob in blobs)


def test_no_surrounding_sample_and_no_recent_body_leave_pixels_unclaimed():
    picture = np.full((30, 40, 3), 80, np.uint8)
    inside = np.ones((30, 40), bool)
    backdrop = np.zeros_like(picture, dtype=np.float32)
    assert np.array_equal(keep_pixels_apart_from_the_floor(
        picture, inside, left=0, top=0, backdrop=backdrop, difference=20), inside)
    assert not backdrop.any()
    assert len(a_recent_thing_inside(picture, inside, left=0, top=0, things=[_body()], at=1)) == 1


def test_sparse_spanning_foreground_does_not_become_a_whole_view_body():
    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 18:22] = 240
    mask = np.zeros((30, 40), bool)
    mask[:, 0] = mask[-1, :] = True
    mask[12:16, 18:22] = True
    parts = connected_parts_before_rejecting(picture, mask, largest=300, cleaned=False)
    assert len(parts) == 1 and int(parts[0].sum()) == 16


def test_new_proposals_are_bounded_before_their_masks_are_kept():
    picture = np.full((60, 80, 3), 80, np.uint8)
    for y in range(2, 58, 6):
        for x in range(2, 78, 6):
            picture[y:y + 3, x:x + 3] = 240
    parts = connected_parts_before_rejecting(picture, np.ones((60, 80), bool), largest=1200,
                                           cleaned=False, proposals=5)
    assert len(parts) == 5
    assert all(int(part.sum()) == 9 for part in parts)


def test_the_joined_foreground_invariant():
    assert _small_joined_thing_invariant() == ()
