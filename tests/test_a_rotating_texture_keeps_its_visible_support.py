"""A measured rigid transform carries current texture across a joined outline."""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.points_that_keep_their_motion import PointMotion
from core.perception.rigid_motion_from_named_points import RigidMotion
from core.perception.texture_that_continues_when_outlines_join import (
    could_continue,
    parts_from_continuing_texture,
)

pytestmark = pytest.mark.unit


def scene(turn):
    old = np.full((20, 30, 3), (240, 50, 10), np.uint8)
    old[2:17, 4:13] = (20, 170, 230)
    old[8:19, 15:29] = (220, 220, 80)
    body = SimpleNamespace(number=1, born=0., seen=1., size=600,
                           patch=old, support=np.ones((20, 30), bool),
                           box=lambda: (25., 30., 55., 50.))
    angle = -turn*math.pi/2.
    center = np.array((39.5, 39.5))
    rotation = RigidMotion(angle, (0., 0.), (), 0.)
    fit = RigidMotion(angle, tuple(center-rotation.apply(center)), (0, 1, 2, 3), 0.)
    starts = np.array(((28., 33.), (51., 33.), (28., 46.), (51., 46.)))
    delta = fit.apply((40., 40.))-(40., 40.)
    motion = PointMotion(*delta, .01, 4, 1., 1.1, (1, 2, 3, 4),
                         tuple(map(tuple, fit.apply(starts))), fit)
    rotated = np.rot90(old, turn)
    picture = np.full((80, 100, 3), 80, np.uint8)
    y, x = 40-rotated.shape[0]//2, 40-rotated.shape[1]//2
    picture[y:y+rotated.shape[0], x:x+rotated.shape[1]] = rotated
    actual = np.zeros(picture.shape[:2], bool)
    actual[y:y+rotated.shape[0], x:x+rotated.shape[1]] = True
    return picture, np.ones(picture.shape[:2], bool), actual, body, motion


def parts(picture, inside, body, motion):
    return parts_from_continuing_texture(picture, inside, left=0, top=0,
                                         things=[body], motions={1: motion}, at=1.1, difference=40.)


@pytest.mark.parametrize('turn', [-1, 1, 2])
def test_current_rotated_pixels_supply_the_proposal_and_keep_the_remaining_outline(turn):
    picture, inside, actual, body, motion = scene(turn)
    assert could_continue(body, {1: motion}, 1.1)
    found = parts(picture, inside, body, motion)
    assert len(found) == 2
    assert np.array_equal(found[0], actual)
    assert np.array_equal(np.sum(found, axis=0), inside)


def test_a_translation_receipt_cannot_supply_unmeasured_rotation():
    picture, inside, _actual, body, motion = scene(1)
    found = parts(picture, inside, body, replace(motion, rigid=None))
    assert len(found) == 1 and found[0] is inside


@pytest.mark.parametrize('departure', ['unseen', 'unowned', 'two', 'invalid-pose', 'stale'])
def test_a_rigid_hypothesis_without_current_pixels_and_mark_custody_cannot_supply_a_part(departure):
    picture, inside, actual, body, motion = scene(1)
    if departure == 'unseen':
        picture[actual] = (180, 0, 170)
    elif departure == 'unowned':
        inside[actual] = False
    elif departure == 'two':
        motion = replace(motion, support=2, points=(1, 2), locations=motion.locations[:2])
    elif departure == 'invalid-pose':
        motion = replace(motion, rigid=replace(motion.rigid, error=2.))
    else:
        motion = replace(motion, before=.9)
    found = parts(picture, inside, body, motion)
    assert len(found) == 1 and found[0] is inside


def test_the_rigid_texture_canary_has_a_registered_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'rotating_texture_requires_measured_rigid_pose')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


def test_the_normal_blob_path_retains_the_rotated_body_and_the_touching_outline():
    from core.perception.what_moves_in_the_picture import Thing, WhatMoves, _look_of

    picture, _inside, actual, old, motion = scene(1)
    picture[20:60, 50:75] = (180, 0, 170)
    joined = actual.copy()
    joined[20:60, 50:75] = True
    body = Thing(1, 40., 40., 30., 20., _look_of(old.patch.reshape(-1, 3)),
                 (240, 50, 10), old.patch, born=0., seen=1., support=old.support)
    eyes = WhatMoves(width=100)
    eyes._backdrop = np.full(picture.shape, 80, np.float32)
    eyes._now, eyes.things, eyes._point_motion = 1.1, {1: body}, {1: motion}
    blobs = eyes._blobs(picture, (joined.astype(np.uint8), joined.astype(np.uint8)))
    assert any((b['x'], b['y'], b['w'], b['h']) == (40., 40., 20., 30.) for b in blobs)
    assert any(b['colour'] == (180, 0, 170) for b in blobs)
