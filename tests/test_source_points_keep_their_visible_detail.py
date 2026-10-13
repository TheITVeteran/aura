"""Visible source marks retain geometry while outlines use a smaller picture."""
import math

import numpy as np
import pytest
from scipy import ndimage

from core.perception.picture_arithmetic import shrink
from core.perception.points_before_the_picture_is_reduced import (
    PointsBeforeReduction,
    _source_point_geometry_invariant,
)

pytestmark = pytest.mark.unit


def render(center=(150., 100.), angle=0., *, replacement=False, present=True, size=(320, 200), seed=18):
    yy, xx = np.mgrid[:size[1], :size[0]]
    places = np.stack((xx, yy), axis=-1).astype(float)
    c, s = math.cos(angle), math.sin(angle)
    local = (places - center) @ np.array(((c, -s), (s, c)))
    inside = (abs(local[..., 0]) <= 18.) & (abs(local[..., 1]) <= 14.)
    texture = np.random.default_rng(seed).integers(0, 80, (29, 37, 3), dtype=np.uint8).astype(float)
    texture += (30., 150., 80.)
    if replacement:
        texture = texture[..., (1, 2, 0)]  # identical grayscale, different visible colour
    picture = np.full((*places.shape[:2], 3), 25., dtype=float)
    if present:
        colours = np.stack([ndimage.map_coordinates(texture[..., k], [local[..., 1] + 14., local[..., 0] + 18.],
                                                    order=1, mode='nearest') for k in range(3)], axis=-1)
        picture[inside] = colours[inside]
    return picture.astype(np.uint8)


def working(source, width=80):
    return shrink(source, width, round(width * source.shape[0] / source.shape[1]))


@pytest.mark.parametrize('angle', [0., .12, -.18])
@pytest.mark.parametrize('shift', [(1.25, -.75), (5., 3.)])
def test_retained_source_marks_follow_translation_and_local_rotation_in_working_coordinates(angle, shift):
    before, center = np.array((150., 100.)), np.array((150. + shift[0], 100. + shift[1]))
    tracker = PointsBeforeReduction(limit=48)
    first = render()
    original = tracker.see_source(first, working(first), 1.)
    rgb = render(center, angle)
    points = tracker.see_source(rgb, working(rgb), 1.1)
    names = {p.number for p in original}
    matched = [p for p in points if p.number in names and p.pictures == 2]
    assert len(matched) >= 3
    rotation = np.array(((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))))
    errors = []
    for point in matched:
        expected = ((np.array((point.old_x, point.old_y)) / .25 - before) @ rotation + center) * .25
        errors.append(np.linalg.norm(expected - (point.x, point.y)))
        assert point.before == 1. and point.at == 1.1 and point.born == 1.
    assert max(errors) < .4


@pytest.mark.parametrize('replacement,present', [(True, True), (False, False)])
def test_colour_alias_or_absent_pixels_cannot_keep_interior_mark_names(replacement, present):
    tracker = PointsBeforeReduction(limit=64)
    rgb = render()
    first = tracker.see_source(rgb, working(rgb), 1.)
    interior = {p.number for p in first if abs(p.x / .25 - 150.) < 10. and abs(p.y / .25 - 100.) < 7.}
    assert interior
    changed = render(replacement=replacement, present=present)
    points = tracker.see_source(changed, working(changed), 1.1)
    assert not interior & {p.number for p in points}


@pytest.mark.parametrize('change', ['gap', 'resize', 'working resize', 'reset', 'repeat'])
def test_capture_or_geometry_departure_starts_new_marks(change):
    tracker = PointsBeforeReduction(limit=32)
    rgb = render()
    old = {p.number for p in tracker.see_source(rgb, working(rgb), 1.)}
    if change == 'reset':
        tracker.clear()
    if change == 'resize':
        rgb = render(size=(360, 225))
    small = working(rgb, width=96 if change == 'working resize' else 80)
    at = 2. if change == 'gap' else 1. if change == 'repeat' else 1.1
    new = tracker.see_source(rgb, small, at)
    assert not old & {p.number for p in new}
    assert all(p.born == at and p.pictures == 1 for p in new)


def test_original_source_resize_still_retires_marks_when_bounded_tracking_sizes_match():
    tracker = PointsBeforeReduction(limit=32, pixel_budget=64_000)
    rgb = render(size=(640, 400))
    first = tracker.see_source(rgb, working(rgb), 1.)
    rgb = render(size=(960, 600))
    second = tracker.see_source(rgb, working(rgb), 1.1)
    assert not {p.number for p in first} & {p.number for p in second}


def test_pixel_and_point_budgets_are_physical_bounds():
    tracker = PointsBeforeReduction(limit=12, pixel_budget=32_000)
    source = render(size=(960, 600))
    tracker.see_source(source, working(source), 1.)
    assert tracker._image[0].size <= 32_000 and len(tracker.points) <= 12
    assert max(p.x for p in tracker.points) < 80. and max(p.y for p in tracker.points) < 50.
    assert not PointsBeforeReduction(limit=0).see_source(source, working(source), 1.)


def test_unusable_shapes_or_times_leave_correspondence_unobserved():
    tracker = PointsBeforeReduction()
    source = render()
    assert tracker.see_source(source, working(source), 1.)
    assert not tracker.see_source(source, working(source), math.nan)
    assert not tracker.points and tracker._colours is None
    assert not tracker.see_source(np.zeros((16, 100_000, 3), dtype=np.uint8), np.zeros((16, 80, 3), dtype=np.uint8), 2.)


def test_the_actual_observer_keeps_source_pixels_for_its_marks():
    from core.perception.what_moves_in_the_picture import WhatMoves

    observer = WhatMoves(width=80)
    rgb = render()
    observer.see(rgb, 1.)
    assert isinstance(observer.points, PointsBeforeReduction)
    assert observer.points._image[0].shape == rgb.shape[:2]
    assert observer.shape == (50, 80) and observer.points.points


def test_source_point_geometry_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _source_point_geometry_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'source_marks_return_to_one_working_geometry')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
