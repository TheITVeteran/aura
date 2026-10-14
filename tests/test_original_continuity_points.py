"""Fresh visible pixels must support correspondence, including after a hidden interval."""
import sys

import numpy as np
import pytest
from scipy import ndimage

from core.perception.learned_tracking.continuity_backend import (
    ContinuityBackend,
    _continuity_invariant,
)
from core.perception.learned_tracking.contracts import PointSeed, TrackingFrame

pytestmark = pytest.mark.unit


def picture(x=20, y=20, *, hidden=False, copy_at=None, bright=0):
    pixels = np.full((90, 120, 3), 20, np.uint8)
    texture = np.random.default_rng(791).integers(40, 210, (24, 24, 3), dtype=np.uint8)
    if not hidden:
        pixels[y:y+24, x:x+24] = texture.astype(np.int16) + bright
    if copy_at is not None:
        cx, cy = copy_at
        pixels[cy:cy+24, cx:cx+24] = texture
    return pixels


def frame(number, **kwargs):
    return TrackingFrame(picture(**kwargs), number*.1, 0, number)


def seeded(**kwargs):
    tracker = ContinuityBackend(**kwargs)
    tracker.start(frame(1), (PointSeed(51, 30., 30., 4),))
    return tracker


def test_original_tracker_reacquires_identity_without_crediting_hidden_predictions():
    assert _continuity_invariant() == ()
    tracker = seeded()
    first = tracker.track(frame(2, x=26))
    assert first.tracks[0].visible and (first.tracks[0].x, first.tracks[0].y) == (36., 30.)
    assert not tracker.track(frame(3, hidden=True)).tracks[0].visible
    returned = tracker.track(frame(4, x=38))
    assert returned.tracks[0].visible and returned.tracks[0].number == 51
    assert returned.tracks[0].x == 48.


def test_identical_competing_appearances_do_not_grant_continuity():
    result = seeded().track(frame(2, copy_at=(36, 20)))
    assert not result.tracks[0].visible


def test_constant_channel_brightness_changes_can_preserve_texture():
    result = seeded().track(frame(2, x=24, bright=12))
    assert result.tracks[0].visible and result.tracks[0].x == 34.


@pytest.mark.parametrize('angle', [-30., -15., 15., 30.])
def test_local_rotation_retains_a_named_material_point(angle):
    pixels = np.full((90, 120, 3), 20, np.uint8)
    texture = np.random.default_rng(714).integers(40, 210, (25, 25, 3), dtype=np.uint8)
    pixels[20:45, 20:45] = texture
    tracker = ContinuityBackend()
    tracker.start(TrackingFrame(pixels, .1, 0, 1), (PointSeed(51, 32., 32.),))
    pixels[20:45, 20:45] = ndimage.rotate(texture, angle, axes=(1, 0), reshape=False, order=1, mode='nearest')
    result = tracker.track(TrackingFrame(pixels, .2, 0, 2))
    assert result.tracks[0].visible and result.tracks[0].number == 51
    assert (result.tracks[0].x, result.tracks[0].y) == (32., 32.)


def test_a_flat_region_cannot_claim_a_query_observation():
    tracker = ContinuityBackend()
    tracker.start(frame(1), (PointSeed(1, 80., 60.),))
    assert not tracker.track(frame(2)).tracks[0].visible


def test_capture_generation_and_geometry_cannot_lend_identity():
    tracker = seeded()
    with pytest.raises(ValueError):
        tracker.track(TrackingFrame(picture(), .2, 1, 2))
    with pytest.raises(ValueError):
        tracker.track(TrackingFrame(np.zeros((50, 50, 3), np.uint8), .2, 0, 2))
    with pytest.raises(ValueError):
        tracker.track(frame(1))


def test_a_replacement_is_a_seed_and_cannot_reuse_the_previous_identity():
    tracker = seeded()
    result = tracker.track(frame(2), replacements={0: PointSeed(99, 30., 30., 4)})
    assert result.tracks[0].number == 99 and not result.tracks[0].visible
    assert tracker.track(frame(3)).tracks[0].visible
    with pytest.raises(ValueError):
        tracker.track(frame(4), replacements={1: PointSeed(100, 30., 30.)})


def test_hidden_memory_has_a_finite_reacquisition_lifetime():
    tracker = seeded(hidden_frames=2)
    tracker.track(frame(2, hidden=True))
    tracker.track(frame(3, hidden=True))
    assert not tracker.track(frame(4)).tracks[0].visible
    assert len(tracker._memory) == 1
    tracker.close()
    assert tracker._memory == [] and tracker._previous is None


def test_selection_reaches_real_motion_eyes_without_torch_or_a_checkpoint(monkeypatch):
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes
    from core.perception.learned_tracking.selection import selected_motion_eyes

    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'continuity_points')
    monkeypatch.setenv('AURA_PLAY_POINT_MODE', 'direct')
    monkeypatch.delenv('AURA_PLAY_POINT_WEIGHTS', raising=False)
    before = {name for name in sys.modules if name.startswith(('torch', 'cotracker', 'tapnet'))}
    eyes = selected_motion_eyes()
    assert isinstance(eyes.points, PointsForMotionEyes)
    assert isinstance(eyes.points._backend, ContinuityBackend)
    pixels = picture()
    eyes.points.see_source(pixels, pixels, .1)
    eyes.points.see_source(picture(x=23), picture(x=23), .2)
    points = eyes.points.see_source(picture(x=26), picture(x=26), .3)
    assert len(points) >= 3 and any(p.x-p.old_x >= 2. for p in points)
    after = {name for name in sys.modules if name.startswith(('torch', 'cotracker', 'tapnet'))}
    assert before == after
    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'cotracker3')
    with pytest.raises(ValueError, match='supported backend'):
        selected_motion_eyes()


@pytest.mark.parametrize('options', [dict(max_points=0), dict(patch_radius=20), dict(search_radius=100),
                                  dict(hidden_frames=True), dict(longest_gap=float('nan'))])
def test_search_and_memory_budgets_are_enforced(options):
    with pytest.raises(ValueError):
        ContinuityBackend(**options)
