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
    assert first.tracks[0].visible and (first.tracks[0].x, first.tracks[0].y) == pytest.approx((36., 30.), abs=.001)
    assert not tracker.track(frame(3, hidden=True)).tracks[0].visible
    returned = tracker.track(frame(4, x=38))
    assert returned.tracks[0].visible and returned.tracks[0].number == 51
    assert returned.tracks[0].x == 48.


def test_identical_competing_appearances_do_not_grant_continuity():
    tracker = seeded()
    tracker.track(frame(2, hidden=True))
    result = tracker.track(frame(3, copy_at=(36, 20)))
    assert not result.tracks[0].visible


def test_a_continuing_visible_point_is_not_lost_when_a_duplicate_appears_elsewhere():
    result = seeded().track(frame(2, copy_at=(36, 20)))
    assert result.tracks[0].visible and result.tracks[0].x == pytest.approx(30., abs=.1)


def test_continuing_pixels_measure_fractional_displacement_without_quantizing_it():
    pixels = picture()
    tracker = ContinuityBackend()
    tracker.start(TrackingFrame(pixels, .1, 0, 1), (PointSeed(51, 30., 30.),))
    translated = ndimage.shift(pixels, (0., .5, 0.), order=1, mode='nearest')
    result = tracker.track(TrackingFrame(translated, .2, 0, 2))
    assert result.tracks[0].visible
    assert result.tracks[0].x == pytest.approx(30.5, abs=.1)


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
    assert (result.tracks[0].x, result.tracks[0].y) == pytest.approx((32., 32.), abs=.15)


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


def test_recovery_yields_at_its_budget_and_rotates_the_unmeasured_queries(monkeypatch):
    from core.perception.learned_tracking import continuity_backend as backend

    tracker = ContinuityBackend(recovery_budget_s=.012)
    tracker.start(frame(1), tuple(PointSeed(i, 30.+i, 30.) for i in range(4)))
    clock, calls = [0.], []
    monkeypatch.setattr(backend.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(tracker._flow, 'measure', lambda *_a: {})

    def slow_unmeasured(_frame, memory, _predicted, **_kw):
        calls.append(memory.seed.number)
        clock[0] += .01
        return None

    monkeypatch.setattr(tracker, '_match', slow_unmeasured)
    result = tracker.track(frame(2, hidden=True))
    assert calls == [0, 1] and not any(p.visible for p in result.tracks)
    result = tracker.track(frame(3, hidden=True))
    assert calls == [0, 1, 2, 3] and not any(p.visible for p in result.tracks)


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


def test_a_small_query_set_gets_its_first_observation_before_replacement():
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes

    points = PointsForMotionEyes(ContinuityBackend(max_points=192), limit=192)
    pixels = picture()
    assert not points.see_source(pixels, pixels, .1)
    names = {seed.number for seed in points._slots}
    moved = picture(x=23)
    first = points.see_source(moved, moved, .2)
    assert first and {seed.number for seed in points._slots} == names
    moved_again = picture(x=26)
    continuing = points.see_source(moved_again, moved_again, .3)
    assert any(p.pictures >= 2 and p.number in names and p.x-p.old_x > 2. for p in continuing)


def test_replacement_budget_follows_live_queries_instead_of_the_configured_ceiling(monkeypatch):
    from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes

    points = PointsForMotionEyes(ContinuityBackend(max_points=192), limit=192)
    pixels = picture()
    points.see_source(pixels, pixels, .1)
    points.see_source(pixels, pixels, .2)
    changed = picture(x=25)
    replacements = points._new_texture_queries(changed, changed.mean(axis=2)/255., .8)
    assert replacements and len(replacements) <= max(1, len(points._slots)//8)


def test_query_lifetime_is_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'queries_keep_their_first_observation_and_live_budget')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.mark.parametrize('options', [dict(max_points=0), dict(patch_radius=20), dict(search_radius=100),
                                  dict(hidden_frames=True), dict(longest_gap=float('nan'))])
def test_search_and_memory_budgets_are_enforced(options):
    with pytest.raises(ValueError):
        ContinuityBackend(**options)
