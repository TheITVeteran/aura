"""Grouped inference retains real frame order without stopping current pixel tracking."""
from concurrent.futures import Future

import numpy as np
import pytest

from core.perception.learned_tracking import corrections_while_points_keep_moving as correcting
from core.perception.learned_tracking.contracts import PointResult, PointTrack
from core.perception.learned_tracking.points_for_the_motion_eyes import PointsForMotionEyes
from core.runtime.lockdep import checked_thread_semaphore
from tests.test_learned_corrections_do_not_hold_up_visible_tracking import marks, picture


def test_the_next_group_contains_each_actual_capture_while_the_worker_was_busy(monkeypatch):
    calls, futures = [], []
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.grouped_capture_sequence', 1))

    def submit(fn, pictures, reset, **_kwargs):
        calls.append((fn, pictures, reset))
        future = Future()
        futures.append(future)
        return future

    monkeypatch.setattr(correcting, 'submit_blocking_io', submit)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, sequence_required=True, limit=16)
    for at in (1., 1.1, 1.2, 1.3):
        p = picture(at)
        points.see_source(p.source, p.working, at)
    assert len(calls) == 1
    futures[0].set_result(correcting._Refinement((), calls[0][1][0]))
    p = picture(1.4)
    points.see_source(p.source, p.working, p.at)
    assert [p.at for p in calls[1][1]] == [1.1, 1.2, 1.3, 1.4]
    assert not calls[1][2]
    futures[1].set_result(correcting._Refinement((), calls[1][1][-1]))


@pytest.mark.parametrize('lost', ['batch-capacity', 'retained-anchor'])
def test_missing_sequence_history_starts_new_queries_instead_of_filling_frames(lost):
    points = correcting.CorrectionsWhilePointsMove(lambda: None, sequence_required=True, batch_frames=2)
    points._submitted_at = 1.
    frames = tuple(picture(at) for at in (1., 1.1, 1.2, 1.3))
    points._retained.extend(frames if lost == 'batch-capacity' else frames[1:3])
    batch, reset = points._batch(points._retained[-1])
    assert reset and len(batch) <= 2
    assert all(any(p is f for f in frames) for p in batch)


def test_a_window_answer_keeps_its_own_capture_when_later_frames_are_still_warming():
    class WindowTeacher:
        def clear(self):
            self.seen = []

        def see_source(self, source, working, at):
            self.seen.append(at)
            return marks(at) if len(self.seen) == 2 else ()

    points = correcting.CorrectionsWhilePointsMove(WindowTeacher, sequence_required=True)
    frames = tuple(picture(at) for at in (1., 1.1, 1.2))
    result = points._refine(frames, False)
    assert points._teacher.seen == [1., 1.1, 1.2]
    assert result.picture is frames[1] and all(p.at == 1.1 for p in result.points)
    current = correcting.visible_transport(result.points, list(frames[1:]), 256_000)
    assert current is not None and all(p.at == 1.2 for p in current.points)


def test_a_refinement_cannot_claim_an_unsubmitted_capture(monkeypatch):
    raw = Future()
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.wrong_capture_sequence', 1))
    monkeypatch.setattr(correcting, 'submit_blocking_io', lambda *_args, **_kwargs: raw)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, sequence_required=True, limit=16)
    p = picture(1.)
    points.see_source(p.source, p.working, p.at)
    raw.set_result(correcting._Refinement(marks(1.), picture(1.)))
    points._receive()
    assert points.correction_status == 'correction unavailable' and points._transport is None


def test_many_intervening_pictures_are_followed_off_the_fast_path(monkeypatch):
    calls, futures = [], []
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.offloop_visible_transport', 1))

    def submit(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        future = Future()
        futures.append(future)
        return future

    monkeypatch.setattr(correcting, 'submit_blocking_io', submit)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    for index in range(8):
        p = picture(1. + .1 * index)
        points.see_source(p.source, p.working, p.at)
    futures[0].set_result(correcting._Refinement(marks(1.), points._operation.pictures[0]))
    p = picture(1.8)
    points.see_source(p.source, p.working, p.at)
    assert len(calls) == 2 and calls[1][2]['label'] == 'learned_point_visible_transport'
    assert points._at == 1.8 and points._operation.stage == 'transport' and points._transport is None
    assert not futures[1].done() and any(p.pictures >= 8 for p in points.points)
    transported = calls[1][0](*calls[1][1])
    futures[1].set_result(transported)
    p = picture(1.9)
    points.see_source(p.source, p.working, p.at)
    assert points._transport is not None and points._transport._at == 1.9
    assert all(p.at == 1.9 for p in points._current_learned(1.9))
    futures[2].set_result(correcting._Refinement((), points._operation.pictures[0]))


def test_reset_forbids_a_transport_result_from_an_older_scene(monkeypatch):
    raw = Future()
    monkeypatch.setattr(correcting, '_SLOTS', checked_thread_semaphore('test.old_scene_transport', 1))
    p = picture(1.)
    points = correcting.CorrectionsWhilePointsMove(lambda: None, limit=16)
    points._operation = correcting._Correction(raw, (p,), stage='transport')
    points.clear()
    raw.set_result(None)
    points._receive()
    assert points._transport is None and points.correction_status == 'correction expired'


def test_new_resting_queries_do_not_discard_continuing_visible_marks():
    from dataclasses import replace

    frames = [picture(1.), picture(1.1)]
    previous = correcting.visible_transport(marks(1.), frames, 256_000)
    new_marks = tuple(replace(p, number=p.number + 20) for p in marks(1.))
    current = correcting.visible_transport(new_marks, frames, 256_000)
    assert previous is not None and current is not None
    previous._native_points = tuple(replace(p, old_x=p.x - 2.) for p in previous._native_points)
    combined = correcting._retain_visible_marks(previous, current, 8)
    assert {p.number for p in combined.points} == {p.number for p in previous.points}
    assert all(p.x - p.old_x == 2. and p.before == 1. and p.at == 1.1 for p in combined.points)
    assert len(combined.points) <= 8  # Two names on the same mark cannot multiply support.


@pytest.mark.parametrize('departure', ['time', 'geometry'])
def test_a_refinement_cannot_merge_mark_memory_from_another_reference(departure):
    frames = [picture(1.), picture(1.1)]
    previous = correcting.visible_transport(marks(1.), frames, 256_000)
    current = correcting.visible_transport(marks(1.), frames, 256_000)
    assert previous is not None and current is not None
    if departure == 'time':
        previous._at = 1.05
    else:
        previous._geometry = ((64, 81), (64, 81), (64, 81))
    assert correcting._retain_visible_marks(previous, current, 8) is current
    assert current.limit == 4


class WindowTracker:
    query_window_frames = 4

    def __init__(self, *, moving=False):
        self.moving = moving
        self.calls = []

    def start(self, frame, seeds):
        self.calls.append(frame)
        self.seeds = seeds
        self.first = frame.number
        return PointResult('invented', frame.epoch, frame.number, frame.at, frame.shape, (), .01, 'warming')

    def track(self, frame):
        elapsed = frame.number - self.first
        observed = elapsed % 4 == 3
        tracks = tuple(PointTrack(s.number, s.x + (2 * elapsed if self.moving else 0), s.y, .99, True)
                       for s in self.seeds) if observed else ()
        return PointResult('invented', frame.epoch, frame.number, frame.at, frame.shape, tracks, .01,
                           'observed' if observed else 'warming')


@pytest.mark.parametrize('moving', [False, True])
def test_window_queries_admit_new_texture_and_protect_continuing_moving_marks(monkeypatch, moving):
    from core.perception.learned_tracking import points_for_the_motion_eyes as bridge

    monkeypatch.setattr(bridge, '_new_points', lambda *_args: [(20., 20.), (30., 20.), (20., 30.), (30., 30.)])
    pixels = np.random.default_rng(45).integers(30, 230, (64, 80, 3), dtype=np.uint8)
    tracker = WindowTracker(moving=moving)
    points = PointsForMotionEyes(tracker, limit=4)
    for index in range(4):
        seen = points.see_source(pixels, pixels, 1. + .1 * index)
    assert len(tracker.calls) == 1 and seen
    assert all(p.pictures == 1 and p.before == p.at for p in seen)
    old = {s.number for s in tracker.seeds}
    changed = pixels.copy()
    changed[15:36, 15:42] = 255 - changed[15:36, 15:42]
    points.see_source(changed, changed, 1.4)
    assert len(tracker.calls) == (1 if moving else 2)
    if not moving:
        assert not old & {s.number for s in tracker.seeds}
    assert not points.points  # Warming positions cannot masquerade as visible motion.


def test_actual_motion_selection_accepts_grouped_causal_tracking(monkeypatch):
    from core.perception.learned_tracking import selection

    monkeypatch.setenv('AURA_PLAY_POINT_BACKEND', 'cotracker3')
    monkeypatch.setenv('AURA_PLAY_POINT_WEIGHTS', '/invented-verified-checkpoint')
    monkeypatch.setenv('AURA_PLAY_POINT_MODE', 'correcting')
    monkeypatch.setenv('AURA_PLAY_POINT_PIXEL_BUDGET', '1000000')
    eyes = selection.selected_motion_eyes()
    assert isinstance(eyes.points, correcting.CorrectionsWhilePointsMove) and eyes.points._sequence_required
    assert eyes.points._teacher is None
    assert eyes.points.pixel_budget == 1_000_000
    monkeypatch.setenv('AURA_PLAY_POINT_PIXEL_BUDGET', '1000001')
    with pytest.raises(ValueError, match='bounded source'):
        selection.selected_motion_eyes()


def test_capture_sequence_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert correcting._capture_batch_invariant() == ()
    assert correcting._visible_mark_retention_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'grouped_points_use_actual_capture_sequences')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
    check = next(t for t in suite.tests() if t.name == 'refinement_keeps_current_visible_marks')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())
