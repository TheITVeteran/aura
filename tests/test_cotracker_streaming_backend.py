"""An invented window tracker exposes causal buffering, reset and visibility mistakes."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.learned_tracking.contracts import PointSeed, TrackingFrame
from core.perception.learned_tracking.cotracker_backend import CoTrackerBackend

pytestmark = pytest.mark.unit


def _frame(number: int, *, epoch: int = 3, at: float | None = None, shape=(24, 40)) -> TrackingFrame:
    return TrackingFrame(np.full((*shape, 3), number % 255, np.uint8),
                         1. + number * .1 if at is None else at, epoch, number)


class _WindowTracker:
    step = 2

    def __init__(self, *, scores: bool = True):
        self.calls = []
        self.model = SimpleNamespace()
        self.scores = scores
        self.fail = False
        self.malformed = False
        self.outside = False
        self.hidden = False

    def __call__(self, video_chunk, *, is_first_step, grid_size, add_support_grid, queries=None):
        self.calls.append((video_chunk.copy(), is_first_step, queries))
        assert grid_size == 0 and add_support_grid
        if is_first_step:
            self.queries = queries.copy()
            self.count = 0
            return None, None
        if self.fail:
            raise RuntimeError('invented inference failed')
        self.count += len(video_chunk[0]) if not self.count else self.step
        count = self.count + int(self.malformed)
        points = self.queries.shape[1]
        coordinates = np.zeros((1, count, points, 2), np.float32)
        coordinates[:] = self.queries[..., 1:]
        coordinates[0, -1, :, 0] += float(video_chunk[0, -1, 0, 0, 0])
        if self.outside:
            coordinates[0, -1, 0, 0] = -3.
        visible = np.full((1, count, points), not self.hidden, bool)
        if self.scores:
            logits = np.full((1, count, points + 36), 2., np.float32)
            self.model.online_vis_predicted = logits
            self.model.online_conf_predicted = logits.copy()
        return coordinates, visible


def _started(model=None, **kwargs):
    model = model or _WindowTracker()
    backend = CoTrackerBackend(Path('/not-needed-with-invented-model.pth'), model=model, **kwargs)
    result = backend.start(_frame(0), (PointSeed(91, 5., 6., 7), PointSeed(22, 10., 11.)))
    assert result.status == 'warming' and not result.tracks
    return backend, model


def _first_output(backend):
    assert backend.track(_frame(1)).status == 'warming'
    assert backend.track(_frame(2)).status == 'warming'
    return backend.track(_frame(3))


def test_only_the_newest_real_input_is_returned_after_a_complete_window():
    backend, model = _started()
    result = _first_output(backend)
    assert result.status == 'observed'
    assert (result.epoch, result.frame_number, result.at, result.shape) == (3, 3, 1.3, (24, 40))
    assert [(point.number, point.x, point.object_number) for point in result.tracks] == [(91, 8., 7), (22, 13., None)]
    assert len(model.calls) == 2
    assert model.calls[-1][0].shape == (1, 4, 3, 24, 40)
    assert result.tracks[0].visibility == pytest.approx((1 / (1 + np.exp(-2.))) ** 2)
    assert result.tracks[0].visible


def test_between_windows_there_is_no_reused_observation_or_renewed_receipt():
    backend, model = _started()
    _first_output(backend)
    waiting = backend.track(_frame(4))
    assert waiting.status == 'warming' and waiting.frame_number == 4 and not waiting.tracks
    result = backend.track(_frame(5))
    assert result.frame_number == 5 and result.at == 1.5
    assert result.tracks[0].x == 10.
    assert model.calls[-1][0][0, :, 0, 0, 0].tolist() == [2., 3., 4., 5.]


def test_public_boolean_visibility_retains_a_known_lower_bound_without_claiming_certainty():
    backend, _model = _started(_WindowTracker(scores=False))
    result = _first_output(backend)
    assert result.tracks[0].visible and result.tracks[0].visibility == .6


@pytest.mark.parametrize('flag', ['hidden', 'outside'])
def test_occluded_or_outside_coordinates_are_predictions(flag):
    model = _WindowTracker()
    setattr(model, flag, True)
    backend, _ = _started(model)
    result = _first_output(backend)
    assert not result.tracks[0].visible


def test_public_visibility_cannot_overrule_low_model_confidence():
    backend, model = _started()
    original = model.__class__.__call__

    def low_score(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not kwargs['is_first_step']:
            self.model.online_conf_predicted.fill(-10.)
        return result

    model.__class__ = type('_LowScoreWindow', (_WindowTracker,), {'__call__': low_score})
    result = _first_output(backend)
    assert not result.tracks[0].visible and result.tracks[0].visibility < .01


@pytest.mark.parametrize('frame', [_frame(0), _frame(1, epoch=4), _frame(1, at=1.), _frame(1, shape=(24, 42))])
def test_foreign_duplicate_or_out_of_order_captures_are_rejected(frame):
    backend, model = _started()
    with pytest.raises(ValueError):
        backend.track(frame)
    assert len(model.calls) == 1


def test_restart_reinitializes_queries_and_keeps_the_callers_ids():
    backend, model = _started()
    _first_output(backend)
    backend.start(_frame(20, epoch=4), (PointSeed(301, 2., 3., 41),))
    backend.track(_frame(21, epoch=4))
    backend.track(_frame(22, epoch=4))
    result = backend.track(_frame(23, epoch=4))
    assert len(result.tracks) == 1 and result.tracks[0].number == 301
    assert result.tracks[0].object_number == 41 and result.epoch == 4
    assert sum(first for _video, first, _queries in model.calls) == 2


@pytest.mark.parametrize('flag', ['fail', 'malformed'])
def test_failed_or_malformed_inference_invalidates_the_segment(flag):
    model = _WindowTracker()
    setattr(model, flag, True)
    backend, _ = _started(model)
    backend.track(_frame(1))
    backend.track(_frame(2))
    with pytest.raises((RuntimeError, ValueError)):
        backend.track(_frame(3))
    with pytest.raises(RuntimeError):
        backend.track(_frame(4))


def test_a_changed_capture_basis_retires_old_queries():
    backend, _model = _started()
    with pytest.raises(ValueError):
        backend.track(_frame(1, epoch=4))
    with pytest.raises(RuntimeError):
        backend.track(_frame(2))


def test_history_exhaustion_requires_fresh_queries_instead_of_synthetic_frames():
    backend, model = _started(max_history_frames=16)
    for number in range(1, 16):
        backend.track(_frame(number))
    result = backend.track(_frame(16))
    assert result.status == 'unavailable' and not result.tracks
    with pytest.raises(RuntimeError):
        backend.track(_frame(17))
    assert max(len(video[0]) for video, _first, _queries in model.calls) <= 4


def test_weights_are_local_and_missing_weights_do_not_trigger_a_download(tmp_path):
    backend = CoTrackerBackend(tmp_path / 'missing.pth')
    with pytest.raises(FileNotFoundError):
        backend.start(_frame(0), (PointSeed(0, 4., 4.),))


@pytest.mark.parametrize('seeds', [(), (PointSeed(1, 4., 4.), PointSeed(1, 5., 5.)), (PointSeed(1, 400., 4.),)])
def test_invalid_queries_never_initialize_a_model(seeds):
    model = _WindowTracker()
    backend = CoTrackerBackend(Path('/unused.pth'), model=model)
    with pytest.raises(ValueError):
        backend.start(_frame(0), seeds)
    assert not model.calls
