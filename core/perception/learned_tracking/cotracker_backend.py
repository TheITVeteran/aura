"""CoTracker3 windows retain the newest capture and the caller's point identities.

The official online predictor advances eight frames in a sixteen-frame
window. Earlier estimates can change during a later window; only the newest
input is returned here. Between windows no current observation is supplied.
Weights are supplied locally. The upstream code and weights use CC-BY-NC 4.0.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from core.perception.learned_tracking.contracts import (
    PointResult,
    PointSeed,
    PointTrack,
    TrackingFrame,
    following_frame,
    validate_points,
)

SOURCE_REVISION = '82e02e8029753ad4ef13cf06be7f4fc5facdda4d'
CHECKPOINT_REVISION = 'acc21f9dba34834006a50432bde26b86aa68d794'
CHECKPOINT_SHA256 = '205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218'
VISIBLE_THRESHOLD = .6
MAX_HISTORY_FRAMES = 2048


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _probabilities(model: Any, count: int, points: int) -> np.ndarray | None:
    """Use the pinned model's visibility and confidence logits when exposed."""
    inner = getattr(model, 'model', None)
    visibility = getattr(inner, 'online_vis_predicted', None)
    confidence = getattr(inner, 'online_conf_predicted', None)
    if visibility is None or confidence is None:
        return None
    visibility, confidence = _numpy(visibility), _numpy(confidence)
    if (visibility.shape != confidence.shape or visibility.ndim != 3
            or visibility.shape[:2] != (1, count) or visibility.shape[2] < points
            or not np.isfinite(visibility).all() or not np.isfinite(confidence).all()):
        raise ValueError('CoTracker confidence does not match the observed history')
    last_v, last_c = visibility[0, -1, :points], confidence[0, -1, :points]
    return np.exp(-np.logaddexp(0., -last_v)) * np.exp(-np.logaddexp(0., -last_c))


class CoTrackerBackend:
    """A synchronous worker owns one bounded online tracking segment."""

    def __init__(self, weights: Path, device: str = 'cpu', *, model: Any = None,
                 max_history_frames: int = MAX_HISTORY_FRAMES) -> None:
        if not isinstance(weights, Path) or not isinstance(device, str) or not device:
            raise ValueError('CoTracker needs a local weights path and an explicit device')
        if type(max_history_frames) is not int or not 16 <= max_history_frames <= MAX_HISTORY_FRAMES:
            raise ValueError('CoTracker history needs a bounded frame limit')
        self.weights, self.device = weights, device
        self.model, self._torch = model, None
        self.max_history_frames = max_history_frames
        self._previous: TrackingFrame | None = None
        self._seeds: tuple[PointSeed, ...] = ()
        self._frames: deque[np.ndarray] = deque()
        self._count = 0
        self._step = 0
        self.query_window_frames: int | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.model = CoTrackerOnlinePredictor(checkpoint=str(self.weights)).to(self.device).eval()
        self._torch = torch

    def _clear(self) -> None:
        self._previous, self._seeds = None, ()
        self._frames.clear()
        self._count = 0

    def _call(self, video: np.ndarray, *, first: bool, queries: np.ndarray | None = None) -> Any:
        kwargs = dict(is_first_step=first, grid_size=0, add_support_grid=True)
        if self._torch is None:
            if queries is not None:
                kwargs['queries'] = queries
            return self.model(video_chunk=video, **kwargs)
        torch = self._torch
        video_tensor = torch.from_numpy(video).to(self.device)
        if queries is not None:
            kwargs['queries'] = torch.from_numpy(queries).to(self.device)
        with torch.inference_mode():
            return self.model(video_chunk=video_tensor, **kwargs)

    def start(self, frame: TrackingFrame, seeds: tuple[PointSeed, ...]) -> PointResult:
        """Query points initialize state; they do not establish visible measurements."""
        began = time.perf_counter()
        self._clear()
        validate_points(frame, seeds)
        self._load()
        step = getattr(self.model, 'step', None)
        if type(step) is not int or not 1 <= step <= 16 or 2 * step > self.max_history_frames:
            raise ValueError('CoTracker needs a supported online window step')
        self._step = step
        self.query_window_frames = 2 * step
        video = frame.pixels.transpose(2, 0, 1)[None, None].astype(np.float32)
        queries = np.asarray([[(0., seed.x, seed.y) for seed in seeds]], dtype=np.float32)
        self._call(video, first=True, queries=queries)
        self._previous, self._seeds = frame, tuple(seeds)
        self._frames = deque((frame.pixels,), maxlen=2 * step)
        self._count = 1
        return self._result(frame, (), began, 'warming')

    def track(self, frame: TrackingFrame) -> PointResult:
        """Only a complete causal window can supply the newest frame's tracks."""
        began = time.perf_counter()
        if self._previous is None:
            raise RuntimeError('CoTracker needs new point queries before tracking')
        try:
            following_frame(self._previous, frame)
        except ValueError:
            self._clear()
            raise
        if self._count >= self.max_history_frames:
            self._clear()
            return self._result(frame, (), began, 'unavailable')
        self._frames.append(frame.pixels)
        self._previous, self._count = frame, self._count + 1
        if self._count < 2 * self._step or self._count % self._step:
            return self._result(frame, (), began, 'warming')
        video = np.stack(self._frames).transpose(0, 3, 1, 2)[None].astype(np.float32)
        try:
            outputs = self._call(video, first=False)
            tracks = self._tracks(outputs, frame)
        except Exception:
            self._clear()
            raise
        return self._result(frame, tracks, began, 'observed')

    def _tracks(self, outputs: Any, frame: TrackingFrame) -> tuple[PointTrack, ...]:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError('CoTracker must return coordinates and visibility')
        coordinates, visible = (_numpy(value) for value in outputs)
        expected = (1, self._count, len(self._seeds))
        if (coordinates.shape != (*expected, 2) or visible.shape != expected
                or visible.dtype != np.bool_ or not np.isfinite(coordinates).all()):
            raise ValueError('CoTracker output does not match the current captured history')
        probabilities = _probabilities(self.model, self._count, len(self._seeds))
        found = []
        for index, seed in enumerate(self._seeds):
            x, y = (float(v) for v in coordinates[0, -1, index])
            predicted_visible = bool(visible[0, -1, index])
            # A public true flag gives a lower bound of .6, never a confidence of one.
            probability = (float(probabilities[index]) if probabilities is not None
                           else VISIBLE_THRESHOLD if predicted_visible else None)
            in_frame = 0. <= x < frame.shape[1] and 0. <= y < frame.shape[0]
            seen = bool(predicted_visible and probability is not None and probability >= VISIBLE_THRESHOLD and in_frame)
            found.append(PointTrack(seed.number, x, y, probability, seen, seed.object_number))
        return tuple(found)

    @staticmethod
    def _result(frame: TrackingFrame, tracks: tuple[PointTrack, ...], began: float, status: str) -> PointResult:
        return PointResult('cotracker3', frame.epoch, frame.number, frame.at, frame.shape,
                           tracks, time.perf_counter() - began, status)

    def close(self) -> None:
        """Retire query history and release the worker's model reference."""
        self._clear()
        self.model = None
