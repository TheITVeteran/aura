"""Publish learned point correspondences through the existing measured motion geometry.

New texture gets a bounded share of query slots. Continuing moving marks
keep their identities; hidden points supply no velocity or control evidence.
The classical path's object support and rigid-motion checks remain in use.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from core.perception.learned_tracking.contracts import PointSeed, TrackingFrame
from core.perception.picture_arithmetic import shrink
from core.perception.points_that_keep_their_motion import (
    FollowingPoints,
    SeenPoint,
    _grey,
    _new_points,
    _sample,
)
from core.verify import invariant


def _same_capture(result: Any, frame: TrackingFrame) -> bool:
    return (result.epoch == frame.epoch and result.frame_number == frame.number
            and result.at == frame.at and result.shape == frame.shape)


class PointsForMotionEyes(FollowingPoints):
    """One worker-owned causal tracker, expressed in the motion eyes' working pixels."""

    def __init__(self, backend: Any, *, limit: int = 64, pixel_budget: int = 256_000,
                 longest_gap: float = 1.) -> None:
        super().__init__(limit=limit, longest_gap=longest_gap)
        if not 3 <= self.limit <= 384 or type(pixel_budget) is not int or not 256 <= pixel_budget <= 1_000_000:
            raise ValueError('learned point observation needs bounded point and pixel budgets')
        self._backend = backend
        self.pixel_budget = pixel_budget
        self._epoch, self._sequence = 0, 0
        self._geometry: tuple | None = None
        self._previous_pixels: np.ndarray | None = None
        self._native_visible: dict[int, SeenPoint] = {}
        self._slots: tuple[PointSeed, ...] = ()
        self._births: dict[int, float] = {}
        self.status = 'not started'

    def clear(self) -> None:
        super().clear()
        self._epoch += 1
        self._geometry, self._previous_pixels = None, None
        self._native_visible.clear()
        self._slots, self._births = (), {}
        self.status = 'reset'

    def _new_seed(self, x: float, y: float, at: float) -> PointSeed:
        seed = PointSeed(self._number, x, y)
        self._number += 1
        self._births[seed.number] = at
        return seed

    def see_source(self, source: np.ndarray, working: np.ndarray, at: float) -> tuple[SeenPoint, ...]:
        """Only a current visible result can extend a point's observed path."""
        if (source.ndim != 3 or source.shape[2] < 3 or working.ndim != 3 or source.dtype != np.uint8
                or min(source.shape[:2]) < 16 or min(working.shape[:2]) < 16 or not math.isfinite(at)):
            self.clear()
            return ()
        original_shape = source.shape[:2]
        factor = min(1., math.sqrt(self.pixel_budget / (source.shape[0] * source.shape[1])))
        if factor < 1.:
            source = shrink(source, int(source.shape[1] * factor), int(source.shape[0] * factor))
        source = np.ascontiguousarray(source[..., :3], dtype=np.uint8)
        geometry = original_shape, source.shape[:2], working.shape[:2]
        if self._geometry is not None and (geometry != self._geometry or not 0. < at - self._at <= self.longest_gap):
            self.clear()
        self._geometry = geometry
        self._sequence += 1
        frame = TrackingFrame(source, at, self._epoch, self._sequence)
        grey = _grey(source)
        try:
            if not self._slots:
                seeds = _new_points(grey, [], self.limit)
                if len(seeds) < 3:
                    self.points, self.status = (), 'insufficient visible texture'
                    self._at, self._previous_pixels = at, source.copy()
                    return ()
                self._slots = tuple(self._new_seed(x, y, at) for x, y in seeds)
                result = self._backend.start(frame, self._slots)
            else:
                replacements = self._new_texture_queries(source, grey, at)
                result = (self._backend.track(frame, replacements=replacements) if replacements
                          else self._backend.track(frame))
                if replacements:
                    self._slots = tuple(replacements.get(index, seed) for index, seed in enumerate(self._slots))
            if not _same_capture(result, frame):
                raise ValueError('a learned point result belongs to another capture')
            self._publish(result, grey, working.shape[:2], at)
        except BaseException:
            self.clear()
            raise
        self._at, self._previous_pixels = at, source.copy()
        self._births = {seed.number: self._births[seed.number] for seed in self._slots}
        return self.points

    def _new_texture_queries(self, source: np.ndarray, grey: np.ndarray, at: float) -> dict[int, PointSeed]:
        """Replace hidden or resting slots with currently exposed changing corners."""
        if self._previous_pixels is None or not callable(getattr(self._backend, '_replace_queries', None)):
            return {}
        changed = np.max(abs(source.astype(np.int16) - self._previous_pixels.astype(np.int16)), axis=2) > 12
        if not changed.any() or changed.mean() > .6:
            return {}
        exposed = ndimage.maximum_filter(changed, 7)
        continuing_motion = [p for p in self._native_visible.values() if math.hypot(p.x - p.old_x, p.y - p.old_y) >= .5]
        candidates = _new_points(grey, continuing_motion, min(1536, max(3 * self.limit, source.shape[0] * source.shape[1] // 64)))
        candidates = [(x, y) for x, y in candidates if exposed[int(round(y)), int(round(x))]]
        if not candidates:
            return {}
        slots = []
        for index, seed in enumerate(self._slots):
            point = self._native_visible.get(seed.number)
            if point is None or (at - self._births[seed.number] >= .5
                                and math.hypot(point.x - point.old_x, point.y - point.old_y) < .5):
                slots.append((point is not None, self._births[seed.number], index))
        slots.sort()
        count = min(len(slots), len(candidates), max(1, self.limit // 8))
        return {slot[2]: self._new_seed(*place, at) for slot, place in zip(slots[:count], candidates[:count], strict=True)}

    def _publish(self, result: Any, grey: np.ndarray, shape: tuple[int, int], at: float) -> None:
        current: dict[int, SeenPoint] = {}
        previous_grey = _grey(self._previous_pixels) if self._previous_pixels is not None else None
        for track in result.tracks if result.status == 'observed' else ():
            if not track.visible:
                continue
            before = self._native_visible.get(track.number)
            continued = before is not None and before.at == self._at and track.number in self._births
            error = 0.
            if continued and previous_grey is not None:
                earlier = _sample(previous_grey, np.asarray([[before.x, before.y]]))
                now = _sample(grey, np.asarray([[track.x, track.y]]))
                error = float(np.mean(abs(earlier - now)))
            current[track.number] = SeenPoint(track.number, track.x, track.y,
                before.x if continued else track.x, before.y if continued else track.y,
                self._births[track.number], self._at if continued else at, at, error,
                before.pictures + 1 if continued else 1)
        scale = shape[1] / grey.shape[1], shape[0] / grey.shape[0]
        self.points = tuple(SeenPoint(p.number, p.x * scale[0], p.y * scale[1], p.old_x * scale[0], p.old_y * scale[1],
                           p.born, p.before, p.at, p.error, p.pictures) for p in current.values())
        self._native_visible, self.status = current, result.status


@invariant('perception.learned_points_keep_their_capture', scope='perception',
           owner='core/perception/learned_tracking/points_for_the_motion_eyes.py', observational=False)
def _learned_point_capture_invariant() -> tuple:
    from dataclasses import replace

    from core.perception.learned_tracking.contracts import PointResult

    frame = TrackingFrame(np.zeros((20, 30, 3), np.uint8), 1., 2, 3)
    result = PointResult('invented', 2, 3, 1., frame.shape, (), .01)
    assert _same_capture(result, frame)
    for changed in (replace(result, epoch=3), replace(result, frame_number=4),
                    replace(result, at=1.1), replace(result, shape=(21, 30))):
        assert not _same_capture(changed, frame)
    return ()
