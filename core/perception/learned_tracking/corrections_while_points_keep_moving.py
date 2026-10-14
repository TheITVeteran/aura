"""Fast visible tracking continues while a learned worker refines correspondence.

A late answer belongs to its older capture. It reaches the current view
only through retained actual pictures and fresh forward/reverse matching.
No learned worker can queue a second operation behind its unfinished one.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.points_before_the_picture_is_reduced import (
    PointsBeforeReduction,
    _colours,
)
from core.perception.points_that_keep_their_motion import SeenPoint, _grey, _pyramid
from core.runtime.executors import submit_blocking_io
from core.runtime.lockdep import checked_thread_semaphore
from core.verify import invariant

_SLOTS = checked_thread_semaphore('perception.learned_point_corrections', 1)
_LEARNED_NAMES = 1 << 48


@dataclass
class _Picture:
    source: np.ndarray
    working: np.ndarray
    at: float
    generation: int


@dataclass
class _Correction:
    future: Any
    picture: _Picture


class _VisibleTransport(PointsBeforeReduction):
    """Follow only the supplied named marks; fresh unrelated corners receive no credit."""

    def _spread(self, points: list[SeenPoint], scale: tuple) -> tuple[SeenPoint, ...]:
        return tuple(point for point in points if point.number in self._allowed)[:self.limit]


def _native_picture(source: np.ndarray, budget: int) -> np.ndarray:
    factor = min(1., math.sqrt(budget / (source.shape[0] * source.shape[1])))
    if factor >= 1.:
        return source
    from core.perception.picture_arithmetic import shrink

    return shrink(source, int(source.shape[1] * factor), int(source.shape[0] * factor))


def visible_transport(points: tuple[SeenPoint, ...], pictures: list[_Picture],
                      budget: int) -> _VisibleTransport | None:
    """An older learned position is a seed; each intervening displacement needs actual pixels."""
    if not points or len(pictures) < 2:
        return None
    anchor = pictures[0]
    if any(point.at != anchor.at for point in points):
        return None
    native = _native_picture(anchor.source, budget)
    scale = anchor.working.shape[1] / native.shape[1], anchor.working.shape[0] / native.shape[0]
    transport = _VisibleTransport(limit=len(points), pixel_budget=budget)
    transport._allowed = {point.number for point in points}
    transport._native_points = tuple(SeenPoint(p.number, p.x / scale[0], p.y / scale[1], p.x / scale[0], p.y / scale[1],
        p.born, anchor.at, anchor.at, p.error, 1) for p in points)
    transport._number = max(transport._allowed) + 1
    transport._geometry = anchor.source.shape[:2], native.shape[:2], anchor.working.shape[:2]
    transport._image, transport._colours, transport._at = _pyramid(_grey(native)), _colours(native), anchor.at
    for picture in pictures[1:]:
        if picture.generation != anchor.generation:
            return None
        transport.see_source(picture.source, picture.working, picture.at)
        if not transport.points:
            return None
    return transport


def _completed(future: Any, slots: Any) -> None:
    try:
        if not future.cancelled():
            future.exception()
    finally:
        slots.release()


class CorrectionsWhilePointsMove(PointsBeforeReduction):
    """The observation worker owns fast pixels; a second worker owns the learned state."""

    def __init__(self, teacher_factory: Any, *, limit: int = 192, pixel_budget: int = 256_000,
                 retained_pictures: int = 32) -> None:
        super().__init__(limit=limit, pixel_budget=pixel_budget)
        if type(retained_pictures) is not int or not 2 <= retained_pictures <= 32:
            raise ValueError('point corrections need bounded retained pictures')
        self._factory, self._teacher = teacher_factory, None
        self._teacher_generation: int | None = None
        self._generation = 0
        self._failed_generation: int | None = None
        self._retained: deque[_Picture] = deque(maxlen=retained_pictures)
        self._operation: _Correction | None = None
        self._learned: tuple[SeenPoint, ...] = ()
        self._transport: _VisibleTransport | None = None
        self.correction_status = 'not started'

    def clear(self) -> None:
        super().clear()
        self._generation += 1
        self._retained.clear()
        self._learned = ()
        self._transport = None
        self._failed_generation = None
        self.correction_status = 'reset'

    def _refine(self, picture: _Picture) -> tuple[SeenPoint, ...]:
        if self._teacher is None:
            self._teacher = self._factory()
        if self._teacher_generation != picture.generation:
            self._teacher.clear()
            self._teacher_generation = picture.generation
        return copy.deepcopy(self._teacher.see_source(picture.source, picture.working, picture.at))

    def _receive(self) -> None:
        operation = self._operation
        if operation is None or not operation.future.done():
            return
        self._operation = None
        if operation.picture.generation != self._generation:
            self.correction_status = 'correction expired'
            return
        try:
            learned = operation.future.result()
            pictures = list(self._retained)
            index = next((i for i, p in enumerate(pictures) if p.at == operation.picture.at
                          and p.generation == operation.picture.generation), None)
            if index is None:
                self.correction_status = 'correction expired'
                return
            transported = visible_transport(learned, pictures[index:], self.pixel_budget)
            if transported is not None:
                self._transport = transported
            self.correction_status = 'current visible correction' if transported is not None else 'no visible correspondence'
        except Exception as why:
            from core.runtime.errors import record_degradation

            record_degradation('learned_point_correction', why, action='discarded the unavailable correction')
            self.correction_status = 'correction unavailable'
            self._failed_generation = self._generation

    def _submit(self, picture: _Picture) -> None:
        slots = _SLOTS
        if self._failed_generation == self._generation or self._operation is not None or not slots.acquire(blocking=False):
            return
        try:
            future = submit_blocking_io(self._refine, picture, label='learned_point_correction')
        except BaseException:
            slots.release()
            raise
        future.add_done_callback(lambda done: _completed(done, slots))
        self._operation = _Correction(future, picture)

    def see_source(self, source: np.ndarray, working: np.ndarray, at: float) -> tuple[SeenPoint, ...]:
        fast = super().see_source(source, working, at)
        if self._transport is not None:
            self._transport.see_source(source, working, at)
        picture = _Picture(np.array(source, copy=True), np.array(working, copy=True), at, self._generation)
        self._retained.append(picture)
        self._receive()
        current = self._current_learned(at)
        # Two names on one measured mark do not become two pieces of support.
        self.points = (*current, *(point for point in fast if all(math.hypot(point.x - p.x, point.y - p.y) >= .5
                                                                 for p in current)))
        self._submit(picture)
        return self.points

    def _current_learned(self, at: float) -> tuple[SeenPoint, ...]:
        carried = self._transport.points if self._transport is not None else ()
        return tuple(SeenPoint(p.number + _LEARNED_NAMES, p.x, p.y, p.old_x, p.old_y,
            p.born, p.before, p.at, p.error, p.pictures) for p in carried if p.at == at)


@invariant('perception.late_points_need_visible_transport', scope='perception',
           owner='core/perception/learned_tracking/corrections_while_points_keep_moving.py', observational=False)
def _visible_correction_invariant() -> tuple:
    pixels = np.full((48, 64, 3), 25, np.uint8)
    pixels[12:36, 16:40] = np.random.default_rng(72).integers(30, 230, (24, 24, 3), dtype=np.uint8)
    point = SeenPoint(1, 22., 18., 22., 18., 1., 1., 1., .01, 1)
    frames = [_Picture(pixels, pixels, 1., 0), _Picture(pixels, pixels, 1.1, 0)]
    current = visible_transport((point,), frames, 256_000)
    assert current is not None and current.points[0].at == 1.1 and current.points[0].before == 1.
    frames[1].generation = 1
    assert visible_transport((point,), frames, 256_000) is None
    return ()
