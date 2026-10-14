"""Original causal point correspondence from current pixels and bounded appearance memory.

This implementation uses RGB correlation and independently checked return
matches. It does not import, reproduce or train from another point tracker.
Hidden positions are predictions. A group can narrow a search but cannot
turn an absent patch into a visible measurement.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from core.perception.learned_tracking.contracts import (
    PointResult,
    PointSeed,
    PointTrack,
    TrackingFrame,
    following_frame,
    validate_points,
)
from core.perception.learned_tracking.patches_with_independent_matches import (
    PatchMatch,
    locate,
    patch,
)
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


@dataclass
class _Memory:
    seed: PointSeed
    anchor: np.ndarray | None
    recent: np.ndarray | None
    place: tuple[float, float]
    velocity: tuple[float, float]
    seen_at: float
    missing: int = 0


class ContinuityBackend:
    """One bounded numerical tracker with no model loading or accelerator allocation."""

    name = 'continuity_points'

    def __init__(self, *, max_points: int = 64, patch_radius: int = 4, search_radius: int = 16,
                 hidden_frames: int = 12, longest_gap: float = 1.) -> None:
        if (any(type(v) is not int for v in (max_points, patch_radius, search_radius, hidden_frames))
                or not 1 <= max_points <= 384 or not 2 <= patch_radius <= 8
                or not 2 <= search_radius <= 64 or not 1 <= hidden_frames <= 32
                or isinstance(longest_gap, bool) or not math.isfinite(longest_gap) or not 0. < longest_gap <= 2.):
            raise ValueError('continuity points require bounded memory and search')
        self.max_points, self.radius, self.search = max_points, patch_radius, search_radius
        self._anchor_radius = math.ceil(patch_radius * math.sqrt(2.)) + 1
        self.hidden_frames, self.longest_gap = hidden_frames, longest_gap
        self._previous: TrackingFrame | None = None
        self._memory: list[_Memory] = []

    def _seed(self, frame: TrackingFrame, seed: PointSeed) -> _Memory:
        anchor = patch(frame.pixels, seed.x, seed.y, self._anchor_radius)
        return _Memory(seed, anchor, anchor, (seed.x, seed.y), (0., 0.), frame.at)

    def start(self, frame: TrackingFrame, seeds: tuple[PointSeed, ...]) -> PointResult:
        began = time.monotonic()
        validate_points(frame, seeds)
        if not isinstance(seeds, tuple) or len(seeds) > self.max_points:
            raise ValueError('continuity queries exceed their point budget')
        self._memory = [self._seed(frame, seed) for seed in seeds]
        self._previous = frame
        return self._result(frame, {}, began, status='seeded')

    def _replace_queries(self, frame: TrackingFrame, replacements: dict[int, PointSeed]) -> None:
        if any(type(i) is not int or not 0 <= i < len(self._memory) for i in replacements):
            raise ValueError('replacement queries need existing slots')
        seeds = tuple(replacements.get(i, m.seed) for i, m in enumerate(self._memory))
        validate_points(frame, seeds)
        for index, seed in replacements.items():
            self._memory[index] = self._seed(frame, seed)

    def _match(self, frame: TrackingFrame, memory: _Memory, predicted: tuple,
               *, search: int | None = None) -> PatchMatch | None:
        if memory.anchor is None or memory.missing >= self.hidden_frames or frame.at-memory.seen_at > self.longest_gap:
            return None
        if float(np.max(memory.anchor.std(axis=(0, 1)))) < 5.:
            return None
        templates = ((memory.anchor,) if memory.missing or np.array_equal(memory.anchor, memory.recent)
                     else (memory.anchor, memory.recent))
        result = locate(frame.pixels, templates, predicted, search or self.search, radius=self.radius)
        if result is None:
            result = locate(frame.pixels, templates, predicted, search or self.search,
                            minimum=.82, angles=(-30., -15., 15., 30.), radius=self.radius)
        if result is None:
            return None
        fresh = patch(frame.pixels, result.x, result.y, self._anchor_radius)
        if fresh is None:
            return None
        if memory.missing:
            return result
        reverse = locate(self._previous.pixels, (fresh,), memory.place, self.search, minimum=.82, radius=self.radius)
        if reverse is None:
            reverse = locate(self._previous.pixels, (fresh,), memory.place, self.search,
                             minimum=.82, angles=(-30., -15., 15., 30.), radius=self.radius)
        if reverse is None or math.dist((reverse.x, reverse.y), memory.place) > 2.:
            return None
        return result

    def _group_searches(self, frame: TrackingFrame, measured: dict[int, PatchMatch],
                        old: tuple[tuple[float, float], ...]) -> dict[int, tuple]:
        """Only caller-named objects can supply an occluded point's search basis."""
        predictions = {}
        groups = {m.seed.object_number for m in self._memory if m.seed.object_number is not None}
        for group in groups:
            indexes = [i for i in measured if self._memory[i].seed.object_number == group
                       and self._memory[i].seen_at == self._previous.at]
            if len(indexes) < 3:
                continue
            starts = np.asarray([old[i] for i in indexes])
            ends = np.asarray([(measured[i].x, measured[i].y) for i in indexes])
            fitted = rigid_motion(starts, ends, error=1.5)
            if fitted is not None:
                for i, memory in enumerate(self._memory):
                    if i not in measured and memory.seed.object_number == group:
                        dt = self._previous.at-memory.seen_at
                        previous = tuple(p+v*dt for p, v in zip(old[i], memory.velocity, strict=True))
                        predictions[i] = tuple(fitted.apply(previous))
        return predictions

    def track(self, frame: TrackingFrame, *, replacements: dict[int, PointSeed] | None = None) -> PointResult:
        began = time.monotonic()
        if self._previous is None:
            raise RuntimeError('continuity tracking needs current seed queries')
        following_frame(self._previous, frame)
        if frame.at-self._previous.at > self.longest_gap:
            self.close()
            raise ValueError('continuity tracking cannot bridge an unbounded capture gap')
        if replacements:
            self._replace_queries(frame, replacements)
        old = tuple(m.place for m in self._memory)
        measured = {}
        for index, memory in enumerate(self._memory):
            if replacements and index in replacements:
                continue
            dt = frame.at-memory.seen_at
            predicted = tuple(p+v*dt for p, v in zip(memory.place, memory.velocity, strict=True))
            match = self._match(frame, memory, predicted)
            if match is not None:
                measured[index] = match
        for index, predicted in self._group_searches(frame, measured, old).items():
            match = self._match(frame, self._memory[index], predicted, search=4)
            if match is not None:
                measured[index] = match
        # Two identities cannot claim the same current patch location.
        collided = {i for i, a in measured.items() if any(i != j and math.dist((a.x, a.y), (b.x, b.y)) < 1.
                                                       for j, b in measured.items())}
        for index in collided:
            measured.pop(index)
        for index, memory in enumerate(self._memory):
            match = measured.get(index)
            if match is None:
                if not replacements or index not in replacements:
                    memory.missing += 1
                continue
            dt = frame.at-memory.seen_at
            place = match.x, match.y
            memory.velocity = tuple((p-q)/dt for p, q in zip(place, memory.place, strict=True))
            memory.place, memory.seen_at, memory.missing = place, frame.at, 0
            memory.recent = patch(frame.pixels, *place, self._anchor_radius)
        self._previous = frame
        return self._result(frame, measured, began)

    def _result(self, frame: TrackingFrame, measured: dict, began: float, *, status: str = 'observed') -> PointResult:
        tracks = tuple(PointTrack(m.seed.number, *m.place,
                                  min(.99, max(.5, measured[i].score)) if i in measured else 0.,
                                  i in measured, m.seed.object_number) for i, m in enumerate(self._memory))
        return PointResult(self.name, frame.epoch, frame.number, frame.at, frame.shape, tracks,
                           time.monotonic()-began, status)

    def close(self) -> None:
        self._previous, self._memory = None, []


@invariant('perception.continuity_points_need_fresh_unique_pixels', scope='perception',
           owner='core/perception/learned_tracking/continuity_backend.py', observational=False)
def _continuity_invariant() -> tuple:
    pixels = np.full((32, 40, 3), 20, np.uint8)
    pixels[8:24, 8:24] = np.random.default_rng(38).integers(30, 230, (16, 16, 3), dtype=np.uint8)
    tracker = ContinuityBackend(max_points=1)
    assert not tracker.start(TrackingFrame(pixels, 1., 0, 1), (PointSeed(7, 16., 16.),)).tracks[0].visible
    vanished = tracker.track(TrackingFrame(np.full_like(pixels, 20), 1.1, 0, 2))
    assert not vanished.tracks[0].visible
    returned = tracker.track(TrackingFrame(pixels, 1.2, 0, 3))
    assert returned.tracks[0].visible and returned.tracks[0].number == 7
    tracker.close()
    assert tracker._previous is None and not tracker._memory
    return ()
