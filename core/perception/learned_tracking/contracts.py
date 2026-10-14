"""Tracking results belong to one captured frame, surface generation and pixel geometry.

Point identity and object proposals do not establish control ownership.
Occluded positions remain predictions even when a tracker supplies them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.verify import invariant


def _number(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError('tracking identities must be nonnegative integers')


def _finite(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('tracking coordinates and times must be finite numbers')


def _probability(value: float | None) -> None:
    if value is not None:
        _finite(value)
        if not 0. <= value <= 1.:
            raise ValueError('tracking probabilities must be in the unit interval')


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    pixels: np.ndarray
    at: float
    epoch: int
    number: int

    def __post_init__(self) -> None:
        _finite(self.at)
        _number(self.epoch)
        _number(self.number)
        if (not isinstance(self.pixels, np.ndarray) or self.pixels.dtype != np.uint8
                or self.pixels.ndim != 3 or self.pixels.shape[2] != 3
                or min(self.pixels.shape[:2]) < 16 or self.pixels.size > 12_000_000):
            raise ValueError('a tracking frame needs bounded RGB pixels')
        owned = np.ascontiguousarray(self.pixels).copy()
        owned.flags.writeable = False
        object.__setattr__(self, 'pixels', owned)

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape[:2]


@dataclass(frozen=True, slots=True)
class PointSeed:
    number: int
    x: float
    y: float
    object_number: int | None = None

    def __post_init__(self) -> None:
        _number(self.number)
        _finite(self.x)
        _finite(self.y)
        if self.object_number is not None:
            _number(self.object_number)


@dataclass(frozen=True, slots=True)
class PointTrack:
    number: int
    x: float
    y: float
    visibility: float | None
    visible: bool
    object_number: int | None = None

    def __post_init__(self) -> None:
        _number(self.number)
        _finite(self.x)
        _finite(self.y)
        _probability(self.visibility)
        if type(self.visible) is not bool or self.visible and (self.visibility is None or self.visibility < .5):
            raise ValueError('visible point measurements need a visibility estimate')
        if self.object_number is not None:
            _number(self.object_number)


@dataclass(frozen=True, slots=True)
class ObjectSeed:
    number: int
    box: tuple[float, float, float, float] | None = None
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        _number(self.number)
        if self.box is None and self.mask is None:
            raise ValueError('an object seed needs a box or mask')
        if self.box is not None:
            if len(self.box) != 4:
                raise ValueError('a seed box needs four pixel coordinates')
            for coordinate in self.box:
                _finite(coordinate)
            if self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
                raise ValueError('a seed box needs positive area')
        if self.mask is not None:
            object.__setattr__(self, 'mask', _owned_mask(self.mask))


def _owned_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2 or not mask.size:
        raise ValueError('an object mask needs measured boolean pixels')
    owned = np.ascontiguousarray(mask).copy()
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True, slots=True)
class MaskTrack:
    number: int
    mask: np.ndarray
    presence: float | None
    visible: bool
    seeded: bool = False

    def __post_init__(self) -> None:
        _number(self.number)
        _probability(self.presence)
        object.__setattr__(self, 'mask', _owned_mask(self.mask))
        if (type(self.visible) is not bool or type(self.seeded) is not bool
                or self.visible and (self.seeded or self.presence is None or self.presence < .5 or not self.mask.any())):
            raise ValueError('seeded or absent masks cannot be visible measurements')


def _result(backend: str, epoch: int, number: int, at: float, shape: tuple[int, int], elapsed: float, status: str) -> None:
    _number(epoch)
    _number(number)
    _finite(at)
    _finite(elapsed)
    if not backend or elapsed < 0. or status not in ('observed', 'warming', 'seeded', 'unavailable'):
        raise ValueError('tracking results need an explicit backend, measured time and status')
    if len(shape) != 2 or any(type(size) is not int or size < 16 for size in shape):
        raise ValueError('tracking results need their capture geometry')


@dataclass(frozen=True, slots=True)
class PointResult:
    backend: str
    epoch: int
    frame_number: int
    at: float
    shape: tuple[int, int]
    tracks: tuple[PointTrack, ...]
    elapsed_s: float
    status: str = 'observed'

    def __post_init__(self) -> None:
        _result(self.backend, self.epoch, self.frame_number, self.at, self.shape, self.elapsed_s, self.status)
        if len({track.number for track in self.tracks}) != len(self.tracks):
            raise ValueError('point identities must be unique within a result')
        for track in self.tracks:
            if track.visible and not (0. <= track.x < self.shape[1] and 0. <= track.y < self.shape[0]):
                raise ValueError('visible points must be inside their captured frame')
        if self.status != 'observed' and any(track.visible for track in self.tracks):
            raise ValueError('unobserved results cannot supply visible points')


@dataclass(frozen=True, slots=True)
class MaskResult:
    backend: str
    epoch: int
    frame_number: int
    at: float
    shape: tuple[int, int]
    masks: tuple[MaskTrack, ...]
    elapsed_s: float
    status: str = 'observed'

    def __post_init__(self) -> None:
        _result(self.backend, self.epoch, self.frame_number, self.at, self.shape, self.elapsed_s, self.status)
        if len({mask.number for mask in self.masks}) != len(self.masks):
            raise ValueError('object identities must be unique within a result')
        if any(mask.mask.shape != self.shape for mask in self.masks):
            raise ValueError('object masks must use the captured pixel geometry')
        if self.status != 'observed' and any(mask.visible for mask in self.masks):
            raise ValueError('unobserved results cannot supply visible masks')


def validate_points(frame: TrackingFrame, seeds: tuple[PointSeed, ...]) -> None:
    if not seeds or len(seeds) > 384 or len({seed.number for seed in seeds}) != len(seeds):
        raise ValueError('point queries need a bounded set of unique identities')
    if any(not (0. <= seed.x < frame.shape[1] and 0. <= seed.y < frame.shape[0]) for seed in seeds):
        raise ValueError('point queries must lie in the captured frame')


def validate_objects(frame: TrackingFrame, seeds: tuple[ObjectSeed, ...]) -> None:
    if not seeds or len(seeds) > 32 or len({seed.number for seed in seeds}) != len(seeds):
        raise ValueError('object queries need a bounded set of unique identities')
    for seed in seeds:
        if seed.mask is not None and seed.mask.shape != frame.shape:
            raise ValueError('seed masks must use the captured pixel geometry')
        if seed.box is not None and not (0. <= seed.box[0] < seed.box[2] <= frame.shape[1]
                                        and 0. <= seed.box[1] < seed.box[3] <= frame.shape[0]):
            raise ValueError('seed boxes must lie in the captured frame')


def following_frame(previous: TrackingFrame, current: TrackingFrame) -> None:
    if (previous.epoch != current.epoch or previous.shape != current.shape
            or current.number <= previous.number or current.at <= previous.at):
        raise ValueError('tracking continuation needs a later capture in the same generation and geometry')


@invariant('perception.occluded_tracking_is_not_a_visible_measurement', scope='perception',
           owner='core/perception/learned_tracking/contracts.py', observational=False)
def _tracking_visibility_invariant() -> tuple:
    frame = TrackingFrame(np.zeros((16, 20, 3), np.uint8), 1., 2, 3)
    predicted = PointTrack(4, -10., 30., .1, False)
    result = PointResult('invented', 2, 3, 1., frame.shape, (predicted,), .01)
    assert not result.tracks[0].visible and not frame.pixels.flags.writeable
    try:
        PointTrack(4, 10., 10., .1, True)
    except ValueError:
        return ()
    raise AssertionError('an occluded prediction became a visible measurement')
