"""Visible exterior edges distinguish a whole patch from a sliding window.

The object mask supplies sample locations. Actual pixels outside that mask
must independently show its boundary; the mask alone cannot certify one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.verify import invariant

_LINE_FRACTIONS = np.arange(9) / 8.
_INWARD = np.array([1, -1, 1, -1])[:, None, None]
_ACROSS = np.array([True, True, False, False])[:, None, None]
_INNER_DEPTH = np.arange(3)[None, None, :]
_OUTER_DEPTH = np.array([2, 3])[None, None, :]


@dataclass(frozen=True)
class BoundarySide:
    """Nine bounded edge samples, including the measured exterior colour."""

    inner: np.ndarray
    outer: np.ndarray
    edge: np.ndarray
    lines: np.ndarray
    visible: np.ndarray


@dataclass(frozen=True)
class VisibleBoundary:
    width: int
    height: int
    sides: tuple[BoundarySide, ...]


def visible_boundary(picture: np.ndarray, left: int, top: int, right: int, bottom: int,
                     support: np.ndarray) -> VisibleBoundary | None:
    """Sample all four sides with actual exterior pixels two and three pixels away.

    A clipped patch, invalid mask or side with no distinct exterior remains
    unavailable. Arrays contain at most 36 samples regardless of object size.
    """
    if (not isinstance(picture, np.ndarray) or picture.ndim != 3 or picture.shape[2] < 3
            or not isinstance(support, np.ndarray) or support.ndim != 2):
        return None
    if any(not isinstance(value, (int, np.integer)) for value in (left, top, right, bottom)):
        return None
    width, height = right - left, bottom - top
    if (min(width, height) < 3 or support.shape != (height, width)
            or left < 3 or top < 3 or right + 3 > picture.shape[1] or bottom + 3 > picture.shape[0]
            or not np.isfinite(support).all()):
        return None
    sides = _all_sides(picture, support.astype(bool, copy=False), left, top)
    if any(not _enough_visible(side) for side in sides):
        return None
    return VisibleBoundary(width, height, sides)


def _all_sides(picture: np.ndarray, support: np.ndarray, left: int, top: int) -> tuple[BoundarySide, ...]:
    """Gather the four edge families together, rather than run four image samplers."""
    height, width = support.shape
    rows = np.rint(_LINE_FRACTIONS * (height - 1)).astype(int)
    columns = np.rint(_LINE_FRACTIONS * (width - 1)).astype(int)
    horizontal, vertical = support[rows, :], support[:, columns].T
    rows_exist, columns_exist = horizontal.any(axis=1), vertical.any(axis=1)
    exists = np.stack((rows_exist, rows_exist, columns_exist, columns_exist))
    edge = np.stack((horizontal.argmax(axis=1), width - 1 - horizontal[:, ::-1].argmax(axis=1),
                     vertical.argmax(axis=1), height - 1 - vertical[:, ::-1].argmax(axis=1)))
    lines = np.stack((rows, rows, columns, columns))
    places = edge[:, :, None] + _INWARD * _INNER_DEPTH
    x, y = _coordinates(lines, places)
    clipped_x, clipped_y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
    valid = exists[:, :, None] & (x == clipped_x) & (y == clipped_y) & support[clipped_y, clipped_x]
    inside = picture[top + clipped_y, left + clipped_x, :3].astype(np.float32)
    finite = np.isfinite(inside).all(axis=(2, 3))
    if valid.all() and finite.all():
        own = np.median(inside, axis=2)
    else:
        # Empty or invalid rows cannot warn or manufacture a visible boundary.
        inside[~valid] = np.nan
        inside[~(exists & finite)] = 0.
        own = _median_of_three_depths(inside)
    outside_places = edge[:, :, None] - _INWARD * _OUTER_DEPTH
    outside_x, outside_y = _coordinates(lines, outside_places)
    surrounding = picture[top + outside_y, left + outside_x, :3].astype(np.float32).mean(axis=2)
    visible = (exists & finite & np.isfinite(own).all(axis=2) & np.isfinite(surrounding).all(axis=2)
               & (np.abs(own - surrounding).max(axis=2) >= 25.))
    return tuple(BoundarySide(own[index], surrounding[index], edge[index].astype(np.float32),
                              lines[index], visible[index]) for index in range(4))


def _coordinates(lines: np.ndarray, places: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.where(_ACROSS, places, lines[:, :, None]), np.where(_ACROSS, lines[:, :, None], places)


def _median_of_three_depths(inside: np.ndarray) -> np.ndarray:
    """Three depths need only the middle observed sample, or the mean of the middle pair.

    Holes are NaN, sorted after the observed pixels. Counting the remaining
    depths preserves each colour's median without NumPy's masked-array path.
    An empty sample stays NaN and cannot supply a visible boundary.
    """
    count = np.sum(~np.isnan(inside), axis=2, keepdims=True)
    ordered = np.sort(inside, axis=2)
    lower = np.take_along_axis(ordered, np.maximum(0, (count - 1) // 2), axis=2)
    upper = np.take_along_axis(ordered, count // 2, axis=2)
    paired = count % 2 == 0
    np.add(lower, upper, out=lower, where=paired)
    np.multiply(lower, .5, out=lower, where=paired)
    return lower.squeeze(axis=2)


def _enough_visible(side: BoundarySide) -> bool:
    return bool(side.visible.mean() >= .6 and len(np.unique(side.lines[side.visible])) >= 3)


def boundaries_agree(before: VisibleBoundary | None, after: VisibleBoundary | None) -> bool:
    """Whether both observations show corresponding, distinct exterior edges.

    Object colours and edge locations agree within raster uncertainty. The
    exterior colour may change as the object travels over its surroundings.
    """
    if before is None or after is None or max(abs(before.width - after.width),
                                             abs(before.height - after.height)) > 2:
        return False
    for old, new in zip(before.sides, after.sides, strict=True):
        paired = old.visible & new.visible
        agreeing = paired & (np.abs(old.edge - new.edge) <= 2.) & (np.abs(old.inner - new.inner).max(axis=1) <= 40.)
        if (agreeing.mean() < .6 or min(len(np.unique(old.lines[agreeing])),
                                      len(np.unique(new.lines[agreeing]))) < 3):
            return False
    return True


@invariant("perception.translation_fallback_requires_visible_exterior_edges", scope="perception",
           owner="core/perception/boundary_support_for_translation.py", observational=False)
def _visible_boundary_invariant() -> tuple:
    picture = np.full((48, 64, 3), 30, np.uint8)
    support = np.ones((12, 12), bool)
    assert visible_boundary(picture, 16, 12, 28, 24, support) is None
    picture[12:24, 16:28] = (210, 180, 70)
    first = visible_boundary(picture, 16, 12, 28, 24, support)
    moved = np.full_like(picture, 30)
    moved[16:28, 21:33] = (210, 180, 70)
    second = visible_boundary(moved, 21, 16, 33, 28, support)
    assert boundaries_agree(first, second)
    return ()


@invariant("perception.bounded_edge_samples_keep_their_observed_median", scope="perception",
           owner="core/perception/boundary_support_for_translation.py", observational=False)
def _bounded_edge_samples_invariant() -> tuple:
    depths = np.array([[[[20., 70., 40.], [np.nan, np.nan, np.nan], [60., 30., 80.]],
                        [[90., 10., 50.], [30., 70., 20.], [60., 40., 80.]]]], dtype=np.float32)
    assert np.array_equal(_median_of_three_depths(depths), [[[40., 50., 60.], [60., 40., 50.]]])
    assert np.isnan(_median_of_three_depths(np.full((1, 1, 3, 3), np.nan))).all()
    return ()
