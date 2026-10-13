"""Continuing measured texture can separate a body from a joined outline.

Offline 2026-10-12 a body's points remained visible as its outline joined
scenery. Matching the whole joined box discarded its identity. Its prior
texture, moved by the points, supplies a proposal made only of current
foreground pixels. The remainder stays available to the other eyes.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from core.verify import invariant


def parts_from_continuing_texture(picture: np.ndarray, inside: np.ndarray, *, left: int, top: int,
                                  things: list[Any], motions: dict, at: float, difference: float,
                                  proposals: int = 16) -> list[np.ndarray]:
    """Keep bounded translated texture proposals with current spread-out owned marks.

    Translation and a one-pixel sampling neighbourhood must explain the
    actual colours. Three current named marks must belong to the resulting
    support. This observes a visible part, without certifying the full
    extent of an occluded or deformed body.
    """
    if (picture.ndim != 3 or inside.ndim != 2 or picture.shape[:2] != inside.shape or picture.shape[2] != 3
            or type(left) is not int or type(top) is not int
            or not math.isfinite(at) or not math.isfinite(difference) or difference <= 0
            or type(proposals) is not int or not 1 <= proposals <= 32):
        return [inside]
    remaining = inside.astype(bool).copy()
    parts = []
    for thing in sorted(things, key=lambda t: t.size):
        if len(parts) >= proposals:
            break
        motion = motions.get(thing.number)
        part = _supported_part(picture, remaining, thing, motion, left, top, at, difference)
        if part is not None:
            parts.append(part)
            remaining &= ~part
    if not parts:
        return [inside]
    return parts + ([remaining] if remaining.any() else [])


def _supported_part(picture: np.ndarray, inside: np.ndarray, thing: Any, motion: Any,
                    left: int, top: int, at: float, difference: float) -> np.ndarray | None:
    born = getattr(thing, "born", None)
    if (born is None or not math.isfinite(born) or motion is None or born > motion.before
            or motion.before != thing.seen or motion.at != at
            or not 0 < at - thing.seen <= .25 or motion.support < 3
            or len(motion.points) != motion.support or len(set(motion.points)) != motion.support
            or len(motion.locations) != motion.support
            or not all(math.isfinite(v) for v in (motion.before, motion.dx, motion.dy, motion.error)) or motion.error < 0):
        return None
    old, support = thing.patch, thing.support
    if (not isinstance(old, np.ndarray) or not isinstance(support, np.ndarray) or old.ndim != 3
            or support.ndim != 2 or old.shape[:2] != support.shape or old.shape[2] != 3
            or not support.any() or int(inside.sum()) <= 2 * int(support.sum())):
        return None
    ox, oy, right, bottom = thing.box()
    if not all(math.isfinite(v) for v in (ox, oy, right, bottom)):
        return None
    if (round(right - ox), round(bottom - oy)) != (old.shape[1], old.shape[0]):
        return None
    h, w = inside.shape
    x0, y0 = max(0, math.floor(ox + motion.dx - left - 1)), max(0, math.floor(oy + motion.dy - top - 1))
    x1, y1 = min(w, math.ceil(right + motion.dx - left + 1)), min(h, math.ceil(bottom + motion.dy - top + 1))
    if x0 >= x1 or y0 >= y1:
        return None
    rows, cols = np.indices((y1 - y0, x1 - x0))
    sx, sy = cols + x0 + left - ox - motion.dx, rows + y0 + top - oy - motion.dy
    own = _matching_texture(picture[y0:y1, x0:x1], inside[y0:y1, x0:x1], old, support, sx, sy, difference)
    if int(own.sum()) < .6 * int(support.sum()):
        return None
    located = []
    for x, y in motion.locations:
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        col, row = int(round(x - left - x0)), int(round(y - top - y0))
        if 0 <= col < own.shape[1] and 0 <= row < own.shape[0] and own[row, col]:
            located.append((x, y))
    if len(located) < 3 or len(located) < .6 * motion.support:
        return None
    centered = np.asarray(located) - np.mean(located, axis=0)
    if np.linalg.svd(centered, compute_uv=False)[-1] < 1.:
        return None
    part = np.zeros_like(inside, dtype=bool)
    part[y0:y1, x0:x1] = own
    return part


def _matching_texture(picture: np.ndarray, inside: np.ndarray, old: np.ndarray, support: np.ndarray,
                      sx: np.ndarray, sy: np.ndarray, difference: float) -> np.ndarray:
    matched = np.zeros_like(inside, dtype=bool)
    here = picture.astype(np.float32)
    old = old.astype(np.float32)
    support = support.astype(np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = sx + dx, sy + dy
            within = (x >= 0) & (x <= old.shape[1] - 1) & (y >= 0) & (y <= old.shape[0] - 1)
            belonged = ndimage.map_coordinates(support, (y, x), order=0, mode="constant") > 0
            texture = np.stack([ndimage.map_coordinates(old[:, :, channel], (y, x), order=1, mode="constant")
                                for channel in range(3)], axis=2)
            matched |= inside & within & belonged & (np.abs(here - texture).max(axis=2) < difference)
    return matched


@invariant("perception.joined_outlines_need_current_owned_texture", scope="perception",
           owner="core/perception/texture_that_continues_when_outlines_join.py", observational=False)
def _continuing_texture_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion

    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[10:18, 12:20] = 240
    body = SimpleNamespace(number=1, size=64, born=0., seen=1., box=lambda: (10., 10., 18., 18.),
                           patch=np.full((8, 8, 3), 240, np.uint8), support=np.ones((8, 8), bool))
    motion = PointMotion(2., 0., .01, 3, 1., 1.1, (1, 2, 3), ((13., 11.), (18., 11.), (13., 16.)))
    inside = np.ones((30, 40), bool)
    parts = parts_from_continuing_texture(picture, inside, left=0, top=0, things=[body], motions={1: motion},
                                          at=1.1, difference=40.)
    assert len(parts) == 2 and int(parts[0].sum()) == 64
    assert np.array_equal(np.sum(parts, axis=0), inside)
    unknown = parts_from_continuing_texture(picture, inside, left=0, top=0, things=[body], motions={},
                                           at=1.1, difference=40.)
    assert len(unknown) == 1 and unknown[0] is inside
    return ()
