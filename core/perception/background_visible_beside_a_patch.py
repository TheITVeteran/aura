"""Exposed background can remain a minority inside a moving foreground patch.

Offline 2026-10-12 the early silhouette stayed in the backdrop while a body
moved through it. The joined patch's median described the body, so exposed
background in its wake remained foreground. Opposite sides can describe
that background locally, even when it changes across the picture.
"""
from __future__ import annotations

import math

import numpy as np

from core.verify import invariant


def exposed_background(picture: np.ndarray, inside: np.ndarray, *,
                       left: int, top: int, difference: float) -> np.ndarray:
    """Match quiet opposite sides and require an open path to the patch exterior.

    Both side samples must agree and each must have little texture. A pixel
    must match both samples. Enclosed matching details remain with the body;
    an absent side or disagreeing surroundings supplies no correction.
    """
    empty = np.zeros_like(inside, dtype=bool)
    if (inside.ndim != 2 or picture.ndim != 3 or picture.shape[2] != 3
            or type(left) is not int or type(top) is not int
            or not math.isfinite(difference) or difference <= 0):
        return empty
    h, w = inside.shape
    if not h or not w or min(left, top) < 0 or left + w > picture.shape[1] or top + h > picture.shape[0]:
        return empty
    floor = _from_sides(picture, inside, left, top, difference)
    floor |= _from_sides(picture.transpose(1, 0, 2), inside.T, top, left, difference).T
    from scipy import ndimage

    opened = floor | ~inside.astype(bool)
    edge = opened.copy()
    edge[1:-1, 1:-1] = False
    return floor & ndimage.binary_propagation(edge, mask=opened)


def _from_sides(picture: np.ndarray, inside: np.ndarray, left: int, top: int,
                difference: float) -> np.ndarray:
    h, w = inside.shape
    if left < 2 or left + w + 2 > picture.shape[1]:
        return np.zeros_like(inside, dtype=bool)
    before = picture[top:top + h, left - 2:left].astype(np.float32)
    after = picture[top:top + h, left + w:left + w + 2].astype(np.float32)
    first, second = np.median(before, axis=1), np.median(after, axis=1)
    agrees = np.abs(first - second).max(axis=1) < difference
    agrees &= np.abs(before - first[:, None, :]).max(axis=(1, 2)) < difference
    agrees &= np.abs(after - second[:, None, :]).max(axis=(1, 2)) < difference
    here = picture[top:top + h, left:left + w].astype(np.float32)
    return (inside.astype(bool) & agrees[:, None]
            & (np.abs(here - first[:, None, :]).max(axis=2) < difference)
            & (np.abs(here - second[:, None, :]).max(axis=2) < difference))


@invariant("perception.exposed_background_does_not_take_enclosed_body_detail", scope="perception",
           owner="core/perception/background_visible_beside_a_patch.py", observational=False)
def _exposed_background_invariant() -> tuple:
    picture = np.full((20, 20, 3), 80, np.uint8)
    picture[5:13, 6:14] = 240
    picture[8:10, 9:11] = 80
    inside = np.ones((12, 8), bool)
    floor = exposed_background(picture, inside, left=6, top=5, difference=20.)
    assert int(floor.sum()) == 32
    assert not floor[3:5, 3:5].any()
    assert not exposed_background(picture, np.ones((20, 20), bool), left=0, top=0, difference=20.).any()
    return ()
