"""Complete a visible solid patch whose corners first stood out from the floor.

Offline 2026-10-12 a wide solid figure became four tiny corner fragments
when the initial median kept its middle as background. Its slight bobbing
then looked like several small objects travelling far beyond their sizes.
A connected colour patch supplies the missing extent when its surrounding
pixels independently show a different floor.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from core.verify import invariant


def complete_visible_patches(picture: np.ndarray, backdrop: np.ndarray, *,
                             difference: float, largest: float, longest: float) -> np.ndarray:
    """Extend existing foreground seeds only through a bounded, distinct connected patch."""
    seeds = np.abs(backdrop - picture).max(axis=2) > difference
    if not seeds.any():
        return backdrop
    tall, wide = picture.shape[:2]
    colours = picture.astype(np.int32) // 64
    bins = colours[:, :, 0] * 16 + colours[:, :, 1] * 4 + colours[:, :, 2]
    for colour in np.unique(bins[seeds]):
        labels, _count = ndimage.label(bins == colour, structure=np.ones((3, 3), bool))
        regions = ndimage.find_objects(labels)
        for number in np.unique(labels[seeds & (bins == colour)]):
            if not number:
                continue
            where = regions[number - 1]
            y, x = where[0].start, where[1].start
            bottom, right = where[0].stop, where[1].stop
            h, w = bottom - y, right - x
            if max(w, h) > longest or w * h > largest:
                continue
            own = labels[where] == number
            x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(wide, right + 2), min(tall, bottom + 2)
            ring = np.ones((y1 - y0, x1 - x0), bool)
            ring[y - y0:bottom - y0, x - x0:right - x0] = False
            if not ring.any():
                continue
            around = np.median(picture[y0:y1, x0:x1][ring].astype(np.float32), axis=0)
            pixels = picture[where][own].astype(np.float32)
            if float(np.abs(np.median(pixels, axis=0) - around).max()) <= difference:
                continue
            backdrop[where][own] = around
    return backdrop


@invariant("perception.visible_corner_keeps_its_whole_connected_patch", scope="perception",
           owner="core/perception/the_whole_patch_around_a_visible_corner.py", observational=False)
def _whole_patch_invariant() -> tuple:
    picture = np.full((60, 80, 3), 30, np.uint8)
    picture[12:42, 20:40] = (60, 150, 230)
    before = picture.astype(np.float32)
    before[12:15, 20:23] = 30
    after = complete_visible_patches(picture, before, difference=40, largest=1200, longest=40)
    assert np.all(after[12:42, 20:40] == 30)
    blank = picture.astype(np.float32)
    assert np.array_equal(complete_visible_patches(picture, blank, difference=40,
                                                  largest=1200, longest=40), picture)
    return ()
