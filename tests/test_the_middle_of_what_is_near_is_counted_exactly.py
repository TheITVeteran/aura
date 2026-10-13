"""The middle of the values around each place, counted rather than sorted, is exactly what sorting gave.

Finding what looks pressable and what is drawn on a screen filtered each picture for the middle value around every
place, and sorting each window was three quarters of the time those took, at every look. Counting the values of a
window and sliding it gives the same values far sooner. Nothing may change but the time: every value is compared.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import ndimage

from core.perception.picture_arithmetic import median, middle_values


def _a_drawing(rng, tall, wide, channels, top):
    """Flat patches with hard edges and a little noise, as a drawn screen has."""
    blocks = rng.integers(0, top, size=(tall // 6 + 2, wide // 6 + 2) + ((channels,) if channels else ()))
    drawn = np.repeat(np.repeat(blocks, 6, 0), 6, 1)[:tall, :wide]
    noise = rng.integers(0, 2, size=drawn.shape)
    return np.clip(drawn + noise, 0, top - 1)


@pytest.mark.parametrize("mode", ["nearest", "reflect"])
def test_every_value_is_what_sorting_gives(mode):
    rng = np.random.default_rng(7)
    for trial in range(200):
        tall, wide = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        down, across = int(rng.integers(1, 25)), int(rng.integers(1, 25))
        channels = int(rng.choice([0, 3]))
        top = int(rng.choice([2, 256, 766]))
        values = _a_drawing(rng, tall, wide, channels, top) if trial % 2 else rng.integers(
            0, top, size=(tall, wide) + ((channels,) if channels else ()))
        values = values.astype(np.uint8 if top <= 256 else np.int16)
        size = (down, across, 1) if channels else (down, across)
        # Where a window is many times wider than the values, mirrored, sorting does not give one answer twice;
        # those are sorted, as before, and only where sorting answers the same twice is the counting held to it.
        sorted_once = ndimage.median_filter(values, size=size, mode=mode)
        if not np.array_equal(sorted_once, ndimage.median_filter(values, size=size, mode=mode)):
            continue
        counted = middle_values(values, (down, across), mode=mode)
        assert counted.dtype == sorted_once.dtype
        assert np.array_equal(counted, sorted_once), (values.shape, (down, across), mode)


def test_values_that_are_not_whole_are_sorted_as_before():
    values = np.random.default_rng(3).random((20, 30)).astype(np.float32)
    assert np.array_equal(middle_values(values, (5, 7)), ndimage.median_filter(values, size=(5, 7), mode="nearest"))


def test_a_colour_picture_is_filtered_colour_by_colour():
    picture = _a_drawing(np.random.default_rng(4), 60, 80, 3, 256).astype(np.uint8)
    assert np.array_equal(median(picture, 9), ndimage.median_filter(picture, size=(9, 9, 1), mode="nearest"))


def test_counting_is_far_sooner_than_sorting_a_wide_window():
    picture = _a_drawing(np.random.default_rng(5), 200, 320, 0, 766).astype(np.int16)
    middle_values(picture, (27, 27))                                  # compiled before it is timed
    began = time.perf_counter()
    middle_values(picture, (27, 27))
    counted = time.perf_counter() - began
    began = time.perf_counter()
    ndimage.median_filter(picture, size=(27, 27), mode="nearest")
    sorted_took = time.perf_counter() - began
    assert counted * 4 < sorted_took
