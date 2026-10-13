"""The middle of the values around each place, found by counting them: the compiled half of
core/perception/picture_arithmetic.py ``middle_values``, kept apart so that it is compiled once and kept compiled.

Imported only where numba is to be had; nothing else imports numba.
"""
from __future__ import annotations

import numba
import numpy as np

__all__ = ["count_the_middle"]


@numba.njit(cache=True, nogil=True)
def count_the_middle(padded, down, across, bins, middle, out):  # pragma: no cover - compiled
    """``out[row, column]``: the value at place ``middle`` (from the least) of the ``down`` by ``across`` values of
    ``padded`` from (row, column), each whole and below ``bins``. A count of each value is kept for the window; moving
    along a row takes its left column out of the count and puts the next one in, and the middle is walked to from
    where it was (Huang, Yang and Tang, 1979)."""
    tall, wide = out.shape
    held = np.zeros(bins, np.int32)
    for row in range(tall):
        held[:] = 0
        for r in range(row, row + down):
            for c in range(across):
                held[padded[r, c]] += 1
        at, below = 0, 0
        while below + held[at] <= middle:
            below += held[at]
            at += 1
        out[row, 0] = at
        for column in range(1, wide):
            for r in range(row, row + down):
                gone = padded[r, column - 1]
                held[gone] -= 1
                if gone < at:
                    below -= 1
                come = padded[r, column + across - 1]
                held[come] += 1
                if come < at:
                    below += 1
            while below > middle:
                at -= 1
                below -= held[at]
            while below + held[at] <= middle:
                below += held[at]
                at += 1
            out[row, column] = at
