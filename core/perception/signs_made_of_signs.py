"""A sign made of signs she already knows: read as the ones it is made of, all asked for at once.

A place teaches its signs one at a time and then puts two together: an arrow
up drawn over an arrow left, two letters side by side, a rest under a note.
Nobody is taught the pairs. They know the single signs by then, see both of
them in the new mark, and do both. Read by its outline alone a pair is one odd
shape, and reads as one of its halves or as nothing.

So a mark is explained by the signs already known. Each known sign is laid
over the mark where most of it fits (cross-correlation finds the place). Every
set of one or two of them, laid so, is held against the mark: what the set
covers that the mark has, over what either has (intersection over union, the
measure by which a detector's box is held against the true one). The set that
fits best is what the mark is made of, where it fits well enough at all.

The measure forgives a mark with a little cut off it, as when a sign overlaps
the edge of the strip it travels on and that edge is left out with the rest of
what stays put: every set loses the same sliver, and the best is still best.

Nothing here knows what the signs are signs of.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np

__all__ = ["how_alike", "laid_over", "made_of"]

#: The least a set of known signs, laid over a mark, must agree with it to be what it is made of (intersection over
#: union, 0 to 1), and the most signs one mark is made of.
FITS_FROM = 0.62
AT_MOST = 2

#: How much better two signs together must fit than the better of them alone for the mark to be both: a single sign
#: with a little noise beside it is not a pair.
BOTH_BEAT_ONE_BY = 0.08


def laid_over(sign: Any, mark: Any) -> np.ndarray:
    """A known sign laid over a mark where most of its ink falls on the mark's: a mask the size of the mark's canvas
    (the mark padded to hold the sign), true where the sign's ink lies."""
    from scipy.signal import correlate2d

    sign, mark = np.asarray(sign, dtype=bool), np.asarray(mark, dtype=bool)
    canvas = _room_for(mark, sign.shape)
    hits = correlate2d(canvas.astype(np.float32), sign.astype(np.float32), mode="valid")
    row, column = np.unravel_index(int(np.argmax(hits)), hits.shape)
    laid = np.zeros_like(canvas)
    laid[row:row + sign.shape[0], column:column + sign.shape[1]] = sign
    return laid


def how_alike(one: Any, other: Any) -> float:
    """How alike two marks are, 0 to 1: one laid over the other where it fits best, what both have over what either has."""
    other = np.asarray(other, dtype=bool)
    laid = laid_over(one, other)
    canvas = _room_for(other, np.asarray(one).shape)
    return float((laid & canvas).sum() / max(1, (laid | canvas).sum()))


def _room_for(mark: np.ndarray, sign_shape: tuple[int, ...], margin: int = 3) -> np.ndarray:
    """A mark on a canvas with room round it for a sign of ``sign_shape`` to be laid partly off it."""
    tall = max(mark.shape[0], sign_shape[0]) + 2 * margin
    wide = max(mark.shape[1], sign_shape[1]) + 2 * margin
    canvas = np.zeros((tall, wide), dtype=bool)
    top, left = (tall - mark.shape[0]) // 2, (wide - mark.shape[1]) // 2
    canvas[top:top + mark.shape[0], left:left + mark.shape[1]] = mark
    return canvas


def made_of(mark: Any, known: Mapping[str, Any]) -> frozenset[str]:
    """The known signs a mark is made of: the one or two that, laid over it, agree with it best; none where nothing
    known fits it well enough."""
    mark = np.asarray(mark, dtype=bool)
    if not mark.any() or not known:
        return frozenset()
    biggest = (max(np.asarray(sign).shape[0] for sign in known.values()),
               max(np.asarray(sign).shape[1] for sign in known.values()))
    canvas = _room_for(mark, biggest)
    laid = {}
    for key, sign in known.items():
        sign = np.asarray(sign, dtype=bool)
        over = laid_over(sign, canvas)
        # ``laid_over`` pads again; bring it back to this canvas, which already has room.
        extra_rows, extra_columns = (over.shape[0] - canvas.shape[0]) // 2, (over.shape[1] - canvas.shape[1]) // 2
        laid[key] = over[extra_rows:extra_rows + canvas.shape[0], extra_columns:extra_columns + canvas.shape[1]]
    fits: dict[frozenset[str], float] = {}
    for count in range(1, AT_MOST + 1):
        for keys in combinations(sorted(laid), count):
            together = np.zeros_like(canvas)
            for key in keys:
                together |= laid[key]
            fits[frozenset(keys)] = float((together & canvas).sum() / max(1, (together | canvas).sum()))
    alone = max((keys for keys in fits if len(keys) == 1), key=lambda keys: fits[keys])
    best = max(fits, key=lambda keys: fits[keys])
    if len(best) > 1 and fits[best] - fits[alone] < BOTH_BEAT_ONE_BY:
        best = alone
    return best if fits[best] >= FITS_FROM else frozenset()
