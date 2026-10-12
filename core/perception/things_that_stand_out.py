"""Small things that stand out from what is round them on every side, each at its own size.

A ball on a mat, a token on a board, a marble in a ring: a small patch of one
colour with something else all the way round it. A filter of one size finds
only the things of that size. A ball as wide as the window it is looked for
through fills the window, and is taken for the ground; one much smaller is
averaged into the mat it sits on. So each size is looked for at its own scale,
as Lindeberg's scale-space blob detection does (1998): the mean colour of a
patch is held against the mean colour a little way off in eight directions,
and the patch stands out by as much as the direction that differs least. An
edge differs on one side, a bar on two, a blob on all eight.

What is found is told with its colour and its size, so whoever asks can say
which of several is the one they mean.

numpy and scipy only: her own process refuses OpenCV (core/media/safe_imports.py).
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["StandsOut", "alike_in_colour", "things_that_stand_out"]

#: The width a picture is looked over at, and the half-widths looked for, in pixels of that width: from a thing six
#: pixels across to one fifty across.
LOOKED_AT = 640
HALF_WIDTHS = (3, 4, 6, 8, 11, 15, 20, 26)

#: How far, of 255, a patch's colour must be from the nearest-coloured of its eight sides, after its own unevenness is
#: taken off, to stand out; and how much its unevenness counts against it.
STANDS_OUT_BY = 40.0
UNEVEN_COUNTS = 1.5

#: How far off, in half-widths, the sides are looked at, and how wide the patch and each side are.
SIDES_AT = 1.7
PATCH_WIDE = 1.2
SIDE_WIDE = 0.8

#: The share of a thing's middle, the pixels with the most colour in them, that its paint is read from.
MOST_COLOURED = 1.0 / 3.0

#: How many times smaller than a thing something inside it is that is told of as well, as a part of it.
A_PART_IS_SMALLER_BY = 2.0

#: The most things told of: a picture with more than this standing out is a texture, and its strongest are enough.
AT_MOST = 60

#: How much colour a thing must have for its hue to mean anything (saturation, 0-1), and how near two hues are that
#: are one colour, in degrees.
COLOURED = 0.3
ONE_HUE = 20.0


@dataclass(frozen=True)
class StandsOut:
    """A small thing that stands out: where its middle is (shares of the picture), how wide it is (a share of the
    picture's width), its colour, and by how much it stands out (of 255)."""

    x: float
    y: float
    wide: float
    colour: tuple[int, int, int]
    by: float

    @property
    def place(self) -> tuple[float, float]:
        return (self.x, self.y)


def things_that_stand_out(picture: Any, *, by: float = STANDS_OUT_BY) -> list[StandsOut]:
    """The small things on ``picture`` that stand out on every side, the ones that stand out most first."""
    from scipy import ndimage

    from core.perception.picture_arithmetic import shrink

    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.shape[0] < 8 or pixels.shape[1] < 8:
        return []
    pixels = pixels[:, :, :3]
    tall, wide = pixels.shape[:2]
    if wide > LOOKED_AT:
        pixels = shrink(pixels, LOOKED_AT, max(1, round(tall * LOOKED_AT / wide)))
    seen = pixels.astype(np.float32)
    high, across = seen.shape[:2]
    found: list[tuple[float, int, int, int, tuple[int, int, int]]] = []
    for half in HALF_WIDTHS:
        reach = SIDES_AT * half
        side_size = max(1, round(SIDE_WIDE * half)) | 1
        margin = math.ceil(reach + side_size)
        if 2 * margin >= min(high, across):
            break
        patch_size = max(1, round(PATCH_WIDE * half)) | 1
        patch = ndimage.uniform_filter(seen, size=(patch_size, patch_size, 1), mode="nearest")
        squares = ndimage.uniform_filter(seen * seen, size=(patch_size, patch_size, 1), mode="nearest")
        uneven = np.sqrt(np.maximum(0.0, squares - patch * patch)).max(axis=2)
        side = ndimage.uniform_filter(seen, size=(side_size, side_size, 1), mode="nearest")
        least: np.ndarray | None = None
        for eighth in range(8):
            right = round(reach * math.cos(eighth * math.pi / 4))
            down = round(reach * math.sin(eighth * math.pi / 4))
            differs = np.abs(patch - np.roll(side, (-down, -right), axis=(0, 1))).max(axis=2)
            least = differs if least is None else np.minimum(least, differs)
        stands = least - UNEVEN_COUNTS * uneven
        stands[:margin] = stands[-margin:] = 0.0
        stands[:, :margin] = stands[:, -margin:] = 0.0
        peaks = (stands == ndimage.maximum_filter(stands, size=2 * half + 1)) & (stands > by)
        reach_in = patch_size // 2
        for row, column in zip(*np.nonzero(peaks), strict=True):
            colour = _its_paint(seen[row - reach_in:row + reach_in + 1, column - reach_in:column + reach_in + 1])
            found.append((float(stands[row, column]), int(column), int(row), half, colour))
    found.sort(reverse=True)
    kept: list[tuple[float, int, int, int, tuple[int, int, int]]] = []
    for one in found:
        # The same thing found at two sizes near each other, or at two places inside itself, is one thing: the
        # strongest finding. A thing and a part of it far smaller (a ball and the shine on it) are both told of, for
        # whoever asks to say which they mean: dropped, the shine was all that was left of an invented token.
        if all(math.hypot(one[1] - other[1], one[2] - other[2]) > 1.2 * max(one[3], other[3])
               or max(one[3], other[3]) > A_PART_IS_SMALLER_BY * min(one[3], other[3]) for other in kept):
            kept.append(one)
        if len(kept) >= AT_MOST:
            break
    return [StandsOut(x=round(column / across, 4), y=round(row / high, 4), wide=round(2 * half / across, 4),
                      colour=colour, by=round(stood, 1))
            for stood, column, row, half, colour in kept]


def _its_paint(patch: np.ndarray) -> tuple[int, int, int]:
    """The paint a thing is made of, from the pixels of its middle: the mean of the third of them with the most colour.
    A ball drawn with a shine on it and dimples is paler on the whole than its paint, and a flat disc of the same paint
    is not: offline 2026-10-12 an orange ball picked from a row, shaded, read nearer a pink ball on the course than the
    flat orange one that was hers. A flat thing reads the same either way.
    """
    pixels = patch.reshape(-1, 3)
    much = pixels.max(axis=1) - pixels.min(axis=1)
    most = pixels[much >= np.quantile(much, 1.0 - MOST_COLOURED)]
    return tuple(int(v) for v in (most if len(most) else pixels).mean(axis=0))


def _hue(colour: tuple[int, int, int]) -> tuple[float, float, float]:
    """A colour's hue in degrees, how much colour it has, and how bright it is."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(max(0, min(255, int(v))) / 255.0 for v in colour[:3]))
    return hue * 360.0, saturation, value


def alike_in_colour(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """How alike two colours are, 0 to 1: the same hue where both have colour, whatever the light on them; the same
    brightness where neither has. A drawing's marker and the thing it marks are one paint at two sizes, one often paler
    or shaded: offline 2026-10-12 a golf ball's picture by the player's name was a salmon red and the ball on the tee a
    full one, further apart as numbers than the pink ball beside it, and the same hue to a degree.
    """
    hue_a, much_a, bright_a = _hue(a)
    hue_b, much_b, bright_b = _hue(b)
    if much_a >= COLOURED and much_b >= COLOURED:
        apart = abs(hue_a - hue_b) % 360.0
        apart = min(apart, 360.0 - apart)
        return max(0.0, 1.0 - apart / ONE_HUE) * (1.0 - 0.5 * abs(much_a - much_b))
    if much_a < COLOURED and much_b < COLOURED:
        return max(0.0, 1.0 - abs(bright_a - bright_b) / 0.25)
    return 0.0
