"""The marker a place draws beside a name: whose thing is whose, as the place itself says it.

A scoreboard puts each player's token beside their name: a ball of their
colour, a face, a flag. Nobody is told what it means. The picture next to a
name is that player's thing, drawn small, and the thing on the field that is
painted the same is theirs. It is how a person finds their ball among six on a
course before touching any of them.

What is read here is only what is beside the name she gave: the colours there,
and how much of each. Which thing on the field they mark is for whoever asks,
with the things in hand (core/perception/things_that_stand_out.py).

numpy only: her own process refuses OpenCV (core/media/safe_imports.py).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.things_that_stand_out import COLOURED, alike_in_colour

__all__ = ["Marker", "how_like_the_marker", "the_marker_by"]

#: How far to each side of a name its marker is looked for, in heights of the name's letters, and how far above and
#: below.
BESIDE = 3.5
ABOVE_AND_BELOW = 0.8

#: The least share of what is beside a name one colour must cover to be told of, and the most colours told.
LEAST_SHARE = 0.03
COLOURS_TOLD = 4

#: The most of the whole picture a marker's colour may cover: what is painted all over a place (its sky, its grass)
#: is behind the name and marks nothing.
ALL_OVER = 0.04
#: The width the whole picture is looked over at for that.
LOOKED_OVER_AT = 160


@dataclass(frozen=True)
class Marker:
    """What a place draws beside a name: its painted colours, each with the share of that place it covers, most first;
    and where that place is on the picture (left, top, right, bottom, as shares), which is a legend and no thing of
    the field's."""

    colours: tuple[tuple[tuple[int, int, int], float], ...] = ()
    where: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __bool__(self) -> bool:
        return bool(self.colours)

    def holds(self, place: tuple[float, float]) -> bool:
        """Whether ``place`` is on the legend itself."""
        left, top, right, bottom = self.where
        return left <= place[0] <= right and top <= place[1] <= bottom


def the_marker_by(name: str, words: Sequence[dict[str, Any]], picture: Any) -> Marker:
    """The marker drawn beside ``name`` where ``words`` read it on ``picture``; an empty one where the name is not on
    the picture. Only colours that have colour: the plate a name is written on is pale or dark, and the token on it is
    painted.
    """
    wanted = " ".join(str(name or "").lower().split())
    if not wanted:
        return Marker()
    pixels = np.asarray(picture)
    if pixels.ndim != 3:
        return Marker()
    tall, wide = pixels.shape[:2]
    whole = re.compile(rf"(?<!\w){re.escape(wanted)}(?!\w)")
    for region in words:
        said = " ".join(str(region.get("text") or "").lower().split())
        if not whole.search(said) or len(said) > len(wanted) + 12:
            continue
        left, top = float(region.get("x", 0.0)) * wide, float(region.get("y", 0.0)) * tall
        letters_wide, letters_tall = float(region.get("width", 0.0)) * wide, float(region.get("height", 0.0)) * tall
        if letters_tall < 4 or letters_wide < 4:
            continue
        upper = max(0, round(top - ABOVE_AND_BELOW * letters_tall))
        lower = min(tall, round(top + (1.0 + ABOVE_AND_BELOW) * letters_tall))
        from_left = max(0, round(left - BESIDE * letters_tall))
        to_right = min(wide, round(left + letters_wide + BESIDE * letters_tall))
        sides = [pixels[upper:lower, from_left:max(0, round(left)), :3],
                 pixels[upper:lower, min(wide, round(left + letters_wide)):to_right, :3]]
        beside = np.concatenate([side.reshape(-1, 3) for side in sides if side.size])
        told = [(colour, share) for colour, share in _its_colours(beside) if _share_of(pixels, colour) <= ALL_OVER]
        if told:
            return Marker(colours=tuple(told), where=(from_left / wide, upper / tall, to_right / wide, lower / tall))
    return Marker()


def _share_of(pixels: np.ndarray, colour: tuple[int, int, int]) -> float:
    """How much of the whole picture is painted like ``colour``."""
    from core.perception.picture_arithmetic import shrink

    tall, wide = pixels.shape[:2]
    small = shrink(pixels[:, :, :3], LOOKED_OVER_AT, max(1, round(tall * LOOKED_OVER_AT / wide))) if wide > LOOKED_OVER_AT else pixels[:, :, :3]
    coarse, counts = np.unique((small.reshape(-1, 3) // 16).astype(np.int32), axis=0, return_counts=True)
    alike = sum(int(count) for cell, count in zip(coarse, counts, strict=True)
                if alike_in_colour(colour, tuple(int(v) * 16 + 8 for v in cell)) >= 0.5)
    return alike / max(1, int(counts.sum()))


def _its_colours(beside: np.ndarray) -> list[tuple[tuple[int, int, int], float]]:
    """The painted colours among ``beside``'s pixels, gathered by hue, each with its share."""
    if len(beside) < 16:
        return []
    seen = beside.astype(np.float32)
    most, least = seen.max(axis=1), seen.min(axis=1)
    painted = seen[(most - least) / np.maximum(1.0, most) >= COLOURED]
    gathered: list[list[Any]] = []
    # Coarse first, so a shaded ball is one colour and not a dozen: the commonest of 4,096, then what is like it.
    coarse, counts = np.unique((painted // 16).astype(np.int32), axis=0, return_counts=True)
    for index in np.argsort(-counts):
        colour = tuple(int(v) * 16 + 8 for v in coarse[index])
        for group in gathered:
            if alike_in_colour(group[0], colour) >= 0.5:
                group[1] += int(counts[index])
                break
        else:
            gathered.append([colour, int(counts[index])])
    told = [(group[0], round(group[1] / len(beside), 4)) for group in gathered if group[1] / len(beside) >= LEAST_SHARE]
    return sorted(told, key=lambda one: -one[1])[:COLOURS_TOLD]


def how_like_the_marker(colour: tuple[int, int, int], marker: Marker) -> float:
    """How like a thing of ``colour`` is to a marker's colours, 0 to 1: the best likeness to any of them."""
    return max((alike_in_colour(colour, painted) for painted, _share in marker.colours), default=0.0)
