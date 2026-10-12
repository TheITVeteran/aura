"""Keep two things apart where their pixels touch in one foreground patch.

Offline 2026-10-12 a moving body crossed a painted mark. Their joined outline
was followed as one thing, and the mark inherited the body's name after it
left. The colours and places seen before the meeting still say which pixels
belong to each. Where those do not distinguish them, keep the joined patch.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.verify import invariant


def same_colour_mix(before: Any, now: Any) -> bool | None:
    """Whether two measured colour shares still describe one look; None without both measures."""
    if before is None or now is None or len(before) != len(now) or len(before) < 2:
        return None
    if not all(math.isfinite(float(v)) and v >= 0 for v in (*before, *now)):
        return None
    first, second = float(sum(before)), float(sum(now))
    if first <= 0 or second <= 0:
        return None
    return .5 * sum(abs(float(a) / first - float(b) / second) for a, b in zip(before, now, strict=True)) <= .4


def parts_when_things_touch(picture: np.ndarray, inside: np.ndarray, *,
                           left: int, top: int, things: list[Any], at: float) -> list[np.ndarray]:
    """Separate a patch only where two recent things have different measured colour mixes.

    Each pixel is weighed by both its colour's share in the earlier thing
    and its distance from where that thing was going. No object name or
    meaning supplies a part, and the parts retain every foreground pixel.
    """
    rows, cols = np.nonzero(inside)
    if len(rows) < 6:
        return [inside]
    candidates = []
    for thing in things:
        if at - thing.seen > .25 or thing.look.shape != (64,):
            continue
        dt = max(0.0, at - thing.seen)
        x, y = thing.x + thing.vx * dt, thing.y + thing.vy * dt
        if (x + thing.w / 2 + 3 < left or x - thing.w / 2 - 3 > left + inside.shape[1]
                or y + thing.h / 2 + 3 < top or y - thing.h / 2 - 3 > top + inside.shape[0]):
            continue
        near = ((abs(cols + left - x) <= thing.w / 2 + 3)
                & (abs(rows + top - y) <= thing.h / 2 + 3))
        if int(near.sum()) >= max(3, .25 * thing.size):
            candidates.append((thing, x, y))
    if len(candidates) < 2:
        return [inside]
    # Pieces whose mixes cannot distinguish them stay one part. A third,
    # different thing does not make those two distinguishable from each other.
    groups: list[list[tuple]] = []
    for candidate in candidates:
        alike = [group for group in groups if any(.5 * float(np.abs(candidate[0].look - other[0].look).sum()) <= .5
                                                 for other in group)]
        combined = [candidate]
        for group in alike:
            combined.extend(group)
            groups.remove(group)
        groups.append(combined)
    if len(groups) < 2:
        return [inside]
    pixels = picture[rows, cols].astype(np.int32) // 64
    bins = pixels[:, 0] * 16 + pixels[:, 1] * 4 + pixels[:, 2]
    scores = []
    for group in groups:
        own = []
        for thing, x, y in group:
            distance = ((cols + left - x) / max(3.0, thing.w)) ** 2 + ((rows + top - y) / max(3.0, thing.h)) ** 2
            own.append(np.log(thing.look[bins] + .001) - distance)
        scores.append(np.max(own, axis=0))
    belonging = np.argmax(np.asarray(scores), axis=0)
    parts = []
    for index, group in enumerate(groups):
        kept = belonging == index
        if int(kept.sum()) < max(3, .25 * max(thing.size for thing, _x, _y in group)):
            continue
        part = np.zeros_like(inside, dtype=bool)
        part[rows[kept], cols[kept]] = True
        parts.append(part)
    if len(parts) < 2 or sum(int(part.sum()) for part in parts) != len(rows):
        return [inside]
    return parts


@invariant("perception.joined_parts_keep_measured_pixels", scope="perception",
           owner="core/perception/parts_when_things_touch.py", observational=False)
def _joined_parts_invariant() -> tuple:
    from types import SimpleNamespace

    picture = np.empty((8, 16, 3), np.uint8)
    picture[:, :8], picture[:, 8:] = (240, 0, 0), (0, 240, 0)
    things = []
    for x, bin_number in ((4, 48), (12, 12)):
        look = np.zeros(64)
        look[bin_number] = 1
        things.append(SimpleNamespace(x=x, y=4, vx=0, vy=0, w=8, h=8, size=64, seen=0, look=look))
    inside = np.ones((8, 16), bool)
    parts = parts_when_things_touch(picture, inside, left=0, top=0, things=things, at=.05)
    assert len(parts) == 2 and np.array_equal(np.sum(parts, axis=0), inside)
    things[1].look = things[0].look
    assert len(parts_when_things_touch(picture, inside, left=0, top=0, things=things, at=.05)) == 1
    assert same_colour_mix((1, 0), (0, 1)) is False
    assert same_colour_mix(None, (0, 1)) is None
    return ()
