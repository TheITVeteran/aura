"""Whether a recent thing is inside a patch is asked of every thing for every patch, thirty of each at a look in a
real game. Where the thing is rules most out, and is asked first; its colours are only looked at for a patch it
could be in. What comes back is what came back when every thing's colours were looked at for every patch.

Measured 2026-10-12 on the pictures her play took of a game of falling eggs: 11 ms of a 91 ms look went to this,
and her watch made the same of all 372 pictures after as before.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.perception.a_small_thing_in_a_joined_patch import (
    _whole_connected_support,
    a_recent_thing_inside,
)

pytestmark = pytest.mark.unit


def _as_it_was(picture, inside, *, left, top, things, at):
    """The same separation with every thing's colours looked at before where it is: the order it was written in."""
    remaining = inside.copy()
    remaining_count = int(remaining.sum())
    tall, wide = inside.shape
    bins = picture.astype(np.int32) // 64
    bins = bins[:, :, 0] * 16 + bins[:, :, 1] * 4 + bins[:, :, 2]
    parts = []
    for thing in things:
        dt = at - thing.seen
        look = np.asarray(thing.look)
        if not 0 <= dt <= .25 or look.shape != (64,) or not np.isfinite(look).all() or look.max() <= 0:
            continue
        if remaining_count <= 2 * thing.size:
            continue
        x, y = thing.x + thing.vx * dt, thing.y + thing.vy * dt
        rx, ry = thing.w + 3 + abs(thing.vx) * dt, thing.h + 3 + abs(thing.vy) * dt
        x0, x1 = max(0, int(np.ceil(x - left - rx))), min(wide, int(np.floor(x - left + rx)) + 1)
        y0, y1 = max(0, int(np.ceil(y - top - ry))), min(tall, int(np.floor(y - top + ry)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        own_here = remaining[y0:y1, x0:x1] & (look[bins[y0:y1, x0:x1]] >= .1 * look.max())
        if not _whole_connected_support(own_here, x0, y0, x1, y1, wide, tall):
            continue
        own_count = int(own_here.sum())
        if not max(3, .25 * thing.size) <= own_count <= 2 * thing.size:
            continue
        own = np.zeros_like(inside)
        own[y0:y1, x0:x1] = own_here
        parts.append(own)
        remaining[y0:y1, x0:x1] &= ~own_here
        remaining_count -= own_count
    if not parts:
        return [inside]
    if remaining_count:
        parts.append(remaining)
    return parts


def _a_patch_with_things(seed: int):
    """A patch of two colours with a small block of a third in it, and things: some that could be the block, some far
    off, some seen too long ago, some with no colours to go by."""
    rng = np.random.default_rng(seed)
    tall, wide = int(rng.integers(18, 40)), int(rng.integers(18, 48))
    picture = np.zeros((tall, wide, 3), np.uint8)
    picture[:] = (200, 60, 60)
    inside = np.ones((tall, wide), bool)
    by, bx, bh, bw = int(rng.integers(3, tall - 9)), int(rng.integers(3, wide - 9)), int(rng.integers(3, 6)), int(rng.integers(3, 6))
    picture[by:by + bh, bx:bx + bw] = (40, 40, 220)
    left, top, at = int(rng.integers(0, 200)), int(rng.integers(0, 200)), 10.0

    def colours(*bins: int) -> np.ndarray:
        look = np.zeros(64)
        look[list(bins)] = 1.0 / len(bins)
        return look

    blue = (40 // 64) * 16 + (40 // 64) * 4 + 220 // 64
    red = (200 // 64) * 16 + (60 // 64) * 4 + 60 // 64

    def thing(x, y, look, seen=9.95, size=None, vx=0.0, vy=0.0):
        return SimpleNamespace(x=float(x), y=float(y), w=float(bw), h=float(bh), vx=vx, vy=vy, seen=seen, look=look,
                               size=float(size if size is not None else bw * bh))

    things = [
        thing(left + bx + bw / 2, top + by + bh / 2, colours(blue), vx=float(rng.normal(0, 20)), vy=float(rng.normal(0, 20))),
        thing(left + 900, top + 900, colours(blue)),                          # nowhere near
        thing(left + bx, top + by, colours(blue), seen=9.0),                  # seen a second ago
        thing(left + bx, top + by, np.zeros(64)),                             # no colours to go by
        thing(left + bx, top + by, np.full(7, 0.1)),                          # colours not kept as they should be
        thing(left + wide / 2, top + tall / 2, colours(red), size=tall * wide),
        thing(left - 2.0, top - 2.0, colours(blue, red)),
    ]
    rng.shuffle(things)
    return picture, inside, left, top, things, at


@pytest.mark.parametrize("seed", range(40))
def test_where_a_thing_is_is_asked_first_and_the_parts_are_the_same(seed):
    picture, inside, left, top, things, at = _a_patch_with_things(seed)
    was = _as_it_was(picture, inside, left=left, top=top, things=things, at=at)
    now = a_recent_thing_inside(picture, inside, left=left, top=top, things=things, at=at)
    assert len(was) == len(now)
    for a, b in zip(was, now, strict=True):
        assert np.array_equal(a, b)


def test_some_of_those_patches_do_have_a_thing_taken_out_of_them():
    taken = 0
    for seed in range(40):
        picture, inside, left, top, things, at = _a_patch_with_things(seed)
        taken += len(a_recent_thing_inside(picture, inside, left=left, top=top, things=things, at=at)) > 1
    assert taken >= 10, taken
