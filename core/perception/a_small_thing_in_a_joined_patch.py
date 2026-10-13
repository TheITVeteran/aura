"""A small thing does not disappear because its foreground touches the scene.

Offline 2026-10-12 a light body remained visible inside a large moving
background patch. Rejecting the whole patch by its size or median colour
discarded the body too. Recent measured looks can separate it first.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core.perception.background_visible_beside_a_patch import exposed_background
from core.verify import invariant


def keep_pixels_apart_from_the_floor(picture: np.ndarray, inside: np.ndarray, *,
                                   left: int, top: int, backdrop: np.ndarray, difference: float) -> np.ndarray:
    """Correct only foreground pixels that themselves match their surroundings.

    A median that matches the surrounding floor says nothing about minority
    pixels in the patch. Those remain foreground. Without a surrounding
    sample, no pixel is put back into the backdrop.
    """
    h, w = inside.shape
    tall, wide = picture.shape[:2]
    y0, y1 = max(0, top - 2), min(tall, top + h + 2)
    x0, x1 = max(0, left - 2), min(wide, left + w + 2)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    ring[top - y0:top - y0 + h, left - x0:left - x0 + w] = False
    if not ring.any() or not inside.any():
        return inside
    around = np.median(picture[y0:y1, x0:x1][ring].astype(np.float32), axis=0)
    here = picture[top:top + h, left:left + w]
    pixels = here[inside].astype(np.float32)
    if float(np.abs(np.median(pixels, axis=0) - around).max()) < difference:
        floor = inside & (np.abs(here.astype(np.float32) - around).max(axis=2) < difference)
    else:
        floor = exposed_background(picture, inside, left=left, top=top, difference=difference)
    backdrop[top:top + h, left:left + w][floor] = here[floor]
    return inside & ~floor


def a_recent_thing_inside(picture: np.ndarray, inside: np.ndarray, *,
                         left: int, top: int, things: list[Any], at: float) -> list[np.ndarray]:
    """Separate a recent thing's measured colours near its predicted place from a larger patch.

    The unassigned remainder stays a part too. No semantic name fills in
    unseen pixels, and an absent or old observation cannot supply a seed.
    """
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


def _whole_connected_support(inside: np.ndarray, left: int, top: int, right: int, bottom: int,
                             wide: int, tall: int) -> bool:
    """The search window cannot cut a piece from scenery or join separate patches.

    Offline 2026-10-12 a moving search window cut changing fragments from a
    wall. Their centres moved with the window although the wall stood still.
    A component reaching an interior window edge has no measured full extent.
    """
    from scipy import ndimage

    if ((left > 0 and inside[:, 0].any()) or (right < wide and inside[:, -1].any())
            or (top > 0 and inside[0].any()) or (bottom < tall and inside[-1].any())):
        return False
    _labels, count = ndimage.label(inside, structure=np.ones((3, 3), bool))
    return count == 1


def connected_parts_before_rejecting(picture: np.ndarray, inside: np.ndarray, *,
                                    largest: float, cleaned: bool, smallest: int = 3,
                                    difference: float = 20, proposals: int = 32) -> list[np.ndarray]:
    """Separate cleaned islands; discover bounded compact proposals inside an oversized patch.

    The spatial extent counts too: sparse foreground can span a whole view.
    Novel colour islands are ranked by area, occupancy and surrounding
    contrast before allocating their masks. Scenery noise cannot create an
    unbounded collection of tracks. Recent seeded parts are already separate.
    """
    rows, cols = np.nonzero(inside)
    if not len(rows):
        return []
    extent = (int(rows.max() - rows.min()) + 1) * (int(cols.max() - cols.min()) + 1)
    oversized = max(len(rows), extent) > largest
    if not cleaned and not oversized:
        return [inside]
    labels, stats = _islands(inside)
    if not oversized:
        return [labels == number for number, _x, _y, _w, _h, area in stats if area >= smallest]
    candidates = []
    colours = picture.astype(np.int32) // 64
    bins = colours[:, :, 0] * 16 + colours[:, :, 1] * 4 + colours[:, :, 2]
    for number, x, y, w, h, area in stats:
        if area < smallest:
            continue
        support = labels[y:y + h, x:x + w] == number
        if max(area, w * h) <= largest:
            score = _compact_contrast(picture, support, x, y, w, h, area, difference)
            if score > 0:
                candidates.append((score, labels, number, 0, 0))
            continue
        local_bins = bins[y:y + h, x:x + w]
        for colour in np.unique(local_bins[support]):
            coloured_labels, coloured_stats = _islands(support & (local_bins == colour))
            for index, cx, cy, cw, ch, coloured_area in coloured_stats:
                if not smallest <= coloured_area <= largest or cw * ch > largest:
                    continue
                own = coloured_labels[cy:cy + ch, cx:cx + cw] == index
                score = _compact_contrast(picture, own, x + cx, y + cy, cw, ch, coloured_area, difference)
                if score <= 0:
                    continue
                candidates.append((score, coloured_labels, index, x, y))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    parts = []
    for _score, labels, index, x, y in candidates[:max(0, proposals)]:
        part = np.zeros_like(inside)
        part[y:y + labels.shape[0], x:x + labels.shape[1]] = labels == index
        parts.append(part)
    return parts


def _islands(inside: np.ndarray) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Connected labels and their support, without unused centre calculations."""
    from scipy import ndimage

    labels, _count = ndimage.label(inside, structure=np.ones((3, 3), bool))
    areas = np.bincount(labels.ravel())
    stats = []
    for number, where in enumerate(ndimage.find_objects(labels), 1):
        if where is not None:
            y, x = where[0].start, where[1].start
            stats.append((number, x, y, where[1].stop - x, where[0].stop - y, int(areas[number])))
    return labels, stats


def _compact_contrast(picture: np.ndarray, support: np.ndarray, x: int, y: int,
                      w: int, h: int, area: int, difference: float) -> float:
    """How much compact visible support differs from the pixels around its own box."""
    density = area / (w * h)
    if density < .2:
        return 0.0
    tall, wide = picture.shape[:2]
    y0, y1, x0, x1 = max(0, y - 2), min(tall, y + h + 2), max(0, x - 2), min(wide, x + w + 2)
    ring = np.ones((y1 - y0, x1 - x0), bool)
    ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
    if not ring.any():
        return 0.0
    own = picture[y:y + h, x:x + w][support]
    contrast = float(np.abs(np.median(own.astype(np.float32), axis=0)
                            - np.median(picture[y0:y1, x0:x1][ring].astype(np.float32), axis=0)).max())
    return area * density * contrast / 255 if contrast >= difference else 0.0


@invariant("perception.joined_foreground_keeps_a_distinct_small_thing", scope="perception",
           owner="core/perception/a_small_thing_in_a_joined_patch.py", observational=False)
def _small_joined_thing_invariant() -> tuple:
    from types import SimpleNamespace

    picture = np.full((30, 40, 3), 80, np.uint8)
    picture[12:16, 18:22] = 240
    inside = np.ones((20, 30), bool)
    backdrop = np.zeros_like(picture, dtype=np.float32)
    kept = keep_pixels_apart_from_the_floor(picture, inside, left=5, top=5, backdrop=backdrop, difference=20)
    assert int(kept.sum()) == 16 and np.all(backdrop[12:16, 18:22] == 0)
    look = np.zeros(64)
    look[63] = 1
    thing = SimpleNamespace(x=20, y=14, w=4, h=4, vx=0, vy=0, seen=0, look=look, size=16)
    parts = a_recent_thing_inside(picture, np.ones((30, 40), bool), left=0, top=0, things=[thing], at=.1)
    assert len(parts) == 2 and int(parts[0].sum()) == 16
    assert np.array_equal(np.sum(parts, axis=0), np.ones((30, 40), bool))
    assert len(a_recent_thing_inside(picture, inside, left=5, top=5, things=[thing], at=1)) == 1
    parts = connected_parts_before_rejecting(picture, np.ones((30, 40), bool), largest=300, cleaned=False)
    assert len(parts) == 1 and int(parts[0].sum()) == 16
    return ()
