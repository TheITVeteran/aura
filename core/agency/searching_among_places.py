"""Searching among places alike, against a clock: each opened once, quickly, in reading order, and what each held kept.

A corridor of doors, a table of face-down cards, a shelf of boxes: a person told the clock is running does not stop to
think between doors. They open each in turn, as fast as the place takes clicks, looking just long enough to see what
was behind it, and they remember what was where. Where two were alike behind (a game of pairs), they open those two
together.

The places are where the pointer turns into a hand (core/perception/where_the_pointer_turns.py), and what a place held
is the patch of the picture round it once it was opened, kept as a little colour signature. Offline 2026-10-11 a
corridor of eighteen doors was opened one every five to fourteen seconds, each click waiting on a whole look and a
decision, and the clock ran out with a third of them unopened.

Nothing here knows a game.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import numpy as np

__all__ = ["AT_LEAST", "alike_pairs", "how_far_the_view_moved", "search_the_places", "signature"]

logger = logging.getLogger("Aura.SearchingAmongPlaces")

#: The fewest places alike in sight for a search to be made among them.
AT_LEAST = 4
#: How long after a place is opened what it holds is looked at, in seconds.
SHOWS_S = 0.35
#: The patch round a place looked at, as a share of the picture's longer side.
PATCH = 0.06
#: How alike two signatures must be to be the same thing behind two places (0 alike, 1 nothing alike).
ALIKE_WITHIN = 0.12
#: How much a place's patch must change on opening for it to have opened.
OPENED_BY = 12.0


def _patch(picture: Any, place: tuple[float, float]) -> np.ndarray:
    pixels = np.asarray(picture)[:, :, :3]
    tall, wide = pixels.shape[:2]
    half = max(2, int(PATCH * max(tall, wide) / 2))
    x, y = int(place[0] * wide), int(place[1] * tall)
    return pixels[max(0, y - half):min(tall, y + half), max(0, x - half):min(wide, x + half)].astype(np.float32)


def signature(picture: Any, place: tuple[float, float]) -> np.ndarray:
    """A small colour signature of the patch round a place: how much of it is in each of a few colour bins."""
    patch = _patch(picture, place)
    if patch.size == 0:
        return np.zeros(27, dtype=np.float32)
    bins = np.clip(patch // 86, 0, 2).astype(np.int32)
    index = bins[..., 0] * 9 + bins[..., 1] * 3 + bins[..., 2]
    counts = np.bincount(index.ravel(), minlength=27).astype(np.float32)
    return counts / max(1.0, counts.sum())


def _apart(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def alike_pairs(held: dict[tuple[float, float], np.ndarray], closed: np.ndarray | None = None) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pairs of places whose openings showed the same thing, each place in one pair at most; what looks like the places
    look closed (``closed``) is no thing found."""
    places = [place for place, sign in held.items() if closed is None or _apart(sign, closed) > ALIKE_WITHIN]
    pairs, taken = [], set()
    for i, a in enumerate(places):
        if a in taken:
            continue
        best, nearest = None, ALIKE_WITHIN
        for b in places[i + 1:]:
            if b in taken:
                continue
            apart = _apart(held[a], held[b])
            if apart <= nearest:
                best, nearest = b, apart
        if best is not None:
            pairs.append((a, best))
            taken.update((a, best))
    return pairs


def how_far_the_view_moved(before: Any, after: Any) -> float | None:
    """How far, as a share of the picture's height, everything in the picture moved up or down between two pictures
    (positive: down), where it moved as one, as a view moves over a world bigger than it; None where it did not."""
    a, b = (np.asarray(picture)[:, :, :3].astype(np.float32).mean(axis=2) for picture in (before, after))
    if a.shape != b.shape or a.shape[0] < 20:
        return None
    rows_a, rows_b = a.mean(axis=1), b.mean(axis=1)
    rows_a, rows_b = rows_a - rows_a.mean(), rows_b - rows_b.mean()
    tall = len(rows_a)

    def alike(shift: int) -> float:
        x = rows_a[max(0, -shift):tall - max(0, shift)]
        y = rows_b[max(0, shift):tall - max(0, -shift)]
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-6)) if len(x) >= tall // 3 else -1.0

    best, best_shift = max((alike(shift), shift) for shift in range(-tall // 2, tall // 2 + 1, max(1, tall // 200)))
    # Moved only where moving explains the picture plainly better than standing still: rows of things alike match
    # themselves a row along.
    if best <= 0.9 or abs(best_shift) < tall * 0.03 or best - alike(0) < 0.05:
        return None
    return best_shift / tall


def _moved(held: dict[tuple[float, float], np.ndarray], paired: set[tuple[float, float]], down: float) -> None:
    """What was found behind each place, carried with the view: a place gone out of it is forgotten."""
    def shifted(place: tuple[float, float]) -> tuple[float, float]:
        return (place[0], round(place[1] + down, 4))

    kept = {shifted(place): sign for place, sign in held.items() if 0.0 <= place[1] + down <= 1.0}
    held.clear()
    held.update(kept)
    still = {shifted(place) for place in paired if 0.0 <= place[1] + down <= 1.0}
    paired.clear()
    paired.update(still)


def _twin_of(place: tuple[float, float], held: dict[tuple[float, float], np.ndarray], paired: set[tuple[float, float]],
             closed: np.ndarray | None) -> tuple[float, float] | None:
    """The place found before to hold what ``place`` holds, not paired yet."""
    sign = held.get(place)
    if sign is None or (closed is not None and _apart(sign, closed) <= ALIKE_WITHIN):
        return None
    near = [(other, _apart(sign, other_sign)) for other, other_sign in held.items()
            if other != place and other not in paired and abs(other[0] - place[0]) + abs(other[1] - place[1]) > 0.02]
    near = [(other, apart) for other, apart in near if apart <= ALIKE_WITHIN]
    return min(near, key=lambda one: one[1])[0] if near else None


async def search_the_places(look: Callable[[], Awaitable[Any]], hands: Any, places: Sequence[tuple[float, float]], *,
                            seconds: float, held: dict[tuple[float, float], np.ndarray] | None = None,
                            paired: set[tuple[float, float]] | None = None) -> dict[str, Any]:
    """Open each place not opened yet, in reading order, a moment's look after each, keeping what each held in
    ``held``; where what one held was found behind another before, that other is opened next, as a person playing pairs
    does; then open together the pairs found alike that are not yet ``paired``. Where the view moves (a building that
    scrolls), what was found is carried with it and the stretch ends for the places to be found again; where the whole
    picture turns over (another screen) or ``seconds`` are gone, it stops. What was done."""
    held = {} if held is None else held
    paired = set() if paired is None else paired
    began, opened, pairs, ended = time.monotonic(), 0, 0, "every place opened"
    first = await look()
    if first is None:
        return {"opened": 0, "pairs": 0, "ended": "the picture could not be taken"}
    closed = np.mean([signature(first[0], place) for place in places], axis=0) if places else None
    order = sorted((tuple(place) for place in places if tuple(place) not in held), key=lambda p: (round(p[1], 1), p[0]))
    for place in order:
        if time.monotonic() - began >= seconds:
            ended = "out of time for this stretch"
            break
        if place in held:
            continue
        before = await look()
        await hands.click(*place)
        await asyncio.sleep(SHOWS_S)
        after = await look()
        if before is None or after is None:
            ended = "the picture could not be taken"
            break
        down = how_far_the_view_moved(before[0], after[0])
        if down is not None:
            _moved(held, paired, down)
            ended = "the view moved"
            break
        a, b = (np.asarray(seen[0])[:, :, :3].astype(np.int16) for seen in (before, after))
        if a.shape == b.shape and (np.abs(a - b).max(axis=2) > 40).mean() > 0.45:
            ended = "the screen turned over"
            break
        if float(np.abs(_patch(after[0], place) - _patch(before[0], place)).mean()) >= OPENED_BY:
            held[place] = signature(after[0], place)
        opened += 1
        twin = _twin_of(place, held, paired, closed)
        if twin is not None:
            await hands.click(*twin)
            await asyncio.sleep(SHOWS_S)
            paired.update((place, twin))
            pairs += 1
    for a_place, b_place in alike_pairs(held, closed):
        if ended in ("the screen turned over", "the view moved") or time.monotonic() - began >= seconds:
            break
        if a_place in paired:
            continue
        await hands.click(*a_place)
        await asyncio.sleep(SHOWS_S)
        await hands.click(*b_place)
        await asyncio.sleep(SHOWS_S)
        paired.update((a_place, b_place))
        pairs += 1
    logger.info("searched among %d places: opened %d, %d held something, %d pairs opened together; %s",
                len(places), opened, len(held), pairs, ended)
    return {"opened": opened, "held": len(held), "pairs": pairs, "ended": ended}
