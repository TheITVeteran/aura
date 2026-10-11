"""A way through is found round walls met on the way (core/agency/a_way_through.py). In a world of its own: a body that
goes straight at where it makes for and is stopped by walls it cannot see; a place behind a wall with a gap at one
end. Making straight for it, she presses into the wall for good; finding a way through, she gets there."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.agency.a_way_through import AWayThrough

pytestmark = pytest.mark.unit

STEP, SPEED, SIZE = 1 / 30, 120.0, 16.0
#: A wall across the middle with a gap at its right end, and another below with a gap at its left: a way that winds.
WALLS = [(0, 140, 400, 160), (80, 220, 480, 240)]


def _blocked(x: float, y: float) -> bool:
    return any(left - SIZE / 2 < x < right + SIZE / 2 and top - SIZE / 2 < y < bottom + SIZE / 2
               for left, top, right, bottom in WALLS) or not (0 < x < 480 and 0 < y < 320)


KEYS = {"left": (-SPEED, 0.0), "right": (SPEED, 0.0), "up": (0.0, -SPEED), "down": (0.0, SPEED)}


def _drawn() -> np.ndarray:
    """The world as a picture: a dark floor, the walls drawn in grey."""
    picture = np.zeros((320, 480, 3), dtype=np.uint8)
    picture[:] = (32, 48, 40)
    for left, top, right, bottom in WALLS:
        picture[top:bottom, left:right] = (108, 111, 90)
    return picture


def _went(planner: AWayThrough | None, seconds: float = 20.0, *, by_keys: bool = False,
          seeing: bool = False) -> tuple[bool, float]:
    x, y, goal = 100.0, 60.0, (100.0, 290.0)
    at, still_since, last = 0.0, 0.0, (x, y)
    while at < seconds:
        aim = goal
        if planner is not None:
            aim = planner.next_step(x, y, *goal, 480, 320) or goal
        far = math.hypot(aim[0] - x, aim[1] - y) or 1.0
        way = ((aim[0] - x) / far * SPEED, (aim[1] - y) / far * SPEED)
        if by_keys:
            # As her play steers: one key at a time, the one that leaves her nearest the place she makes for.
            way = min(KEYS.values(), key=lambda v: abs(aim[0] - x - v[0] * STEP) + abs(aim[1] - y - v[1] * STEP))
        nx, ny = x + way[0] * STEP, y + way[1] * STEP
        if not _blocked(nx, y):
            x = nx
        if not _blocked(x, ny):
            y = ny
        # Stopped, as her play judges it: no further than a quarter of her size from where she was 0.3 s ago.
        if math.dist((x, y), last) > SIZE / 4:
            still_since, last = at, (x, y)
        elif at - still_since > 0.3 and planner is not None:
            # Stopped under the way held: what is just ahead is a wall; seeing, so is all drawn in its colour.
            planner.stopped(x, y, way, (SIZE, SIZE))
            if seeing and planner.a_wall_colour(PICTURE, 1.0, x, y, way, (SIZE, SIZE)):
                planner.walls_seen(PICTURE, 1.0, 480, 320)
            still_since, last = at, (x, y)
        if math.dist((x, y), goal) < 10:
            return True, at
        at += STEP
    return False, at


def test_making_straight_for_a_place_behind_a_wall_never_gets_there():
    assert not _went(None)[0]


def test_a_way_through_is_found_round_the_walls_met():
    # Each wall is found a cell at a time, by being stopped by it: along two walls of four hundred pixels, 18.6 seconds.
    got, took = _went(AWayThrough())
    assert got, took


def test_a_way_through_is_found_steering_by_one_key_at_a_time():
    got, took = _went(AWayThrough(), 30.0, by_keys=True)
    assert got, took


PICTURE = _drawn()


def test_one_wall_met_shows_every_wall_drawn_like_it_and_the_way_is_found_at_once():
    _got, bumping = _went(AWayThrough(), 30.0, by_keys=True)
    got, seeing = _went(AWayThrough(), 30.0, by_keys=True, seeing=True)
    assert got and seeing < 0.5 * bumping, (bumping, seeing)


def test_the_floor_is_not_taken_for_a_wall():
    planner = AWayThrough()
    planner.stopped(100.0, 60.0, (120.0, 0.0), (SIZE, SIZE))
    assert not planner.a_wall_colour(PICTURE, 1.0, 100.0, 60.0, (120.0, 0.0), (SIZE, SIZE))
    assert not planner.wall_colours


def test_with_no_wall_met_the_way_is_straight_and_with_none_there_is_none():
    planner = AWayThrough()
    assert planner.next_step(10, 10, 200, 200, 480, 320) is None
    planner.cell = 16.0
    planner.walls = {(c, 5) for c in range(0, 31)}
    assert planner.next_step(100, 40, 100, 200, 480, 320) is None
