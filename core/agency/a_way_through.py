"""A way through: from where she is to where she is going, round what has stopped her, found as she goes.

Going straight at a place behind a wall, she presses into the wall and gets no nearer, and the place is given up as
out of her reach (core/agency/what_she_cannot_reach.py), though the wall has an end and the place is round it. A person
walks along the wall to its end. The way such a way is found without a map is old and plain: take every place not yet
known to be blocked as open, plan the shortest way across the open places, go, and where something stops her, mark
that place blocked and plan again (planning in the unknown on the freespace assumption, as robots crossing ground
they have not seen do). Each wall is found once, by meeting it; the way round it is the shortest the walls found so far
allow.

Places are cells of her own size over the picture. Only the cell just ahead of where she was stopped is marked: what
lies beyond it may be open. Nothing here knows a game or a wall.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AWayThrough", "goes_every_way"]

Cell = tuple[int, int]


@dataclass
class AWayThrough:
    """The cells she has been stopped by in one screen, and the way round them."""

    cell: float = 0.0
    walls: set[Cell] = field(default_factory=set)
    #: Whether she has been seen to go every way on the screen: a floor seen from above, for the rest of the game, though
    #: a key may be judged afresh for a moment.
    floor: bool = False
    #: The distances from her cell to every cell, and what they were worked out for.
    _field: tuple | None = field(default=None, repr=False)
    #: The colours she has been stopped by where no thing stood: the colours of the place's walls.
    wall_colours: list[tuple[int, int, int]] = field(default_factory=list)
    #: The way she is keeping to, cell by cell to where she is going, and how many walls were known when it was taken.
    _path: list = field(default_factory=list, repr=False)
    _path_walls: int = -1

    def new_screen(self) -> None:
        self.walls.clear()

    def a_floor(self, ways: dict[str, tuple[float, float]]) -> bool:
        """Whether this is a floor seen from above: once her keys have taken her every way, it stays one."""
        self.floor = self.floor or goes_every_way(ways)
        return self.floor

    def a_wall_colour(self, picture: Any, scale: float, x: float, y: float, way: tuple[float, float],
                      size: tuple[float, float], things: Any = ()) -> bool:
        """She was stopped going ``way`` from (x, y): the colour just ahead of her, where it is not the floor she stands
        on and no thing stands there, is a colour the place draws its walls in. Whether a colour was learned.

        Bumped into a cell at a time, a wall a room long takes a minute to find; seen, it is found at once. The one
        bump says which colour stops her, and every cell drawn in it is a wall from then on."""
        if picture is None or not scale or not any(way):
            return False
        far = math.hypot(*way)
        ux, uy = way[0] / far, way[1] / far
        reach = max(size) / 2 + max(2.0, (self.cell or max(size)) / 2)
        ahead = (x + ux * reach, y + uy * reach)
        behind = (x - ux * reach, y - uy * reach)
        for thing in things:
            if abs(float(thing.x) - ahead[0]) <= float(thing.w) / 2 and abs(float(thing.y) - ahead[1]) <= float(thing.h) / 2:
                return False
        wall, floor = _colour_at(picture, scale, *ahead), _colour_at(picture, scale, *behind)
        if wall is None or floor is None or _apart(wall, floor) <= SAME_COLOUR * 2:
            return False
        if any(_apart(wall, known) <= SAME_COLOUR for known in self.wall_colours):
            return False
        self.wall_colours.append(wall)
        return True

    def drawn_like(self, picture: Any, scale: float, thing: Any) -> bool:
        """A thing she has found in her way: the colour it is drawn in is a wall's, so every wall drawn like it is one,
        thing or not. Whether a colour was learned."""
        if picture is None or not scale:
            return False
        colour = _colour_at(picture, scale, float(thing.x), float(thing.y))
        if colour is None or any(_apart(colour, known) <= SAME_COLOUR for known in self.wall_colours):
            return False
        self.wall_colours.append(colour)
        return True

    def walls_seen(self, picture: Any, scale: float, wide: float, tall: float) -> int:
        """Every cell drawn mostly in a wall's colour, marked a wall: how many were."""
        if picture is None or not scale or not self.cell or not self.wall_colours:
            return 0
        import numpy as np

        image = np.asarray(picture)[..., :3].astype(np.int16)
        near = np.zeros(image.shape[:2], dtype=bool)
        for colour in self.wall_colours:
            near |= np.abs(image - np.array(colour, dtype=np.int16)).max(axis=2) <= SAME_COLOUR
        step = self.cell / scale
        found = 0
        for cx in range(int(wide // self.cell) + 1):
            for cy in range(int(tall // self.cell) + 1):
                block = near[int(cy * step):int((cy + 1) * step), int(cx * step):int((cx + 1) * step)]
                if block.size and block.mean() >= MOSTLY and (cx, cy) not in self.walls:
                    self.walls.add((cx, cy))
                    found += 1
        return found

    def a_wall_at(self, box: tuple[float, float, float, float]) -> None:
        """A thing she knows is in her way, standing over ``box`` (left, top, right, bottom): every cell it covers."""
        if not self.cell:
            return
        left, top, right, bottom = box
        for cx in range(int((left + 1) // self.cell), int((right - 1) // self.cell) + 1):
            for cy in range(int((top + 1) // self.cell), int((bottom - 1) // self.cell) + 1):
                self.walls.add((cx, cy))

    def ahead_is_a_wall(self, x: float, y: float, way: tuple[float, float], size: tuple[float, float]) -> bool:
        """Whether going ``way`` from (x, y) runs straight into a wall she knows: what holding that key would do."""
        if not self.walls or not self.cell or not any(way):
            return False
        return bool(self._ahead(x, y, way, size) & self.walls)

    def _ahead(self, x: float, y: float, way: tuple[float, float], size: tuple[float, float]) -> set[Cell]:
        """The cells just ahead of her going ``way``, across her own breadth."""
        cell = self.cell or max(8.0, max(size))
        along = 0 if abs(way[0]) >= abs(way[1]) else 1
        step = 1 if way[along] > 0 else -1
        here = [x, y]
        here[along] += step * (size[along] / 2 + cell / 2)
        breadth = size[1 - along] / 2
        cells = set()
        for offset in (-breadth + 1, 0.0, breadth - 1):
            spot = list(here)
            spot[1 - along] += offset
            cells.add((int(spot[0] // cell), int(spot[1] // cell)))
        return cells

    def stopped(self, x: float, y: float, way: tuple[float, float], size: tuple[float, float]) -> None:
        """She was at (x, y) holding a way (dx, dy) and did not go: the cells just ahead of her that way, across her own
        breadth, are blocked."""
        self.cell = self.cell or max(8.0, max(size))
        self.walls |= self._ahead(x, y, way, size)

    def next_step(self, x: float, y: float, to_x: float, to_y: float, wide: float, tall: float) -> tuple[float, float] | None:
        """Where to make for next, on the shortest way from (x, y) to (to_x, to_y) round the walls found: the furthest
        cell along it that can be gone to in a straight line through open cells. None where no wall is known, where she
        is there already, or where there is no way."""
        if not self.walls or not self.cell:
            return None
        start, goal = self._at(x, y), self._at(to_x, to_y)
        if start == goal:
            return None
        columns, rows = max(1, int(wide // self.cell) + 1), max(1, int(tall // self.cell) + 1)
        path = self._kept_or_new(start, goal, self._shortest(start, goal, columns, rows))
        if not path:
            return None
        furthest = path[0]
        here = (x / self.cell, y / self.cell)
        for cell in path:
            if self._clear(here, (cell[0] + 0.5, cell[1] + 0.5)):
                furthest = cell
            else:
                break
        if furthest == goal:
            return to_x, to_y
        return (furthest[0] + 0.5) * self.cell, (furthest[1] + 0.5) * self.cell

    def way_length(self, x: float, y: float, to_x: float, to_y: float, wide: float, tall: float) -> float:
        """How far, in pixels, the shortest way round the walls found is from (x, y) to (to_x, to_y): the straight
        distance where no wall is known, and infinite where the walls found leave no way. Read off the distances from
        her cell to every cell, worked out once each time she comes to a new cell or a wall is found."""
        if not self.walls or not self.cell:
            return math.hypot(to_x - x, to_y - y)
        start, goal = self._at(x, y), self._at(to_x, to_y)
        if start == goal:
            return math.hypot(to_x - x, to_y - y)
        columns, rows = max(1, int(wide // self.cell) + 1), max(1, int(tall // self.cell) + 1)
        far = self._distances(start, columns, rows).get(goal)
        return math.inf if far is None else far * self.cell

    def _distances(self, start: Cell, columns: int, rows: int) -> dict[Cell, float]:
        """How far every cell is from ``start``, eight ways round the walls, a corner not cut between two walls."""
        key = (start, len(self.walls), columns, rows)
        if self._field is not None and self._field[0] == key:
            return self._field[1]
        far: dict[Cell, float] = {start: 0.0}
        frontier = [(0.0, start)]
        while frontier:
            cost, here = heapq.heappop(frontier)
            if cost > far.get(here, math.inf):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    there = (here[0] + dx, here[1] + dy)
                    if not (0 <= there[0] < columns and 0 <= there[1] < rows) or there in self.walls:
                        continue
                    if dx and dy and ((here[0] + dx, here[1]) in self.walls or (here[0], here[1] + dy) in self.walls):
                        continue
                    step = cost + (math.sqrt(2) if dx and dy else 1.0)
                    if step < far.get(there, math.inf):
                        far[there] = step
                        heapq.heappush(frontier, (step, there))
        self._field = (key, far)
        return far

    def _kept_or_new(self, start: Cell, goal: Cell, new: list[Cell]) -> list[Cell]:
        """The way she is keeping to, while no wall has been found since it was taken, it leads on from where she is,
        and it is not plainly longer than the shortest. Two ways of about one length, chosen afresh each picture, turn
        her back and forth between them and she goes neither (offline 2026-10-11, up and down at the foot of a wall).
        A wall found anywhere, by being stopped or by seeing it, and the way is chosen again."""
        if self._path and new and self._path[-1] == goal and self._path_walls == len(self.walls):
            near = [i for i, cell in enumerate(self._path) if max(abs(cell[0] - start[0]), abs(cell[1] - start[1])) <= 1]
            if near:
                rest = self._path[near[0]:]
                rest = rest[1:] if rest[0] == start and len(rest) > 1 else rest
                if _length([start, *rest]) <= KEEP_TO_A_WAY * _length([start, *new]):
                    self._path = rest
                    return rest
        self._path, self._path_walls = new, len(self.walls)
        return new

    def _at(self, x: float, y: float) -> Cell:
        return int(x // self.cell), int(y // self.cell)

    def _shortest(self, start: Cell, goal: Cell, columns: int, rows: int) -> list[Cell]:
        """The shortest way, eight ways round, over cells not walls; a corner is not cut between two walls."""
        frontier = [(0.0, start)]
        came: dict[Cell, Cell | None] = {start: None}
        cost: dict[Cell, float] = {start: 0.0}
        while frontier:
            _, here = heapq.heappop(frontier)
            if here == goal:
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    there = (here[0] + dx, here[1] + dy)
                    if not (0 <= there[0] < columns and 0 <= there[1] < rows) or there in self.walls:
                        continue
                    if dx and dy and ((here[0] + dx, here[1]) in self.walls or (here[0], here[1] + dy) in self.walls):
                        continue
                    step = cost[here] + (math.sqrt(2) if dx and dy else 1.0)
                    if step < cost.get(there, math.inf):
                        cost[there], came[there] = step, here
                        heapq.heappush(frontier, (step + math.dist(there, goal), there))
        if goal not in came:
            return []
        path, at = [], goal
        while at is not None and at != start:
            path.append(at)
            at = came[at]
        return path[::-1]

    def _clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Whether her body, a cell across, goes from ``a`` to ``b`` (in cells, from the corner of the picture) in a
        straight line without touching a wall: every cell under her at every step of the way is open. A line checked
        along its middle alone let her side catch a wall's corner on the way to a place a cell up and over."""
        reach = 0.45
        steps = max(1, int(math.ceil(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 4)))
        for i in range(steps + 1):
            t = i / steps
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            for cx in {math.floor(px - reach), math.floor(px + reach)}:
                for cy in {math.floor(py - reach), math.floor(py + reach)}:
                    if (cx, cy) in self.walls:
                        return False
        return True


#: How much longer than the shortest the way she keeps to may be before she takes the shorter.
KEEP_TO_A_WAY = 1.1


def _length(cells: list[Cell]) -> float:
    return sum(math.dist(a, b) for a, b in zip(cells, cells[1:], strict=False))


#: How near, in each of red, green and blue, two colours must be to be one; what share of a cell must be in a wall's
#: colour for the cell to be a wall.
SAME_COLOUR = 24
MOSTLY = 0.5


def _colour_at(picture: Any, scale: float, x: float, y: float) -> tuple[int, int, int] | None:
    """The middle colour round (x, y), a point in working pixels, in the picture they were shrunk from."""
    import numpy as np

    image = np.asarray(picture)
    px, py = int(round(x / scale)), int(round(y / scale))
    if not (0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
        return None
    patch = image[max(0, py - 1):py + 2, max(0, px - 1):px + 2, :3].reshape(-1, 3)
    return tuple(int(v) for v in np.median(patch, axis=0))


def _apart(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(int(p) - int(q)) for p, q in zip(a, b, strict=True))


def goes_every_way(ways: dict[str, tuple[float, float]], least: float = 1.0) -> bool:
    """Whether her keys take her left, right, up and down on the screen: the ground she is on is a floor seen from
    above, where a wall met is gone round, not one seen from the side, where what is above her is climbed or not got
    to at all."""
    found = set()
    for vx, vy in ways.values():
        if abs(vx) >= max(least, 2 * abs(vy)):
            found.add("right" if vx > 0 else "left")
        elif abs(vy) >= max(least, 2 * abs(vx)):
            found.add("down" if vy > 0 else "up")
    return len(found) == 4
