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

__all__ = ["AWayThrough", "goes_every_way"]

Cell = tuple[int, int]


@dataclass
class AWayThrough:
    """The cells she has been stopped by in one screen, and the way round them."""

    cell: float = 0.0
    walls: set[Cell] = field(default_factory=set)

    def new_screen(self) -> None:
        self.walls.clear()

    def stopped(self, x: float, y: float, way: tuple[float, float], size: tuple[float, float]) -> None:
        """She was at (x, y) holding a way (dx, dy) and did not go: the cells just ahead of her that way, across her own
        breadth, are blocked."""
        self.cell = self.cell or max(8.0, max(size))
        along = 0 if abs(way[0]) >= abs(way[1]) else 1
        step = 1 if way[along] > 0 else -1
        here = [x, y]
        here[along] += step * (size[along] / 2 + self.cell / 2)
        breadth = size[1 - along] / 2
        for offset in (-breadth + 1, 0.0, breadth - 1):
            spot = list(here)
            spot[1 - along] += offset
            self.walls.add((int(spot[0] // self.cell), int(spot[1] // self.cell)))

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
        path = self._shortest(start, goal, columns, rows)
        if not path:
            return None
        furthest = path[0]
        for cell in path:
            if self._clear(start, cell):
                furthest = cell
            else:
                break
        if furthest == goal:
            return to_x, to_y
        return (furthest[0] + 0.5) * self.cell, (furthest[1] + 0.5) * self.cell

    def way_length(self, x: float, y: float, to_x: float, to_y: float, wide: float, tall: float) -> float:
        """How far, in pixels, the shortest way round the walls found is from (x, y) to (to_x, to_y): the straight
        distance where no wall is known, and infinite where the walls found leave no way."""
        if not self.walls or not self.cell:
            return math.hypot(to_x - x, to_y - y)
        start, goal = self._at(x, y), self._at(to_x, to_y)
        if start == goal:
            return math.hypot(to_x - x, to_y - y)
        columns, rows = max(1, int(wide // self.cell) + 1), max(1, int(tall // self.cell) + 1)
        path = self._shortest(start, goal, columns, rows)
        if not path:
            return math.inf
        cells = [start, *path]
        return self.cell * sum(math.dist(a, b) for a, b in zip(cells, cells[1:], strict=False))

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

    def _clear(self, a: Cell, b: Cell) -> bool:
        """Whether the straight line between two cells crosses no wall."""
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 or 1
        for i in range(steps + 1):
            t = i / steps
            if (round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)) in self.walls:
                return False
        return True


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
