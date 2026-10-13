"""The ground a place's code says is to be gone over, as it lies in her picture: which cells there are, which of them
she has marked, and which are left.

Without the code she measures ground in cells of her own size over the whole picture. That is wrong in two ways a
reading of the code puts right. The code's cells are its own size and begin where it says, so her cells and its
cells disagree about what is done. And the picture holds more than the ground: offline 2026-10-13 a painter whose
floor began 40 pixels below the top of its picture, under its score bar, went to the "ground" beside the bar, held
the up arrow against the top of the floor for eighteen seconds, and painted nothing more. With the code's grid
(core/perception/a_world_as_its_code_makes_it.py) only its cells are ground, a cell is marked when the point of her
the code marks by is over it, and what is left is exactly what the code still counts as not done.

Nothing here names a game: a grid, a point of her body and a rectangle she is kept in, from whatever code said them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GroundInHerPicture", "ground_in_her_picture", "mark_the_ground_under_her", "stage_in_her_picture"]


def stage_in_her_picture(stage: tuple[float, float], shape: tuple[int, int],
                         lies: tuple[float, float, float, float] | None = None) -> tuple[float, float, float] | None:
    """How a stage of ``stage`` pixels lies in a picture of ``shape`` (rows, columns): its scale and where its corner
    is. ``lies`` is where the thing that draws it was found in the picture (left, top, width, height, as shares of the
    picture), the whole picture where not found; a stage drawn into a box of another shape is shown whole and centred
    in it, as a player shows a movie."""
    wide, tall = float(stage[0]), float(stage[1])
    rows, columns = float(shape[0]), float(shape[1])
    if wide <= 0 or tall <= 0 or rows <= 0 or columns <= 0:
        return None
    left, top, width, height = lies or (0.0, 0.0, 1.0, 1.0)
    box_x, box_y, box_w, box_h = left * columns, top * rows, width * columns, height * rows
    scale = min(box_w / wide, box_h / tall)
    return scale, box_x + (box_w - wide * scale) / 2.0, box_y + (box_h - tall * scale) / 2.0


@dataclass
class GroundInHerPicture:
    """A code's grid in her picture's working pixels: where cell (0, 0) begins, a cell's width and height, how many
    columns and rows, where the point of her that marks a cell is from the middle of her body, the rectangle that
    point is kept in, and the share of cells that wins."""

    origin: tuple[float, float]
    cell: tuple[float, float]
    cells: tuple[int, int]
    mark: tuple[float, float] = (0.0, 0.0)
    reach: tuple[float, float, float, float] | None = None
    wins_at: float | None = None
    #: The cells she has marked, by column and row.
    marked: set[tuple[int, int]] = field(default_factory=set)

    def cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        column = math.floor((x - self.origin[0]) / self.cell[0] + 1e-9)
        row = math.floor((y - self.origin[1]) / self.cell[1] + 1e-9)
        if 0 <= column < self.cells[0] and 0 <= row < self.cells[1]:
            return column, row
        return None

    def middle_of(self, column: int, row: int) -> tuple[float, float]:
        return self.origin[0] + (column + 0.5) * self.cell[0], self.origin[1] + (row + 0.5) * self.cell[1]

    def mark_of(self, x: float, y: float) -> tuple[float, float]:
        """The point that marks, for her body's middle at (x, y)."""
        return x + self.mark[0], y + self.mark[1]

    def marked_by(self, mine: Any) -> tuple[int, int] | None:
        """The cell her body marks where it is now, kept as marked; None off the grid."""
        cell = self.cell_of(*self.mark_of(float(mine.x), float(mine.y)))
        if cell is not None:
            self.marked.add(cell)
        return cell

    def within_reach(self, column: int, row: int) -> bool:
        """Whether the point that marks can be brought over a cell at all: some of the cell is inside where the point
        is kept."""
        if self.reach is None:
            return True
        left, top, right, bottom = self.reach
        x0, y0 = self.origin[0] + column * self.cell[0], self.origin[1] + row * self.cell[1]
        return x0 <= right + 1e-6 and x0 + self.cell[0] >= left - 1e-6 and y0 <= bottom + 1e-6 and \
            y0 + self.cell[1] >= top - 1e-6

    def left_to_do(self) -> list[tuple[int, int]]:
        return [(c, r) for r in range(self.cells[1]) for c in range(self.cells[0])
                if (c, r) not in self.marked and self.within_reach(c, r)]

    def where_to_be_for(self, column: int, row: int) -> tuple[float, float]:
        """Where her body's middle is to be for the point that marks to be over the middle of a cell, as near it as
        where that point is kept allows."""
        x, y = self.middle_of(column, row)
        if self.reach is not None:
            left, top, right, bottom = self.reach
            x, y = min(max(x, left), right), min(max(y, top), bottom)
        return x - self.mark[0], y - self.mark[1]

    def done(self) -> float:
        reachable = sum(1 for r in range(self.cells[1]) for c in range(self.cells[0]) if self.within_reach(c, r))
        return len(self.marked) / reachable if reachable else 0.0


def ground_in_her_picture(world: Any, shape: tuple[int, int],
                          lies: tuple[float, float, float, float] | None = None) -> GroundInHerPicture | None:
    """The ground a code's reading names, laid on a picture of ``shape`` (rows, columns) of its stage, drawn where
    ``lies`` says (see ``stage_in_her_picture``); None where the reading names no ground, or not enough of it to lay
    down (its stage, its cells)."""
    ground = getattr(world, "ground", None)
    stage = getattr(world, "stage", None)
    if ground is None or not stage or not ground.cells:
        return None
    laid = stage_in_her_picture(stage, shape, lies)
    if laid is None:
        return None
    scale, left, top = laid
    hers = getattr(world, "hers", None)
    size = getattr(hers, "size", None)
    # Where the code does not say which point of her marks, it is taken to be her middle; and where it does not say her
    # size either (a Flash body placed by its registration point), the point it keeps inside its rectangle is that
    # same middle.
    marks_at = ground.marks_at
    if marks_at is None and size is not None:
        marks_at = (size[0] / 2, size[1] / 2)
    mark = (0.0, 0.0)
    if marks_at is not None and size is not None:
        mark = ((marks_at[0] - size[0] / 2) * scale, (marks_at[1] - size[1] / 2) * scale)
    reach = None
    kept = getattr(hers, "kept_inside", None)
    if kept is not None:
        at = marks_at if size is not None and marks_at is not None else (0.0, 0.0)
        x0, y0, x1, y1 = kept
        reach = (left + (x0 + at[0]) * scale, top + (y0 + at[1]) * scale,
                 left + (x1 + at[0]) * scale, top + (y1 + at[1]) * scale)
    return GroundInHerPicture(origin=(left + ground.origin[0] * scale, top + ground.origin[1] * scale),
                              cell=(ground.cell[0] * scale, ground.cell[1] * scale), cells=tuple(ground.cells),
                              mark=mark, reach=reach, wins_at=ground.wins_at)


def mark_the_ground_under_her(run: Any, moves: Any, mine: Any, happened: list[dict[str, Any]]) -> GroundInHerPicture | None:
    """Each picture of play: the cell of her own size her body is in, as before; and where the guide to the place holds
    a reading of its code that names its ground, that ground laid on her picture, with the cell of it under the point
    of her that marks marked. A new screen is new ground (a level begun is bare). The ground, or None."""
    if any(h.get("what") == "new screen" for h in happened):
        run.ground = None
    if mine is None:
        return run.ground or None
    run.cell = run.cell or max(4.0, mine.w, mine.h)
    run.covered.add((int(mine.x // run.cell), int(mine.y // run.cell)))
    if run.ground is None:
        from core.cognition.a_guide_to_a_place import THE_GUIDE

        guide = THE_GUIDE.get()
        world = getattr(guide, "world_its_code_makes", None)
        lies = getattr(guide, "stage_in_her_picture", None)
        # Looked for once a screen: False where the guide names no ground.
        run.ground = ground_in_her_picture(world, moves.shape, lies) or False
    if run.ground:
        run.ground.marked_by(mine)
    return run.ground or None
