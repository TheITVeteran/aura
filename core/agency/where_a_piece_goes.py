"""Where a piece goes: each place a falling piece could land, judged by what it would clear and what it would leave.

A well of cells; a piece comes down from the top; left and right move it, a key turns it or changes its order, and it
lands where it stops. Where it lands, cells clear by the game's rule, what is above falls, and that may clear more.
Players of such games look over the places it could go and drop it in the best. That is what is done here, the way it
has long been done for falling blocks (an evaluation of each landing in the manner of Dellacherie's and the El-Tetris
weights, which score a landing by lines cleared, holes, height and how rough the top is): every column and every
orientation is tried, the piece dropped, the well settled under the rule, and the result scored by what cleared, by
the chain of clears, by what is left ready to clear next, and by how high and holed the well is left.

Which rule clears cells is the game's own: full rows, three or more of a colour in a line, a group of four touching.
Each is a candidate; the one that leads is the one the game's words name ("line up three tiles of one colour"), and
the candidates kept are the ones that have foretold what cleared each time a piece settled.

Nothing here knows a game. A colour is whatever the well's cells are told apart by.
"""
from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = ["RULES", "Well", "WhichRule", "best_place", "dropped", "ready", "rule_from_words", "score", "settled"]

Cell = tuple[int, int]  # column, row; row 0 at the top


@dataclass
class Well:
    """A well of ``columns`` by ``rows``, and what is in each cell that is not empty."""

    columns: int
    rows: int
    cells: dict[Cell, object] = field(default_factory=dict)

    def copy(self) -> Well:
        return Well(self.columns, self.rows, dict(self.cells))

    def inside(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.columns and 0 <= cell[1] < self.rows

    def height(self) -> int:
        return max((self.rows - row for _col, row in self.cells), default=0)


# -- the rules that clear cells ---------------------------------------------------


def _full_rows(well: Well) -> set[Cell]:
    return {(col, row) for row in range(well.rows) if all((col, row) in well.cells for col in range(well.columns))
            for col in range(well.columns)}


def _lines_of(least: int, slanting: bool) -> Callable[[Well], set[Cell]]:
    ways = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if slanting else [])

    def clear(well: Well) -> set[Cell]:
        gone: set[Cell] = set()
        for (col, row), colour in well.cells.items():
            for dc, dr in ways:
                run = [(col, row)]
                while well.cells.get((run[-1][0] + dc, run[-1][1] + dr), None) == colour:
                    run.append((run[-1][0] + dc, run[-1][1] + dr))
                if len(run) >= least:
                    gone.update(run)
        return gone

    return clear


def _groups_of(least: int) -> Callable[[Well], set[Cell]]:
    def clear(well: Well) -> set[Cell]:
        gone: set[Cell] = set()
        seen: set[Cell] = set()
        for start, colour in well.cells.items():
            if start in seen:
                continue
            group, frontier = {start}, [start]
            while frontier:
                col, row = frontier.pop()
                for near in ((col + 1, row), (col - 1, row), (col, row + 1), (col, row - 1)):
                    if near not in group and well.cells.get(near, None) == colour:
                        group.add(near)
                        frontier.append(near)
            seen |= group
            if len(group) >= least:
                gone |= group
        return gone

    return clear


#: The rules a well can clear by, each by name.
RULES: dict[str, Callable[[Well], set[Cell]]] = {
    "full rows": _full_rows,
    "three in a line, slanting too": _lines_of(3, True),
    "three in a line": _lines_of(3, False),
    "four in a line": _lines_of(4, False),
    "four together": _groups_of(4),
    "three together": _groups_of(3),
}


def rule_from_words(words: str) -> str | None:
    """The rule a game's words name, where they name one."""
    said = (words or "").lower()
    number = {"three": 3, "3": 3, "four": 4, "4": 4}
    count = next((number[w] for w in re.findall(r"\b(three|four|3|4)\b", said)), None)
    if re.search(r"\b(?:full|complete|whole)\s+(?:rows?|lines?)\b|\bfill (?:a |the )?(?:rows?|lines?)\b", said):
        return "full rows"
    if count and re.search(r"\b(?:colou?rs?|same)\b", said):
        if re.search(r"\b(?:together|touching|group|connected|join)\b", said):
            return "four together" if count == 4 else "three together"
        if re.search(r"\b(?:line|row|column|across|diagonal|slant)\b", said):
            return "four in a line" if count == 4 else "three in a line, slanting too"
    return None


def settled(well: Well, rule: Callable[[Well], set[Cell]]) -> tuple[Well, int, int]:
    """The well once everything that clears has cleared and what was above has fallen, again and again until nothing
    more clears: the well, how many cells cleared, and how many times over (the chain)."""
    well = well.copy()
    cleared = chain = 0
    while True:
        gone = rule(well)
        if not gone:
            return well, cleared, chain
        chain += 1
        cleared += len(gone)
        for cell in gone:
            well.cells.pop(cell, None)
        if rule is _full_rows:
            # Rows above a cleared row come down as they are, gaps and all: nothing falls into a gap under it.
            emptied = sorted({row for _col, row in gone})
            well.cells = {(col, row + sum(1 for below in emptied if below > row)): colour
                          for (col, row), colour in well.cells.items()}
            continue
        for col in range(well.columns):
            kept = [well.cells.pop((col, row)) for row in range(well.rows - 1, -1, -1) if (col, row) in well.cells]
            for i, colour in enumerate(kept):
                well.cells[(col, well.rows - 1 - i)] = colour


Piece = Sequence[tuple[int, int, object]]  # each part: column and row from the piece's corner, and its colour


def dropped(well: Well, piece: Piece, column: int) -> Well | None:
    """The well with ``piece`` dropped with its corner over ``column``, where it fits there at all; None where not.
    Each part falls with the piece until one part would meet a cell or the floor."""
    if any(not 0 <= column + dc < well.columns for dc, _dr, _c in piece):
        return None
    top = -max(dr for _dc, dr, _c in piece)
    row = top
    while all(column + dc < well.columns and row + 1 + dr < well.rows and (column + dc, row + 1 + dr) not in well.cells
              for dc, dr, _c in piece):
        row += 1
    if any(row + dr < 0 for _dc, dr, _c in piece):
        return None
    placed = well.copy()
    for dc, dr, colour in piece:
        placed.cells[(column + dc, row + dr)] = colour
    return placed


def _holes(well: Well) -> int:
    holes = 0
    for col in range(well.columns):
        roof = False
        for row in range(well.rows):
            if (col, row) in well.cells:
                roof = True
            elif roof:
                holes += 1
    return holes


def ready(well: Well) -> int:
    """How many pairs of touching cells share a colour: what is lined up to clear next."""
    return sum(1 for (col, row), colour in well.cells.items() for near in ((col + 1, row), (col, row + 1))
               if well.cells.get(near, None) == colour)


def _rough(well: Well) -> int:
    tops = [next((well.rows - row for row in range(well.rows) if (col, row) in well.cells), 0)
            for col in range(well.columns)]
    return sum(abs(a - b) for a, b in itertools.pairwise(tops))


def _shafts(well: Well) -> int:
    """How deep the well's shafts are, counted as such evaluations count them: a column lower than both its neighbours
    (a wall is a neighbour as high as the well) is a shaft as deep as the lower neighbour stands above it, and a shaft
    of depth d counts 1 + 2 + ... + d, since only one shape fills a deep one and the wait for it is what loses."""
    tops = [next((well.rows - row for row in range(well.rows) if (col, row) in well.cells), 0)
            for col in range(well.columns)]
    deep = 0
    for col, top in enumerate(tops):
        left = tops[col - 1] if col > 0 else well.rows
        right = tops[col + 1] if col + 1 < well.columns else well.rows
        depth = min(left, right) - top
        if depth > 0:
            deep += depth * (depth + 1) // 2
    return deep


def score(well: Well, cleared: int, chain: int) -> float:
    """How good a well is to have left: what cleared and the chain it made, what is ready to clear next, less how
    high, holed, rough and shafted it is, each in the proportions such evaluations give them."""
    return (10.0 * cleared + 6.0 * max(0, chain - 1) + 1.0 * ready(well)
            - 1.5 * well.height() - 4.0 * _holes(well) - 0.6 * _rough(well) - 1.0 * _shafts(well))


def best_place(well: Well, orientations: Sequence[Piece], rule: Callable[[Well], set[Cell]]) -> tuple[int, int] | None:
    """The orientation (by its place in ``orientations``) and column to drop the piece at: the best landing."""
    best: tuple[float, int, int] | None = None
    for turned, piece in enumerate(orientations):
        for column in range(-3, well.columns + 3):
            placed = dropped(well, piece, column)
            if placed is None:
                continue
            after, cleared, chain = settled(placed, rule)
            value = score(after, cleared, chain)
            if best is None or value > best[0]:
                best = (value, turned, column)
    return (best[1], best[2]) if best is not None else None


@dataclass
class WhichRule:
    """The rules still standing: each, until it fails to foretell what cleared when a piece settled."""

    standing: list[str] = field(default_factory=lambda: list(RULES))
    named: str | None = None

    def told(self, words: str) -> None:
        self.named = rule_from_words(words) or self.named

    def saw(self, placed: Well, after: Well) -> None:
        """A piece settled: the well as it landed, and as it was once what cleared had cleared."""
        kept = [name for name in self.standing if settled(placed, RULES[name])[0].cells == after.cells]
        if kept:
            self.standing = kept

    def rule(self) -> Callable[[Well], set[Cell]]:
        """The rule to place by: the one the words named while it stands, else the first still standing."""
        if self.named in self.standing:
            return RULES[self.named]
        return RULES[self.standing[0]]
