"""The rules of a board game, as a family with its variations: what may be moved where, what it does, and what wins.

A person who sits down to a board recognises the family it belongs to (pieces
on the dark squares of eight by eight, two sides facing: draughts) and brings
its rules with them; then they read what the game says is different ("your Big
Shot can swap places with any of your pieces next to it", "get your Big Shot to
king row" wins) and play by the rules so changed. The family is general
knowledge, the variations are read off the place, and both are held to what the
game does: a move the game will not take is not a move.

So the rules here are a family with parameters, not a game. Draughts:

* men move one square diagonally, forward only or either way;
* a jump over a piece of the other side onto the empty square beyond takes it;
  where one is possible, it may have to be made; one jump may go on to another;
* a man reaching the far row may be crowned, after which it moves either way;
* a piece of a special standing may move either way, and may swap places with a
  piece of its own side beside it;
* a side wins by taking every piece of the other, by leaving it no move, and,
  where the game says so, by bringing its special piece to the far row.

Every move is a path of squares: a step, a run of jumps, or a swap. Nothing
here knows a game's name or its pictures (core/perception/a_board_seen.py sees
the board; core/agency/playing_a_board.py plays it).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["KING", "MAN", "SPECIAL", "Draughts", "Move", "Placing", "Position"]

#: A square: row, column.
Square = tuple[int, int]
#: A move: the squares the piece goes through, first where it starts; ``swap`` where it changes places with another.
Move = tuple[Square, ...]

MAN, KING, SPECIAL = "", "K", "+"


@dataclass(frozen=True)
class Position:
    """Who stands where (``cells[row][column]``: "" empty, else the side and its standing: "a", "aK", "a+"), and whose
    turn it is."""

    cells: tuple[tuple[str, ...], ...]
    turn: str

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def columns(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def at(self, square: Square) -> str:
        r, c = square
        return self.cells[r][c] if 0 <= r < self.rows and 0 <= c < self.columns else "#"

    def pieces(self, side: str) -> list[Square]:
        return [(r, c) for r in range(self.rows) for c in range(self.columns) if self.cells[r][c][:1] == side]

    @classmethod
    def from_rows(cls, rows: list[list[str]], turn: str) -> Position:
        return cls(tuple(tuple(cell for cell in row) for row in rows), turn)


@dataclass(frozen=True)
class Draughts:
    """Draughts, with its variations as parameters. ``up`` is the side that moves toward row 0."""

    up: str = "b"
    down: str = "a"
    men_forward_only: bool = True
    must_take: bool = True
    jumps_go_on: bool = True
    crowned: bool = True
    special_both_ways: bool = True
    special_swaps: bool = False
    special_wins_at_far_row: bool = False

    def other(self, side: str) -> str:
        return self.down if side == self.up else self.up

    def _forward(self, side: str) -> int:
        return -1 if side == self.up else 1

    def _far_row(self, side: str, rows: int) -> int:
        return 0 if side == self.up else rows - 1

    def _directions(self, piece: str) -> list[tuple[int, int]]:
        side, standing = piece[:1], piece[1:]
        both = standing == KING or (standing == SPECIAL and self.special_both_ways) or not self.men_forward_only
        rows = (-1, 1) if both else (self._forward(side),)
        return [(dr, dc) for dr in rows for dc in (-1, 1)]

    def _jumps_from(self, position: Position, start: Square, piece: str, taken: frozenset[Square]) -> list[Move]:
        """Every run of jumps from ``start``, as far as jumps go on."""
        runs: list[Move] = []
        for dr, dc in self._directions(piece):
            over, onto = (start[0] + dr, start[1] + dc), (start[0] + 2 * dr, start[1] + 2 * dc)
            if over in taken or position.at(over)[:1] != self.other(piece[:1]) or position.at(onto) != "":
                continue
            further = self._jumps_from(position, onto, piece, taken | {over}) if self.jumps_go_on else []
            runs += [(start, *run) for run in further] if further else [(start, onto)]
        return runs

    def moves(self, position: Position) -> list[Move]:
        """Every move the side whose turn it is may make."""
        side = position.turn
        jumps: list[Move] = []
        steps: list[Move] = []
        swaps: list[Move] = []
        for square in position.pieces(side):
            piece = position.at(square)
            jumps += self._jumps_from(position, square, piece, frozenset())
            for dr, dc in self._directions(piece):
                onto = (square[0] + dr, square[1] + dc)
                if position.at(onto) == "":
                    steps.append((square, onto))
                elif self.special_swaps and piece[1:] == SPECIAL and position.at(onto)[:1] == side:
                    swaps.append((square, onto, square))
        if jumps and self.must_take:
            return jumps
        return jumps + steps + swaps

    def after(self, position: Position, move: Move) -> Position:
        """The position a move leaves, with the turn passed to the other side."""
        cells = [list(row) for row in position.cells]
        start = move[0]
        piece = cells[start[0]][start[1]]
        if len(move) == 3 and move[2] == start:
            # A swap: the special piece and its own beside it change places.
            other = move[1]
            cells[start[0]][start[1]], cells[other[0]][other[1]] = cells[other[0]][other[1]], piece
            return Position(tuple(tuple(row) for row in cells), self.other(position.turn))
        cells[start[0]][start[1]] = ""
        for here, there in zip(move, move[1:], strict=False):
            if abs(there[0] - here[0]) == 2:
                cells[(here[0] + there[0]) // 2][(here[1] + there[1]) // 2] = ""
        end = move[-1]
        side, standing = piece[:1], piece[1:]
        if self.crowned and standing == MAN and end[0] == self._far_row(side, position.rows):
            piece = side + KING
        cells[end[0]][end[1]] = piece
        return Position(tuple(tuple(row) for row in cells), self.other(position.turn))

    def winner(self, position: Position) -> str:
        """The side that has won, "" where none has yet."""
        for side in (self.up, self.down):
            if self.special_wins_at_far_row and any(
                    position.at(sq)[1:] == SPECIAL and sq[0] == self._far_row(side, position.rows)
                    for sq in position.pieces(side)):
                return side
            if not position.pieces(self.other(side)):
                return side
        if not self.moves(position):
            return self.other(position.turn)
        return ""

    def worth(self, position: Position, side: str) -> float:
        """How the position stands for ``side``, from its pieces: men, crowned and special ones, how far on each
        special piece is where reaching the far row wins, and how much each side can do."""
        total = 0.0
        for who, sign in ((side, 1.0), (self.other(side), -1.0)):
            for square in position.pieces(who):
                standing = position.at(square)[1:]
                total += sign * {MAN: 1.0, KING: 1.6, SPECIAL: 2.0}.get(standing, 1.0)
                if standing == MAN:
                    gone = abs(square[0] - self._far_row(self.other(who), position.rows)) / max(1, position.rows - 1)
                    total += sign * 0.1 * gone
                if standing == SPECIAL and self.special_wins_at_far_row:
                    near = 1.0 - abs(square[0] - self._far_row(who, position.rows)) / max(1, position.rows - 1)
                    total += sign * 1.5 * near * near
        return total

    def with_changes(self, **changes: bool) -> Draughts:
        return replace(self, **changes)


@dataclass(frozen=True)
class Placing:
    """A family where a move puts one piece of one's own on an empty place, and a line of so many of one side wins:
    noughts and crosses (three, anywhere), four in a row dropped down columns, five in a row anywhere. ``falls`` where
    a piece put in a column falls to the lowest empty place in it."""

    line: int = 3
    falls: bool = False
    first: str = "a"
    second: str = "b"

    def other(self, side: str) -> str:
        return self.second if side == self.first else self.first

    def moves(self, position: Position) -> list[Move]:
        if self.winner(position):
            return []
        if self.falls:
            found = []
            for column in range(position.columns):
                row = next((r for r in range(position.rows - 1, -1, -1) if position.cells[r][column] == ""), None)
                if row is not None:
                    found.append(((row, column),))
            # The middle columns first: they take part in the most lines.
            return sorted(found, key=lambda move: abs(move[0][1] - (position.columns - 1) / 2))
        middle = ((position.rows - 1) / 2, (position.columns - 1) / 2)
        empty = [((r, c),) for r in range(position.rows) for c in range(position.columns) if position.cells[r][c] == ""]
        return sorted(empty, key=lambda move: abs(move[0][0] - middle[0]) + abs(move[0][1] - middle[1]))

    def after(self, position: Position, move: Move) -> Position:
        cells = [list(row) for row in position.cells]
        r, c = move[0]
        cells[r][c] = position.turn
        return Position(tuple(tuple(row) for row in cells), self.other(position.turn))

    def _lines(self, position: Position) -> list[list[str]]:
        rows, columns, n = position.rows, position.columns, self.line
        found = []
        for r in range(rows):
            for c in range(columns):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    end_r, end_c = r + dr * (n - 1), c + dc * (n - 1)
                    if 0 <= end_r < rows and 0 <= end_c < columns:
                        found.append([position.cells[r + dr * k][c + dc * k] for k in range(n)])
        return found

    def winner(self, position: Position) -> str:
        for cells in self._lines(position):
            if cells[0] and all(cell == cells[0] for cell in cells):
                return cells[0]
        if all(cell for row in position.cells for cell in row):
            return "-"  # full and no line: drawn
        return ""

    def worth(self, position: Position, side: str) -> float:
        """Open lines: each line holding pieces of one side only is worth more the more it holds."""
        total = 0.0
        for cells in self._lines(position):
            mine = sum(1 for cell in cells if cell == side)
            theirs = sum(1 for cell in cells if cell == self.other(side))
            if mine and not theirs:
                total += 4 ** mine
            elif theirs and not mine:
                total -= 4 ** theirs
        return total / (4 ** self.line)
