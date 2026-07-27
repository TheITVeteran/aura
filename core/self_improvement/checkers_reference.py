"""English draughts, implemented independently, to grade a reconstruction.

Bryan asked for checkers and got a board whose pieces did not move. Checkers is
the right bar precisely because it is not hard: if a reconstruction lane cannot
produce it, nothing more complex is worth attempting. And it is unforgiving in
a useful way — the rules contain three things a plausible-looking implementation
routinely gets wrong, and each is mechanically detectable:

* **capture is mandatory** — if any jump exists, a quiet move is illegal;
* **jumps chain** — a capture that can continue must continue, as one move;
* **promotion ends the move** — a man reaching the back rank becomes a king and
  does not keep jumping as a king in the same turn.

This is the grader, never the answer. It is never shown to the synthesiser and
never shipped; it exists so "faithful" is a measurement rather than an opinion.
The same arrangement as the host-binary lane, where the oracle is
``/usr/bin/base64``.

Board encoding, chosen to be the obvious one a reconstruction would also reach
for: an 8x8 grid of rows top to bottom, ``0`` empty, ``"b"``/``"w"`` men,
``"B"``/``"W"`` kings. Black occupies the top rows and moves down.
"""
from __future__ import annotations

import copy
from typing import Any

Board = list[list[Any]]

BLACK, WHITE = "b", "w"
_KINGS = {"B", "W"}
_BLACK_SIDE = {"b", "B"}
_WHITE_SIDE = {"w", "W"}


def initial_board() -> Board:
    """Standard opening: twelve a side on the dark squares."""
    board: Board = [[0] * 8 for _ in range(8)]
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0:
                continue
            if row < 3:
                board[row][col] = BLACK
            elif row > 4:
                board[row][col] = WHITE
    return board


def _side_of(piece: Any) -> str | None:
    if piece in _BLACK_SIDE:
        return BLACK
    if piece in _WHITE_SIDE:
        return WHITE
    return None


def _directions(piece: Any) -> tuple[tuple[int, int], ...]:
    if piece in _KINGS:
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return ((1, -1), (1, 1)) if piece == BLACK else ((-1, -1), (-1, 1))


def _on_board(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


def _promote(board: Board, row: int, col: int) -> bool:
    """Crown a man that reached the far rank. Returns True if it was crowned."""
    piece = board[row][col]
    if piece == BLACK and row == 7:
        board[row][col] = "B"
        return True
    if piece == WHITE and row == 0:
        board[row][col] = "W"
        return True
    return False


def _jumps_from(board: Board, row: int, col: int) -> list[tuple[int, int, int, int]]:
    piece = board[row][col]
    side = _side_of(piece)
    if side is None:
        return []
    opponent = _WHITE_SIDE if side == BLACK else _BLACK_SIDE
    jumps = []
    for d_row, d_col in _directions(piece):
        over_r, over_c = row + d_row, col + d_col
        land_r, land_c = row + 2 * d_row, col + 2 * d_col
        if not _on_board(land_r, land_c):
            continue
        if board[over_r][over_c] in opponent and board[land_r][land_c] == 0:
            jumps.append((over_r, over_c, land_r, land_c))
    return jumps


def _capture_sequences(
    board: Board, row: int, col: int, path: tuple[tuple[int, int], ...]
) -> list[tuple[tuple[int, int], ...]]:
    """Every maximal jump chain from here. A chain that can continue, must."""
    jumps = _jumps_from(board, row, col)
    if not jumps:
        return [path] if len(path) > 1 else []
    sequences: list[tuple[tuple[int, int], ...]] = []
    for over_r, over_c, land_r, land_c in jumps:
        working = copy.deepcopy(board)
        piece = working[row][col]
        working[row][col] = 0
        working[over_r][over_c] = 0
        working[land_r][land_c] = piece
        # Crowning ends the move: a man promoted mid-chain does not continue
        # as a king on the same turn.
        if _promote(working, land_r, land_c):
            sequences.append((*path, (land_r, land_c)))
            continue
        deeper = _capture_sequences(working, land_r, land_c, (*path, (land_r, land_c)))
        sequences.extend(deeper or [(*path, (land_r, land_c))])
    return sequences


def legal_moves(board: Board, side: str) -> list[tuple[tuple[int, int], ...]]:
    """Legal moves as square paths. Captures are mandatory when they exist."""
    captures: list[tuple[tuple[int, int], ...]] = []
    quiet: list[tuple[tuple[int, int], ...]] = []
    for row in range(8):
        for col in range(8):
            if _side_of(board[row][col]) != side:
                continue
            for sequence in _capture_sequences(board, row, col, ((row, col),)):
                captures.append(sequence)
            if captures:
                continue
            piece = board[row][col]
            for d_row, d_col in _directions(piece):
                to_r, to_c = row + d_row, col + d_col
                if _on_board(to_r, to_c) and board[to_r][to_c] == 0:
                    quiet.append(((row, col), (to_r, to_c)))
    if captures:
        return sorted(captures)
    return sorted(quiet)


def apply_move(board: Board, path: tuple[tuple[int, int], ...]) -> Board:
    """Apply a legal path. Raises on an illegal one, because rules refuse."""
    if len(path) < 2:
        raise ValueError("a move needs a source and a destination")
    working = copy.deepcopy(board)
    from_r, from_c = path[0]
    piece = working[from_r][from_c]
    if _side_of(piece) is None:
        raise ValueError("no piece on the source square")
    side = _side_of(piece)
    if path not in legal_moves(board, side):
        raise ValueError("illegal move")
    working[from_r][from_c] = 0
    for index in range(1, len(path)):
        to_r, to_c = path[index]
        prev_r, prev_c = path[index - 1]
        if abs(to_r - prev_r) == 2:
            working[(to_r + prev_r) // 2][(to_c + prev_c) // 2] = 0
        working[to_r][to_c] = piece
        if index < len(path) - 1:
            working[to_r][to_c] = piece
    last_r, last_c = path[-1]
    working[last_r][last_c] = piece
    _promote(working, last_r, last_c)
    return working


def winner(board: Board, side_to_move: str) -> str | None:
    """The side to move loses when it has no pieces or no legal move."""
    if not legal_moves(board, side_to_move):
        return WHITE if side_to_move == BLACK else BLACK
    return None


def render(board: Board) -> str:
    """A board a person can read — the reference for "there is a view"."""
    glyphs = {0: ".", "b": "b", "w": "w", "B": "B", "W": "W"}
    lines = ["  " + " ".join(str(c) for c in range(8))]
    for index, row in enumerate(board):
        lines.append(f"{index} " + " ".join(glyphs.get(cell, "?") for cell in row))
    return "\n".join(lines)


def board_signature(board: Board) -> str:
    return "".join(str(cell) for row in board for cell in row)


__all__ = [
    "BLACK",
    "WHITE",
    "apply_move",
    "board_signature",
    "initial_board",
    "legal_moves",
    "render",
    "winner",
]
