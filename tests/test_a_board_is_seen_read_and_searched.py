"""A board of two sides is seen in a picture, known for its family, played by the rules its words give, and searched.

Read offline 2026-10-11 from a game that "plays like checkers but with a huge twist": a draughts board drawn in two
pinks with no writing on it, two sides of chips, a figure standing on one chip of each side, and rules that change the
family's (a special piece that swaps with its own and wins at the far row). The picture in tests/fixtures is that
board as the game first drew it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.agency.playing_a_board import a_move_said, draughts_said, is_draughts, position_of, search
from core.agency.rules_of_a_board import Draughts, Position
from core.perception.a_board_seen import a_board_in

pytestmark = pytest.mark.unit

PICTURE = Path(__file__).parent / "fixtures" / "a_board_of_two_sides.png"
WORDS = ("This game plays like checkers but with a huge twist: the Big Shot! Three Ways to Win: 1. Capture all of your "
         "opponent's pieces. 2. Block in your opponent so they have no moves left. 3. Get your Big Shot to king row. "
         "Regular pieces can move only forward. If you're able to jump over an opponent's piece and capture it, you've got "
         "to do it. Get a piece to king row, and it becomes a king, which can move forward or backward. Your Big Shot can "
         "always move forward or backward. Or your Big Shot can swap places with any of your pieces next to it, as long as "
         "you don't have any captures you've got to do.")


def test_the_board_is_seen_square_by_square_with_its_two_sides_and_the_figure_standing_on_each():
    board = a_board_in(np.asarray(Image.open(PICTURE).convert("RGB")))
    assert (board.rows, board.columns, board.width, board.height) == (8, 8, 56.0, 42.0)
    assert board.says().splitlines() == ["a.a.A.a.", ".a.a.a.a", "a.a.a.a.", "........",
                                         "........", ".b.b.b.b", "b.b.b.b.", ".b.B.b.b"]
    assert is_draughts(board) and is_draughts(board, WORDS)


def test_its_words_change_the_familys_rules_as_they_say():
    rules = draughts_said(WORDS, up="b", down="a")
    assert rules.men_forward_only and rules.must_take and rules.crowned
    assert rules.special_swaps and rules.special_wins_at_far_row
    plain = draughts_said("Checkers. Jump your opponent's pieces.", up="b", down="a")
    assert not plain.special_swaps and not plain.special_wins_at_far_row


def _position(rows: list[str], turn: str) -> Position:
    return Position.from_rows([[{".": "", "A": "a+", "B": "b+"}.get(c, c) for c in row] for row in rows], turn)


def test_a_jump_must_be_made_where_the_words_say_so_and_runs_on():
    rules = Draughts(up="b", down="a", must_take=True)
    position = _position(["........", "........", "...a....", "........", ".a......", "b.......", "........",
                          "........"], "b")
    moves = rules.moves(position)
    assert moves == [((5, 0), (3, 2), (1, 4))]
    after = rules.after(position, moves[0])
    assert not after.pieces("a") and rules.winner(after) == "b"


def test_the_search_takes_a_win_it_can_see_and_a_move_is_said_as_players_say_it():
    rules = Draughts(up="b", down="a", special_swaps=True, special_wins_at_far_row=True)
    position = _position(["........", "..B.....", "........", "......a.", "........", "........", "a.......",
                          "........"], "b")
    found = search(rules, position, seconds=0.5)
    assert found.move is not None and found.move[-1][0] == 0
    assert rules.winner(rules.after(position, found.move)) == "b"
    assert a_move_said(position, ((5, 1), (4, 2))) == "b3 to c4"


def test_the_real_board_is_searched_within_its_time():
    board = a_board_in(np.asarray(Image.open(PICTURE).convert("RGB")))
    position, up, down = position_of(board, turn="")
    rules = draughts_said(WORDS, up, down)
    found = search(rules, Position(position.cells, up), seconds=0.6)
    assert found.move in rules.moves(Position(position.cells, up)) and found.depth >= 3
