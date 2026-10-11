"""Where a piece goes is found by trying every landing (core/agency/where_a_piece_goes.py).

Two wells of their own. One as the tiles world plays: a pair of coloured tiles comes down, the up key swaps them, and
three or more of a colour in a line, across, down or slanting, clear. One of falling blocks: four-cell pieces that
turn, and full rows clear. Dropped anywhere, the wells fill and the game ends soon; dropped where the search says, far
more is cleared and the game goes on. The rule each well clears by is found from what cleared, and from the words.
"""
from __future__ import annotations

import random

import pytest

from core.agency.where_a_piece_goes import (
    RULES,
    Well,
    WhichRule,
    best_place,
    dropped,
    rule_from_words,
    settled,
)

pytestmark = pytest.mark.unit


def _tiles_game(searching: bool, seed: int, pieces: int = 150, *, finder: WhichRule | None = None) -> tuple[int, int]:
    """Cleared and pieces played, until the top is blocked or ``pieces`` have come."""
    rand = random.Random(seed)
    well, rule, cleared = Well(6, 12), RULES["three in a line, slanting too"], 0
    for played in range(pieces):
        top, bottom = rand.randrange(4), rand.randrange(4)
        orientations = [[(0, 0, top), (0, 1, bottom)], [(0, 0, bottom), (0, 1, top)]]
        if (2, 0) in well.cells or (2, 1) in well.cells:
            return cleared, played
        if searching:
            choice = best_place(well, orientations, (finder.rule() if finder else rule))
            turned, column = choice if choice is not None else (0, 2)
        else:
            turned, column = rand.randrange(2), rand.randrange(6)
        placed = dropped(well, orientations[turned], column)
        if placed is None:
            return cleared, played
        well, gone, _chain = settled(placed, rule)
        if finder is not None:
            finder.saw(placed, well)
        cleared += gone
    return cleared, pieces


_SHAPES = {"I": [(0, 0), (1, 0), (2, 0), (3, 0)], "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
           "T": [(0, 0), (1, 0), (2, 0), (1, 1)], "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
           "Z": [(0, 0), (1, 0), (1, 1), (2, 1)], "L": [(0, 0), (0, 1), (0, 2), (1, 2)],
           "J": [(1, 0), (1, 1), (1, 2), (0, 2)]}


def _turns(shape):
    found, cells = [], shape
    for _ in range(4):
        low_c, low_r = min(c for c, _r in cells), min(r for _c, r in cells)
        normal = sorted((c - low_c, r - low_r) for c, r in cells)
        if normal not in found:
            found.append(normal)
        cells = [(-r, c) for c, r in cells]
    return [[(c, r, "#") for c, r in turn] for turn in found]


def _blocks_game(searching: bool, seed: int, pieces: int = 120) -> int:
    """Rows cleared in a ten-wide, twenty-deep well, until a piece cannot come in."""
    rand = random.Random(seed)
    well, rows = Well(10, 20), 0
    for _ in range(pieces):
        orientations = _turns(_SHAPES[rand.choice("IOTSZLJ")])
        if searching:
            choice = best_place(well, orientations, RULES["full rows"])
            if choice is None:
                return rows
            turned, column = choice
        else:
            turned = rand.randrange(len(orientations))
            column = rand.randrange(10 - max(c for c, _r, _x in orientations[turned]))
        placed = dropped(well, orientations[turned], column)
        if placed is None:
            return rows
        well, gone, _chain = settled(placed, RULES["full rows"])
        rows += gone // 10
    return rows


def test_tiles_placed_where_the_search_says_clear_far_more_and_last_far_longer():
    dropped_anywhere = [_tiles_game(False, seed) for seed in range(5)]
    searched = [_tiles_game(True, seed) for seed in range(5)]
    assert sum(c for c, _p in searched) > 4 * sum(c for c, _p in dropped_anywhere), (dropped_anywhere, searched)
    assert sum(p for _c, p in searched) > 2 * sum(p for _c, p in dropped_anywhere), (dropped_anywhere, searched)


def test_the_same_search_clears_rows_of_falling_blocks():
    anywhere = sum(_blocks_game(False, seed) for seed in range(3))
    searched = sum(_blocks_game(True, seed) for seed in range(3))
    assert searched >= 10 * max(1, anywhere), (anywhere, searched)


def test_the_rule_is_found_from_what_cleared_and_from_the_words():
    assert rule_from_words("Line up three tiles of one color to clear them.") == "three in a line, slanting too"
    assert rule_from_words("Fill a whole row to clear it.") == "full rows"
    assert rule_from_words("Join four blobs of the same colour together.") == "four together"
    finder = WhichRule()
    _tiles_game(True, 7, finder=finder)
    assert "three in a line, slanting too" in finder.standing
    assert "full rows" not in finder.standing and "four together" not in finder.standing
