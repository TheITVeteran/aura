"""A blueprint is not a game.

Asked to reverse-engineer a clean-room 2048 and put it on the Desktop, the
Program DNA lane produced — on the runs where it produced anything — a scaffold:
``ReconstructedProgram.execute()`` returning ``{"status": "planned"}``, beside
eight JSON files of analysis. Nothing anyone could play. The one run that
reached the chat surface said:

    "I routed `version of the game` through Program DNA, but I am not claiming
     a successful reconstruction"

— which is two defects in one sentence: the target was the filler words in front
of "2048", and the lane it routed to could not have produced a game anyway.

What is tested here is the grading, not the model. The battery has to be able to
tell a faithful implementation from a plausible one before it is worth running:
if a wrong 2048 passes, then "supported" means nothing and a file on the Desktop
is just a stronger way to be wrong.
"""
from __future__ import annotations

import random

import pytest

from core.self_improvement.program_materialization import (
    TWENTY_FORTY_EIGHT,
    _reference_2048_move,
    _verify_playable,
    build_case_sets,
    resolve_program_spec,
)

# A faithful implementation of the published rules.
FAITHFUL = '''
import random


def _slide(row):
    tiles = [v for v in row if v]
    out, gained, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))
    return out, gained


def move(state):
    b = [list(r) for r in state["board"]]
    d = state["direction"]
    n = len(b)
    if d == "left":
        rows = b
    elif d == "right":
        rows = [list(reversed(r)) for r in b]
    elif d == "up":
        rows = [[b[r][c] for r in range(n)] for c in range(n)]
    else:
        rows = [[b[r][c] for r in range(n - 1, -1, -1)] for c in range(n)]
    total, moved = 0, []
    for row in rows:
        new_row, gained = _slide(row)
        total += gained
        moved.append(new_row)
    if d == "left":
        res = moved
    elif d == "right":
        res = [list(reversed(r)) for r in moved]
    elif d == "up":
        res = [[moved[c][r] for c in range(n)] for r in range(n)]
    else:
        res = [[moved[c][n - 1 - r] for c in range(n)] for r in range(n)]
    return {"board": res, "score": total}


def spawn_tile(board):
    b = [list(r) for r in board]
    empty = [(r, c) for r in range(len(b)) for c in range(len(b)) if b[r][c] == 0]
    if not empty:
        return b
    r, c = random.choice(empty)
    b[r][c] = 2 if random.random() < 0.9 else 4
    return b


def is_game_over(board):
    n = len(board)
    if any(0 in row for row in board):
        return False
    for r in range(n):
        for c in range(n):
            if c + 1 < n and board[r][c] == board[r][c + 1]:
                return False
            if r + 1 < n and board[r][c] == board[r + 1][c]:
                return False
    return True
'''

# The two ways a plausible 2048 is actually wrong.
OVER_MERGING = FAITHFUL.replace(
    """    out, gained, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))
    return out, gained""",
    """    gained = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(tiles) - 1):
            if tiles[i] == tiles[i + 1]:
                tiles[i] *= 2
                gained += tiles[i]
                del tiles[i + 1]
                changed = True
                break
    tiles += [0] * (len(row) - len(tiles))
    return tiles, gained""",
)

RESOLVES_BACKWARDS = FAITHFUL.replace(
    """    out, gained, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (len(row) - len(out))
    return out, gained""",
    """    out, gained, i = [], 0, len(tiles) - 1
    while i >= 0:
        if i - 1 >= 0 and tiles[i] == tiles[i - 1]:
            out.insert(0, tiles[i] * 2)
            gained += tiles[i] * 2
            i -= 2
        else:
            out.insert(0, tiles[i])
            i -= 1
    out += [0] * (len(row) - len(out))
    return out, gained""",
)


def _score_against_held_out(source: str) -> tuple[int, int]:
    _, held_out = build_case_sets(TWENTY_FORTY_EIGHT)
    namespace: dict = {"__name__": "candidate"}
    exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102
    passed = 0
    for case in held_out:
        try:
            passed += namespace["move"](case["input"]) == case["expected"]
        except Exception:  # noqa: BLE001 - a crash is a failed case
            pass
    return passed, len(held_out)


# ── The oracle states the rules correctly ──────────────────────────────────

def test_a_merged_tile_does_not_merge_again() -> None:
    """[2,2,2,0] left is [4,2,0,0]. Not [2,4,0,0], and not [8,0,0,0]."""
    result = _reference_2048_move(
        {"board": [[2, 2, 2, 0], [0] * 4, [0] * 4, [0] * 4], "direction": "left"}
    )
    assert result["board"][0] == [4, 2, 0, 0]
    assert result["score"] == 4


def test_merging_resolves_in_the_direction_of_travel() -> None:
    result = _reference_2048_move(
        {"board": [[0, 2, 2, 2], [0] * 4, [0] * 4, [0] * 4], "direction": "right"}
    )
    assert result["board"][0] == [0, 0, 2, 4]


def test_four_equal_tiles_make_two_pairs() -> None:
    result = _reference_2048_move(
        {"board": [[4, 4, 4, 4], [0] * 4, [0] * 4, [0] * 4], "direction": "left"}
    )
    assert result["board"][0] == [8, 8, 0, 0]
    assert result["score"] == 16


def test_columns_move_too() -> None:
    result = _reference_2048_move(
        {"board": [[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]], "direction": "up"}
    )
    assert [row[0] for row in result["board"]] == [4, 8, 0, 0]
    assert result["score"] == 12


# ── The grading separates faithful from plausible ──────────────────────────

def test_a_faithful_implementation_reproduces_every_held_out_position() -> None:
    passed, total = _score_against_held_out(FAITHFUL)
    assert passed == total


@pytest.mark.parametrize(
    ("label", "source"),
    [("over-merging", OVER_MERGING), ("resolving backwards", RESOLVES_BACKWARDS)],
)
def test_a_plausible_but_wrong_implementation_is_caught(label: str, source: str) -> None:
    """Both of these slide correctly and look right. Neither is 2048."""
    passed, total = _score_against_held_out(source)
    assert passed < total, f"{label} passed every held-out position — the battery is blind"


def test_held_out_positions_are_never_the_training_ones() -> None:
    train, held_out = build_case_sets(TWENTY_FORTY_EIGHT)
    shown = {repr(case["input"]) for case in train}
    assert not shown & {repr(case["input"]) for case in held_out}


def test_positions_are_generated_to_expose_the_merge_rule() -> None:
    """Random boards mostly test sliding, which everyone gets right."""
    rng = random.Random(7)
    runs = 0
    for _ in range(40):
        case = TWENTY_FORTY_EIGHT.case_generator(rng)
        board, direction = case["board"], case["direction"]
        lines = board if direction in {"left", "right"} else [
            [board[r][c] for r in range(4)] for c in range(4)
        ]
        for line in lines:
            for i in range(2):
                if line[i] and line[i] == line[i + 1] == line[i + 2]:
                    runs += 1
                    break
    assert runs >= 20, "too few runs of three along the axis being moved"


# ── Playable means it plays ────────────────────────────────────────────────

def test_a_complete_program_plays() -> None:
    ok, evidence = _verify_playable(TWENTY_FORTY_EIGHT, FAITHFUL)
    assert ok, evidence
    assert "moves headlessly" in evidence


@pytest.mark.parametrize(
    ("removed", "expected_complaint"),
    [
        ("def is_game_over", "game-over"),
        ("def spawn_tile", "spawn"),
        ("def move", "move"),
    ],
)
def test_a_rule_engine_without_a_game_is_not_playable(
    removed: str, expected_complaint: str
) -> None:
    """A perfect move() with no board, no spawn and no loss is not the game."""
    crippled = FAITHFUL.replace(removed, "def _renamed_away")
    ok, evidence = _verify_playable(TWENTY_FORTY_EIGHT, crippled)
    assert not ok
    assert expected_complaint in evidence


def test_source_that_does_not_even_parse_is_not_playable() -> None:
    ok, evidence = _verify_playable(TWENTY_FORTY_EIGHT, "def move(state:\n  return")
    assert not ok
    assert "does not parse" in evidence


# ── Naming the target ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "target", ["2048", "the game 2048", "game 2048", " 2048 ", "2048 GAME"]
)
def test_the_target_resolves_however_it_is_said(target: str) -> None:
    assert resolve_program_spec(target) is TWENTY_FORTY_EIGHT


def test_an_unknown_target_falls_through_rather_than_guessing() -> None:
    assert resolve_program_spec("some proprietary internal tool") is None
