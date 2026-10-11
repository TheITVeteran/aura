"""A game of falling pieces is known by what it shows and played by search (core/agency/playing_a_well.py).

Two wells of their own, drawn in pixels with scenery round them and nothing written anywhere. In one, round tokens of
six kinds come down three at a time in a ruled well; a key changes their order; a token clears with the one kind it
goes with when the two touch, and nothing else ever clears. In the other, blocks of four cells come down an unruled
well, a key turns them, and full rows clear. Neither tells her which keys do what or what clears.

She is shown each well and must see it for what it is, find her keys by pressing them, find what clears by what goes
when a piece lands, and then play: against the same well left to itself, which fills and ends, hers goes on clearing.
A picture with no well, and a well with nothing coming down, are not taken for a game of falling pieces.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import numpy as np
import pytest

from core.agency import playing_a_well as playing
from core.agency.playing_a_well import (
    GOES_WITH,
    GoesWith,
    a_well_with_a_piece_coming_down,
    orientations,
    play_a_well,
    what_a_turn_did,
)
from core.agency.where_a_piece_goes import RULES, Well, settled
from core.perception.a_well_seen import a_well_in, the_piece_in, what_rests, what_stands_in

pytestmark = pytest.mark.unit

_GROUND = (38, 36, 30)
_COLOURS = [(220, 40, 40), (240, 240, 240), (40, 90, 220), (240, 200, 30), (230, 110, 190), (150, 90, 40), (60, 190, 90)]
_TRIO = [(0, 0), (0, 1), (0, 2)]
_FOURS = [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (1, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)],
          [(1, 0), (1, 1), (1, 2), (0, 2)], [(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)]]


class _WellWorld:
    """A well in a picture, on a clock of its own: pieces come down a row at a time, land, clear and come again."""

    def __init__(self, *, tokens: bool, seed: int, columns: int = 6, rows: int = 12, cell: int = 22, falls_every: float = 0.5,
                 keys: dict[str, str] | None = None) -> None:
        self.tokens, self.columns, self.rows, self.cell = tokens, columns, rows, cell
        self.falls_every, self.rand = falls_every, random.Random(seed)
        self.keys = keys or {"left": "left", "right": "right", "up": "turn", "down": "drop"}
        self.left, self.top = 90, 30
        self.now, self.fall_due = 0.0, falls_every
        self.cells: dict[tuple[int, int], int] = {}
        self.piece: list[tuple[int, int, int]] = []
        self.at = (0, 0)
        self.fast = False
        self.clears_at: float | None = None
        self.over = False
        self.cleared = self.landed = 0
        self.pressed: list[str] = []
        scene = np.random.default_rng(seed).integers(60, 255, size=(self.top * 2 + rows * cell, self.left * 2 + columns * cell, 3))
        self.scene = scene.astype(np.uint8)
        self._comes()

    # -- the game's own rules --------------------------------------------------------

    def _goes_with(self, one: int, other: int) -> bool:
        return one // 2 == other // 2 and one != other

    def _clear(self, well: Well) -> set[tuple[int, int]]:
        if not self.tokens:
            return RULES["full rows"](well)
        gone: set[tuple[int, int]] = set()
        for (col, row), kind in well.cells.items():
            for near in ((col + 1, row), (col, row + 1)):
                if near in well.cells and self._goes_with(kind, well.cells[near]):
                    gone |= {(col, row), near}
        return gone

    def _comes(self) -> None:
        if self.tokens:
            self.piece = [(dc, dr, self.rand.randrange(6)) for dc, dr in _TRIO]
        else:
            kind = self.rand.randrange(len(_FOURS))
            self.piece = [(dc, dr, kind) for dc, dr in _FOURS[kind]]
        tall = 1 + max(dr for _dc, dr, _k in self.piece)
        self.at = (self.columns // 2 - 1, -tall)
        if any((self.at[0] + dc, 0) in self.cells for dc, _dr, _k in self.piece):
            self.over = True

    def _fits(self, piece: list[tuple[int, int, int]], at: tuple[int, int]) -> bool:
        return all(0 <= at[0] + dc < self.columns and at[1] + dr < self.rows and (at[0] + dc, at[1] + dr) not in self.cells
                   for dc, dr, _k in piece)

    def _step(self) -> None:
        if self.over:
            return
        if self.clears_at is not None:
            if self.now >= self.clears_at:
                well, cleared, _chain = settled(Well(self.columns, self.rows, self.cells), self._clear)
                self.cells, self.cleared, self.clears_at = well.cells, self.cleared + cleared, None
            return
        if self.now < self.fall_due:
            return
        self.fall_due = self.now + (self.falls_every / 8.0 if self.fast else self.falls_every)
        below = (self.at[0], self.at[1] + 1)
        if self._fits(self.piece, below):
            self.at = below
            return
        if any(self.at[1] + dr < 0 for _dc, dr, _k in self.piece):
            self.over = True
            return
        for dc, dr, kind in self.piece:
            self.cells[(self.at[0] + dc, self.at[1] + dr)] = kind
        self.landed += 1
        # What clears goes a moment after the piece lands, as it does in such games; the next piece is already coming.
        self.clears_at = self.now + 0.4
        self._comes()

    def pass_time(self, seconds: float) -> None:
        ends = self.now + seconds
        while self.now < ends:
            self.now = min(ends, self.now + 0.01)
            self._step()

    # -- her eyes and hands ----------------------------------------------------------

    async def look(self) -> tuple[np.ndarray, float]:
        self.pass_time(0.03)
        return self.picture(), self.now

    async def down(self, key: str) -> None:
        self.pressed.append(key)
        does = self.keys.get(key)
        if self.over or self.clears_at is not None and does != "drop":
            return
        if does in ("left", "right"):
            moved = (self.at[0] + (-1 if does == "left" else 1), self.at[1])
            if self._fits(self.piece, moved):
                self.at = moved
        elif does == "turn" and self.tokens:
            kinds = [k for _dc, _dr, k in self.piece]
            self.piece = [(dc, dr, kinds[(i + 1) % 3]) for i, (dc, dr, _k) in enumerate(self.piece)]
        elif does == "turn":
            tallest = max(dr for _dc, dr, _k in self.piece)
            turned = [(tallest - dr, dc, k) for dc, dr, k in self.piece]
            if self._fits(turned, self.at):
                self.piece = turned
        elif does == "drop":
            self.fast, self.fall_due = True, self.now

    async def up(self, key: str) -> None:
        if self.keys.get(key) == "drop":
            self.fast = False

    def picture(self) -> np.ndarray:
        image = self.scene.copy()
        cell, left, top = self.cell, self.left, self.top
        image[top:top + self.rows * cell, left:left + self.columns * cell] = _GROUND
        if self.tokens:
            for column in range(1, self.columns):
                image[top:top + self.rows * cell, left + column * cell] = (52, 50, 44)
            for row in range(1, self.rows):
                image[top + row * cell, left:left + self.columns * cell] = (52, 50, 44)
        yy, xx = np.mgrid[0:cell, 0:cell]
        disc = (xx - cell / 2 + 0.5) ** 2 + (yy - cell / 2 + 0.5) ** 2 <= (0.42 * cell) ** 2
        falling = {(self.at[0] + dc, self.at[1] + dr): kind for dc, dr, kind in self.piece} if not self.over else {}
        for (column, row), kind in {**self.cells, **falling}.items():
            if row < 0:
                continue
            patch = image[top + row * cell:top + (row + 1) * cell, left + column * cell:left + (column + 1) * cell]
            if self.tokens:
                patch[disc] = _COLOURS[kind]
                patch[cell // 2 - 2:cell // 2 + 2, cell // 2 - 2:cell // 2 + 2] = _COLOURS[(kind + 3) % 7]
            else:
                patch[1:-1, 1:-1] = _COLOURS[kind]
        return image


@pytest.fixture
def on_its_clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Her play run on the world's own clock, so a minute of it takes a moment."""

    def run(world: _WellWorld, coroutine: Any) -> Any:
        async def wait(seconds: float) -> None:
            world.pass_time(seconds)

        monkeypatch.setattr(playing, "_now", lambda: world.now)
        monkeypatch.setattr(playing, "_wait", wait)
        return asyncio.run(coroutine)

    return run


def _left_to_itself(world: _WellWorld, seconds: float) -> None:
    world.pass_time(seconds)


def test_the_well_is_seen_where_it_is_ruled_and_where_it_is_not() -> None:
    for tokens in (True, False):
        world = _WellWorld(tokens=tokens, seed=3)
        world.pass_time(6.0)
        well = a_well_in(world.picture())
        assert well is not None, "no well seen"
        assert (well.columns, well.rows) == (world.columns, world.rows)
        assert abs(well.wide - world.cell) < 1.0 and abs(well.left - world.left) < 2.0 and abs(well.top - world.top) < 2.0


def test_what_rests_and_what_comes_down_are_told_apart() -> None:
    world = _WellWorld(tokens=True, seed=5)
    world.pass_time(12.0)
    # A moment when a piece is well clear of what rests and nothing is about to clear.
    while world.clears_at is not None or world.at[1] < 1 or (world.at[0], world.at[1] + 4) in world.cells:
        world.pass_time(0.1)
    assert world.landed >= 1 and not world.over
    well = a_well_in(world.picture())
    resting = what_rests(what_stands_in(world.picture(), well), well)
    assert set(resting) == set(world.cells)
    # One kind for each look: cells the world gave one kind are read as one kind, and unlike kinds as unlike.
    by_kind: dict[int, set[int]] = {}
    for cell, kind in resting.items():
        by_kind.setdefault(world.cells[cell], set()).add(kind)
    assert all(len(read) == 1 for read in by_kind.values())
    assert len({next(iter(read)) for read in by_kind.values()}) == len(by_kind)
    piece = the_piece_in(world.picture(), well, resting)
    assert piece is not None and piece.column == world.at[0]


def test_a_well_is_seen_where_no_picture_library_can_be_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Her main process refuses to load one (core/media/safe_imports.py). LIVE 2026-10-11 her seeing of a well asked
    # for one, and the refusal ended every game she opened, well in it or not.
    import sys

    monkeypatch.setitem(sys.modules, "cv2", None)
    world = _WellWorld(tokens=True, seed=5)
    world.pass_time(12.0)
    while world.clears_at is not None or world.at[1] < 1 or (world.at[0], world.at[1] + 4) in world.cells:
        world.pass_time(0.1)
    well = a_well_in(world.picture())
    resting = what_rests(what_stands_in(world.picture(), well), well)
    assert set(resting) == set(world.cells)
    assert the_piece_in(world.picture(), well, resting) is not None


def test_a_picture_with_no_well_and_a_well_with_nothing_coming_are_not_this_game(on_its_clock: Any) -> None:
    world = _WellWorld(tokens=True, seed=7)
    assert a_well_in(world.scene) is None
    assert on_its_clock(world, a_well_with_a_piece_coming_down(world.look)) is not None

    still = _WellWorld(tokens=True, seed=7)
    still.pass_time(12.0)
    # The same well with its clock stopped: time goes by and nothing in it moves.
    still.falls_every = 1e9
    still.fall_due = 1e9
    assert on_its_clock(still, a_well_with_a_piece_coming_down(still.look)) is None


def test_what_a_key_did_to_a_piece_is_known_and_gives_every_way_it_can_lie() -> None:
    trio = ((0, 0, 4), (0, 1, 5), (0, 2, 6))
    assert what_a_turn_did(trio, ((0, 0, 5), (0, 1, 6), (0, 2, 4))) == ("order", (1, 2, 0))
    assert len(orientations(trio, ("order", (1, 2, 0)))) == 3
    # Turned about, not stepped round: two ways it can lie.
    assert what_a_turn_did(trio, ((0, 0, 6), (0, 1, 5), (0, 2, 4))) == ("order", (2, 1, 0))
    assert len(orientations(trio, ("order", (2, 1, 0)))) == 2
    ell = ((0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 2, 1))
    turned = ((0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1))
    assert what_a_turn_did(ell, turned) == ("shape", 1)
    assert len(orientations(ell, ("shape", 1))) == 4
    assert what_a_turn_did(trio, trio) is None
    # An order learned on three parts is not put on a piece read with two: LIVE-like 2026-10-11 it was, and play stopped.
    assert orientations(((0, 0, 4), (0, 1, 5)), ("order", (1, 2, 0))) == [((0, 0, 4), (0, 1, 5))]


def test_kinds_that_go_together_are_learned_from_what_went_and_what_stayed() -> None:
    goes = GoesWith()
    placed = Well(4, 4, {(0, 3): 1, (0, 2): 2, (1, 3): 3})
    goes.learn(placed, {(0, 3), (0, 2)})
    assert goes.goes(1, 2) and not goes.goes(1, 3) and goes.met(1, 3) and not goes.met(2, 3)
    assert goes.clear(Well(4, 4, {(2, 3): 2, (3, 3): 1, (2, 2): 3})) == {(2, 3), (3, 3)}
    # Seen touching and staying more often than going, a pair is no longer taken to go together.
    goes.learn(Well(4, 4, {(0, 3): 1, (1, 3): 2}), set())
    goes.learn(Well(4, 4, {(0, 3): 1, (1, 3): 2}), set())
    assert not goes.goes(1, 2)


def test_tokens_that_go_with_one_another_are_found_and_the_well_kept_clear(on_its_clock: Any) -> None:
    alone = _WellWorld(tokens=True, seed=11)
    _left_to_itself(alone, 240.0)
    assert alone.over and alone.cleared <= 12, "left to itself the well should fill"

    world = _WellWorld(tokens=True, seed=11)
    well = on_its_clock(world, a_well_with_a_piece_coming_down(world.look))
    said: list[str] = []
    known: dict[str, Any] = {}
    came = on_its_clock(world, play_a_well(world.look, world, well, seconds=240.0, say=said.append, known=known))
    assert not world.over, f"her well filled: {came}"
    assert came["rule"] == GOES_WITH
    assert world.cleared >= 5 * max(1, alone.cleared) and world.cleared >= 60, (world.cleared, alone.cleared, came)
    assert known["well_keys"] == {"left": "left", "right": "right", "turn": "up", "drop": "down", "turn_does": ["order", [1, 2, 0]]}
    # What she says clears is what does: each pair she names is one the world pairs, told in the colours she sees.
    named = [line for line in said if "clear when they touch" in line]
    assert 1 <= len(named) <= 3, said
    assert len(known["well_clears"].goes_with.pairs(2)) == len(named)
    assert any("move the piece across" in line for line in said), said


def test_blocks_are_turned_and_rows_filled(on_its_clock: Any) -> None:
    alone = _WellWorld(tokens=False, seed=13, columns=8, rows=14)
    _left_to_itself(alone, 240.0)
    assert alone.over and alone.cleared == 0

    world = _WellWorld(tokens=False, seed=13, columns=8, rows=14)
    well = on_its_clock(world, a_well_with_a_piece_coming_down(world.look))
    assert well is not None and (well.columns, well.rows) == (8, 14)
    said: list[str] = []
    came = on_its_clock(world, play_a_well(world.look, world, well, seconds=240.0, say=said.append))
    assert not world.over, f"her well filled: {came}"
    assert came["rule"] == "full rows" and world.cleared >= 8 * 20, (world.cleared, came)
    # The one thing she says clears is what does.
    assert [line for line in said if "clear" in line] == ["A row filled the whole way across clears. I'll fill rows."], said


def test_a_wider_deeper_well_is_played_the_same(on_its_clock: Any) -> None:
    world = _WellWorld(tokens=False, seed=5, columns=10, rows=18)
    well = on_its_clock(world, a_well_with_a_piece_coming_down(world.look))
    assert well is not None and (well.columns, well.rows) == (10, 18)
    came = on_its_clock(world, play_a_well(world.look, world, well, seconds=240.0))
    assert not world.over and world.cleared >= 10 * 12, (world.cleared, came)


def test_keys_found_once_are_not_tried_again_and_other_keys_are_found(on_its_clock: Any) -> None:
    keys = {"a": "left", "d": "right", "w": "turn", "s": "drop"}
    world = _WellWorld(tokens=True, seed=17, keys=keys)
    well = on_its_clock(world, a_well_with_a_piece_coming_down(world.look))
    known: dict[str, Any] = {}
    offered = ("left", "right", "up", "down", "a", "d", "w", "s")
    on_its_clock(world, play_a_well(world.look, world, well, seconds=60.0, offered=offered, known=known))
    assert known["well_keys"]["left"] == "a" and known["well_keys"]["right"] == "d"
    assert known["well_keys"]["turn"] == "w" and known["well_keys"]["drop"] == "s"

    again = _WellWorld(tokens=True, seed=19, keys=keys)
    well = on_its_clock(again, a_well_with_a_piece_coming_down(again.look))
    on_its_clock(again, play_a_well(again.look, again, well, seconds=30.0, offered=offered, known=known))
    assert not {"left", "right", "up", "down"} & set(again.pressed), "keys known to do nothing were pressed again"
