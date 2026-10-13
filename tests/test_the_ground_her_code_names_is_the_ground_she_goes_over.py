"""Where a place's code says the ground is, she goes over that ground: the cells its code counts, from where its code
marks them, and none it does not.

Offline 2026-10-13, put to the paint world's own code (tools/oracles/her_choices_on_a_world.py), her choosing painted
29%, 11% and 5% of the floor in the 150 seconds the world gives, where 85% wins. She measured ground in cells of her
own size over the whole picture, so the strip above the floor, where the world keeps its score and never lets her
go, was ground to her: she held the up arrow against the top of the floor for eighteen seconds. And the nearest cell
not yet gone over was a few pixels off, too near to outweigh any risk, so at the start she stood still for four.

Read from the world's code instead (core/perception/reading_a_world_s_code.py), the ground is 24 by 14 cells of 20
pixels from 40 pixels down, marked by her middle; and each plan she imagines is worth the cells it would go over. Each
of those is held here against the world running its own code.
"""
from __future__ import annotations

import pytest

from tools.oracles.a_world_run_from_its_code import FRAME_S, WorldRunFromItsCode, node_is_here
from tools.oracles.her_choices_on_a_world import SHAPE, HerChoicesOnAWorld, reading_of

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not node_is_here(), reason="node is not installed")]

PAINT = reading_of("paint")
SCALE = SHAPE[1] / PAINT.stage[0]


def _laid():
    from core.agency.ground_her_code_names import ground_in_her_picture

    return ground_in_her_picture(PAINT, SHAPE)


def _painted(state) -> set[tuple[int, int]]:
    columns = PAINT.ground.cells[0]
    return {(k % columns, k // columns) for k, v in enumerate(state["game"]["painted"]) if v}


def _her_middle(x: float, y: float) -> tuple[float, float]:
    """Her body's middle in her picture, for its corner at (x, y) on the world's stage."""
    w, h = PAINT.hers.size
    return (x + w / 2) * SCALE, (y + h / 2) * SCALE


def test_the_code_s_ground_laid_on_her_picture_is_where_the_world_paints():
    ground = _laid()
    left, top, right, bottom = PAINT.hers.kept_inside
    with WorldRunFromItsCode("paint", seed=5) as w:
        w.value("(__world.lives = 1e9, 0)")
        for i in range(200):
            x = left + (right - left) * ((i * 0.6180339887) % 1.0)
            y = top + (bottom - top) * ((i * 0.7548776662) % 1.0)
            w.value(f"(__world.game.painted.fill(0), __world.game.count = 0, __world.game.me.x = {x}, "
                    f"__world.game.me.y = {y}, 0)")
            painted = _painted(w.step(FRAME_S))
            assert painted == {ground.cell_of(*ground.mark_of(*_her_middle(x, y)))}, (x, y, painted)


def test_she_never_goes_for_ground_the_world_does_not_paint():
    """At the top of the floor with the top row painted, cells of her own size put the nearest ground where her middle
    would be above where the world keeps it: the world will not take her there. The code's ground puts it in the next
    row down, a cell the world counts as not painted; and taken there, the world paints it."""
    from types import SimpleNamespace

    from core.agency.choosing_where_to_be import _Choosing
    from core.agency.what_meeting_things_does import WhatMeetingDoes
    from tools.oracles.her_choices_on_a_world import _a_thing, _HerKeys

    hers = _HerKeys(PAINT.hers.paces, SCALE)
    with WorldRunFromItsCode("paint", seed=5) as w:
        w.value("(__world.lives = 1e9, __world.game.bugs = [], __world.game.me.x = 120, __world.game.me.y = 40, "
                "__world.game.painted.fill(0), __world.game.count = 0, 0)")
        w.value("(__world.game.painted.fill(1, 0, 24), __world.game.count = 24, 0)")
        state = w.step(FRAME_S)
        mine = _a_thing(1, 0, (120, 40, 16, 16), SCALE)
        moves = SimpleNamespace(shape=SHAPE, things={1: mine}, kinds=[], out_of_sight={})

        def target(ground):
            choosing = _Choosing(moves, hers, WhatMeetingDoes(), list(hers.paces), None, next_picture_s=1 / 15,
                                 covered=(own, max(mine.w, mine.h)) if ground is None else None)
            choosing.ground = ground
            return choosing.target()

        cell = max(mine.w, mine.h)
        own = {(int(mine.x // cell), int(mine.y // cell))}
        (gx, gy), why, _ = target(None)
        lowest_middle = PAINT.hers.kept_middle()[1]
        assert why == "cover" and gy / SCALE < lowest_middle          # her own cells: above where her middle can be
        w.value(f"(__world.game.me.y = {gy / SCALE - 8}, 0)")
        assert w.step(FRAME_S)["game"]["me"]["y"] + 8 == lowest_middle  # the world puts her back on the floor
        w.value("(__world.game.me.x = 120, __world.game.me.y = 40, 0)")
        ground = _laid()
        ground.marked |= _painted(state)
        (gx, gy), why, _ = target(ground)
        assert why == "cover"
        x, y = gx / SCALE - 8, gy / SCALE - 8           # where her body's corner is on the stage for that
        column, row = ground.cell_of(*ground.mark_of(gx, gy))
        assert row == 1 and (column, row) not in _painted(state)
        w.value(f"(__world.game.me.x = {x}, __world.game.me.y = {y}, 0)")
        assert (column, row) in _painted(w.step(FRAME_S))


def test_what_she_counts_as_gone_over_is_what_the_world_counts_at_every_choice():
    seen = []

    def each(ground, state):
        seen.append(ground.marked == _painted(state))

    with HerChoicesOnAWorld("paint", seed=4, decide_every=1 / 15) as play:
        play.run(12.0, each=each)
    assert len(seen) >= 150 and all(seen)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_she_goes_over_new_ground_from_the_first_moment(seed):
    """In the first four seconds, with the code's ground and plans worth the cells they go over, she paints a tenth of
    the floor; with cells of her own size she does not paint half that."""
    with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / 15) as play:
        with_code = play.run(4.0)["painted"]
    with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / 15, ground_from_code=False) as play:
        without = play.run(4.0)["painted"]
    assert with_code >= 0.10, with_code
    assert without < with_code / 2, (without, with_code)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_she_wins_the_painted_floor_by_its_code_s_ground(seed):
    with HerChoicesOnAWorld("paint", seed=seed, decide_every=1 / 15) as play:
        came_to = play.run(150.0)
    assert came_to["ended"] and came_to["ended"][-1]["why"] == "won", came_to
    assert came_to["seconds"] < 90 and came_to["lives"] == 3, came_to


@pytest.mark.slow
def test_neither_the_code_s_ground_alone_nor_the_worth_of_ground_alone_wins_it(monkeypatch):
    """Both halves are needed: cells of her own size scored as ground in her plans paint 62% at most, and the code's
    ground with plans that do not count it paints about a third; neither reaches the 85% that wins."""
    import core.agency.ground_her_code_names as laid
    import core.agency.imagining_ahead as imagining
    import tools.oracles.her_choices_on_a_world as harness

    def her_own_cells(reading, shape):
        size = reading.hers.size[0] * shape[1] / reading.stage[0]
        return laid.GroundInHerPicture(origin=(0.0, 0.0), cell=(size, size),
                                       cells=(int(shape[1] // size), int(shape[0] // size)))

    with monkeypatch.context() as patched:
        patched.setattr(laid, "ground_in_her_picture", her_own_cells)
        with HerChoicesOnAWorld("paint", seed=2, decide_every=1 / 15) as play:
            assert play.run(150.0)["painted"] < 0.85
    with monkeypatch.context() as patched:
        patched.setattr(imagining, "A_CELL_GONE_OVER", 0.0)
        with HerChoicesOnAWorld("paint", seed=2, decide_every=1 / 15) as play:
            assert play.run(150.0)["painted"] < 0.85
    with HerChoicesOnAWorld("paint", seed=2, decide_every=1 / 15, ground_from_code=False) as play:
        assert play.run(150.0)["painted"] < 0.85
    assert harness is not None
