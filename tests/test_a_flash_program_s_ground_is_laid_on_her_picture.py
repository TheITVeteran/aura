"""The ground a Flash program's code marks is laid on her picture as its code lays it: Camp Lazlo: Paintcan Panic's 30 x
25 cells, and its hero kept inside (10, 110, 590, 390).

Two things its real code says are read apart: the grid and the rectangle the hero is kept in
(core/perception/how_a_program_plays.py), and how coverage is counted, the painted cells over 2.4 as a percentage. Laid
together, the cells the hero can be brought over number 20 by 12, which is 240: a hundred percent at 2.4 a percent. The
first test holds the reading of the real program where the offline museum is on this machine; the others hold what
that reading was found to be (tests/test_how_a_program_plays_is_read_from_its_code.py) and code written to say the same.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.agency.ground_her_code_names import ground_in_her_picture
from core.perception.a_world_as_its_code_makes_it import from_how_it_plays
from core.perception.how_a_program_plays import AKey, HowItPlays, how_it_plays

pytestmark = pytest.mark.unit

_MUSEUM = Path(os.path.expanduser("~/.aura/flash-campaign/offline-bundle/programs"))
_CARD = json.loads((Path(__file__).parent / "fixtures" / "flash_games_as_their_code_says" / "25.json").read_text())
#: Her working picture of a 600 x 400 stage.
SHAPE = (160, 240)


def _as_read() -> HowItPlays:
    """What the reading of the real program was found to say (its test asserts each of these)."""
    keys = {k: AKey(k, "moves", "", "Hero.update") for k in ("left", "up", "right", "down")}
    keys["z"] = AKey("z", "jumps", "", "Hero.update")
    return HowItPlays(keys=keys, hers={"class": "Hero", "symbol": "Lazlo", "numbers": {"_speedMax": 7.0}},
                      stage=(600.0, 400.0),
                      covers={"cell": (30.0, 25.0), "field": (10.0, 110.0, 590.0, 390.0), "counted_by": "_paintCount"})


def test_the_cells_its_hero_can_reach_are_the_cells_its_coverage_counts():
    world = from_how_it_plays(_as_read(), 31.0)
    ground = ground_in_her_picture(world, SHAPE)
    reachable = ground.left_to_do()
    assert len(reachable) == round(100 * 2.4) == 240
    assert {c for c, _ in reachable} == set(range(20)) and {r for _, r in reachable} == set(range(4, 16))
    assert ground.cell == pytest.approx((12.0, 10.0))           # 30 x 25 stage pixels at 0.4 a pixel
    # Its card, written by reading the same code by hand, says the same of the grid and the field.
    assert tuple(_CARD["goal"]["cell"]) == world.ground.cell and tuple(_CARD["goal"]["field"]) == world.hers.kept_inside


def test_its_hero_runs_at_its_greatest_speed_a_frame_at_its_frames_a_second():
    world = from_how_it_plays(_as_read(), 31.0)
    assert world.hers.paces["left"] == (-7.0 * 31.0, 0.0) and world.hers.paces["down"] == (0.0, 217.0)
    assert "z" not in world.hers.paces and world.keys["z"] == "jumps"


def test_code_written_to_say_the_same_is_read_to_the_same_ground():
    code = [
        "// init script for symbol 3",
        "  // anonymous function (board, mc)",
        "  r1._mc = r2",
        "_global.Hero = function (board, mc) {...}",
        "  // anonymous function ()",
        "  if !Key.isDown(37) goto +20",
        "  r1.moveDir('w')",
        "  if !Key.isDown(39) goto +20",
        "  r1.moveDir('e')",
        "  if !(r1._x < r1._bounds.left) goto +4",
        "function (board, mc) {...}.prototype.update = function () {...}",
        "function (board, mc) {...}.prototype._speedMax = 7",
        "  // anonymous function ()",
        "  r1._bounds = new flash.geom.Rectangle(10, 110, 590, 390)",
        "function (board, mc) {...}.prototype.place = function () {...}",
        "  // anonymous function (x, y)",
        "  r1._grid[Math.floor((x / 30))][Math.floor((y / 25))] = 1",
        "  r1._paintCount = (r1._paintCount + 1)",
        "GameBoard.prototype.paintAt = function () {...}",
        "  // anonymous function ()",
        "  return (r1._paintCount / 2.4)",
        "GameBoard.prototype.getCoverage = function () {...}",
        "  // anonymous function ()",
        "  if !(r1.getCoverage() >= r1._quota) goto +12",
        "  r1.endLevel('pass')",
        "GameBoard.prototype.check = function () {...}",
    ]
    plays = how_it_plays(code, (600.0, 400.0))
    world = from_how_it_plays(plays, 31.0)
    assert world.ground.cell == (30.0, 25.0) and world.hers.kept_inside == (10.0, 110.0, 590.0, 390.0)
    assert len(ground_in_her_picture(world, SHAPE).left_to_do()) == 240


def test_the_real_program_is_read_to_that_ground():
    from core.perception.reading_a_program import read_program

    path = _MUSEUM / "25.swf"
    if not path.exists():
        pytest.skip("the offline museum of Flash games is not on this machine")
    world = read_program(path.read_bytes(), "25.swf").world
    ground = ground_in_her_picture(world, SHAPE)
    assert world.ground.cell == (30.0, 25.0) and world.hers.kept_inside == (10.0, 110.0, 590.0, 390.0)
    assert len(ground.left_to_do()) == 240 and world.hers.paces["right"] == (217.0, 0.0)


def test_a_reading_with_nothing_of_its_world_lays_nothing():
    assert from_how_it_plays(None) is None and from_how_it_plays(HowItPlays()) is None
    assert ground_in_her_picture(from_how_it_plays(HowItPlays(keys={"left": AKey("left", "moves", "", "")})), SHAPE) is None
