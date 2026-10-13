"""A world made from the card a game's code was read into plays as the card says.

tests/fixtures/worlds_that_move/paintcan.html is Camp Lazlo: Paintcan Panic, game 25 of the 56, written from
tests/fixtures/flash_games_as_their_code_says/25.json at 0.8 of its stage. What her choosing is measured against there
is only worth what the world keeps of the card: each number here is the card's, held against the world running its own
code (tools/oracles/a_world_run_from_its_code.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.oracles.a_world_run_from_its_code import FRAME_S, WorldRunFromItsCode, node_is_here

pytestmark = [pytest.mark.unit, pytest.mark.skipif(not node_is_here(), reason="node is not installed")]

#: One of its frames, 31 a second, in the oracle's steps of a 60th: two of those always run one of its frames.
ITS_FRAME = 2 * FRAME_S
CARD = json.loads((Path(__file__).parent / "fixtures" / "flash_games_as_their_code_says" / "25.json").read_text())
S = 0.8


def test_its_chasers_come_in_on_the_levels_the_card_brings_them_in_on():
    for level, chasers in ((1, 1), (2, 1), (3, 2), (6, 2), (7, 3), (20, 3)):
        with WorldRunFromItsCode("paintcan", seed=1, query=f"level={level}") as world:
            assert len(world.state()["game"]["chasers"]) == chasers, level
    starts = {hazard["what"]: hazard.get("from_level") for hazard in CARD["hazards"]}
    assert starts == {"Edward": 1, "Chip": 3, "Skip": 7}


def test_a_cell_costs_its_paint_and_the_quota_passes_the_level():
    quota = CARD["goal"]["quota_by_level"]
    for level in (1, 9, 20):
        with WorldRunFromItsCode("paintcan", seed=1, query=f"level={level}") as world:
            world.value("(__world.game.chasers = [], 0)")
            need = math.ceil(quota[level - 1] * 2.4)
            # Painted to one short of the quota, and then the one: the first leaves the level going, the second passes it.
            world.value(f"(__world.game.painted.fill(1, 0, {need - 1}), __world.game.count = {need - 1}, 0)")
            assert world.step(ITS_FRAME)["world"]["state"] == "play"
            before = world.value("__world.game.supply")
            world.value(f"(__world.game.me.x = {10 * S + (need - 1) % 20 * 30 * S + 4}, "
                        f"__world.game.me.y = {110 * S + (need - 1) // 20 * 25 * S + 2}, 0)")
            state = world.step(ITS_FRAME)
            assert state["world"]["state"] == "over" and state["world"]["ended"][-1]["why"] == "won", level
            assert before - state["game"]["supply"] == CARD["fail"][0]["costs"]["a cell newly painted"]


def test_a_touch_costs_a_thousand_and_sends_every_chaser_back_for_two_seconds():
    with WorldRunFromItsCode("paintcan", seed=1, query="level=7") as world:
        world.value("(() => { const g = __world.game, me = g.me; for (const c of g.chasers) c.wait = 0;"
                    " g.chasers[0].x = me.x + 4; g.chasers[0].y = me.y + 1; return 0; })()")
        state = world.step(ITS_FRAME)
        game = state["game"]
        assert game["supply"] == CARD["fail"][0]["supply"] - CARD["fail"][0]["costs"]["an enemy's touch"] - 15
        middle = (game["me"]["x"] + 8, game["me"]["y"] + 8)
        for chaser in game["chasers"]:
            assert math.dist(middle, (chaser["x"] + 7, chaser["y"] + 7)) == pytest.approx(80 * S, abs=1.0)
            assert chaser["wait"] in (61, 62)                # two seconds of its frames, counted from this one
        # Out of paint, the level fails.
        world.value("(__world.game.supply = 5, 0)")
        world.value("(__world.game.me.x += 30, 0)")
        assert world.step(ITS_FRAME)["world"]["ended"][-1]["why"] == "no paint left"


def test_she_runs_at_its_greatest_speed_a_frame_and_a_chaser_at_no_more_than_its_push_allows():
    with WorldRunFromItsCode("paintcan", seed=1, query="level=1") as world:
        world.value("(__world.game.chasers[0].wait = 0, 0)")
        world.hold("ArrowRight")
        before = world.value("__world.game.me.x")
        world.step(1.0)
        after = world.value("__world.game.me.x")
        # Speed grows by 1 a frame to 7: in its 30 or 31 frames of a second, 1 + 2 + ... + 7 and then 7 a frame.
        assert (28 + 7 * 23) * S - 1e-6 <= after - before <= (28 + 7 * 24) * S + 1e-6
        world.step(2.0)
        chaser = world.value("__world.game.chasers[0]")
        assert math.hypot(chaser["vx"], chaser["vy"]) <= 0.4 / 0.03 * S + 1e-6
