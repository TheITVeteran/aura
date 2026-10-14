"""Where nothing of hers is seen, where what costs her lands is where she is, and what is on its way there is ducked.

Offline 2026-10-13, a food fight seen over a table (tests/fixtures/worlds_that_move/foodfight.html, as Cartoon
Network's Food Bash says it plays): its thrower hit her with five throws of six, game after game, and she ducked by
what grew in the picture, which food going down a table does not.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.thrown_at_what_shows import WhereHarmLands, comes_at_her_where_harm_lands

pytestmark = pytest.mark.unit


def _food(number, x, y, vx=0.0, vy=75.0):
    return SimpleNamespace(number=number, x=x, y=y, w=7.0, h=7.0, vx=vx, vy=vy, sizes=(), closing_s=None)


def _thrown_and_landed(lands, number, at, losses, x=120.0):
    """A throw going down to (x, 140), seen every eighth of a second, gone at the moment a loss is read."""
    for step in range(8):
        lands.saw([_food(number, x, 80.0 + 7.5 * step)], losses, at + step / 8)
    lands.saw([], losses + 1, at + 1.0)


def test_where_harm_has_landed_twice_is_where_she_is():
    lands = WhereHarmLands()
    assert lands.where() is None
    _thrown_and_landed(lands, 1, 0.0, 0)
    assert lands.where() is None                                  # once is not yet a place
    _thrown_and_landed(lands, 2, 2.0, 1, x=124.0)
    x, y, about = lands.where()
    assert 118.0 <= x <= 126.0 and 125.0 <= y <= 135.0 and about >= 7.0


def test_a_thing_that_went_with_no_loss_is_not_where_harm_lands():
    lands = WhereHarmLands()
    for number, at in ((1, 0.0), (2, 2.0), (3, 4.0)):
        for step in range(8):
            lands.saw([_food(number, 60.0, 80.0 + 7.5 * step)], 0, at + step / 8)
        lands.saw([], 0, at + 1.0)                                # gone off the table: nothing lost
    assert lands.where() is None


def test_what_is_on_its_way_there_is_ducked_soon_before_it_lands_and_not_long_before():
    lands = WhereHarmLands()
    _thrown_and_landed(lands, 1, 0.0, 0)
    _thrown_and_landed(lands, 2, 2.0, 1)
    assert lands.coming([_food(3, 120.0, 112.0)])                 # 0.25 s from where she is
    assert not lands.coming([_food(4, 120.0, 40.0)])              # more than a second off
    assert not lands.coming([_food(5, 40.0, 112.0)])              # going down somewhere else
    assert not lands.coming([_food(6, 120.0, 112.0, vy=-75.0)])   # going away


def test_what_comes_at_her_is_kept_on_her_run_picture_by_picture():
    run = SimpleNamespace(losses=0)
    for step in range(8):
        comes_at_her_where_harm_lands(run, [_food(1, 120.0, 80.0 + 7.5 * step)], step / 8)
    run.losses = 1
    comes_at_her_where_harm_lands(run, [], 1.0)
    for step in range(8):
        comes_at_her_where_harm_lands(run, [_food(2, 120.0, 80.0 + 7.5 * step)], 2.0 + step / 8)
    run.losses = 2
    comes_at_her_where_harm_lands(run, [], 3.0)
    assert comes_at_her_where_harm_lands(run, [_food(3, 120.0, 112.0)], 4.0)


def test_a_thing_that_went_out_of_the_picture_or_was_never_going_did_not_land():
    """Her own throws left the top of the food fight's picture as losses were read; a life's mark goes when she loses
    one, and stays where it was."""
    lands = WhereHarmLands()
    for number, at in ((1, 0.0), (2, 2.0), (3, 4.0)):
        for step in range(4):
            lands.saw([_food(number, 60.0, 30.0 - 9.0 * step, vy=-120.0)], number - 1, at + step / 8, shape=(160, 240))
        lands.saw([], number, at + 0.5, shape=(160, 240))             # gone out of the top, a loss read
    for number, at in ((4, 6.0), (5, 8.0)):
        lands.saw([_food(number, 200.0, 10.0, vy=0.0)], number - 1, at, shape=(160, 240))
        lands.saw([], number, at + 0.1, shape=(160, 240))             # a mark gone with the loss it counted
    assert lands.where() is None


def test_of_two_things_gone_at_a_loss_the_one_gone_where_harm_has_landed_is_the_one():
    lands = WhereHarmLands()
    _thrown_and_landed(lands, 1, 0.0, 0)
    _thrown_and_landed(lands, 2, 2.0, 1)
    for step in range(4):
        lands.saw([_food(3, 120.0, 110.0 + 7.5 * step), _food(4, 30.0, 40.0 - 7.5 * step, vy=-60.0)], 2, 4.0 + step / 8)
    lands.saw([_food(4, 30.0, 10.0, vy=-60.0)], 2, 4.5)               # what hit her went, where harm lands
    lands.saw([], 2, 4.6)                                              # and then the other, going up the table
    lands.saw([], 3, 4.7)                                              # the loss is read after both went
    x, y, _size = lands.landings[-1]
    assert (x, y) == (120.0, 132.5)


def test_where_harm_lands_is_known_again_on_the_next_stretch():
    run = SimpleNamespace(losses=0)
    for number, at in ((1, 0.0), (2, 2.0)):
        for step in range(8):
            comes_at_her_where_harm_lands(run, [_food(number, 120.0, 80.0 + 7.5 * step)], at + step / 8)
        run.losses += 1
        comes_at_her_where_harm_lands(run, [], at + 1.0)
    later = SimpleNamespace(losses=0, harm_lands=run.harm_lands)     # what the stretch kept, its losses from none
    assert comes_at_her_where_harm_lands(later, [_food(3, 120.0, 112.0)], 10.0)
    for step in range(8):
        comes_at_her_where_harm_lands(later, [_food(4, 120.0, 80.0 + 7.5 * step)], 11.0 + step / 8)
    later.losses = 1
    comes_at_her_where_harm_lands(later, [], 12.0)
    assert len(later.harm_lands.landings) == 3


def test_in_the_food_fight_a_duck_dodges_what_is_thrown_at_her():
    """Until 14 October 2026 the food fight's food fell on through the low shape of her ducking: one who ducked each
    throw as it came lost all five lives in 10.5 s, as one who never ducked did."""
    from tools.oracles.a_world_run_from_its_code import FRAME_S, WorldRunFromItsCode, node_is_here

    if not node_is_here():
        pytest.skip("node is not installed")

    def lives_after(ducking: bool, seconds: float = 20.0) -> int:
        with WorldRunFromItsCode("foodfight", seed=1) as world:
            for _frame in range(int(seconds / FRAME_S)):
                near = world.value("__world.game.theirs.some(f => f.y + 7 >= 250 && f.y < 288"
                                   " && Math.abs(f.x - (__world.game.me.x + 18)) < 25)")
                world.hold(*((" ",) if ducking and near else ()))
                state = world.step(FRAME_S)
            return int(state["world"]["lives"])

    assert lives_after(False) == 0 and lives_after(True) >= 3
