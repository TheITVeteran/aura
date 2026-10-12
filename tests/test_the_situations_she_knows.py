"""The situations she knows are told by a place's words and set what she supposes, how wide she keeps, what she plays by,
and whether holding on is the task.

Read offline 2026-10-10: the crafts of play reached only what her model reads, so with no model to read them nothing
in her play changed for a game of tossing a friend away from a girl until help came, or of things falling to be caught.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.situations_she_knows import (
    AVOID,
    MEET,
    SHOOT,
    SITUATIONS,
    caution_of,
    lasting_is_the_task,
    prior_stance,
    situations_in,
    ways_of,
)

pytestmark = pytest.mark.unit


def _names(said: str) -> set[str]:
    return {situation.name for situation in situations_in(said)}


@pytest.mark.parametrize("situation", SITUATIONS, ids=[s.name for s in SITUATIONS])
def test_each_situation_is_told_by_its_examples_and_not_by_its_counter_examples(situation):
    for said in situation.examples:
        assert situation.name in _names(said), said
    for said in situation.counter_examples:
        assert situation.name not in _names(said), said


def test_a_game_of_keeping_a_friend_away_until_help_comes_is_a_lasting_one_whose_clock_is_won():
    said = ("Oh, no! The world's brattiest girl wants to adopt Bloo! Keep Bloo safely in the arms of his friends until "
            "Mac arrives. Use the mouse to aim and toss Bloo with one of three throwing options.")
    found = situations_in(said)
    assert {"keeping something away from whoever is after it", "holding on until the end"} <= {s.name for s in found}
    assert lasting_is_the_task(found) and caution_of(found) > 1.0


def _thing(x, y, vx, vy):
    return SimpleNamespace(x=x, y=y, vx=vx, vy=vy)


def test_what_falls_is_caught_or_kept_clear_of_and_what_heads_for_her_is_avoided_or_shot_as_the_situation_says():
    mine = _thing(100.0, 150.0, 0.0, 0.0)
    falling, heading, idle = _thing(110.0, 20.0, 2.0, 60.0), _thing(20.0, 150.0, 50.0, 0.0), _thing(50.0, 50.0, 0.0, 0.0)
    catching = situations_in("Catch the falling eggs in your basket.")
    dodging = situations_in("Dodge the falling rocks and watch out for the guards.")
    shooting = situations_in("Shoot the robots before they reach you.")
    assert prior_stance(catching, falling, mine) == MEET
    assert prior_stance(dodging, falling, mine) == AVOID
    assert prior_stance(dodging, heading, mine) == AVOID
    assert prior_stance(shooting, heading, mine) == SHOOT
    assert prior_stance(catching, idle, mine) == ""
    assert prior_stance((), falling, mine) == ""


def test_the_ways_a_situation_is_played_by_reach_the_ways_she_chooses_from():
    from core.agency.the_way_it_is_played import AS_IT_HAPPENS, FOLLOW, ways_asked

    assert "steer" in ways_of(situations_in("Catch the falling eggs."))
    assert AS_IT_HAPPENS in ways_asked("Catch the falling eggs in your basket.")
    assert FOLLOW in ways_asked("Copy the dance moves in order.")


def test_a_mechanic_its_words_show_reaches_the_ways_she_chooses_from_and_one_its_program_alone_shows_does_not():
    """Before, a mechanic the guide held reached her choice of way only through her model's reading of the place."""
    import time

    from core.agency.the_way_it_is_played import SEND, ways_asked
    from core.cognition.a_guide_to_a_place import PROGRAM, THE_GUIDE, TOLD, Guide

    guide = Guide(place="somewhere")
    token = THE_GUIDE.set(guide)
    try:
        guide._know("sending by strength", ["says “drag back”"], PROGRAM, time.monotonic())
        assert SEND not in ways_asked("A screen with nothing to say.")
        guide._know("sending by strength", ["says “drag back”"], TOLD, time.monotonic())
        assert SEND in ways_asked("A screen with nothing to say.")
    finally:
        THE_GUIDE.reset(token)


def test_steering_something_somewhere_is_played_in_time_and_moving_tiles_is_not():
    """Offline 2026-10-11 "Use the keys to guide Tommy's Lunar Lander safely onto the landing platform" put her in no
    situation, and the lander crashed on each try while she pressed one key every few seconds."""
    from core.agency.situations_she_knows import played_in_time, situations_in

    for said in ("Use the … keys to guide TOMMY'S Lunar Lander safely onto the landing platform!",
                 "Park each car in its space without hitting the others.", "Drive the boat through all the rings."):
        assert played_in_time(situations_in(said)), said
    assert not played_in_time(situations_in("Use your arrow keys to move the tiles."))
