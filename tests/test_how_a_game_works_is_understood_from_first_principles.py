"""How a game works is understood once, axis by axis, from what it says and what play shows, for every part of her.

Offline 2026-10-11 a food fight said "take aim with your mouse and click to throw; dodge food by pressing the space
bar", and one part of her pressed each arrow to see what it did, another took the opponent for her body, and another
waited for the picture to come to rest: each worked out from the words afresh how the place is played, and they did
not agree. The places here are invented; nothing is any one game's.
"""
from __future__ import annotations

import pytest

from core.agency import how_a_game_works as how
from core.cognition.a_guide_to_a_place import SCREEN, Guide

pytestmark = pytest.mark.unit


def _understood(said: str) -> how.HowItWorks:
    guide = Guide(place="an invented place")
    guide.take_in(SCREEN, said)
    return how.understood(guide)


@pytest.mark.parametrize(("said", "me", "tempo", "counts"), [
    ("Take aim with your mouse and click to throw snowballs. Press SPACE to duck the ones coming at you.",
     how.AIMS_FROM_HERE, how.RUNS, ""),
    ("Use the arrow keys to move and SPACE to jump. Collect all the coins and reach the flag.", how.STEERED, "", "win"),
    ("Use the mouse to move the basket and catch the falling apples. Score as many points as you can in 60 seconds.",
     how.FOLLOWS, how.RUNS, "score"),
    ("WASD to move, mouse to aim and shoot the robots.", how.AIMS_FROM_A_BODY, how.RUNS, ""),
    ("Click and hold on the ball, drag back to set the power and let go to putt it into the hole.", how.SENDS, how.SHOTS, ""),
    ("Click to jump over the rocks as you run.", how.TIMES, "", ""),
    ("Drag the parts onto the machine, then press TEST to see if it works.", how.PLACES, how.WAITS, ""),
    ("Create your own monster by choosing its parts, then click MAKE.", how.MAKES, how.WAITS, "make"),
    ("Click on the doors to open them and find the hidden key.", how.POINTS, "", "win"),
    ("Use the mouse to move the boy. Click to throw the ball and pop the balloons before they fly away.", how.FOLLOWS,
     how.RUNS, ""),
])
def test_what_she_is_whether_it_waits_and_what_counts_are_read_from_what_it_says(said, me, tempo, counts):
    understood = _understood(said)
    assert (understood.me.value, understood.tempo.value, understood.counts.value) == (me, tempo, counts)


def test_each_control_is_given_its_own_act_and_said_together_where_they_do_one_thing():
    understood = _understood("Use the arrow keys to move and SPACE to jump.")
    assert understood.inputs == {"the arrows": "move", "space": "jump"}
    turning = Guide(place="a lander")
    turning.take_in(SCREEN, "Press the up arrow to thrust, left and right to rotate.")
    assert {key: control.act for key, control in turning.controls.items()} == {"up": "thrust", "left": "rotate",
                                                                               "right": "rotate"}


def test_what_follows_from_it_and_what_is_said():
    thrower = _understood("Take aim with your mouse and click to throw. Dodge food by pressing the space bar. "
                          "The first to splat the other five times wins!")
    assert thrower.has_a_body() is False and not thrower.arrows_are_a_guess_worth_trying()
    assert thrower.goes_on_without_her()
    assert thrower.said().startswith("As I understand it: I'm someone aiming from this side of the picture")
    walker = _understood("Use the arrow keys to move.")
    assert walker.has_a_body() and walker.arrows_are_a_guess_worth_trying()
    assert "whether it waits for me" in walker.unknown()


def test_what_play_shows_outweighs_what_was_said_and_is_kept_for_every_part_of_her():
    guide = Guide(place="a place that said little")
    guide.take_in(SCREEN, "Press space to jump.")
    assert how.understood(guide).me.sure < how.SAID_SURE
    how.seen_to_answer(guide, "the pointer")
    how.seen_going_on_by_itself(guide, True)
    understood = how.understood(guide)
    assert (understood.me.value, understood.tempo.value) == (how.FOLLOWS, how.RUNS)
    assert understood.me.sure == how.SEEN_SURE and understood.has_a_body()


def test_a_place_understood_to_be_worked_otherwise_has_no_arrows_guessed_nor_its_programs_keys_played():
    """Offline 2026-10-11 a game whose words said the mouse moves its hero had each arrow pressed and watched while its
    eggs fell; its program listened for up and down, and it was played with them."""
    from core.agency.what_i_can_do_here import THE_PICTURE, WhatWorksHere, a_click_on
    from core.cognition.a_guide_to_a_place import PROGRAM, THE_GUIDE, Control
    from core.skills.screen_pursuit_as_it_happens import _keys_of_play

    guide = Guide(place="eggs")
    guide.take_in(SCREEN, "Use the mouse to move the boy. Click to throw the ball and crack the eggs before they fall. "
                          "Press the space bar to jump.")
    for key in ("up", "down"):
        guide.controls[key] = Control(key, "", PROGRAM, 0.0)
    token = THE_GUIDE.set(guide)
    try:
        here = WhatWorksHere(told=("up", "down", "left", "right"))
        here.asked_for_by("")
        assert set(here.available()) == {"space", a_click_on(THE_PICTURE)}
        assert _keys_of_play(guide, "") == ["space"]
    finally:
        THE_GUIDE.reset(token)


def test_a_clock_said_as_what_loses_is_a_place_that_goes_on_without_her():
    """Offline 2026-10-11 a corridor of doors said only that "Out of Time" loses, and was taken for a place that waits."""
    understood = _understood("Click on the doors. Out of Time! You have failed.")
    assert understood.me.value == how.POINTS and understood.goes_on_without_her()
