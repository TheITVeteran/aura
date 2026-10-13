"""What a place's code says of the things she sees reaches her before play has to teach it: which one is her, what
costs her and what is worth getting, and how far a thing could swerve.

Offline 2026-10-13, her whole live play on the paint world (tools/measure_playing_as_it_happens.py, its canvas drawn by
a browser, at four pictures a second here) painted 14% of the floor and lost every life in eleven seconds; with its code
read as her browser reads a page's scripts, the same seed painted 77% in ninety seconds with a life left. She said why:
its code draws her in #f2f2f2 and its bugs in #2a9a2a, and its bugs cost a life when they touch her.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.what_its_code_says_she_sees import (
    kinds_its_code_draws,
    what_its_code_says_she_sees,
)
from core.agency.what_meeting_things_does import AVOID, MEET, WhatMeetingDoes

pytestmark = pytest.mark.unit

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"


def _read(world: str):
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    page = (WORLDS / f"{world}.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


def _kind(number, colour):
    return SimpleNamespace(number=number, colour=colour, size=8.0)


def test_each_kind_is_the_thing_its_code_draws_in_its_colour():
    paint = _read("paint")
    seen = [_kind(0, (47, 127, 209)), _kind(1, (242, 242, 242)), _kind(2, (44, 150, 40))]
    matched = kinds_its_code_draws(paint, seen, scale=0.5)
    assert matched[1].name == "hers"
    assert matched[2].name == "bugs" and matched[2].touching == "costs a life" and matched[2].wanders == 40.0
    assert matched[0].name == "ground gone over"                 # the paint is ground gone over, no thing to meet


def test_where_one_list_holds_two_looks_each_look_does_what_its_code_ties_to_it():
    catch = _read("catch")
    matched = kinds_its_code_draws(catch, [_kind(3, (242, 194, 48)), _kind(4, (192, 38, 45))])
    assert (matched[3].name, matched[3].touching) == ("falling", "scores")
    assert (matched[4].name, matched[4].touching) == ("falling", "costs a life")


def test_a_colour_two_of_its_things_share_tells_neither_apart():
    paddle = _read("paddle")
    assert paddle.hers.drawn_in == paddle.thing("them").looks[0][0]
    assert kinds_its_code_draws(paddle, [_kind(1, (232, 232, 232))]) == {}


def test_she_keeps_clear_of_what_its_code_says_costs_her_before_it_touches_her_and_takes_its_body_for_hers():
    from core.agency.which_one_answers_to_her import WhichIsHers

    guide = SimpleNamespace(world_its_code_makes=_read("paint"), stage_in_her_picture=None)
    me = SimpleNamespace(number=7, kind=1, x=120.0, y=90.0)
    bug = SimpleNamespace(number=8, kind=2, x=30.0, y=40.0)
    moves = SimpleNamespace(shape=(160, 240), kinds=[_kind(1, (242, 242, 242)), _kind(2, (42, 154, 42))],
                            things={7: me, 8: bug})
    meeting, hers, said = WhatMeetingDoes(), WhichIsHers(), []
    what_its_code_says_she_sees(None, moves, hers, meeting, 1.0, lambda line, once: said.append(line), guide)
    assert meeting.told[2] == AVOID and hers.number == 7
    assert said[0] == "Its code says the bugs cost me."
    # What meeting a kind has shown outweighs what its code was read to say.
    assert meeting.evidence[2] is not None                       # met now: there is evidence of what it does
    meeting.told[2] = MEET
    what_its_code_says_she_sees(None, moves, hers, meeting, 2.0, None, guide)
    assert meeting.told[2] == MEET


def test_with_no_reading_of_its_code_nothing_is_said_of_what_she_sees():
    moves = SimpleNamespace(shape=(160, 240), kinds=[_kind(1, (242, 242, 242))], things={})
    meeting = WhatMeetingDoes()
    assert what_its_code_says_she_sees(None, moves, None, meeting, 0.0, None,
                                       SimpleNamespace(world_its_code_makes=None)) == {}
    assert meeting.told == {}


def test_a_key_her_play_took_for_a_lift_is_a_way_to_go_where_its_code_paces_her_by_it():
    """Her presses can take a key that walks her up for one that lifts her; where the code sets her pace by that key and
    lifts her by none, it is a way to go. Followed by the pointer instead, her keys' paces are not hers to go by."""
    from core.agency.choosing_where_to_be import _Choosing
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from tools.oracles.her_choices_on_a_world import SHAPE, _a_thing, _HerKeys

    paint = _read("paint")
    scale = SHAPE[1] / paint.stage[0]
    keys = ["left", "right", "up", "down"]
    mine = _a_thing(1, 0, (120, 100, 16, 16), scale)
    moves = SimpleNamespace(shape=SHAPE, things={1: mine}, kinds=[_kind(0, (242, 242, 242))], out_of_sight={})
    presses = SimpleNamespace(lifts=lambda kind, size, number: ["up"], path=lambda key, kind, number: None)
    guide = Guide(place="paint")
    guide.world_its_code_makes = paint
    token = THE_GUIDE.set(guide)
    try:
        hers = _HerKeys({"left": (-150.0, 0.0), "right": (150.0, 0.0)}, scale)
        choosing = _Choosing(moves, hers, WhatMeetingDoes(), keys, None, presses=presses)
        assert "up" not in choosing.lifts
        assert choosing.ways["up"] == (0.0, -150.0 * scale) and choosing.ways["down"] == (0.0, 150.0 * scale)
        hers.follows_pointer, hers.follows_along = True, (True, True)
        choosing = _Choosing(moves, hers, WhatMeetingDoes(), keys, None, presses=presses)
        assert choosing.lifts == ["up"] and "down" not in choosing.ways
    finally:
        THE_GUIDE.reset(token)
