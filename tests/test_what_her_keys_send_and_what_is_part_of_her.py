"""What a place's code says her keys send is hers to aim from the first press, and a part of her drawn in her colour is
part of her.

Offline 2026-10-14 on the shooter world (tests/fixtures/worlds_that_move/shooter.html), her eyes saw the ship and its
gun as two kinds of one green. With two kinds drawn as her she was taken for neither, and lined up her shots on her own
gun. Her shots were two pixels wide and went 26 pixels a picture; until she saw one, 10.7 s into play, she fired once
in 1.5 s, and 14 times in the minute.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.what_its_code_says_she_sees import what_its_code_says_she_sees
from core.agency.what_meeting_things_does import IGNORE, WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"


def _read(world: str):
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    page = (WORLDS / f"{world}.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


def _kind(number, colour):
    return SimpleNamespace(number=number, colour=colour, size=8.0)


def _thing(number, kind, x, y, w, h):
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, w=w, h=h)


def _shooter_seen(gun_at=(118.0, 144.0)):
    """The shooter as her eyes saw it: her ship (kind 1) and its gun (kind 2) in one green, a foe, the white of its
    shots and its writing."""
    guide = SimpleNamespace(world_its_code_makes=_read("shooter"), stage_in_her_picture=None)
    kinds = [_kind(0, (224, 76, 224)), _kind(1, (124, 255, 107)), _kind(2, (124, 255, 107)), _kind(6, (255, 255, 255))]
    things = {10: _thing(10, 1, 118.0, 148.0, 12.0, 7.0), 11: _thing(11, 2, *gun_at, 2.0, 3.0),
              12: _thing(12, 0, 60.0, 63.0, 11.0, 8.0)}
    return guide, SimpleNamespace(shape=(160, 240), kinds=kinds, things=things)


def test_the_keys_its_code_says_send_a_thing_are_read_with_the_thing_they_send():
    assert _read("shooter").sends == {"space": "shots"}
    assert _read("doors").sends == {"space": "stars"}
    assert _read("foodfight").sends == {} and _read("paint").sends == {}


def test_a_ship_and_its_gun_drawn_alike_are_her_body_and_a_part_of_her():
    guide, moves = _shooter_seen()
    meeting, hers, said = WhatMeetingDoes(), WhichIsHers(), []
    what_its_code_says_she_sees(None, moves, hers, meeting, 1.0, lambda line, once: said.append(line), guide)
    assert hers.number == 10
    assert meeting.told[2] == IGNORE                                # her gun: nothing to line up on
    assert "Its code says which one is me; I'll check it answers my keys." in said


def test_two_drawn_in_her_colour_apart_are_left_to_play():
    guide, moves = _shooter_seen(gun_at=(30.0, 40.0))
    hers = WhichIsHers()
    what_its_code_says_she_sees(None, moves, hers, WhatMeetingDoes(), 1.0, None, guide)
    assert hers.number is None


def test_what_space_sends_is_hers_to_aim_from_the_first_picture_until_she_has_seen_it():
    guide, moves = _shooter_seen()
    hers, said = WhichIsHers(), []
    what_its_code_says_she_sees(None, moves, hers, WhatMeetingDoes(), 1.0, lambda line, once: said.append(line), guide)
    shot = hers.makes["space"]
    # Its shots rise 360 pixels a second on a stage of 480, in her picture of 240: 180 a second, up.
    assert (shot.key, shot.kind, shot.vx, shot.vy) == ("space", 6, 0.0, -180.0)
    assert "Its code says space fires its shots up." in said
    # What her presses show replaces it; and where they show it makes nothing, it is not given again.
    del hers.makes["space"]
    what_its_code_says_she_sees(None, moves, hers, WhatMeetingDoes(), 2.0, None, guide)
    assert "space" not in hers.makes


def test_a_shot_its_code_sends_sideways_or_nowhere_is_not_guessed():
    world = _read("shooter")
    world.things = [thing if thing.name != "shots" else SimpleNamespace(**{**vars(thing), "moves": "crosses"})
                    for thing in world.things]
    guide = SimpleNamespace(world_its_code_makes=world, stage_in_her_picture=None)
    _guide, moves = _shooter_seen()
    hers = WhichIsHers()
    what_its_code_says_she_sees(None, moves, hers, WhatMeetingDoes(), 1.0, None, guide)
    assert "space" not in hers.makes
