"""What costs her as much getting by her as touching her is shot, not kept clear of.

The shooter world (tests/fixtures/worlds_that_move/shooter.html) says "One that lands costs a life", and its code costs
a life in one test for an invader that lands or touches her: ``if (f.y > world.H - 20 || hits(f, me)) world.lose()``.
Offline 2026-10-14, her choosing handed the world's state at seven decisions a second, each a decision late, ran from
the invaders she was lined up on, and they landed and cost her the life they would have cost on her: over seeds 1 to 6
she lasted 15.7 to 60.1 s of a minute. Told what its code says, she played the minute out on all six, 360 to 420 points.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"


def _read(world: str = "", source: str = ""):
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    if world:
        page = (WORLDS / f"{world}.html").read_text()
        source = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), source])


def _shooter_saying(test: str) -> str:
    page = (WORLDS / "shooter.html").read_text()
    source = re.findall(r"<script>([\s\S]*?)</script>", page)[0]
    said = "if (!f.gone && (f.y > world.H - 20 || hits(f, me))) { f.gone = true; world.lose(); }"
    assert said in source
    return source.replace(said, test)


def test_its_code_says_an_invader_costs_a_life_landing_as_touching_her():
    shooter = _read("shooter")
    assert shooter.thing("foes").past_an_edge == "costs a life" == shooter.thing("foes").touching
    assert _read("paint").thing("bugs").past_an_edge == ""
    assert _read("dodge").thing("rocks").past_an_edge == ""


def test_only_landing_or_touching_either_one_and_at_an_edge_is_that():
    both = _shooter_saying("if (!f.gone && f.y > world.H - 20 && hits(f, me)) { f.gone = true; world.lose(); }")
    assert _read(source=both).thing("foes").past_an_edge == ""
    middle = _shooter_saying("if (!f.gone && (f.y > world.H / 2 || hits(f, me))) { f.gone = true; world.lose(); }")
    assert _read(source=middle).thing("foes").past_an_edge == ""


def test_she_is_told_to_shoot_what_its_code_says_costs_her_wherever_it_gets_by():
    from core.agency.what_its_code_says_she_sees import what_its_code_says_she_sees
    from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers

    guide = SimpleNamespace(world_its_code_makes=_read("shooter"), stage_in_her_picture=None)
    kinds = [SimpleNamespace(number=0, colour=(224, 76, 224), size=8.0)]
    moves = SimpleNamespace(shape=(160, 240), kinds=kinds, things={5: SimpleNamespace(number=5, kind=0, x=60.0, y=60.0,
                                                                                       w=11.0, h=8.0)})
    meeting, said = WhatMeetingDoes(), []
    what_its_code_says_she_sees(None, moves, WhichIsHers(), meeting, 1.0, lambda line, once: said.append(line), guide)
    assert meeting.lands_anyway == {0} and meeting.told[0] == AVOID
    assert "Its code says the foes cost me wherever they get by; I'll shoot them, not dodge them." in said


def test_what_lands_anyway_is_no_danger_to_keep_clear_of():
    from core.agency.choosing_where_to_be import _Choosing
    from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
    from tools.oracles.her_choices_on_a_world import _a_thing, _HerKeys

    hers = _HerKeys({"left": (-240.0, 0.0), "right": (240.0, 0.0)}, 0.5)
    mine = _a_thing(1, 0, (188.0, 290.0, 24.0, 14.0), 0.5)
    foe = _a_thing(2, 3, (189.0, 250.0, 22.0, 16.0), 0.5)
    foe.vy, foe.moved = 30.0, True                                  # coming down on her, 0.6 s off
    moves = SimpleNamespace(shape=(160, 240), things={1: mine, 2: foe}, kinds=[], out_of_sight={})
    meeting = WhatMeetingDoes()
    meeting.told[3] = AVOID

    def staying() -> float:
        return _Choosing(moves, hers, meeting, ["left", "right"], None, next_picture_s=1 / 7).danger((0.0, 0.0))

    assert staying() > 0.0
    meeting.lands_anyway.add(3)
    assert staying() == 0.0


@pytest.mark.slow
def test_she_plays_the_shooter_out_shooting_what_lands_anyway():
    from tools.oracles.a_world_run_from_its_code import node_is_here
    from tools.oracles.her_choices_on_a_world import HerChoicesOnAWorld, reading_of

    if not node_is_here():
        pytest.skip("node is not installed")

    def played(told: bool) -> list[float]:
        out = []
        for seed in range(1, 7):
            reading = reading_of("shooter")
            if not told:
                reading.things = [SimpleNamespace(**{**vars(t), "past_an_edge": ""}) for t in reading.things]
            with HerChoicesOnAWorld("shooter", seed=seed, decide_every=1 / 7, late_by=1, jitter=0.3, fires=True,
                                    reading=reading) as play:
                out.append(play.run(60.0)["seconds"])
        return out

    told, not_told = played(True), played(False)
    assert all(seconds >= 59.0 for seconds in told), told
    assert sum(seconds < 59.0 for seconds in not_told) >= 4, not_told
