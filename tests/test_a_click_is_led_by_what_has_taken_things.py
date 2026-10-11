"""A click at a moving thing is led by how far ahead has taken things; a shot sent toward the pointer is aimed at what it is for.

Before, every click led a moving thing by the time a click takes to land, whatever the click sent; and a shot fired by a
click from a body moved by keys went wherever the pointer had been left.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.leading_a_click import LEADS, TRIED_EACH, LeadingAClick, the_thing_to_aim_at
from core.agency.what_meeting_things_does import AVOID, MEET, SHOOT

pytestmark = pytest.mark.unit


def test_each_lead_is_tried_and_the_one_that_takes_things_is_kept():
    leading = LeadingAClick()
    at, number = 0.0, 0
    for _ in range(TRIED_EACH * len(LEADS)):
        lead = leading.lead()
        number += 1
        leading.aimed(number, at, lead)
        # Only a lob's lead takes what it is aimed at: the thing goes when the lob comes down.
        happened = [{"what": "gone", "thing": number}] if lead == 0.9 else []
        leading.saw(happened, at + lead)
        leading.saw([], at + 5.0)
        at += 10.0
    assert all(leading.tries[lead] == TRIED_EACH for lead in LEADS)
    assert leading.lead() == 0.9
    assert "0.90 s ahead took 3 of 3" in leading.says()


def test_a_thing_gone_long_after_its_lead_is_not_credited_to_it():
    leading = LeadingAClick()
    leading.aimed(7, 0.0, 0.08)
    leading.saw([{"what": "gone", "thing": 7}], 3.0)
    assert leading.took[0.08] == 0 and not leading.waiting


def _thing(number, x, y, stance):
    return SimpleNamespace(number=number, x=x, y=y, vx=0.0, vy=0.0, kind=stance)


def test_a_shot_is_aimed_at_what_it_lines_up_on_else_the_nearest_worth_a_shot_else_worth_getting():
    mine = _thing(0, 100.0, 100.0, MEET)
    near_enemy, far_enemy, friend, trap = _thing(1, 120, 100, SHOOT), _thing(2, 300, 100, SHOOT), _thing(3, 110, 110, MEET), \
        _thing(4, 101, 100, AVOID)
    choosing = SimpleNamespace(mine=mine, others=lambda: [far_enemy, near_enemy, friend, trap], stance=lambda t: t.kind)
    assert the_thing_to_aim_at(choosing, far_enemy) is far_enemy
    assert the_thing_to_aim_at(choosing, None) is near_enemy
    choosing.others = lambda: [friend, trap]
    assert the_thing_to_aim_at(choosing, None) is friend
    choosing.others = lambda: [trap]
    assert the_thing_to_aim_at(choosing, None) is None
