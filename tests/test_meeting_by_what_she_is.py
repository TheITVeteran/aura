"""What meeting a kind does, by what she is when she meets it (core/agency/meeting_by_what_she_is.py): a crystal the
colour of her tower's top is met and one of another colour kept clear of; a ghost met after the thing that turns the
tables pays and before it costs; a kind no term splits is left to its one stance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.meeting_by_what_she_is import BAD, GOOD, MeetingByWhatSheIs, what_she_is

pytestmark = pytest.mark.unit

RED, BLUE, GREY = (232, 74, 74), (61, 143, 240), (154, 154, 168)


def test_a_thing_the_colour_of_what_goes_with_her_is_met_and_one_of_another_colour_kept_clear_of():
    meeting = MeetingByWhatSheIs()
    red_kind = 4
    for _ in range(6):
        # Her tower topped with red: a red crystal adds to it. Topped with blue: a red one blows it apart.
        meeting.settled(red_kind, what_she_is(RED, [GREY, RED], ()), +1.0)
        meeting.settled(red_kind, what_she_is(RED, [GREY, BLUE], ()), -1.0)
    assert meeting.stance(red_kind, what_she_is(RED, [GREY, RED], ())) == GOOD
    assert meeting.stance(red_kind, what_she_is(RED, [GREY, BLUE], ())) == BAD


def test_a_kind_met_after_another_pays_and_before_it_costs():
    meeting = MeetingByWhatSheIs()
    ghost, pellet = 7, 9
    for at in range(0, 60, 10):
        meeting.settled(ghost, what_she_is(BLUE, [GREY], ()), -1.0)
        meeting.met(pellet, float(at))
        meeting.settled(ghost, what_she_is(BLUE, [GREY], meeting.met_lately(at + 2.0, besides=ghost)), +1.0)
    assert meeting.stance(ghost, what_she_is(BLUE, [GREY], {pellet})) == GOOD
    assert meeting.stance(ghost, what_she_is(BLUE, [GREY], ())) == BAD


def test_a_kind_no_term_splits_is_left_to_its_one_stance():
    meeting = MeetingByWhatSheIs()
    for at in range(8):
        meeting.settled(3, what_she_is(RED, [RED] if at % 2 else [BLUE], ()), +1.0 if at % 3 else -1.0)
    assert meeting.stance(3, what_she_is(RED, [RED], ())) is None


def test_meeting_through_her_play_learns_the_split():
    """Through what_meeting_things_does: touches noted with what she was, settled by what followed."""
    from core.agency.what_meeting_things_does import AVOID, MEET, WhatMeetingDoes

    meeting = WhatMeetingDoes()
    red = SimpleNamespace(number=1, kind=4, x=0.0, y=0.0, vx=0.0, vy=0.0, colour=RED, size=100.0)
    meeting._looks = {4: (RED, 100.0), 1: (GREY, 300.0)}
    at = 0.0
    for top in [RED, BLUE] * 5:
        at += 3.0
        meeting._now = at
        meeting._her_colours = [GREY, top]
        meeting._note("touched", red, at)
        meeting.verdicts.append({"what": "gain" if top == RED else "loss", "at": at + 0.2, "from": at - 0.1,
                                 "since": at})
        meeting._settle(at + 5.0)
    meeting._her_colours = [GREY, RED]
    assert meeting.stance(4) == MEET
    meeting._her_colours = [GREY, BLUE]
    assert meeting.stance(4) == AVOID
