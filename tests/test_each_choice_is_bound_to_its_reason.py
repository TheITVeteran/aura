"""Each choice of her fast play is bound to the act that carries it out, its reason, the craft of people's play it is a
case of, what she knows that it rests on, and what should follow by when; settled against what did; and kept.

Bryan, 10 October 2026: research people playing and apply their actual decision-making, physics and reasoning as
architecture, binding choices to actions and both to reasoning, strategy and knowledge.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from core.agency.a_choice_and_its_reasons import (
    AS_EXPECTED,
    CHOICES,
    NOT_AS_EXPECTED,
    ChoicesMade,
    the_arrival_to_meet,
    the_choice_now,
)

pytestmark = pytest.mark.unit


def _thing(number, x, y, *, kind=1, w=10.0, h=10.0):
    return SimpleNamespace(number=number, kind=kind, x=x, y=y, w=w, h=h, vx=0.0, vy=0.0)


class _Choosing:
    """Her choosing as the record reads it: her thing, how fast she goes, how soon she answers."""

    def __init__(self, speed=200.0, met_kinds=()):
        self.mine = _thing(0, 100.0, 400.0, kind=0, w=40.0)
        self.ways = {"left": (-speed, 0.0), "right": (speed, 0.0)} if speed else {}
        self.across, self.updown = True, False
        self.response_s, self.next_picture_s = 0.05, 0.03
        self.meeting = SimpleNamespace(known=lambda kind: kind in met_kinds)
        self.going_to = None
        self.arrival = None

    def speed(self, axis):
        return max([abs(way[axis]) for way in self.ways.values()] + [1.0])

    def line(self):
        return 1


def test_a_catcher_lets_go_of_what_cannot_be_reached_for_what_can():
    choosing = _Choosing(speed=200.0)
    far_and_soon = (0.3, 600.0, _thing(1, 600.0, 100.0))       # 500 px away in 0.3 s: out of reach
    near_and_later = (0.8, 160.0, _thing(2, 160.0, 50.0))      # 60 px away in 0.8 s: in reach
    when, there, thing = the_arrival_to_meet(choosing, [far_and_soon, near_and_later], 0, 100.0)
    assert thing.number == 2 and choosing.arrival[2] < 0.8
    # Where she can reach none, the soonest; where her speed is not known yet, every one counts as reachable.
    assert the_arrival_to_meet(choosing, [far_and_soon], 0, 100.0)[2].number == 1
    unknown = _Choosing(speed=0.0)
    assert the_arrival_to_meet(unknown, [far_and_soon, near_and_later], 0, 100.0)[2].number == 1


def test_a_choice_carries_its_craft_reason_cue_and_what_it_counts_on_and_is_said_once_a_craft():
    choosing = _Choosing(met_kinds={1})
    aim = _thing(2, 160.0, 50.0)
    choosing.arrival = (0.8, 160.0, 0.35)
    physics = SimpleNamespace(gravity=lambda kind: (0.0, 900.0))
    choice = the_choice_now(choosing, "right", "meet", aim, 10.0, physics=physics, named=lambda t: "the egg")
    assert choice.craft == "getting under what to catch"
    said = choice.said()
    assert said.startswith("Holding right to be where the egg gets to when it does: it gets to me in 0.8 s")
    assert "I go about 200 px a second across, measured" in said
    assert "pulled at 900 px/s²" in said and "meeting one has counted for me before" in said
    made, heard = ChoicesMade(), []
    assert made.chose(choice, heard.append) and len(heard) == 1
    again = the_choice_now(choosing, "", "meet", aim, 10.1, physics=physics, named=lambda t: "the egg")
    assert not made.chose(again, heard.append) and len(heard) == 1    # the same aim for the same reason is held


def test_a_choice_is_settled_by_what_followed_and_two_misses_are_said():
    choosing = _Choosing()
    aim = _thing(2, 160.0, 50.0)
    choosing.arrival = (0.5, 160.0, 0.3)
    made, heard = ChoicesMade(), []
    made.chose(the_choice_now(choosing, "right", "meet", aim, 0.0, named=lambda t: "the egg"))
    assert made.settle(0.2, met={2: 0.2}, gains=1, losses=0) == AS_EXPECTED
    for start, other in ((1.0, 3), (3.0, 4)):
        made.chose(the_choice_now(choosing, "left", "meet", _thing(other, 60.0, 50.0), start, named=lambda t: "the egg"))
        assert made.settle(start + 5.0, met={}, gains=1, losses=0, say=heard.append) == NOT_AS_EXPECTED
    tally = made.came_out["getting under what to catch"]
    assert tally == Counter({AS_EXPECTED: 1, NOT_AS_EXPECTED: 2})
    assert heard == [] or heard[-1].startswith("Twice now I counted on meeting")
    waiting = the_choice_now(choosing, "", "wait", None, 20.0)
    made.chose(waiting)
    assert made.settle(20.1, met={}, gains=1, losses=1) == NOT_AS_EXPECTED   # something cost her while she waited
    assert made.how_it_came_out()[0].startswith("getting under what to catch: 1 of 3")


def test_every_choice_is_a_case_of_a_craft_people_play_by():
    from core.agency.the_craft_of_play import CRAFTS

    names = {craft.name for craft in CRAFTS}
    assert {kind.craft for kind in CHOICES.values()} <= names


def test_how_her_choices_came_out_is_kept_with_the_game():
    from core.agency.what_she_keeps_of_a_game import kept_from, to_keep

    made = ChoicesMade(came_out={"getting under what to catch": Counter({AS_EXPECTED: 7, NOT_AS_EXPECTED: 2})})
    held = to_keep({"choices": made})
    assert held["choices"] == {"getting under what to catch": {AS_EXPECTED: 7, NOT_AS_EXPECTED: 2}}
    again = kept_from({**held, "kinds": [{"colour": [1, 2, 3], "size": 4.0}]})
    assert again["choices"].came_out["getting under what to catch"][AS_EXPECTED] == 7


def test_a_places_instructions_plan_locks_and_board_are_kept_with_its_guide():
    from core.cognition.a_guide_to_a_place import Guide
    from core.cognition.a_plan_to_an_end import Means, Plan
    from core.cognition.locks_she_met import Lock
    from core.cognition.reading_the_rules import Frame

    guide = Guide(place="a place of devices")
    guide.rules.heard.append("Drag a device to any active square.")
    plan = Plan(end="the cage drops", start="the mouse trap")
    plan.means = [Means("a roller", does="rolls the way its arrow points")]
    step = Frame("Put a roller at the end of the arrow", act="carry", thing="a roller", where="the end of the arrow")
    plan.took([step])
    plan.expects[step.key] = "the roller sits at the arrow's end"
    guide.plan = plan
    guide.locks.met.append(Lock("You need the key", "DOOR", "the key"))
    guide.board = {"family": "draughts", "variation": {"special_swaps": True}, "side": "b"}
    again = Guide.from_memory(guide.as_memory(), "a place of devices")
    assert again.rules.heard[:1] == ["Drag a device to any active square."]
    assert again.plan.end == "the cage drops" and [m.name for m in again.plan.means] == ["a roller"]
    assert [f.sentence for f in again.plan.steps()] == ["Put a roller at the end of the arrow"]
    assert again.plan.expects == {step.key: "the roller sits at the arrow's end"}
    assert again.locks.met[0].wants == "the key" and again.board["variation"] == {"special_swaps": True}
