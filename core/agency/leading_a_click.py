"""How far ahead of a moving thing a click is aimed: learned from which leads took the things they were aimed at.

A click on a thing that moves lands where the thing will be when what the
click does gets there. Where the click itself is the act (a catch, a tap on a
balloon) that is the moment the click lands; where the click sends something
(a ball tossed, a shot fired toward the pointer) it is that thing's flight,
which no game says. A player finds it by aiming: they lead a little, then
more, and keep the lead that hits. Before this her clicks led every moving
thing by the time a click takes to land: offline 2026-10-10, in a game played
by tossing a friend from hand to hand to moving catchers, she lost every round
within seconds.

So each click aimed at a thing records the lead it was given, and the thing
going (taken, caught, hit) soon after counts for that lead. Leads not yet
tried are tried before any is settled on; after that the lead that has taken
the most of what it was aimed at, for what it has been tried, is used. Nothing
here knows what is clicked or what it sends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LEADS", "LeadingAClick", "the_thing_to_aim_at"]

#: The leads tried, in seconds ahead of a moving thing: a click landing, a quick throw, a slow one, a lob.
LEADS: tuple[float, ...] = (0.08, 0.25, 0.5, 0.9)

#: Tries each lead is given before the one that pays is settled on.
TRIED_EACH = 3

#: How long after the lead's own time a thing going still counts for the click aimed at it.
COUNTS_WITHIN_S = 0.5


@dataclass
class LeadingAClick:
    """The leads tried and what each took."""

    tries: dict[float, int] = field(default_factory=lambda: dict.fromkeys(LEADS, 0))
    took: dict[float, int] = field(default_factory=lambda: dict.fromkeys(LEADS, 0))
    #: Clicks still waiting to be credited: the thing aimed at, when, and the lead it was given.
    waiting: dict[int, tuple[float, float]] = field(default_factory=dict)

    def lead(self) -> float:
        """The lead to give the next click: one not yet tried enough, else the one that has taken the most."""
        untried = [lead for lead in LEADS if self.tries[lead] < TRIED_EACH]
        if untried:
            return min(untried, key=lambda lead: (self.tries[lead], lead))
        return max(LEADS, key=lambda lead: ((self.took[lead] + 0.5) / (self.tries[lead] + 1.0), -lead))

    def aimed(self, number: int, at: float, lead: float) -> None:
        """A click aimed at thing ``number`` at ``at``, ``lead`` seconds ahead of it."""
        self.tries[lead] = self.tries.get(lead, 0) + 1
        self.waiting[number] = (at, lead)

    def saw(self, happened: list[dict[str, Any]], at: float) -> None:
        """What one picture showed: a thing aimed at that went soon enough counts for its lead; one that did not, not."""
        gone = {h.get("thing") for h in happened if h.get("what") == "gone"}
        for number, (when, lead) in list(self.waiting.items()):
            if number in gone and at - when <= lead + COUNTS_WITHIN_S:
                self.took[lead] = self.took.get(lead, 0) + 1
                del self.waiting[number]
            elif at - when > lead + COUNTS_WITHIN_S:
                del self.waiting[number]

    def says(self) -> str:
        """What she has found, for whoever asks."""
        tried = [lead for lead in LEADS if self.tries[lead]]
        if not tried:
            return "no click aimed at a moving thing yet"
        return ", ".join(f"{lead:.2f} s ahead took {self.took[lead]} of {self.tries[lead]}" for lead in tried)


def the_thing_to_aim_at(choosing: Any, aim: Any) -> Any:
    """What a shot sent toward the pointer is for: the thing the choice is lined up on, else the nearest thing worth a
    shot, else the nearest worth getting to; None where nothing is. ``choosing`` is the real-time player's arithmetic of
    one decision (core/agency/playing_as_it_happens.py)."""
    from core.agency.what_meeting_things_does import MEET, SHOOT

    if aim is not None:
        return aim
    mine = choosing.mine
    for wanted in (SHOOT, MEET):
        things = [t for t in choosing.others() if choosing.stance(t) == wanted]
        if things:
            if mine is None:
                return things[0]
            return min(things, key=lambda t: (t.x - mine.x) ** 2 + (t.y - mine.y) ** 2)
    return None
