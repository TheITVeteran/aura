"""What she plays for: the end that leads, and what every other pull on her is worth in its terms.

A place pulls a player several ways at once: a count to raise, herself to keep, a threat to take out, a thing not yet
known, a way further on, someone to beat, what somebody asked. Her play weighs them at every picture: how wide a berth
to give what costs her, whether to take a threat out or go round it, how far out of her way to go to find out what a
thing is. Weighed with fixed numbers, she plays her last life like her first, and a run with lives to spare and ten
seconds on the clock as warily as one with all the time there is. A player does not. The stakes change with what is
left, and so does a player's care.

So one end leads, and every other pull is priced in its terms. The end is the person's ask where they made one; else
what the game's words say to do; else what the screen counts. Four ends cover what a place can be played for:

- a target: a count to reach ("collect 60 stars", "2/40"). What counts is the chance of reaching it.
- the most: a count to raise with no end named. What counts is the count expected by the end.
- to last: nothing to raise, herself to keep, and a clock. What counts is the chance of lasting to the end.
- to find out: nothing counted, or the ask was to look round. What counts is the kinds of thing found out.

What she can lose is counted in blows: a hit off a bar of health, or, where there is no bar, a life. She has the hits
left in her bar and a full bar's worth in each life to come. Her record of the run gives the rest: how fast she gains
while she is in play (one gain in ten seconds believed before any is seen) and how long a blow is in coming (thirty
seconds believed before any comes). Blows are taken to come at random at that rate, as arrivals do (a Poisson process):
the chance of taking no more than k of them in t seconds is the sum of the first k+1 terms of a Poisson series with
mean t over the seconds between blows. From that:

- **Self-preservation.** A blow is worth the play it would cut short. Toward a target, it is what taking it takes off
  the chance of getting there, against what one gain more adds to it: where getting there is all but sure, a gain adds
  little and a blow is worth many gains, so she plays safe; where it hangs in the balance, both count. With a count to
  raise against a clock, it is the gains she would have made in the time it takes away: the fewer blows she has left,
  the more each is worth, and with the clock nearly out and lives in hand, almost nothing. Being hit often does not make
  a blow cheap, since how often she is hit depends on her care. Her care is that worth in seconds of play against the
  thirty a life is believed to last, which is what the berth her play gives was set for; her play multiplies the
  berth by it.
- **Threat elimination.** A threat left alone deals her blows while it is about. How many a second is measured for
  each kind: the blows things of that kind dealt over the seconds they were about her, counted from one blow in the
  believed time between blows, or half that for a thing that comes for her. How long one stays is how long the ones
  before stayed. Taking it out is worth the blows it would deal in that time by what a blow is worth, and a gain where
  shooting its kind has paid one; it is done where that comes to more than the time taking it out takes, or than what
  the best thing to go for pays in that time.
- **Finding out.** A thing not yet known is worth the value of knowing it (value of information): the chance it is
  worth getting, by what getting it and the others like it on the screen is worth, less the chance it is a threat, by
  what a blow is worth in gains. The chances are what the kinds met in this run turned out to be, counted from one of
  each (a uniform Dirichlet prior). Until two kinds have been met there is nothing to count from, and the player's own
  measure stands.
- **When it stops paying.** A lead is spent when its count has not moved for much longer than it took to move before,
  while she plays: the ways she knows have stopped paying, and finding out what she does not know is all that is left
  to try. She says so.

Nothing here knows a game, a counter or a key. The counters' meanings are the readouts' own
(core/agency/what_meeting_things_does.py); the clock is the pressures' (core/agency/what_the_place_does_to_her.py);
what comes for her is core/agency/what_comes_for_her.py's.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import deque
from dataclasses import dataclass, field

from core.agency.what_meeting_things_does import what_a_counter_means
from core.language.how_a_game_ended import asks_to_win

__all__ = ["FIND_OUT", "LAST", "MOST", "TARGET", "WhatSheGoesFor", "chance_of_reaching", "count_expected", "target_in"]

TARGET, MOST, LAST, FIND_OUT = "a target", "the most", "to last", "to find out"

#: What is believed before the run says: one gain in ten seconds, a blow in thirty; each believed as if seen once.
GAIN_EVERY_S = 10.0
BLOW_EVERY_S = 30.0

#: How far her care may move from where it starts, either way: past this the berth swamps every other cost, or none.
CARE_BOUNDS = (0.25, 4.0)

#: How far out of her way a thing not yet known may take her, as a multiple of the time to it: at the near end, as far
#: as a thing known to be worth getting; at the far end, it waits until little else is left.
DETOUR_BOUNDS = (1.0, 6.0)

#: How many kinds met in this run the chances of what an unknown kind is are counted from, at the least.
KINDS_TO_COUNT_FROM = 2

#: How many times longer than the usual gap between gains, and at least how long, the count must be still for a lead
#: to be spent.
SPENT_GAPS = 4.0
SPENT_AT_LEAST_S = 15.0

#: The least fall in a bar, as a share of it, that is a blow and not the bar being read a little differently.
LEAST_BLOW = 0.02

#: Words that ask her to find out what is there rather than to score.
_ASKS_TO_LOOK = re.compile(r"\b(?:explore|look (?:a)?round|see what|find out|what(?:'s| is) (?:in|there)|discover)\b", re.I)

#: A count the words name as the one to reach: "collect 60", "score 100 points", "get 10 stars to win".
_A_TARGET = re.compile(r"\b(?:collect|get|score|reach|catch|earn|gather|make|find|win with)\s+(?:at least\s+|over\s+)?"
                       r"(\d{1,6})\b", re.I)


def target_in(words: str) -> float | None:
    """The count the words name as the one to reach, where they name one."""
    found = _A_TARGET.search(words or "")
    return float(found.group(1)) if found else None


def _poisson_at_most(k: int, x: float) -> float:
    """The chance a Poisson count with mean ``x`` is at most ``k``."""
    if k < 0:
        return 0.0
    if x <= 0:
        return 1.0
    term = total = math.exp(-x)
    for i in range(1, k + 1):
        term *= x / i
        total += term
    return min(1.0, total)


def chance_of_reaching(needed: float, blows: int, rate: float, blow_s: float, seconds_left: float | None) -> float:
    """The chance of making ``needed`` more gains, at ``rate`` gains a second while in play, before she has taken the
    ``blows`` she can take, one coming every ``blow_s`` seconds on average, and before the clock runs out."""
    if needed <= 0:
        return 1.0
    if blows <= 0 or rate <= 0:
        return 0.0
    in_play_for = needed / rate
    if seconds_left is not None and in_play_for > seconds_left:
        return 0.0
    return _poisson_at_most(blows - 1, in_play_for / blow_s)


def count_expected(blows: int, rate: float, blow_s: float, seconds_left: float | None) -> float:
    """The gains to come, at ``rate`` while in play, until the ``blows`` she can take are taken or the clock runs out.
    She is still in play at a moment where fewer blows than she can take have come by then; over the time left that is
    the sum, for each k below ``blows``, of the time by which more than k have not come."""
    if blows <= 0:
        return 0.0
    if seconds_left is None:
        return rate * blows * blow_s
    x = seconds_left / blow_s
    return rate * blow_s * sum(1.0 - _poisson_at_most(k, x) for k in range(blows))


@dataclass
class WhatSheGoesFor:
    """One run's end, her record toward it, and what each pull is worth."""

    asked: str = ""
    #: The count to reach, where the words or the screen name one.
    target: float | None = None
    began: float | None = None
    #: The counter she plays to raise and the one she keeps, by the readouts' labels; her bar, as a share of full.
    count_label: str | None = None
    lives_label: str | None = None
    count: float | None = None
    lives: float | None = None
    vitals: float | None = None
    #: Each rise of the count (when, by how much); each blow (when), and how much of her bar each hit took.
    gains: deque = field(default_factory=lambda: deque(maxlen=200))
    blows: list[float] = field(default_factory=list)
    hits: list[float] = field(default_factory=list)
    #: What each kind met has turned out to be: worth getting (``"gain"``), a threat (``"loss"``), or nothing.
    kinds: dict[int, str] = field(default_factory=dict)
    #: Kinds that have cost her, and kinds whose taking out has paid a gain.
    cost_her: set[int] = field(default_factory=set)
    pays_to_shoot: set[int] = field(default_factory=set)
    #: For threat elimination: the blows each kind dealt her, the seconds things of each kind were about her, when
    #: each threat on the screen came, how long threats stayed, and the time of the last look at them.
    blows_by: dict[int, int] = field(default_factory=dict)
    exposure: dict[int, float] = field(default_factory=dict)
    appeared: dict[int, float] = field(default_factory=dict)
    stays: deque = field(default_factory=lambda: deque(maxlen=50))
    threats_at: float | None = None
    count_moved_at: float | None = None
    #: Earlier runs of the same game: the seconds played, the gains and the blows in them.
    played_before: float = 0.0
    gains_before: int = 0
    blows_before: int = 0
    said_spent: bool = False

    # -- what she is told ------------------------------------------------------

    def begin_run(self, at: float | None = None) -> None:
        """A run of the same game begins: its own counts begin afresh, and what the runs before showed is kept: how fast
        she gains, how often blows come, what each kind is and how dangerous, how long threats stay."""
        if self.began is not None and self.threats_at is not None:
            self.played_before += max(0.0, self.threats_at - self.began)
        elif self.began is not None and self.count_moved_at is not None:
            self.played_before += max(0.0, self.count_moved_at - self.began)
        self.gains_before += len(self.gains)
        self.blows_before += len(self.blows)
        self.began, self.count, self.lives, self.vitals, self.count_moved_at = at, None, None, None, None
        self.gains.clear()
        self.blows = []
        self.appeared.clear()
        self.threats_at, self.said_spent = None, False

    def told(self, *, asked: str = "", words: str = "") -> None:
        """The person's ask and the game's words: whether the ask is to look round, and the count to reach."""
        self.asked = asked or self.asked
        self.target = target_in(words) or self.target

    def lead(self) -> str:
        """The end that leads: the person's ask, the words, then what the screen counts."""
        if _ASKS_TO_LOOK.search(self.asked) and not asks_to_win(self.asked):
            return FIND_OUT
        if self.count_label is None:
            return LAST if self.lives_label is not None or self.vitals is not None else FIND_OUT
        return TARGET if self.target is not None else MOST

    # -- her record ------------------------------------------------------------

    def saw(self, at: float, counters: dict[str, int], *, out_of: dict[str, int] | None = None,
            vitals: float | None = None) -> None:
        """One reading: the count, the lives, how full her bar is, and their changes."""
        self.began = at if self.began is None else self.began
        for label, value in counters.items():
            meaning = what_a_counter_means(label)
            if not meaning and out_of and label in out_of:
                # A count written out of a whole ("Fish 2/40") is counted toward it, whatever its word.
                meaning = "up is good"
            if meaning == "up is good" and self.count_label in (None, label):
                self.count_label = label
                if self.count is not None and value > self.count:
                    self.gains.append((at, float(value) - self.count))
                    self.count_moved_at = at
                self.count = float(value)
                if out_of and label in out_of:
                    self.target = float(out_of[label])
            elif meaning == "down is bad" and self.lives_label in (None, label):
                self.lives_label = label
                if self.lives is not None and value < self.lives and not self.hits:
                    # With no bar, each life lost is a blow; with one, its hits are, and the lives follow from them.
                    self.blows.extend([at] * int(self.lives - value))
                self.lives = float(value)
        if vitals is not None and 0.0 <= vitals <= 1.0:
            if self.vitals is not None and vitals < self.vitals - LEAST_BLOW:
                self.hits.append(self.vitals - vitals)
                self.blows.append(at)
            self.vitals = vitals
        if self.count_moved_at is None:
            self.count_moved_at = at

    def met(self, kind: int, what: str) -> None:
        """What a kind has turned out to be: ``"gain"``, ``"loss"`` or ``"nothing"``."""
        self.kinds[kind] = what

    def shot_paid(self, kind: int) -> None:
        """Taking out a thing of ``kind`` paid a gain."""
        self.pays_to_shoot.add(kind)

    def _played(self, at: float) -> float:
        return max(0.0, at - (self.began if self.began is not None else at))

    def rate(self, at: float) -> float:
        """Gains a second while in play, over this run and the runs before."""
        return (self.gains_before + len(self.gains) + 1.0) / (self.played_before + self._played(at) + GAIN_EVERY_S)

    def blow_s(self, at: float) -> float:
        """Seconds between blows, over this run and the runs before."""
        return (self.played_before + self._played(at) + BLOW_EVERY_S) / (self.blows_before + len(self.blows) + 1.0)

    def gain_size(self) -> float:
        return sum(by for _at, by in self.gains) / len(self.gains) if self.gains else 1.0

    def blows_left(self) -> int | None:
        """How many more blows she can take before the run ends: the hits left in her bar and a full bar's worth in each
        life to come; a life a blow where she has no bar. None where she has nothing to lose that she knows of."""
        lives = int(self.lives) if self.lives is not None else (1 if self.vitals is not None else None)
        if lives is None or lives < 1:
            return None
        if self.vitals is None or not self.hits:
            return lives
        hit = statistics.median(self.hits)
        in_a_bar = max(1, round(1.0 / hit))
        now = max(1, math.ceil(self.vitals / hit - 1e-6))
        return now + (lives - 1) * in_a_bar

    # -- self-preservation -----------------------------------------------------

    def worth_of_a_blow_s(self, at: float, seconds_left: float | None = None, *, as_hit: bool = False) -> float | None:
        """What taking a blow now costs, in seconds of play: the gains it takes away, over the rate she gains. None where
        she has nothing to lose that she knows of.

        For her care, blows are taken to come no oftener than once in ``BLOW_EVERY_S``: how often she is hit depends on
        her care, so being hit often cannot make a blow cheap, or she would care less and be hit more. Blows rarer than
        that make each worth more. For a threat, whose blows are counted on their own, they come as often as they
        have (``as_hit``): where they come every few seconds whatever she does, each costs little."""
        blows = self.blows_left()
        if blows is None:
            return None
        rate, blow_s = self.rate(at), (self.blow_s(at) if as_hit else max(self.blow_s(at), BLOW_EVERY_S))
        lead = self.lead()
        if lead == TARGET and self.target is not None and self.count is not None:
            needed = max(0.0, (self.target - self.count) / self.gain_size())
            reach = chance_of_reaching(needed, blows, rate, blow_s, seconds_left)
            lost = reach - chance_of_reaching(needed, blows - 1, rate, blow_s, seconds_left)
            gained = chance_of_reaching(needed - 1, blows, rate, blow_s, seconds_left) - reach
            if gained > 1e-9:
                # The blow costs as much of the chance as this many gains; a gain takes 1/rate seconds to make.
                return (lost / gained) / rate
            # Where one gain more or less does not move the chance (it is all but sure, or all but lost), the count is
            # what can still be raised.
        if lead == LAST and seconds_left is not None:
            # What counts is lasting: a blow is worth the chance of lasting it takes away, as a share of the time left.
            x = seconds_left / blow_s
            return seconds_left * (_poisson_at_most(blows - 1, x) - _poisson_at_most(blows - 2, x))
        lost = count_expected(blows, rate, blow_s, seconds_left) - count_expected(blows - 1, rate, blow_s, seconds_left)
        return lost / rate

    def care(self, at: float, seconds_left: float | None = None) -> float:
        """How much more, or less, a blow is worth now than the blow her play's berth was set for: a life of
        ``BLOW_EVERY_S`` seconds of play. What the berth she gives what costs her is multiplied by."""
        worth = self.worth_of_a_blow_s(at, seconds_left)
        if worth is None:
            return 1.0
        return min(CARE_BOUNDS[1], max(CARE_BOUNDS[0], worth / BLOW_EVERY_S))

    # -- threat elimination ----------------------------------------------------

    def saw_threats(self, at: float, threats: list[tuple[int, int]]) -> None:
        """The threats on the screen now, as (number, kind): how long each kind has been about her, and how long each
        thing stayed once it went."""
        if self.threats_at is not None:
            step = min(1.0, max(0.0, at - self.threats_at))
            for _number, kind in threats:
                self.exposure[kind] = self.exposure.get(kind, 0.0) + step
        here = {number for number, _kind in threats}
        for number, since in list(self.appeared.items()):
            if number not in here:
                self.stays.append(at - since)
                del self.appeared[number]
        for number, _kind in threats:
            self.appeared.setdefault(number, at)
        self.threats_at = at

    def took_out(self, number: int) -> None:
        """She took a threat out: it went because of her, which says nothing of how long it would have stayed."""
        self.appeared.pop(number, None)

    def blow_from(self, kind: int) -> None:
        """A blow she took came from a thing of ``kind``."""
        self.blows_by[kind] = self.blows_by.get(kind, 0) + 1
        self.cost_her.add(kind)

    def hazard(self, kind: int, *, comes_for_her: bool) -> float:
        """Blows a second one thing of ``kind`` deals her while it is about: the blows its kind has dealt over the time
        things of its kind have been about her, counted from one blow in the believed time between blows, twice that
        for one that comes for her."""
        believed = BLOW_EVERY_S / (2.0 if comes_for_her else 1.0)
        return (self.blows_by.get(kind, 0) + 1.0) / (self.exposure.get(kind, 0.0) + believed)

    def stay_s(self, seconds_left: float | None = None) -> float:
        """How long a threat stays about her once it has come: as long as the ones before stayed, the believed time
        between blows before any has gone; no longer than the time left."""
        stays = statistics.median(self.stays) if len(self.stays) >= 3 else BLOW_EVERY_S
        return min(stays, seconds_left) if seconds_left is not None else stays

    def worth_taking_out_s(self, kind: int, *, comes_for_her: bool, at: float, seconds_left: float | None = None) -> float:
        """What taking a threat out is worth, in seconds of play: the blows it would deal while it stays, by what a blow
        is worth; and a gain's worth where shooting its kind has paid one."""
        worth = self.worth_of_a_blow_s(at, seconds_left, as_hit=True) or 0.0
        blows = 1.0 - math.exp(-self.hazard(kind, comes_for_her=comes_for_her) * self.stay_s(seconds_left))
        paid = 1.0 / self.rate(at) if kind in self.pays_to_shoot else 0.0
        return blows * worth + paid

    # -- finding out -----------------------------------------------------------

    def detour_for_the_unknown(self, at: float, *, alike_on_screen: int = 0, seconds_left: float | None = None,
                               standing: float = 2.0) -> float:
        """How far out of her way a thing not yet known is worth going, as a multiple of the time to it, where a thing
        known to be worth getting is one. ``standing`` is the player's own measure, kept until there is a record."""
        if self.lead() == FIND_OUT or self.spent(at):
            return DETOUR_BOUNDS[0]
        if len(self.kinds) < KINDS_TO_COUNT_FROM:
            return standing
        met = list(self.kinds.values())
        good = (met.count("gain") + 1.0) / (len(met) + 3.0)
        bad = (met.count("loss") + 1.0) / (len(met) + 3.0)
        blow_in_gains = (self.worth_of_a_blow_s(at, seconds_left) or 0.0) * self.rate(at)
        worth = good * (1.0 + max(0, alike_on_screen)) - bad * blow_in_gains
        if worth <= 0:
            return DETOUR_BOUNDS[1]
        return min(DETOUR_BOUNDS[1], max(DETOUR_BOUNDS[0], 1.0 / worth))

    # -- when it stops paying --------------------------------------------------

    def spent(self, at: float) -> bool:
        """Whether the count that leads has stopped moving for much longer than it took to move before."""
        if self.lead() not in (TARGET, MOST) or self.count_moved_at is None or self.began is None:
            return False
        gap = (self.count_moved_at - self.began) / len(self.gains) if self.gains else GAIN_EVERY_S
        return at - self.count_moved_at > max(SPENT_AT_LEAST_S, SPENT_GAPS * gap)

    def what_to_say(self, at: float) -> str:
        """A line for the person watching, once, when the lead is spent."""
        if self.said_spent or not self.spent(at):
            return ""
        self.said_spent = True
        return "Nothing I'm doing is adding to the count any more, so I'm going to try the things I don't know yet."
