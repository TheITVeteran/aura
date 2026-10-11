"""Who is who: what each thing on the screen is to her, told from what it does, with nothing in the game to say so.

A player sorts what is about them before deciding anything: that is mine, that is after me, that one is racing me for
the coins, that one is on my side, that is just a wall, that is only the background. Nothing on the screen says which.
The sorting is old and well worked out where it matters most: the standard identities of military symbology
(MIL-STD-2525, APP-6) give every contact one of pending, unknown, assumed friend, friend, neutral, suspect and hostile,
and move it along as evidence comes in. A contact just seen is pending; one looked at with nothing to go on is unknown;
one with signs but no proof is assumed friend or suspect; one whose own acts have shown it is friend, neutral or
hostile. Play adds what a thing is for, since a player's ends are not only to stay alive: a prize, a thing in her way.
And two that need no designation in any end: a thing that has had every chance to matter and has not (negligible), and
the place itself, still and doing nothing (part of the place).

What each designation rests on, all of it measured in play:

- hers: it answers to her keys, or goes out from her.
- a threat: it has dealt her a blow, or meeting it cost her, or a blow came as it went off the edge of the picture: it
  got past her, as an invader that lands does. Suspect where only signs say so: it keeps coming for her
  (core/agency/what_comes_for_her.py), or the game's words said to keep clear of its kind.
- a prize: meeting it raised a count.
- friendly: losing it in the middle of the picture cost her, while she was not touched, twice, and things of its kind go seldom enough that its going
  just before a blow is not chance. Assumed friendly after once, or where it keeps with her as she moves and meeting it
  does nothing to her.
- friendly competition: it goes for what she goes for and takes it first, and meeting it costs her nothing. It is
  beaten by getting there first, not by taking it out.
- neutral: it goes its own way, and meeting it does nothing.
- in her way: it stopped her, and meeting it does nothing.
- negligible: for as long as things of its kind have been about, none came near her, stopped her or took anything; or
  it is small and gone at once, a spark or a puff.
- part of the place: as negligible, and it has never moved.
- unknown: none of the above yet. Pending: seen for less than a moment.

Her play reads these (core/agency/playing_as_it_happens.py): threats and suspects are what threat elimination prices
(core/agency/what_she_plays_for.py); what is friendly, friendly competition, neutral or the place is never shot at;
and she says what she has found out, once each. Nothing here knows a game, a colour or a kind.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FRIENDLY",
    "HERS",
    "IN_HER_WAY",
    "NEGLIGIBLE",
    "NEUTRAL",
    "NEVER_SHOT",
    "PART_OF_THE_PLACE",
    "PENDING",
    "PRIZE",
    "RIVAL",
    "THREAT",
    "UNKNOWN",
    "Designation",
    "WhoIsWho",
]

HERS, THREAT, PRIZE, FRIENDLY, RIVAL = "hers", "a threat", "a prize", "friendly", "friendly competition"
NEUTRAL, IN_HER_WAY, NEGLIGIBLE, PART_OF_THE_PLACE = "neutral", "in her way", "negligible", "part of the place"
UNKNOWN, PENDING = "unknown", "pending"

#: What she never shoots at: on her side, racing her fairly, shown to do nothing, or the place itself. A negligible
#: thing is not spared: sparing a target wrongly costs more than a shot at what never mattered.
NEVER_SHOT = frozenset({HERS, FRIENDLY, RIVAL, NEUTRAL, PART_OF_THE_PLACE})

#: How near the edge of the picture, as a share of it, a thing must go to have gone off it, or got past her.
AT_THE_EDGE = 0.06

#: How long a thing is seen before it is looked at: under this it is pending.
PENDING_S = 0.3

#: How near, in her own sizes, a thing must come to her to have come near her.
NEAR = 3.0

#: How long things of a kind must have been about her, never coming near, stopping her or taking anything, to be
#: negligible.
NEGLIGIBLE_AFTER_S = 10.0

#: A thing gone within this long of being first seen, and smaller than this share of her, is a spark or a puff.
FLEETING_S = 0.5
SMALL_SHARE = 0.25

#: How long, and in how many of the pictures, a thing must keep within ``NEAR`` of her while she moves about, to keep
#: with her.
KEEPS_WITH_S = 3.0
KEEPS_WITH_SHARE = 0.8

#: How long before a blow a thing that went may have been what the blow was for.
WENT_WITHIN_S = 0.5

#: How many prizes a kind must have taken first to be a rival for sure, and how many losses of it must have cost her
#: for it to be on her side for sure.
TAKEN_FOR_SURE = 2
#: How many times a kind must have done a thing (cost her, counted, stopped her, done nothing) for that to be shown.
SHOWN_AFTER = 2
LOST_WITH_FOR_SURE = 2

#: The most a kind's going may be expected by chance in the moment before a blow for the blow to be put to its loss:
#: things that go all the time go before every blow.
BY_CHANCE = 0.1


@dataclass(frozen=True)
class Designation:
    """What a thing is to her, whether its own acts have shown it (``sure``) or only signs say so, and why."""

    name: str
    sure: bool
    why: str


@dataclass
class _Kind:
    """What things of one kind have done, all of them together."""

    about_s: float = 0.0
    moved: bool = False
    near: int = 0
    gains: int = 0
    losses: int = 0
    nothings: int = 0
    blows: int = 0
    took: int = 0
    stopped: int = 0
    lost_with: int = 0
    got_past: int = 0
    went: int = 0
    told: str | None = None


@dataclass
class WhoIsWho:
    """Each kind's record, each thing's first sighting, and what has been said."""

    kinds: dict[int, _Kind] = field(default_factory=dict)
    first_seen: dict[int, float] = field(default_factory=dict)
    #: Each thing as last seen: (kind, box, moved); and things that went lately: (when, kind, box).
    last: dict[int, tuple[int, tuple[float, float, float, float], bool]] = field(default_factory=dict)
    went: deque = field(default_factory=lambda: deque(maxlen=40))
    #: Things that kept with her: pictures near her while she moved, out of pictures she moved, since when.
    keeping: dict[int, list[float]] = field(default_factory=dict)
    fleeting: set[int] = field(default_factory=set)
    hers: set[int] = field(default_factory=set)
    #: When each thing on her now came onto her.
    on_her_since: dict[int, float] = field(default_factory=dict)
    #: Kinds that come for her, as what_comes_for_her last said.
    coming: set[int] = field(default_factory=set)
    said: set[tuple[int, str]] = field(default_factory=set)
    at: float = 0.0

    def begin_run(self) -> None:
        """A run of the same game begins: what each kind has shown is kept; the things themselves are numbered anew."""
        self.first_seen.clear()
        self.last.clear()
        self.went.clear()
        self.keeping.clear()
        self.on_her_since.clear()
        self.hers.clear()
        self.at = 0.0

    def _kind(self, kind: int) -> _Kind:
        return self.kinds.setdefault(kind, _Kind())

    # -- what is seen ----------------------------------------------------------

    def saw(self, at: float, things: Any, mine: Any, *, prizes: set[int] | None = None,
            comes_for_her: Any = None, with_her: Any = None, shape: tuple[float, float] | None = None) -> None:
        """One picture: every thing, hers among them. ``prizes`` are the kinds worth getting, for telling who took one
        that went; ``comes_for_her`` says of a thing whether it comes for her; ``with_her`` whether it goes where hers
        goes, picture for picture, as a part of hers or her shadow does."""
        step = min(1.0, max(0.0, at - self.at)) if self.at else 0.0
        self.at = at
        size = max(float(mine.w), float(mine.h), 1.0) if mine is not None else 16.0
        here = {}
        moving = mine is not None and math.hypot(float(mine.vx), float(mine.vy)) > 1.0
        # How long things of each kind have been about her: the time any of them was on the screen.
        for kind in {thing.kind for thing in things if mine is None or thing.number != mine.number}:
            self._kind(kind).about_s += step
        for thing in things:
            if mine is not None and (thing.number == mine.number or (with_her is not None and with_her(thing))):
                self.hers.add(thing.number)
                continue
            here[thing.number] = thing
            self.first_seen.setdefault(thing.number, at)
            kept = self._kind(thing.kind)
            kept.moved = kept.moved or bool(getattr(thing, "moved", False))
            near = mine is not None and math.hypot(float(thing.x) - float(mine.x), float(thing.y) - float(mine.y)) \
                <= NEAR * size
            kept.near += int(near)
            if mine is not None and _overlap(_box(thing), _box(mine), 2.0):
                self.on_her_since.setdefault(thing.number, at)
            else:
                self.on_her_since.pop(thing.number, None)
            if moving:
                keeping = self.keeping.setdefault(thing.number, [0.0, 0.0, at])
                keeping[0] += float(near)
                keeping[1] += 1.0
            if comes_for_her is not None:
                # As the reading of what comes for her now says, over its own window: lining up for a while by chance
                # is not kept against a kind for good.
                (self.coming.add if comes_for_her(thing) else self.coming.discard)(thing.kind)
            self.last[thing.number] = (thing.kind, _box(thing), bool(getattr(thing, "moved", False)))
        for number in [n for n in self.last if n not in here]:
            kind, box, _moved = self.last.pop(number)
            self.went.append((at, kind, box, _at_the_edge(box, shape)))
            self._kind(kind).went += 1
            area = (box[2] - box[0]) * (box[3] - box[1])
            if at - self.first_seen.get(number, at) < FLEETING_S and area < SMALL_SHARE * size * size:
                self.fleeting.add(kind)
            if prizes and kind in prizes:
                self._who_took(box, here, mine)

    def _who_took(self, box: tuple[float, float, float, float], here: dict[int, Any], mine: Any) -> None:
        """A prize went: where she was not on it and another thing was, that thing's kind took it first."""
        if mine is not None and _overlap(box, _box(mine), 2.0):
            return
        for thing in here.values():
            if getattr(thing, "moved", False) and _overlap(box, _box(thing), 2.0):
                self._kind(thing.kind).took += 1
                return

    # -- what happened ---------------------------------------------------------

    def met(self, kind: int, what: str) -> None:
        """What meeting a thing of ``kind`` did: ``"gain"``, ``"loss"`` or ``"nothing"``."""
        kept = self._kind(kind)
        if what == "gain":
            kept.gains += 1
        elif what == "loss":
            kept.losses += 1
        else:
            kept.nothings += 1

    def blow(self, at: float, mine: Any, things: Any) -> int | None:
        """She took a blow. A thing on her dealt it; else a thing that went just now, not on her and not a prize, was
        on her side, and losing it cost her. The kind it is put to, if any."""
        # A blow comes as a thing comes onto her: one that had been on her a while with no blow did not deal this one.
        on_her = [thing for thing in things
                  if mine is not None and thing.number != mine.number and _overlap(_box(thing), _box(mine), 2.0)
                  and at - self.on_her_since.get(thing.number, at) <= WENT_WITHIN_S]
        # A still thing she has just walked onto, as another thing goes in that same picture, is not taken for the
        # cause: the faller that landed as she reached the block cost her, not the block.
        going_now = any(at - when <= 1e-6 for when, _kind, _box, _edge in self.went)
        if going_now and on_her and not any(getattr(thing, "moved", False) for thing in on_her):
            on_her = []
        if on_her:
            # Of several things on her, the blow is put to the one least shown to be worth getting, that moves, and
            # that has dealt blows before: a coin she was picking up as the chaser reached her did not hurt her.
            dealt = max(on_her, key=lambda thing: (not self._harmless(thing.kind), thing.kind in self.coming,
                                                   bool(getattr(thing, "moved", False)), self._kind(thing.kind).blows))
            if not self._harmless(dealt.kind):
                self._kind(dealt.kind).blows += 1
                return dealt.kind
        for when, kind, box, at_the_edge in reversed(self.went):
            if at - when > WENT_WITHIN_S:
                break
            if mine is not None and _overlap(box, _box(mine), 2.0) and not self._harmless(kind):
                self._kind(kind).blows += 1
                return kind
            kept = self._kind(kind)
            by_chance = kept.went / max(kept.about_s, 1.0) * WENT_WITHIN_S
            if kept.gains or by_chance >= BY_CHANCE:
                continue
            # Gone off the edge of the picture as the blow came, it got past her: an invader that lands. Gone in the
            # middle of it, untouched by her, it was one of hers lost.
            if at_the_edge:
                kept.got_past += 1
            else:
                kept.lost_with += 1
            return kind
        return None

    def _harmless(self, kind: int) -> bool:
        """Whether a kind has been shown not to hurt her: meeting it was worth something, or did nothing, or it stopped
        her and nothing came of it; or it is a spark, gone as soon as it comes. Such a kind on her when a blow came did
        not deal it."""
        kept = self._kind(kind)
        if kept.blows or kept.losses:
            return False
        return bool(kept.gains or kept.nothings or kept.stopped) or kind in self.fleeting

    def stopped(self, mine: Any, way: tuple[float, float], things: Any) -> int | None:
        """She held a way and did not go: the thing just ahead of her that way stopped her. Its kind, if any."""
        if mine is None:
            return None
        size = max(float(mine.w), float(mine.h), 1.0)
        far = math.hypot(*way) or 1.0
        ahead = (float(mine.x) + way[0] / far * size, float(mine.y) + way[1] / far * size)
        for thing in things:
            if thing.number == mine.number:
                continue
            left, top, right, bottom = _box(thing)
            if left - size / 2 <= ahead[0] <= right + size / 2 and top - size / 2 <= ahead[1] <= bottom + size / 2:
                self._kind(thing.kind).stopped += 1
                return thing.kind
        return None

    def told(self, kind: int, name: str) -> None:
        """The game's words, or a legend, said what a kind is: a threat, a prize, friendly."""
        self._kind(kind).told = name

    # -- what each is ----------------------------------------------------------

    def of(self, thing: Any) -> Designation:
        """What this thing is to her now."""
        if thing.number in self.hers:
            return Designation(HERS, True, "it answers to me")
        kept = self.kinds.get(thing.kind) or _Kind()
        # Once is a sign, twice is shown: a thing a cost or a gain came with once is taken for what it seemed and played
        # as such, but is not said for sure until it does so again. LIVE 2026-10-11 a scrolling game had twenty kinds
        # each said for sure to be a threat, a prize or a rival, every one from a single meeting that came by chance.
        if kept.blows or kept.losses:
            return Designation(THREAT, kept.blows + kept.losses >= SHOWN_AFTER,
                               "it has cost me" if kept.blows + kept.losses >= SHOWN_AFTER else "it seemed to cost me")
        if kept.got_past:
            return Designation(THREAT, kept.got_past >= SHOWN_AFTER, "it costs me when it gets past")
        if kept.lost_with >= LOST_WITH_FOR_SURE:
            return Designation(FRIENDLY, True, "losing it cost me")
        if kept.gains:
            return Designation(PRIZE, kept.gains >= SHOWN_AFTER,
                               "getting it counted" if kept.gains >= SHOWN_AFTER else "getting it seemed to count")
        if kept.took >= TAKEN_FOR_SURE:
            return Designation(RIVAL, True, "it takes what I'm after, and meeting it costs me nothing")
        if kept.stopped and not kept.losses:
            return Designation(IN_HER_WAY, kept.stopped >= SHOWN_AFTER, "it stops me")
        keeps_with = self._keeps_with(thing.number)
        if kept.nothings:
            if keeps_with:
                return Designation(FRIENDLY, False, "it keeps with me and does nothing to me")
            if kept.moved:
                return Designation(NEUTRAL, kept.nothings >= SHOWN_AFTER, "it goes its own way and meeting it does nothing")
            return Designation(PART_OF_THE_PLACE, True, "it never moves and meeting it does nothing")
        # One that keeps with her at its own distance is a companion, though its going points at her as a pursuer's does.
        if keeps_with:
            return Designation(FRIENDLY, False, "it keeps with me")
        if thing.kind in self.coming or kept.told == THREAT:
            return Designation(THREAT, False, "it keeps coming for me" if thing.kind in self.coming else "I was told so")
        if kept.told in (PRIZE, FRIENDLY):
            return Designation(kept.told, False, "I was told so")
        if kept.took:
            return Designation(RIVAL, False, "it took what I was after")
        if kept.lost_with:
            return Designation(FRIENDLY, False, "losing it seemed to cost me")
        if thing.kind in self.fleeting:
            return Designation(NEGLIGIBLE, True, "it is small and gone at once")
        if kept.about_s >= NEGLIGIBLE_AFTER_S and not kept.near and not kept.got_past:
            if kept.moved:
                return Designation(NEGLIGIBLE, True, "it has never come near me or done anything")
            return Designation(PART_OF_THE_PLACE, True, "it has never moved or done anything")
        if self.at - self.first_seen.get(thing.number, self.at) < PENDING_S:
            return Designation(PENDING, False, "I've only just seen it")
        return Designation(UNKNOWN, False, "I don't know yet")

    def _keeps_with(self, number: int) -> bool:
        keeping = self.keeping.get(number)
        return (keeping is not None and self.at - keeping[2] >= KEEPS_WITH_S and keeping[1] > 0
                and keeping[0] / keeping[1] >= KEEPS_WITH_SHARE)

    def threats(self, things: Any) -> list[tuple[Any, Designation]]:
        """The threats on the screen, sure and suspect."""
        return [(thing, d) for thing in things if (d := self.of(thing)).name == THREAT]

    def to_say(self, things: Any, name_of: Any, most: int = 1) -> list[str]:
        """What she has found out of each kind and not yet said: once each, when its own acts have shown it, and no more
        than ``most`` at a time, the rest left for later (LIVE 2026-10-11 one loss had seven kinds said to be threats
        in a fifth of a second)."""
        lines = []
        for thing in things:
            if len(lines) >= most:
                break
            found = self.of(thing)
            if not found.sure or found.name in (HERS, PENDING, UNKNOWN, PART_OF_THE_PLACE, NEGLIGIBLE):
                continue
            if (thing.kind, found.name) in self.said:
                continue
            self.said.add((thing.kind, found.name))
            lines.append(f"The {name_of(thing)} is {_spoken(found.name)}: {found.why}.")
        return lines


def _spoken(name: str) -> str:
    return {THREAT: "a threat", PRIZE: "worth getting", FRIENDLY: "on my side", RIVAL: "friendly competition",
            NEUTRAL: "neutral", IN_HER_WAY: "in my way"}.get(name, name)


def _at_the_edge(box: tuple[float, float, float, float], shape: tuple[float, float] | None) -> bool:
    """Whether a box is at, or past, the edge of a picture of ``shape`` (tall, wide)."""
    if shape is None:
        return False
    tall, wide = shape
    margin_x, margin_y = AT_THE_EDGE * wide, AT_THE_EDGE * tall
    return box[0] <= margin_x or box[1] <= margin_y or box[2] >= wide - margin_x or box[3] >= tall - margin_y


def _box(thing: Any) -> tuple[float, float, float, float]:
    x, y, w, h = float(thing.x), float(thing.y), float(thing.w), float(thing.h)
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], slack: float) -> bool:
    return a[0] - slack < b[2] and b[0] - slack < a[2] and a[1] - slack < b[3] and b[1] - slack < a[3]
