"""What a sound comes with: learned from when each is heard against what happens around it.

The page's sounds are heard as events (core/perception/what_the_page_plays.py); what each means is learned the way what
meeting a thing does is learned, from what follows. Four meanings, each read against what chance alone would give: a
sound heard often, among events that are frequent, falls near one of them by luck. A meaning is taken where most of a
sound's hearings agree, and agree well above that chance.

- It comes with a gain: a gain within a moment of it, either side.
- It comes with a loss: a blow within a moment of it.
- It is her own: it follows a press of hers at once, as a shot's sound does.
- It is a warning: the blows she takes come after it, not at once but after a while, and after much the same while
  each time, far oftener than chance would put it there. The while is its lead. A warning is read from the blows and
  not from its hearings: one she heeded, or got out of the way of by luck, is followed by no blow.

For a warning there is one thing more to know: whether the blow lands where she was when it sounded, so that moving
away from there is the answer. That is tried: while a warning is heard, the spot she was on is a place to keep clear
of (core/agency/playing_as_it_happens.py weighs it as it weighs what comes at her), as far round as such blows have
been found to reach. Whether that pays is measured: where a warning heeded is followed by a blow as often as it was
before she knew it, the blow finds her wherever she goes or reaches further than she can get in the time, and that
sound is not heeded so again.

Nothing here knows a game or a sound.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["GAIN", "HER_OWN", "LOSS", "WARNING", "WhatSoundsComeWith"]

GAIN, LOSS, HER_OWN, WARNING = "a gain", "a loss", "her own", "a warning"

#: How near, either side, an event must be to a sound to come with it; how soon after her press her own sound comes.
WITH_S = 0.6
HER_OWN_S = 0.3

#: How long after a warning the blow it warns of may come, at the least and at the most; how far each blow's lead may
#: be from the middle one for the lead to be steady.
WARNED_FROM_S, WARNED_UNTIL_S = 0.6, 5.0
STEADY_S = 0.5

#: How many hearings, judged, a sound needs before it means anything; what share of them must agree; and by how much
#: that share must stand above chance.
LEAST_HEARD = 3
MOST_AGREE = 0.6
ABOVE_CHANCE = 0.3

#: How near, in her own sizes, a warned blow is taken to reach from its spot before any has found her; and how much
#: further than that a blow may find her and still be the spot's: further off, it found her where she had gone.
ON_THE_SPOT = 1.5
CLEAR_OF_IT = 3.0


@dataclass
class WhatSoundsComeWith:
    """Each sound's hearings, the events around them, and what has been said."""

    heard: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))
    gains: deque = field(default_factory=lambda: deque(maxlen=300))
    blows: deque = field(default_factory=lambda: deque(maxlen=300))
    presses: deque = field(default_factory=lambda: deque(maxlen=600))
    began: float | None = None
    #: Where she was at each hearing of each sound, and at each blow, in her own sizes' terms: (at, x, y, size).
    where_heard: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=60)))
    where_hit: deque = field(default_factory=lambda: deque(maxlen=300))
    #: Per warning sound: the hearings whose blow she got clear of, the hearings she heeded (kept clear of the spot
    #: for) and those of them a blow followed all the same, and each while from a hearing to the blow after it.
    cleared: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    heeded: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    hit_heeding: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    leads: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=30)))
    #: How far from the spot she was, in her own sizes, each time a warned blow found her: how far such a blow reaches.
    reached: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=30)))
    said: set[tuple[str, str]] = field(default_factory=set)

    # -- what is heard and what happens ----------------------------------------

    def heard_sound(self, at: float, sig: str, mine: Any = None) -> None:
        """A sound heard at ``at``, with her thing as it then was."""
        self.began = at if self.began is None else min(self.began, at)
        self.heard[sig].append(at)
        if mine is not None:
            self.where_heard[sig].append((at, float(mine.x), float(mine.y), max(float(mine.w), float(mine.h), 1.0)))

    def gained(self, at: float) -> None:
        self.gains.append(at)

    def pressed(self, at: float) -> None:
        self.presses.append(at)

    def blow(self, at: float, mine: Any = None) -> None:
        """She took a blow at ``at``; where she was, and whether a warning before it had been got clear of."""
        self.blows.append(at)
        for sig, hearings in self.heard.items():
            before = [when for when in hearings if WARNED_FROM_S <= at - when <= WARNED_UNTIL_S]
            if before:
                self.leads[sig].append(at - max(before))
                self.hit_heeding[sig].update(when for when in before if when in self.heeded[sig])
        if mine is None:
            return
        self.where_hit.append((at, float(mine.x), float(mine.y)))
        for sig in list(self.heard):
            spot = self._spot_before(sig, at)
            if spot is None or not self.leads.get(sig):
                continue
            _when, x, y, size = spot
            away = math.dist((x, y), (float(mine.x), float(mine.y))) / size
            # Near the spot, a blow says how far such a blow reaches; well away from it, it found her where she had
            # gone, and says nothing of its reach.
            if away <= ON_THE_SPOT + CLEAR_OF_IT:
                self.reached[sig].append(away)

    # -- what each means -------------------------------------------------------

    def _span(self, now: float) -> float:
        return max(1.0, now - (self.began if self.began is not None else now))

    @staticmethod
    def _has(events: deque, start: float, end: float) -> bool:
        return any(start <= when <= end for when in events)

    def _agree(self, sig: str, events: deque, lo: float, hi: float, now: float) -> float | None:
        """The share of the judged hearings of ``sig`` with an event between ``lo`` and ``hi`` seconds of it, less the
        share chance would give; None with too few judged."""
        judged = [when for when in self.heard[sig] if when + hi <= now]
        if len(judged) < LEAST_HEARD:
            return None
        share = sum(self._has(events, when + lo, when + hi) for when in judged) / len(judged)
        rate = len(events) / self._span(now)
        chance = 1.0 - math.exp(-rate * (hi - lo))
        return share - chance if share >= MOST_AGREE else None

    def _lead(self, sig: str, now: float) -> float | None:
        """The steady while between a sound and the blow after it, where there is one."""
        leads = list(self.leads.get(sig, ()))
        if len(leads) < LEAST_HEARD:
            return None
        middle = statistics.median(leads)
        steady = sum(abs(lead - middle) <= STEADY_S for lead in leads) / len(leads)
        return middle if steady >= MOST_AGREE else None

    def _warns(self, sig: str, now: float) -> float | None:
        """How far above chance the blows she takes come a steady while after this sound; None where they do not, or
        too few have. Judged from the blows and not from the hearings: a warning she heeded, or got out of the way of
        by luck, is followed by no blow, and is a warning still."""
        lead = self._lead(sig, now)
        if lead is None or len(self.blows) < LEAST_HEARD:
            return None
        hearings = self.heard[sig]
        preceded = sum(self._has(hearings, blow - lead - STEADY_S, blow - lead + STEADY_S) for blow in self.blows)
        chance = 1.0 - math.exp(-len(hearings) / self._span(now) * 2 * STEADY_S)
        return preceded / len(self.blows) - chance if preceded >= LEAST_HEARD else None

    def meaning(self, sig: str, now: float) -> str | None:
        """What a sound has come to mean: her own, a gain, a loss, a warning; None where nothing yet. Of several that
        hold, the one that stands furthest above chance: a crash heard as each blow lands may also follow a press of
        hers now and then."""
        above = {HER_OWN: self._agree(sig, self.presses, -HER_OWN_S, 0.0, now),
                 LOSS: self._agree(sig, self.blows, -WITH_S, WITH_S, now),
                 GAIN: self._agree(sig, self.gains, -WITH_S, WITH_S, now),
                 WARNING: self._warns(sig, now)}
        held = {name: excess for name, excess in above.items() if excess is not None and excess >= ABOVE_CHANCE}
        return max(held, key=held.__getitem__) if held else None

    # -- what a warning asks ---------------------------------------------------

    def _spot_before(self, sig: str, at: float) -> tuple[float, float, float, float] | None:
        """Where she was when ``sig`` was last heard within a warning's while before ``at``."""
        for spot in reversed(self.where_heard[sig]):
            if WARNED_FROM_S * 0.5 <= at - spot[0] <= WARNED_UNTIL_S:
                return spot
        return None

    def reach(self, sig: str) -> float:
        """How far, in her own sizes, a blow a sound warns of reaches from the spot it marks: a little beyond the
        furthest it has found her, and never less than ``ON_THE_SPOT``."""
        reached = self.reached.get(sig)
        return max(ON_THE_SPOT, 1.25 * max(reached)) if reached else ON_THE_SPOT

    def spots_to_keep_clear_of(self, now: float) -> list[tuple[float, float, float, float, float]]:
        """Each spot a warning heard lately marks: (x, y, how near counts, from when, until when), the times as
        seconds from now. The spot is where she was as it sounded, from a little before the blow is due until a little
        after; not for a sound whose blow has found her wherever she went."""
        spots = []
        for sig in list(self.heard):
            if self.meaning(sig, now) != WARNING or not self.heeding_pays(sig, now):
                continue
            lead = self._lead(sig, now)
            for when, x, y, size in reversed(self.where_heard[sig]):
                due = when + lead - now
                if due < -STEADY_S:
                    break
                self.heeded[sig].add(when)
                spots.append((x, y, self.reach(sig) * size, max(0.0, due - STEADY_S), due + STEADY_S))
        return spots

    def heeding_pays(self, sig: str, now: float) -> bool:
        """Whether keeping clear of the spot a warning marks has made its blow come less often than it came before she
        knew the warning: blows that find her wherever she goes, or reach further than she can get in the time, come
        as often heeded as not, and heeding them is given up."""
        over = WARNED_UNTIL_S
        heeded = [when for when in self.heeded[sig] if when + over <= now]
        if len(heeded) < 4:
            return True
        hit_heeding = len([when for when in heeded if when in self.hit_heeding[sig]]) / len(heeded)
        # Hit four times in five though she got off the spot: it does not pay, whatever came before.
        if hit_heeding >= 0.8:
            return False
        unheeded = [when for when in self.heard[sig] if when + over <= now and when not in self.heeded[sig]]
        if len(unheeded) < LEAST_HEARD:
            return True
        hit_before = sum(self._has(self.blows, when + WARNED_FROM_S, when + over) for when in unheeded) / len(unheeded)
        return hit_heeding < 0.8 * hit_before

    def judge_clear(self, now: float, mine: Any) -> None:
        """As each warning's while runs out: where she had got clear of its spot and no blow came, she got clear."""
        if mine is None:
            return
        for sig in list(self.heard):
            lead = self._lead(sig, now)
            if lead is None:
                continue
            for when, x, y, size in self.where_heard[sig]:
                over = when + lead + STEADY_S
                if when in self.cleared[sig] or not now - 0.5 <= over <= now:
                    continue
                if (math.dist((x, y), (float(mine.x), float(mine.y))) >= self.reach(sig) * size
                        and not self._has(self.blows, when + WARNED_FROM_S, over)):
                    self.cleared[sig].add(when)

    def warned(self, now: float) -> bool:
        """Whether a warning heard lately has its blow still to come."""
        return bool(self.spots_to_keep_clear_of(now))

    def to_say(self, now: float) -> list[str]:
        """What she has found each sound to mean and not yet said."""
        words = {GAIN: "That sound comes with a point.", LOSS: "That sound comes when I'm hit.",
                 HER_OWN: "That sound is mine: it comes when I press.",
                 WARNING: "That sound comes before I'm hit: when I hear it, I get off the spot I'm on."}
        lines = []
        for sig in list(self.heard):
            meaning = self.meaning(sig, now)
            if meaning is None or (sig, meaning) in self.said:
                continue
            self.said.add((sig, meaning))
            lines.append(words[meaning])
        return lines
