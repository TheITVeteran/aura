"""A thing gone for, kept to: a target chosen and held until there is a reason to let it go.

Choosing afresh at every picture which thing to go for turns a near tie into a dither: two coins at about the same
distance, and she turns from one to the other as each picture tips the sums, getting neither. A player picks one and
keeps to it. This is the old question of how often to reconsider an intention (in agents built on beliefs, desires
and intentions, an agent that reconsiders at every step does worse where the world changes than one that keeps to
what it chose), and the answer used in learning by options: an act held over time with conditions that end it.

A thing gone for is let go, and another chosen, for one of these reasons only:

- it is got, or gone (it is no longer on the screen);
- it is no longer one to go for (what meeting it does turned out otherwise, or it went out of reach);
- she is not getting nearer it: its time to reach has not come down for a while, and there is another to go for;
- another can be had plainly sooner: sooner by more than a margin of what this one takes.

The margin is learned in each game from what changing did: a change after which she got the new thing within a
while lowers it, a change after which she got nothing raises it. Urgency (core/agency/what_the_place_does_to_her.py)
lowers it too: with the time nearly gone, a thing much nearer is worth turning for. Nothing here knows what is gone
for or why.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.AThingGoneFor")

__all__ = ["GoneFor"]

#: The margin a new thing must beat the held one by, as a share of the time the held one takes; its bounds.
MARGIN = 0.2
MARGIN_BOUNDS = (0.1, 0.7)

#: How long a thing gone for may not come nearer before it is let go, and how much nearer counts.
NOT_NEARER_FOR_S = 2.0
NEARER_BY = 0.05

#: How long after a change the new thing must be got for the change to have paid.
PAID_WITHIN_S = 3.0

#: How long a thing let go of for not getting nearer is passed over: chosen again at once, it would be let go again.
PASSED_OVER_S = 3.0

Candidate = tuple[float, Any, Any]  # how long it takes (seconds), where to be, the thing


@dataclass
class GoneFor:
    """What one game's player is going for, and how firmly changes are made in it."""

    number: int | None = None
    since: float = 0.0
    #: The least time it has taken since it was chosen, and when that was.
    least: float = float("inf")
    least_at: float = 0.0
    margin: float = MARGIN
    #: A change made, and when: judged by whether the new thing is got soon after.
    changed_at: float | None = None
    changed_to: int | None = None
    changes: int = 0
    kept: int = 0
    #: Things let go of for not getting nearer, and until when they are passed over.
    passed_over: dict[int, float] = field(default_factory=dict)

    def choose(self, candidates: Sequence[Candidate], at: float, *, urgency: float = 0.0) -> Candidate | None:
        """The candidate to go for now, from ``candidates`` (each: seconds it takes, where to be, the thing), keeping to
        the one held unless a reason above lets it go."""
        candidates = [c for c in candidates if self.passed_over.get(getattr(c[2], "number", None), -1.0) < at]
        if not candidates:
            self._let_go("nothing to go for", at)
            return None
        best = min(candidates, key=lambda candidate: candidate[0])
        held = next((c for c in candidates if getattr(c[2], "number", None) == self.number), None) \
            if self.number is not None else None
        if held is not None:
            if held[0] < self.least * (1.0 - NEARER_BY):
                self.least, self.least_at = held[0], at
            margin = self.margin * (1.0 - 0.5 * min(1.0, max(0.0, urgency)))
            rest = [c for c in candidates if c is not held]
            if at - self.least_at > NOT_NEARER_FOR_S and rest:
                self.passed_over[self.number] = at + PASSED_OVER_S
                self._let_go("not getting nearer", at)
                best = min(rest, key=lambda candidate: candidate[0])
            elif at - self.least_at > NOT_NEARER_FOR_S:
                # Nothing else to go for: passing it over would only stand her still.
                self.least_at = at
                self.kept += 1
                return held
            elif best is held or best[0] >= held[0] * (1.0 - margin):
                self.kept += 1
                return held
            else:
                self.changes += 1
                self.changed_at, self.changed_to = at, getattr(best[2], "number", None)
                logger.info("let go of a thing gone for: another takes %.2fs against %.2fs", best[0], held[0])
        elif self.number is not None:
            self._let_go("got, gone, or no longer one to go for", at)
        if self.number is not None and getattr(best[2], "number", None) == self.number:
            return best
        self._hold(best, at)
        return best

    def _hold(self, candidate: Candidate, at: float) -> None:
        self.number = getattr(candidate[2], "number", None)
        self.since, self.least, self.least_at = at, candidate[0], at

    def _let_go(self, why: str, at: float) -> None:
        if self.number is not None:
            logger.info("let go of a thing gone for: %s", why)
        self.number = None
        self.least = float("inf")

    def got(self, number: int, at: float) -> None:
        """A thing was got (it went, touching her): where it was the one changed to soon after a change, the change
        paid, and changes are made more readily; a change that got nothing in that while made them less so."""
        if self.changed_at is None:
            return
        if number == self.changed_to and at - self.changed_at <= PAID_WITHIN_S:
            self.margin = max(MARGIN_BOUNDS[0], self.margin - 0.05)
            self.changed_at = None

    def judge_changes(self, at: float) -> None:
        """A change not followed by getting the new thing within its while makes changes harder to make."""
        if self.changed_at is not None and at - self.changed_at > PAID_WITHIN_S:
            self.margin = min(MARGIN_BOUNDS[1], self.margin + 0.1)
            self.changed_at = None
