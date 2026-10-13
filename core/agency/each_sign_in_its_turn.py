"""Each sign in its turn: whose keys are down now, of the signs coming along a line to where they are judged.

Signs come one behind another. The one at the front is judged where the line
ends, by what is down at that moment and nothing else. So its keys go down
when it becomes the front one, stay down until it has been judged, and then
the next one's go down, alone, before that one gets there in its turn.

Three things make this harder than it sounds, and each cost a reference
runner a game before it was done right:

- a sign is often not to be seen while it is being judged: it swells, or
  changes colour, as it comes onto the mark, and how far onto the mark it is
  judged is not to be seen either. So its keys stay down until it is midway
  to where the one behind it will go from sight in its turn: as long as they
  can, with time left for the next one's. What is seen beyond where the
  front ones go from sight is not gone by;
- which sign is which is told by their order, not by following each one: the
  front one has changed when the furthest sign is suddenly further back. A
  thing that shows ahead of the front one came from nowhere, and is not one
  coming: a sign that was not answered is often drawn on, past where the
  others went from sight, and one crossing the rim of its mark goes from
  sight and comes back;
- a sign is read while it is out on the clear part of the line, several times,
  and what it was mostly read as is kept; it is not read again where it is
  judged.

Nothing here sees a picture or knows what the signs are signs of: it is given,
at each look, where the signs on the line are and a way to read one.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["InTurn"]

#: How far back along the line, as a share of the picture, the furthest sign must suddenly be for the front one to
#: have changed; a thing at the front that was not seen to come that far along before it went was never a sign on its
#: way (a sign already judged is often drawn again, bigger, further on, in the signs' own ink).
THE_NEXT_ONE_BY = 0.1
#: How many of the last places a front sign went from sight are gone by: the middle one of them is where they go.
GONE_FROM_SIGHT_OF = 5

#: How far past where the front ones go from sight a thing seen is still one coming, as a share of the picture.
JUDGED_WITHIN = 0.03
#: What share of the way to the next sign's going from sight a sign's keys stay down for after it has gone itself,
#: and the furthest that is taken to be, as a share of the picture: a sign far ahead of the next is not held to for
#: all that way. Offline 2026-10-12 keys let go a thirtieth of the picture past where a sign went from sight were let
#: go before it was judged, about one time in six, and the next one's were down in their place.
HELD_FOR = 0.5
HELD_AT_MOST = 0.15
#: How long they stay down where the signs' pace is not yet known, in seconds.
HELD_ON_UNPACED_S = 0.3

#: How far short of where signs go from sight a sign is still read, and how far past where it was first seen at the
#: least: a sign coming out of what it comes from, or going onto its mark, is not all there. It is read once it has
#: come as far as the widest sign seen is wide.
READ_SHORT_OF = 0.05
READ_FROM = 0.04

#: Readings of a sign after which it is read no more, how many of those made while it was the one behind count once
#: it is the front one (it is read again there, and what it reads as now outvotes a reading it was handed by a
#: mistake of order), and how long the line must be empty for a showing to be over.
READ_ENOUGH = 4
READ_WHILE_BEHIND = 2
A_SHOWING_OVER_S = 0.6
#: For how long a line with nothing on it, its front sign last seen well short of where they go from sight, is a look
#: or two that missed them and not a line gone empty.
A_LOOK_MISSES_FOR_S = 0.25


class InTurn:
    """Whose keys are down now."""

    def __init__(self) -> None:
        self.of_front: Counter[frozenset[str]] = Counter()      # what the front sign has been read as
        self.of_next: Counter[frozenset[str]] = Counter()       # and the one behind it
        self.front_was_at = 0.0
        self.front_seen_at = 0.0
        #: Where the front sign was when first it was seen, and the one behind it: one of them has come along the line.
        self.front_first_at = 0.0
        self.next_first_at: float | None = None
        self.next_was_at: float | None = None
        self._went_at: list[float] = []
        #: How fast the signs go along the line, shares of the picture a second, once seen.
        self.pace: float | None = None
        #: Where the front ones go from sight, as far as seen.
        self.seen_to: float | None = None
        self.keys_of_the_last: frozenset[str] = frozenset()
        self.held_until = 0.0
        #: How far behind the one before it a sign comes, and how wide the widest seen is (shares of the picture).
        self.behind: float | None = None
        self.widest = 0.0
        self.empty_since: float | None = None
        self.judged = 0

    def take(self, signs: Sequence[Any], at: float, read: Callable[[Any], frozenset[str]]) -> frozenset[str]:
        """The keys to have down now, given the signs on the line (furthest along first, each with ``along``) and a
        way to read one."""
        if self.seen_to is not None:
            # Past where the front ones go from sight is where they are judged: what shows there is one already gone by.
            signs = [sign for sign in signs if sign.along <= self.seen_to + JUDGED_WITHIN]
        came_along = self.front_was_at - self.front_first_at >= THE_NEXT_ONE_BY
        if self.front_was_at > 0 and came_along:
            # And what shows ahead of a front one that has come along the line, further than it can have got, came
            # from nowhere. (A thing at the front that has not come along is no sign, and whatever shows ahead of it
            # takes its place.)
            reach = self.front_was_at + max(0.5 * THE_NEXT_ONE_BY, 3.0 * (self.pace or 0.0) * (at - self.front_seen_at))
            signs = [sign for sign in signs if sign.along <= reach]
        if self.front_was_at > 0 and at - self.front_seen_at < A_LOOK_MISSES_FOR_S and self._missed(signs, came_along):
            return self.keys_of_the_last if at < self.held_until else self._wanted()
        wanted: frozenset[str] = frozenset()
        if signs:
            front = signs[0]
            self.widest = max(self.widest, *(float(getattr(sign, "span", 0.0)) for sign in signs))
            if front.along < self.front_was_at - THE_NEXT_ONE_BY:
                self._the_front_went(at, behind=self.front_was_at - front.along)
                self.front_first_at = self.next_first_at if self.next_first_at is not None else front.along
                self.next_first_at = None
            elif self.front_was_at <= 0 or front.along > self.front_was_at + THE_NEXT_ONE_BY:
                self.front_first_at = front.along
                self.of_front = Counter() if self.front_was_at > 0 else self.of_front
            elif front.along > self.front_was_at and at > self.front_seen_at:
                seen = (front.along - self.front_was_at) / (at - self.front_seen_at)
                self.pace = seen if self.pace is None else 0.8 * self.pace + 0.2 * seen
            if len(signs) > 1 and self.next_first_at is None:
                self.next_first_at = signs[1].along
            self.next_was_at = signs[1].along if len(signs) > 1 else None
            self.front_was_at, self.front_seen_at = front.along, at
            self._read_into(self.of_front, front, read, self.front_first_at)
            if len(signs) > 1:
                self._read_into(self.of_next, signs[1], read, self.next_first_at or signs[1].along)
            # Only what has come along the line is answered: a thing that shows on it and stays where it showed is no sign.
            wanted = self._wanted() if front.along - self.front_first_at >= THE_NEXT_ONE_BY else frozenset()
            self.empty_since = None
        else:
            if self.empty_since is None:
                self.empty_since = at
                if self.front_was_at > 0:
                    self._the_front_went(at)        # the last of a showing has gone on to be judged
            self.next_was_at = None
            if at - self.empty_since > A_SHOWING_OVER_S:
                self.of_front, self.of_next, self.front_was_at, self.next_first_at = Counter(), Counter(), 0.0, None
        return self.keys_of_the_last if at < self.held_until else wanted

    def _missed(self, signs: Sequence[Any], came_along: bool) -> bool:
        """Whether a look missed the signs that are on the line: the one behind the front one is not where it was
        either, or there is nothing at all and the front one was well short of where they go from sight. A front
        sign goes from sight alone."""
        if self.next_was_at is not None:
            return not any(sign.along >= self.next_was_at - 0.5 * THE_NEXT_ONE_BY for sign in signs)
        return (not signs and came_along and self.seen_to is not None
                and self.front_was_at < self.seen_to - THE_NEXT_ONE_BY)

    def _wanted(self) -> frozenset[str]:
        return self.of_front.most_common(1)[0][0] if self.of_front else frozenset()

    def _the_one_behind_is_in_front(self) -> None:
        """What the one behind was read as goes with it to the front, for a little: it is read again there."""
        self.of_front = Counter({keys: min(times, READ_WHILE_BEHIND) for keys, times in self.of_next.items()})
        self.of_next = Counter()

    def _the_front_went(self, at: float, behind: float | None = None) -> None:
        """The sign that was at the front has gone from sight: it is judged somewhere on from there, and its keys stay
        down until it is midway to where the one ``behind`` it goes from sight in its turn; that one is the front one
        now, read as it was while it came."""
        went_at = self.front_was_at
        if went_at - self.front_first_at < THE_NEXT_ONE_BY:
            # It showed at the front and went without coming along the line: not one of them. The one behind it is.
            self._the_one_behind_is_in_front()
            return
        self.judged += 1
        self._went_at = [*self._went_at, went_at][-GONE_FROM_SIGHT_OF:]
        self.seen_to = sorted(self._went_at)[len(self._went_at) // 2]
        self.keys_of_the_last = self._wanted()
        self.behind = behind if behind is not None else self.behind
        if self.pace and self.pace > 0 and self.behind:
            more = min(HELD_FOR * self.behind, HELD_AT_MOST) / self.pace
        else:
            more = HELD_ON_UNPACED_S
        self.held_until = self.front_seen_at + more
        self._the_one_behind_is_in_front()

    def _read_into(self, readings: Counter[frozenset[str]], sign: Any, read: Callable[[Any], frozenset[str]],
                   first_at: float) -> None:
        """A sign read once more, where it is all there to be read and has not been read enough already: past where
        it was first seen (``first_at``) by as much as the widest sign seen is wide, and short of where they go."""
        if readings and readings.most_common(1)[0][1] >= READ_ENOUGH:
            return
        if self.seen_to is not None and sign.along > self.seen_to - READ_SHORT_OF:
            return
        if sign.along < first_at + max(READ_FROM, self.widest):
            return
        keys = read(sign)
        if keys:
            readings[keys] += 1
