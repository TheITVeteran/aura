"""What meeting a kind does, by what she is when she meets it: a thing worth meeting in one state of hers and costly in
another.

A crystal the colour of the top of her tower adds to it; one of another colour blows it apart. A ghost is to be kept
clear of, until she has eaten the thing that turns the tables, and then it is worth chasing. Learned as one stance a
kind, such a thing is a mess of gains and losses that settles nowhere. A person sees what was different the times it
went well, and keeps to that.

So each meeting is kept with what she was at that moment, in two plain terms that hold in any game:

- whether the thing looked like her: its colour close to the colour of her own thing or of what goes with her (the
  crystals already stacked on her base, what she carries);
- which kinds of thing she had met in the few seconds before.

Where one such term splits a kind's meetings cleanly, gains on one side and losses on the other, the kind is met or
kept clear of by that term as it stands now. Where none does, the kind is left to its one stance, as before
(core/agency/what_meeting_things_does.py). Nothing here knows a colour, a game or a power.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = ["MeetingByWhatSheIs", "what_she_is"]

#: How near, on its most different channel, a thing's colour must be to one of hers to look like her.
LIKE_HER = 48

#: How long after meeting a kind it still counts as met lately, in seconds.
LATELY_S = 8.0

#: The fewest meetings on each side of a term for it to say anything, and the most of the other outcome either side
#: may hold, as a share of it.
LEAST_EACH_SIDE = 2
STRAY_SHARE = 0.2

GOOD, BAD = "meet", "avoid"


def _like(a: Sequence[int], b: Sequence[int]) -> bool:
    return max(abs(int(x) - int(y)) for x, y in zip(a, b, strict=False)) <= LIKE_HER


def what_she_is(colour: Sequence[int], her_colours: Iterable[Sequence[int]], met_lately: Iterable[int]) -> frozenset[str]:
    """The terms that hold of her for a thing of ``colour``: "like her", and "after <kind>" for each kind met lately."""
    terms = {f"after {kind}" for kind in met_lately}
    if any(_like(colour, hers) for hers in her_colours):
        terms.add("like her")
    return frozenset(terms)


@dataclass
class MeetingByWhatSheIs:
    """For each kind, its meetings: the terms that held of her, and whether it went better or worse."""

    meetings: dict[int, list[tuple[frozenset[str], float]]] = field(default_factory=lambda: defaultdict(list))
    #: The kinds she met, and when, for "after <kind>".
    lately: deque = field(default_factory=lambda: deque(maxlen=40))

    def met(self, kind: int, at: float) -> None:
        """She met a thing of ``kind`` at ``at``."""
        self.lately.append((at, kind))

    def met_lately(self, at: float, besides: int | None = None) -> set[int]:
        return {kind for when, kind in self.lately if 0.0 < at - when <= LATELY_S and kind != besides}

    def settled(self, kind: int, terms: frozenset[str], better: float) -> None:
        """A meeting of ``kind`` came to ``better`` (above nought a gain, below a loss), with ``terms`` holding of her."""
        if better == 0:
            return
        kept = self.meetings[kind]
        kept.append((terms, better))
        del kept[:-60]

    def stance(self, kind: int, terms_now: frozenset[str]) -> str | None:
        """Meet or keep clear of ``kind``, by the term that splits its meetings, as that term holds of her now; None
        where no term splits them."""
        kept = self.meetings.get(kind) or []
        if len(kept) < 2 * LEAST_EACH_SIDE:
            return None
        for term in sorted({t for terms, _ in kept for t in terms}):
            with_it = [better for terms, better in kept if term in terms]
            without = [better for terms, better in kept if term not in terms]
            way = _splits(with_it, without)
            if way is None:
                continue
            good_with = way
            return GOOD if (term in terms_now) == good_with else BAD
        return None


def _splits(with_it: list[float], without: list[float]) -> bool | None:
    """Whether a term splits meetings into gains on one side and losses on the other: True where gains go with it,
    False where losses do; None where it does not split them."""
    if len(with_it) < LEAST_EACH_SIDE or len(without) < LEAST_EACH_SIDE:
        return None

    def mostly(values: list[float], gains: bool) -> bool:
        same = sum(1 for v in values if (v > 0) == gains)
        return same >= LEAST_EACH_SIDE and len(values) - same <= STRAY_SHARE * len(values)

    if mostly(with_it, True) and mostly(without, False):
        return True
    if mostly(with_it, False) and mostly(without, True):
        return False
    return None
