"""Which thing goes where the pointer goes: the pointer's half of finding which thing is hers.

A thing that goes where she takes the pointer is hers, the way a thing a key
moves is. Moved out of core/agency/which_one_answers_to_her.py, whose
`WhichIsHers` takes this as a base: the class had grown past the number of
methods a class may have. The state these read (the pointer's trail, what has
followed it, the belief and since when it was held) is set up by WhichIsHers.
"""
from __future__ import annotations

import bisect
import logging
import math
from typing import Any

from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant

logger = logging.getLogger("core.agency.which_one_answers_to_her")

#: The least a body the pointer moves measures on its longer side, in the picture's working pixels: a thin paddle is
#: long, a cursor's tip is small every way.
A_BODY_PX = 6

__all__ = ["FollowsThePointer"]

#: How many pictures of a thing make a sighting, where the place's words say the pointer moves her.
TOLD_SAMPLES = 12


def _told_the_pointer_moves_her() -> bool:
    """Whether the guide to the place holds that its pointer moves the player's own thing."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    pointer = getattr(THE_GUIDE.get(), "pointer", None) or {}
    return bool(pointer.get("moves"))


def _the_kind_her_eyes_took() -> int | None:
    """The kind of thing her eyes took for her in the place before her, where they have (core/cognition/what_things_are.py)."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    mine = getattr(getattr(THE_GUIDE.get(), "seen", None), "mine", None)
    return mine[0] if isinstance(mine, tuple) and mine else None


class FollowsThePointer:
    """Where the pointer has been, and the thing that goes with it."""

    def pointed(self, x: float, y: float, at: float) -> None:
        """The pointer was taken to (x, y), in working pixels."""
        self._pointer.append((at, x, y))
        del self._pointer[:-400]

    def _pointer_at(self, at: float) -> tuple[float, float] | None:
        index = bisect.bisect_right(self._pointer, (at, math.inf, math.inf))
        return self._pointer[index - 1][1:] if index else None

    def _with_the_pointer(self, seen: list[tuple[float, float, float]], extent: tuple[float, float] | None) -> tuple[bool | None, bool | None]:
        """Along which axes places seen go with where the pointer was, at whichever delay fits best.

        How long a page takes to answer the pointer is the page's own: one
        follows within a picture, another a few later. Matched at the wrong
        delay, a paddle that goes exactly where the mouse goes lags its
        pointer at every turn and falls short of the test (offline
        2026-10-04, bounce). Each delay is a separate try; the best stands.
        """
        from core.agency.which_one_answers_to_her import (
            POINTER_LAGS_S,
            _chases_along,
            _measured_along,
        )

        best: list[bool | None] = [None, None]
        for lag in POINTER_LAGS_S:
            pairs = [(p[0], p[1], x, y) for at, x, y in seen if (p := self._pointer_at(at - lag)) is not None]
            if len(pairs) < (TOLD_SAMPLES if _told_the_pointer_moves_her() else 20):
                continue
            for axis, value in enumerate(_measured_along(pairs, extent)):
                if value or (value is False and best[axis] is None):
                    best[axis] = value
            # Or it heads for where the pointer is at a pace of its own (eased, or kept to a lane).
            if lag == 0.0:
                for axis, value in enumerate(_chases_along(pairs)):
                    if value:
                        best[axis] = True
        return best[0], best[1]

    def _what_follows_the_pointer(self, moves: Any, at: float) -> None:
        """A thing that goes where the pointer went is hers, the way a key's thing is."""
        from core.agency.which_one_answers_to_her import _extent

        if not self._pointer or self._pointer_at(at) is None:
            return
        if self.follows_pointer:
            self._still_follows(moves, at)
            return
        # Offline 2026-10-12 a stream passing the pointer supplied a trail
        # from successive nearest objects. Each object's own trail must
        # answer; a shared appearance cannot supply continuity.
        current = {}
        for thing in moves.things.values():
            if not motion_was_measured(thing, at) or not thing.moved or thing.number in self.not_mine:
                continue
            current[(thing.number, getattr(thing, 'born', None))] = thing
        for identity in list(self._followed):
            if identity not in current:
                del self._followed[identity]
                if self._sighted == identity:
                    self._sighted = None
        for identity, thing in current.items():
            seen = self._followed[identity]
            if seen and not 0 < at - seen[-1][0] <= .35:
                seen.clear()
                if self._sighted == identity:
                    self._sighted = None
            seen.append((at, thing.x, thing.y))
            del seen[:-60]
        self._looked += 1
        if self._looked % 4:
            return
        # Of the kinds that go with the pointer, the biggest: a speck sitting on it is the game's own cursor, and the
        # body the pointer moves (a boat, a basket) is bigger. LIVE 2026-10-11 a 3x2 thing was taken for her in a boat
        # game, and the boat was never steered.
        following = []
        # Where the place's words say the pointer moves her ("use the mouse to move Mac"), a sighting is a finding, and
        # fewer pictures make one: offline 2026-10-11 an egg-catching game's rounds were lost in under ten seconds
        # each, while the two sightings asked of what might be chance were still being made.
        told = _told_the_pointer_moves_her()
        for identity, seen in self._followed.items():
            thing = current[identity]
            if len(seen) < (TOLD_SAMPLES if told else 20) or max(thing.w, thing.h) < A_BODY_PX:
                continue
            along = tuple(bool(v) for v in self._with_the_pointer(seen, _extent(moves)))
            if any(along):
                following.append((float(thing.w) * float(thing.h), identity, thing, seen, along))
        # The parts of a body move together, and a thing it holds moves with it until it is let go: the part her eyes
        # took for her, told by the place's words who she is, comes before the biggest. Offline 2026-10-11 the ball a boy
        # held was bigger than any one part of him her eyes followed, was taken for her, and was thrown.
        eyed = _the_kind_her_eyes_took()
        for _size, identity, thing, seen, along in sorted(following, key=lambda one: (one[2].kind != eyed, -one[0]))[:1]:
            # Once is a sighting; the same again on pictures taken since is a
            # finding. Tested every few pictures at a few delays, chance
            # agreement turns up now and then (offline 2026-10-04, the other
            # paddle during a long Pong game), and seldom twice running.
            if self._sighted != identity and not told:
                self._sighted = identity
                seen[:] = seen[-1:]
                continue
            self.number, self.kind, self.follows_pointer = thing.number, thing.kind, True
            self.follows_along = along
            logger.info(
                "follows the pointer: thing %s of kind %s at (%.0f, %.0f), %dx%d, along %s, over %d pictures",
                thing.number, thing.kind, thing.x, thing.y, thing.w, thing.h, along, len(seen),
            )
            return

    def _still_follows(self, moves: Any, at: float) -> None:
        """Whether her thing still goes where she points, measured as she plays; if not, it was never hers.

        A belief that a thing follows the pointer is tested by every move she
        makes with it. Where she has since taken the pointer both ways along an
        axis and her thing did not go with it, the belief is dropped and which
        thing is hers is found again.
        """
        from core.agency.which_one_answers_to_her import _extent

        mine = moves.things.get(self.number) if self.number is not None else None
        if mine is None or not motion_was_measured(mine, at):
            self._since_believed.clear()
            return
        pairs = self._since_believed
        if pairs and not 0 < at - pairs[-1][0] <= .35:
            pairs.clear()
        pairs.append((at, mine.x, mine.y))
        del pairs[:-60]
        self._looked += 1
        if len(pairs) < 30 or self._looked % 4:
            return
        measured = self._with_the_pointer(pairs, _extent(moves))
        believed = [m for m, b in zip(measured, self.follows_along, strict=True) if b and m is not None]
        if believed and not any(believed):
            self.follows_pointer, self.follows_along = False, (False, False)
            self._followed.clear()
            pairs.clear()
            self.number = None

    def _under_the_pointer(self, moves: Any, at: float) -> int | None:
        """Recover her measured body within reach of its last sighting."""
        number = self._one_of_her_kind(moves, at)
        return number if number is not None and motion_was_measured(moves.things[number], at) else None


@invariant("agency.pointer_control_belongs_to_one_observed_body", scope="agency",
           owner="core/agency/what_follows_the_pointer.py", observational=False)
def _pointer_body_continuity_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    for step in range(100):
        at, x = step * .04, 120. + 70. * math.sin(step / 2.)
        hers.pointed(x, 80., at)
        thing = SimpleNamespace(number=step // 10, born=0., kind=3, seen=at, moved=True,
                                x=x, y=80., w=12., h=16.)
        hers._what_follows_the_pointer(SimpleNamespace(things={thing.number: thing}, shape=(160, 240)), at)
        assert hers.number is None
    return ()
