"""Whether to say something now, inside a shared activity.

Alone at his desk, a person is interrupted by a remark, and the bar for one
is high (core/perception/ambient_utterance.py; core/agency/ambient_governor.py).
Watching a film with somebody he asked to watch it with him, a remark is part
of what they are doing. The question is the same one both times, whether
speaking now is worth more than staying quiet, and what changes is every
number in it.

Horvitz (1999) put the choice for a system that may act unasked as an
expected utility over whether the act is wanted. With p the chance a remark
is welcome now, v what the remark is worth, a what it costs to take his
attention when it is not welcome, and m the part of v lost by waiting:

    speak now   p v           - (1 - p) a
    wait        - p v m

so she speaks when  p v (1 + m)  >  (1 - p) a.

Each term is read off the with she is in (core/together/a_with.py):

    p   how her remarks have been taken up in this with, times how far into
        one of this pair's usual silences the present one is. Nothing while
        the other person is speaking: conversation runs with almost no
        overlap and with gaps near a fifth of a second, which people manage
        by planning their turn while the other is still talking and waiting
        for the end of it (Sacks, Schegloff and Jefferson 1974; Levinson and
        Torreira 2015).
    v   how much the thing interests her, and nothing if she has already
        said her piece about it.
    a   how much the shared thing is holding attention: how much it is
        changing now against how much it has been changing.
    m   given by whoever asks, since only they know whether the thing will
        still be there to point at.

An unknown is read as one half, which is what a chance is before anything
has been seen of it. When she is spoken to, none of this applies: an answer
is owed, and the only question is whether the floor is free.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.together.a_with import HER, THEM, With

__all__ = ["Say", "worth_saying_now"]

_UNKNOWN = 0.5


@dataclass(frozen=True)
class Say:
    speak: bool
    why: str
    welcome: float = 0.0
    worth: float = 0.0
    cost: float = 0.0
    #: The two sides of the comparison, so the margin can be read.
    for_speaking: float = 0.0
    against: float = 0.0


def worth_saying_now(
    held: With,
    now: float,
    *,
    interest: float,
    thing: str = "",
    lost_by_waiting: float = 0.0,
    spoken_to: bool = False,
) -> Say:
    """Whether a remark of this much interest about `thing` should be made now."""
    floor = held.speaking(now)
    if not held.open:
        return Say(False, "the with is over")
    if spoken_to:
        if floor == THEM:
            return Say(False, "spoken to, and they have not finished")
        return Say(True, "spoken to: an answer is owed")
    if floor == THEM:
        return Say(False, "they are speaking")
    if floor == HER:
        return Say(False, "she is already speaking")
    interest = max(0.0, min(1.0, float(interest)))
    if thing and held.she_has_said_her_piece(thing):
        return Say(False, "she has said her piece about it", worth=0.0)
    if interest <= 0.0:
        return Say(False, "nothing she has to say about it", worth=0.0)
    moment = held.moment(now)
    welcome = held.welcome() * (_UNKNOWN if moment is None else moment)
    eventful = held.eventful()
    cost = _UNKNOWN if eventful is None else eventful
    lost = max(0.0, min(1.0, float(lost_by_waiting)))
    for_speaking = welcome * interest * (1.0 + lost)
    against = (1.0 - welcome) * cost
    return Say(
        speak=for_speaking > against,
        why=(
            f"welcome {welcome:.3f} x worth {interest:.3f} x (1 + {lost:.2f} lost by waiting) "
            f"= {for_speaking:.3f} against (1 - welcome) x attention held {cost:.3f} = {against:.3f}"
        ),
        welcome=welcome,
        worth=interest,
        cost=cost,
        for_speaking=for_speaking,
        against=against,
    )
