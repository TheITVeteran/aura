"""Whether she joins what somebody proposed, from what she wants.

An invitation is answered by the one invited, and "not now" is one of its
answers. Bryan's three: "Sure!", "What video? May be interested.", "Nah, I'm ok
for now." Which of them is hers to say is read off her record
(core/conation/what_she_wants.py), and her model is not asked.

What joining would be worth is what her record predicts of a turn with those
things and that person in it: the sum of her share in what each has been
worth, the same sum the record predicts every turn with before it learns from
it. What staying would be worth is what the thing she is in the middle of has
been worth while it lasted, or what she had decided to do next is worth to
her, and nothing more than a usual turn if she is free.

Each side is counted with what she is still unsure of in its favour, the
square root of its variance, which is how her wants are already ranked
(core/conation/what_she_wants.py) and how a choice made again and again comes
to know what it has not tried (Auer, Cesa-Bianchi and Fischer 2002). The
record's variance falls fast once her turns are predictable, so this counts
for little after a few turns with anybody; it is the same rule, not a thumb
on the scale for joining.

She joins when joining is worth more, says not now when staying is, and asks
what it is when her record cannot tell them apart: nothing proposed is
anything she has met, and the person alone does not settle it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["Answer", "her_answer"]


@dataclass(frozen=True)
class Answer:
    #: True to join, False for not now, None when she has to ask what it is.
    joins: bool | None
    worth_of_joining: float
    worth_of_staying: float
    #: What was proposed that she has never met.
    unknown: tuple[str, ...]
    #: What she would be leaving to join, if anything.
    leaving: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "joins": self.joins,
            "worth_of_joining": round(self.worth_of_joining, 4),
            "worth_of_staying": round(self.worth_of_staying, 4),
            "unknown": list(self.unknown),
            "leaving": self.leaving,
            "why": self.why,
        }


def _in_the_middle_of(record: Any, who: str) -> tuple[str, float, float]:
    """What she would be leaving, what it has been worth to her while it lasted, and how unsure that is."""
    best_key, best, unsure = "", 0.0, 0.0
    for thing in list(getattr(record, "_things", {}).values()):
        if thing.key == who or not (thing.under_way or thing.open):
            continue
        worth = thing.during.liking if thing.during.times else thing.liking
        if not best_key or worth + math.sqrt(thing.unsure) > best + math.sqrt(unsure):
            best_key, best, unsure = thing.key, worth, thing.unsure
    decided = record.decided()
    if decided:
        key = str(decided.get("thing") or "")
        held = record.thing(key)
        worth = max(0.0, held.liking) if held is not None else 0.0
        doubt = held.unsure if held is not None else 0.0
        if key and key != who and (not best_key or worth + math.sqrt(doubt) > best + math.sqrt(unsure)):
            best_key, best, unsure = key, worth, doubt
    return best_key, best, unsure


def her_answer(record: Any, who: str, things: Iterable[str]) -> Answer:
    """Her answer to a proposal to do `things` together with `who`."""
    things = [str(key) for key in dict.fromkeys(things) if str(key).strip() and str(key) != who]
    met = [record.thing(key) for key in things]
    unknown = tuple(key for key, thing in zip(things, met) if thing is None or thing.contacts <= 0)
    known = [thing for thing in met if thing is not None and thing.contacts > 0]
    person = record.thing(who) if who else None
    parts = [(thing.liking, thing.unsure) for thing in known]
    if person is not None and person.contacts > 0:
        parts.append((person.liking, person.unsure))
    joining = max(-1.0, min(1.0, sum(liking for liking, _unsure in parts)))
    leaving, staying, staying_unsure = _in_the_middle_of(record, who)
    hope_in_joining = joining + math.sqrt(sum(unsure for _liking, unsure in parts))
    hope_in_staying = staying + math.sqrt(staying_unsure)
    disliked = [thing.label or thing.key for thing in known if record.known(thing.key) is False]
    said = (
        f"her record predicts {joining:+.3f} for a turn with what was proposed"
        f"{' and ' + who if person is not None and person.contacts > 0 else ''}, against "
        f"{staying:+.3f} for {leaving or 'a usual turn'}; with what she is unsure of counted for each, "
        f"{hope_in_joining:+.3f} against {hope_in_staying:+.3f}"
    )
    if not parts:
        if leaving and staying > 0.0:
            # Nothing known of what was proposed, and what she is in the
            # middle of has been worth more than a usual turn.
            return Answer(False, joining, staying, unknown, leaving, said)
        return Answer(
            None, joining, staying, unknown, leaving,
            said + "; nothing she has met tells them apart" + (f" ({', '.join(unknown)} is new to her)" if unknown else ""),
        )
    if hope_in_joining > hope_in_staying and not disliked:
        return Answer(True, joining, staying, unknown, leaving, said)
    if hope_in_joining < hope_in_staying or disliked:
        why = said + (f"; she has found she dislikes {', '.join(disliked)}" if disliked else "")
        return Answer(False, joining, staying, unknown, leaving, why)
    return Answer(
        None, joining, staying, unknown, leaving,
        said + "; nothing she has met tells them apart" + (f" ({', '.join(unknown)} is new to her)" if unknown else ""),
    )
