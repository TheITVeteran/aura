"""Taking up what somebody proposed: the reading, her answer, and the with that follows.

One utterance of somebody's goes through three steps here, in the order Clark
(1996) gives them. It is understood: what it asks of her is read from what
her model expects next (core/together/what_was_proposed.py). It is taken up or
not: when it proposes doing something together, her answer is read off her
record (core/together/her_answer.py). And if she joins, there is from then on
something the two of them are doing (core/together/a_with.py), which holds
until one of them leaves.

Her reply is not written here. What was proposed and what her record says
she wants travel to her model as a reading of the turn
(core/brain/what_she_is_given.py), beside the other readings and in the same
role, and she says it her way.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.together.a_with import HER, THEM, Together, get_together
from core.together.her_answer import Answer, her_answer
from core.together.what_was_proposed import Proposal, WhatItAsksOfHer

logger = logging.getLogger("Aura.Together")

__all__ = ["TakingItUp", "get_taking_it_up", "install"]

HEADER = "## WHAT WAS PROPOSED TO YOU AND WHAT YOUR RECORD SAYS YOU WANT"

#: What a with is of when what was proposed named nothing she can name.
BEING_TOGETHER = "being together"


class TakingItUp:
    def __init__(self, reader: WhatItAsksOfHer | None = None, *, record: Any = None, together: Together | None = None) -> None:
        self._reader = reader
        self._record = record
        self._together = together
        #: What was read of the newest thing somebody said, by what they said.
        self._this: dict[str, Any] = {}
        self._seeded = False

    # ── the parts, found when first needed ────────────────────────────

    def _reads(self) -> WhatItAsksOfHer:
        if self._reader is None:
            from core.brain.llm.what_she_expects_next import get_expects_next

            self._reader = WhatItAsksOfHer(get_expects_next().score)
        return self._reader

    def _her_record(self) -> Any:
        if self._record is None:
            from core.conation.what_she_wants import get_what_she_wants

            self._record = get_what_she_wants()
        return self._record

    def _withs(self) -> Together:
        if self._together is None:
            self._together = get_together()
        return self._together

    def _seed(self) -> None:
        """Set the reader against what people have already said to her, once."""
        if self._seeded:
            return
        reader = self._reads()
        if reader.heard_before() > 0:
            self._seeded = True
            return
        try:
            from core.conversation.persistence import get_persistence

            stored = get_persistence().get_recent_history(limit=1024)
        # A store that will not read raises the database's own error as often as the OS's.
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError, sqlite3.Error) as exc:
            record_degradation(
                "together", exc, severity="warning",
                action="what an utterance asks of her is read against later ones only",
            )
            return
        said = [str(turn.get("content", "") or "") for turn in stored if str(turn.get("role", "")).lower() == "user"]
        folded = sum(1 for text in dict.fromkeys(said) if reader.remember(text))
        self._seeded = True
        logger.info("👂 What an utterance asks of her is set against %d things people have said to her.", folded)

    # ── an utterance of theirs ────────────────────────────────────────

    def read(self, said: str, *, who: str, at: float | None = None) -> dict[str, Any]:
        """Read what `said` asks of her, answer it if it is hers to answer, and keep the with in step.

        Waits on her fast model. Call it off the event loop.
        """
        at = time.time() if at is None else at
        said = " ".join(str(said or "").split())
        if not said:
            return {}
        self._seed()
        proposal: Proposal = self._reads().heard(said)
        record = self._her_record()
        about = [f"topic:{term}" for term in record.talk.subjects_of(said)]
        withs = self._withs()
        held = withs.current(who)
        answer: Answer | None = None
        if held is not None:
            held.said(THEM, at, about=about, until=at)
        if proposal.kind == "join":
            answer = her_answer(record, who, about)
            if answer.joins and held is None:
                what = about[0] if about else BEING_TOGETHER
                held = withs.begin(who, what, proposed_by=THEM, at=at)
                held.said(THEM, at, about=about, until=at)
                record.started(who, f"with:{what}", label=who.split(":", 1)[-1])
        read = {
            "said": said,
            "who": who,
            "proposal": proposal.as_dict(),
            "answer": answer.as_dict() if answer is not None else None,
            "with": held.status(at) if held is not None and held.open else None,
        }
        self._this = read
        logger.info(
            "🤝 Read as %s (%.2f)%s%s.",
            proposal.kind or "nothing",
            proposal.sure,
            "" if answer is None else f"; her answer: {'joins' if answer.joins else 'asks what it is' if answer.joins is None else 'not now'}",
            "" if not read["with"] else f"; with {who} in {read['with']['what']}",
        )
        return read

    async def heard(self, said: str, *, who: str, at: float | None = None) -> dict[str, Any]:
        """`read`, off the event loop. Never raises: an utterance not read is answered as it would have been."""
        try:
            return await asyncio.to_thread(self.read, said, who=who, at=at)
        except (ImportError, ArithmeticError, LookupError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "together", exc, severity="warning",
                action="one utterance was not read for what it asks of her",
            )
            return {}

    def left(self, who: str, *, by: str = THEM, at: float | None = None) -> bool:
        """One of them left. Said by whatever surface the with is on: a window closed, a film over, her going.

        Not read from what was said. On her fast model a parting is not told
        from a greeting well enough to end something on
        (core/together/what_was_proposed.py).
        """
        held = self._withs().end(who, by=by, at=at)
        if held is None:
            return False
        self._her_record().stopped(f"with:{held.what}", who)
        logger.info("🤝 The with with %s in %s is over; %s left.", who, held.what, by)
        return True

    # ── her turn ──────────────────────────────────────────────────────

    def she_said(self, said: str, *, who: str, at: float | None = None, unprompted: bool = False) -> None:
        """Her turn in the with she is in with `who`, if she is in one."""
        held = self._withs().current(who)
        if held is None:
            return
        at = time.time() if at is None else at
        about = [f"topic:{term}" for term in self._her_record().talk.subjects_of(str(said or ""))]
        held.said(HER, at, about=about, until=at, unprompted=unprompted)

    def of_this_turn(self, said: str) -> str:
        """The reading her model is handed when what was said proposed doing something together."""
        read = self._this
        if not read or " ".join(str(said or "").split()) != read.get("said"):
            return ""
        answer, proposal = read.get("answer"), read.get("proposal") or {}
        if not answer or proposal.get("kind") != "join":
            return ""
        wants = {
            True: "joining is worth more to you than what you would leave for it",
            False: "what you would leave for it is worth more to you than joining",
            None: "nothing you have met tells joining from staying apart",
        }[answer["joins"]]
        lines = [
            HEADER,
            f"- What was said proposes doing something together (read at {proposal.get('sure', 0.0):.2f} "
            "from which answers come next after it).",
            f"- Your record: {wants}.",
            f"- {answer['why']}.",
        ]
        if answer["leaving"]:
            lines.append(f"- You are in the middle of: {answer['leaving']}.")
        if answer["unknown"]:
            lines.append(f"- New to you: {', '.join(key.split(':', 1)[-1] for key in answer['unknown'])}.")
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        return {"this_turn": dict(self._this), "withs": self._withs().status()}


def who_it_is(person: str, sent_from: str) -> str:
    """The key her record keeps a person under: by name, or by where they write from."""
    return f"person:{str(person or '').strip() or 'someone at ' + str(sent_from or '').strip()}"


#: Readings under way, held so they are not collected before they finish.
_HEARING: set[Any] = set()


def who_she_is_with(record: Any) -> str:
    """Who a turn with no name on it is from: whoever she is talking with, or the last person she met."""
    people = [thing for thing in getattr(record, "_things", {}).values() if thing.kind == "person"]
    if not people:
        return who_it_is("", "")
    talking = [thing for thing in people if thing.under_way]
    return max(talking or people, key=lambda thing: (thing.last, thing.key)).key


def hear_in_the_background(said: str, *, person: str = "", sent_from: str = "") -> None:
    """Start reading what somebody said while the turn it belongs to is prepared.

    The reading waits on her fast model for about half a second, and her
    reply is seconds away, so it runs beside the turn and is there when her
    model is handed the turn. A turn that outruns it is answered without it.
    A turn that arrives with no name and no place on it, as one through the
    kernel does, is from whoever she is with.
    """
    taking = install()
    try:
        who = who_it_is(person, sent_from) if (person or sent_from) else who_she_is_with(taking._her_record())
        task = asyncio.get_running_loop().create_task(taking.heard(said, who=who), name="what_it_asks_of_her")
    except (RuntimeError, ImportError, AttributeError, OSError) as exc:
        record_degradation(
            "together", exc, severity="warning",
            action="one utterance was not read for what it asks of her",
        )
        return
    _HEARING.add(task)
    task.add_done_callback(_HEARING.discard)


_TAKING: TakingItUp | None = None


def get_taking_it_up() -> TakingItUp:
    global _TAKING
    if _TAKING is None:
        _TAKING = TakingItUp()
    return _TAKING


def install() -> TakingItUp:
    """Let the reading of a turn travel to her model with the other readings."""
    from core.brain import what_she_is_given

    taking = get_taking_it_up()
    if taking.of_this_turn not in what_she_is_given.OF_THIS_TURN:
        what_she_is_given.OF_THIS_TURN.append(taking.of_this_turn)
    return taking
