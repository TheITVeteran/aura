"""What an utterance asks of her, read from what her own model expects to come next.

Bryan, 13 October 2026: "how does Aura even intrinsically know the difference
between a task and a command ... I can ask Aura 'hey, watch this youtube video
for 10 minutes and comment on it as you watch.' ... but that's different from
asking 'Hey do you want to watch this video with me?'"

Conversation is built of pairs: a question and its answer, an invitation and
its acceptance or refusal, a request and its granting. What kind of thing a
first part is shows in which second parts it makes relevant, and a second part
that does not fit is heard as missing (Schegloff and Sacks 1973; Schegloff
2007). After "do you want to watch this with me?" "I'd love to" is in order
and "done" is not. So what an utterance asks of her can be read without
reading its words: by how likely each kind of answer is as the next turn.

Her language model already holds that. It is used here as an instrument and
told nothing: it is handed the bare transcript and asked for no judgement.
For each kind, a few second parts only that kind makes relevant are scored as
her next turn. A second part also has a likelihood of its own that has nothing
to do with what was said (short ones score higher, "Oh" higher than "Count me
in"), so each is read against what it usually scores after the things people
have said to her, the way a subject of her talk is read against her talk
(core/conation/what_her_talk_is_about.py; Holtzman et al. 2021 on surface form
competition). The kind is the one whose second parts gained most.

Measured 13 October on her fast model with no system text and her 120 most
recent stored messages as the reference, over 24 sentences written before the
run: 19 read as intended, and of the 12 that were an invitation or a request,
10. tools/what_an_utterance_asks_of_her.py holds a second set kept apart from
the reader. The kinds:

    join    a proposal to do something together; hers to accept or refuse
    do      a request that she do something for the speaker
    ask     a question to be answered
    tell    something told or shared; what is owed is attention and a stance
    greet   a greeting or a parting

A sixth kind for partings alone was tried the same day and dropped. On the fast
model a greeting is returned in its own form and bad news is answered with
"take care", so "Good evening." and "My dog died last night." read as partings,
and a parting misread would end a shared activity. A with is ended by whatever
surface it is on knowing that somebody left (core/together/taking_it_up.py).

Understanding what was proposed is one step and taking it up is another: she
can refuse though she has understood (Clark 1996). This module is the first
step. core/together/her_answer.py is the second.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["KINDS", "SECONDS", "Proposal", "WhatItAsksOfHer"]

KINDS: tuple[str, ...] = ("join", "do", "ask", "tell", "greet")

#: Second parts each kind makes relevant and the others do not. An acceptance
#: that is also a way of granting a request ("Sure") is left out: it would
#: score for both and tell them apart for neither.
SECONDS: Mapping[str, tuple[str, ...]] = {
    "join": ("I'd love to.", "I'd like that.", "Count me in.", "Not right now.", "Maybe later.", "What is it?"),
    "do": ("Okay, doing that now.", "On it.", "Done.", "Here you go.", "I can't do that."),
    "ask": ("I think", "It is", "Because", "It depends"),
    "tell": ("Oh, really?", "That's lovely.", "Oh no.", "That sounds"),
    "greet": ("Hi!", "Hello.", "Hey.", "Bye!"),
}

#: How many of the things said to her a second part's usual score is taken
#: over; the window her record reads her own history over.
_WINDOW = 256

Scorer = Callable[[str, Sequence[str]], Sequence[float]]


@dataclass(frozen=True)
class Proposal:
    """What one utterance asks of her."""

    said: str
    #: The kind whose second parts gained most, or "" when nothing could be read.
    kind: str = ""
    #: How much of the reading that kind holds, in [0, 1].
    sure: float = 0.0
    by_kind: Mapping[str, float] = field(default_factory=dict)
    #: How many utterances the second parts' usual scores were taken over.
    heard_before: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sure": round(self.sure, 4),
            "by_kind": {kind: round(share, 4) for kind, share in self.by_kind.items()},
            "heard_before": self.heard_before,
        }


def _log_mean_exp(values: Sequence[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values) / len(values))


class WhatItAsksOfHer:
    """Reads each utterance against what the second parts usually score.

    `scorer(said, continuations)` returns the log-likelihood of each
    continuation as her next turn after `said`, whole, in nats.
    """

    def __init__(self, scorer: Scorer, stored: Mapping[str, Any] | None = None) -> None:
        self._scorer = scorer
        self._lock = checked_lock("together.what_it_asks_of_her")
        self._probes: tuple[tuple[str, str], ...] = tuple(
            (kind, second) for kind in KINDS for second in SECONDS[kind]
        )
        #: The running mean likelihood of each second part, and over how many.
        self._usual: dict[str, float] = {}
        self._heard = 0
        stored = dict(stored or {})
        self._heard = max(0, int(stored.get("heard", 0)))
        for second, mean in dict(stored.get("usual", {})).items():
            try:
                self._usual[str(second)] = max(0.0, float(mean))
            # not a failure: a stored mean that is not a number is one second
            # part with no history, which is where it starts anyway.
            except (TypeError, ValueError):
                continue

    def held(self) -> dict[str, Any]:
        with self._lock:
            return {"heard": self._heard, "usual": dict(self._usual)}

    def heard_before(self) -> int:
        with self._lock:
            return self._heard

    def seconds(self) -> list[str]:
        """Every second part that is scored, in the order the scores are read in."""
        return [second for _kind, second in self._probes]

    def _scores(self, said: str, scored: Sequence[float] | None = None) -> dict[str, float] | None:
        seconds = self.seconds()
        scored = list(self._scorer(said, seconds)) if scored is None else list(scored)
        if len(scored) != len(seconds) or not all(math.isfinite(float(s)) for s in scored):
            return None
        return {second: float(score) for second, score in zip(seconds, scored)}

    def _fold(self, scores: Mapping[str, float]) -> None:
        self._heard += 1
        rate = 1.0 / min(self._heard, _WINDOW)
        for second, score in scores.items():
            held = self._usual.get(second, 0.0)
            self._usual[second] = held + (math.exp(score) - held) * rate

    def remember(self, said: str, *, scored: Sequence[float] | None = None) -> bool:
        """Fold one earlier utterance into what the second parts usually score."""
        said = " ".join(str(said or "").split())
        if not said:
            return False
        scores = self._scores(said, scored)
        if scores is None:
            return False
        with self._lock:
            self._fold(scores)
        return True

    def heard(self, said: str, *, scored: Sequence[float] | None = None) -> Proposal:
        """Read what `said` asks of her, then count it among what she has heard.

        `scored` is the scores when somebody else took them, in the order of
        `seconds()`: on the desktop the worker that owns the model does.
        """
        said = " ".join(str(said or "").split())
        if not said:
            return Proposal(said=said)
        scores = self._scores(said, scored)
        if scores is None:
            return Proposal(said=said, heard_before=self.heard_before())
        with self._lock:
            usual, before = dict(self._usual), self._heard
            self._fold(scores)
        if before <= 0 or any(usual.get(second, 0.0) <= 0.0 for second in scores):
            # With nothing to set a second part against, its score says only
            # how common it is.
            return Proposal(said=said, heard_before=before)
        gained = {
            kind: _log_mean_exp([scores[second] - math.log(usual[second]) for second in SECONDS[kind]])
            for kind in KINDS
        }
        top = max(gained.values())
        total = sum(math.exp(value - top) for value in gained.values())
        by_kind = {kind: math.exp(value - top) / total for kind, value in gained.items()}
        kind = max(by_kind, key=lambda name: (by_kind[name], name))
        return Proposal(said=said, kind=kind, sure=by_kind[kind], by_kind=by_kind, heard_before=before)
