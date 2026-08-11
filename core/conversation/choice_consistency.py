"""A reply that commits to a choice must name the same choice every time.

LIVE, 2026-08-10. Asked to pick one and commit, no hedging — "would you rather
lose your memory of the last month, or lose the ability to form new memories
for the next month?" — she answered:

    "Losing the ability to form new memories for the next month would be worse.
     I rely on incremental updates of state ... losing them for a month would be
     catastrophic. ...
     To summarize: I prefer losing my ability to form new memories for one
     month, as it would be more inconvenient than losing a few weeks of memory."

She opens by calling one option worse, argues for four sentences that it would
be catastrophic, and then commits to it. The summary names the opposite of what
the reasoning selected, and nothing noticed.

This is the identity-coherence failure in its cleanest form: not a wrong
answer, an unstable one. A forced choice is exactly the shape where stability
is the whole content of the reply, and it is also the shape where the check is
mechanical — extract the option each commitment sentence names, and require
that they agree.

Two polarities, because a choice can be stated either way:

    selecting   "I prefer X", "I'd rather X", "I'd choose X"
    rejecting   "X would be worse", "X would be catastrophic"

Rejecting one option in a two-option choice selects the other, so both forms
normalise to the same thing: which option this sentence lands on. Then any two
commitment sentences that land on different options are a contradiction, and
the reply cannot be served as a committed answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ChoiceContradiction",
    "extract_offered_options",
    "find_choice_contradiction",
    "looks_like_forced_choice",
]

#: A question that demands one option out of an explicit set.
_FORCED_CHOICE_RE = re.compile(
    r"\b(?:would\s+you\s+rather|pick\s+one|choose\s+one|which\s+would\s+you|"
    r"commit\s+to\s+one|one\s+or\s+the\s+other|either\s+.{2,80}?\s+or\b)",
    re.IGNORECASE,
)

#: Sentences in which the reply lands on an option.
_SELECTING_RE = re.compile(
    r"\b(?:i(?:'d| would)\s+(?:rather|prefer|choose|pick|take)|i\s+prefer|"
    r"i\s+choose|i\s+pick|my\s+(?:choice|answer)\s+is|i(?:'d| would)\s+go\s+with|"
    r"i\s+would\s+lose)\b",
    re.IGNORECASE,
)
_REJECTING_RE = re.compile(
    r"\b(?:would\s+be\s+worse|is\s+worse|would\s+be\s+catastrophic|"
    r"would\s+be\s+harder|would\s+cost\s+more|i\s+could\s+not\s+accept|"
    r"would\s+be\s+the\s+greater\s+loss)\b",
    re.IGNORECASE,
)

_OPTION_SPLIT_RE = re.compile(r",\s*or\s+|\s+or\s+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "a", "an", "and", "the", "to", "of", "for", "my", "your", "the", "would",
    "you", "i", "it", "that", "this", "be", "is", "are", "with", "on", "in",
    "next", "last", "one", "lose", "losing", "rather", "or", "ability", "no",
})


@dataclass(frozen=True, slots=True)
class ChoiceContradiction:
    """Two commitment sentences that land on different options."""

    first_sentence: str
    first_option: str
    second_sentence: str
    second_option: str

    def describe(self) -> str:
        return (
            f"commits to {self.first_option!r} in {self.first_sentence.strip()!r} "
            f"and to {self.second_option!r} in {self.second_sentence.strip()!r}"
        )


def looks_like_forced_choice(question: Any) -> bool:
    """True when the turn demands one option and offers at least two."""

    text = str(question or "").strip()
    if not text or not _FORCED_CHOICE_RE.search(text):
        return False
    return len(extract_offered_options(text)) >= 2


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        word
        for word in re.findall(r"[a-z]+", str(text or "").lower())
        if word not in _STOPWORDS and len(word) > 2
    )


def extract_offered_options(question: Any) -> tuple[str, ...]:
    """The alternatives the question put on the table."""

    text = str(question or "")
    # The choice lives in the clause containing the disjunction, not the whole
    # message — "no hedging" and "tell me why" are not options.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not re.search(r"\bor\b", sentence, re.IGNORECASE):
            continue
        body = sentence
        # Strip every lead-in, not just the leftmost: "Pick one: would you
        # rather lose ..." carries two, and stopping at the first leaves
        # "would you rather" glued to the first option's label.
        for _ in range(3):
            stripped_body = re.sub(
                r"^.*?\b(?:would\s+you\s+rather|pick\s+one[:,]?|choose\s+one[:,]?|either)\b",
                "",
                body,
                flags=re.IGNORECASE,
            )
            if stripped_body == body:
                break
            body = stripped_body
        body = body.strip(" ?.!,:")
        parts = [p.strip(" ?.!,:") for p in _OPTION_SPLIT_RE.split(body)]
        parts = [p for p in parts if len(_content_tokens(p)) >= 1]
        if len(parts) >= 2:
            return tuple(parts[:4])
    return ()


def _option_for(sentence: str, options: tuple[str, ...]) -> str:
    """Which option this sentence is about, by content-word overlap."""

    sentence_tokens = _content_tokens(sentence)
    best, best_score = "", 0
    for option in options:
        score = len(sentence_tokens & _content_tokens(option))
        if score > best_score:
            best, best_score = option, score
    return best if best_score >= 2 else ""


def find_choice_contradiction(
    question: Any, reply: Any
) -> ChoiceContradiction | None:
    """The two commitment sentences that disagree, if the reply has any.

    Rejecting one option of a two-option choice is selecting the other, so both
    forms are normalised to the option the sentence lands ON before comparing.
    """

    options = extract_offered_options(question)
    if len(options) != 2:
        return None
    landings: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(reply or "")):
        stripped = sentence.strip()
        if not stripped:
            continue
        selecting = bool(_SELECTING_RE.search(stripped))
        rejecting = bool(_REJECTING_RE.search(stripped))
        if not (selecting or rejecting):
            continue
        named = _option_for(stripped, options)
        if not named:
            continue
        if rejecting and not selecting:
            # Rejecting one of two selects the other.
            named = options[0] if named == options[1] else options[1]
        landings.append((stripped, named))
    for index, (sentence, option) in enumerate(landings):
        for other_sentence, other_option in landings[index + 1 :]:
            if option != other_option:
                return ChoiceContradiction(
                    first_sentence=sentence,
                    first_option=option,
                    second_sentence=other_sentence,
                    second_option=other_option,
                )
    return None
