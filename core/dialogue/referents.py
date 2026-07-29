"""Who "I" is, who "you" is, and who said the sentence being remembered.

Measured live on 2026-07-28. Bryan typed:

    It was the notes thing. I was trying to get you to write one about
    yourself in your own words. Like about who you are

Six turns later, Aura said:

    I was trying to get you to write a paragraph about yourself in your own
    words. Like an actual summary of who you are

She had not misunderstood him. She was *quoting* him — and the quote had lost
its speaker on the way through memory. Recall renders a snippet as::

    <recalled type="recent_episode" source="episodic">I was trying to get you
    to write one about yourself in your own words.</recalled>

There is no `speaker`. In her own context "I" means Aura and "you" means
Bryan, so that snippet does not read as *a thing Bryan asked*; it reads as *a
thing she intended*. Every downstream oddity followed from that one missing
attribute — she told Bryan what "Bryan asked" for as though he were a third
party, offered to write his self-summary, and when he said "I am Bryan"
answered "I know that" while still holding the swapped frame.

An unattributed first-person sentence is not neutral data. It is a claim about
the speaker, and in the absence of a label the reader supplies themselves.

So this module makes the frame explicit and carries it everywhere:

* :class:`ReferentFrame` — the standing bindings for one exchange: I am Aura,
  you are Bryan, "yourself" from Bryan means Aura.
* :func:`speaker_of` — recover the speaker from whatever metadata a memory
  source happens to carry, since every store spells it differently.
* :func:`attribute` — label a remembered utterance, and mark it
  ``unattributed`` when nothing knows who said it, rather than letting it
  pass as her own voice.
* :func:`resolve_second_person` — what "yourself" refers to in a request, so
  "write about yourself" reaches the planner meaning Aura.

The rule the whole module serves: **remembered speech carries its speaker, or
it carries the fact that its speaker is unknown. It never travels bare.**
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AURA",
    "OWNER",
    "UNATTRIBUTED",
    "ReferentFrame",
    "attribute",
    "borrowed_first_person_spans",
    "current_frame",
    "has_person_reference",
    "resolve_second_person",
    "speaker_of",
]

#: Canonical speaker tokens. Stores spell their roles a dozen ways; these
#: three are what the rest of the system reasons over.
AURA = "aura"
OWNER = "owner"
UNATTRIBUTED = "unattributed"

#: Every spelling of "this was the assistant" seen across the memory stores,
#: the message bus, and the transcript importers.
_AURA_ALIASES = frozenset(
    {
        "aura",
        "assistant",
        "self",
        "me",
        "ai",
        "model",
        "agent",
        "aura_response",
        "response",
        "reply",
    }
)

#: ...and of "this was the person she is talking to".
_OWNER_ALIASES = frozenset(
    {
        "user",
        "owner",
        "human",
        "person",
        "you",
        "bryan",
        "creator",
        "operator",
        "interlocutor",
        "user_message",
        "prompt",
        "request",
    }
)

#: Metadata keys that have carried a speaker in at least one store.
_SPEAKER_KEYS = (
    "speaker",
    "role",
    "author",
    "said_by",
    "from",
    "actor",
    "participant",
    "origin_role",
)

#: First- and second-person forms. Their referent flips with the speaker, so
#: their presence is exactly what makes an unattributed snippet dangerous.
_PERSON_RE = re.compile(
    r"(?i)\b(i|me|my|mine|myself|we|us|our|ours|ourselves|"
    r"you|your|yours|yourself|yourselves)\b"
)

#: "write about yourself", "describe yourself", "tell me about you".
_SECOND_PERSON_TARGET_RE = re.compile(
    r"(?i)"
    r"\b(?:about|describing|describe|regarding|concerning|of)\s+"
    r"(?:your\s?self|yourself|you)\b"
    r"|\byour\s+own\s+words\b"
    r"|\bwho\s+you\s+are\b"
)


def _canonical(value: Any) -> str:
    """Map any spelling of a speaker onto AURA / OWNER / UNATTRIBUTED."""
    token = str(value or "").strip().lower().replace("-", "_")
    if not token:
        return UNATTRIBUTED
    if token in _AURA_ALIASES:
        return AURA
    if token in _OWNER_ALIASES:
        return OWNER
    return UNATTRIBUTED


@dataclass(frozen=True)
class ReferentFrame:
    """Who is who, for one exchange.

    ``owner_name`` is a display name only. Identity is the *role*: the person
    she is speaking to is OWNER whatever they are called, which is what keeps
    the frame right when the name is unknown or wrong.
    """

    owner_name: str = "Bryan"
    aura_name: str = "Aura"

    def display(self, speaker: str) -> str:
        """The name to print for a canonical speaker token."""
        canonical = _canonical(speaker)
        if canonical == AURA:
            return self.aura_name
        if canonical == OWNER:
            return self.owner_name or "the person I am talking to"
        return "someone unidentified"

    def first_person_of(self, speaker: str) -> str:
        """Who "I" refers to when ``speaker`` says it."""
        return self.display(speaker)

    def second_person_of(self, speaker: str) -> str:
        """Who "you" refers to when ``speaker`` says it."""
        canonical = _canonical(speaker)
        if canonical == AURA:
            return self.owner_name or "the person I am talking to"
        if canonical == OWNER:
            return self.aura_name
        return "someone unidentified"

    def binding_note(self) -> str:
        """The standing bindings, short enough to sit in every prompt.

        Deliberately about *reading remembered text*, not about manners. The
        failure this prevents is a comprehension failure, not a politeness
        one.
        """
        return (
            f"Referents in this exchange: I am {self.aura_name}. "
            f"The person I am talking to is {self.owner_name}. "
            f"When {self.owner_name} says \"you\" or \"yourself\" he means "
            f"{self.aura_name}; when I say \"you\" I mean {self.owner_name}. "
            f"Recalled text is someone's past speech, not my own thought: a "
            f"snippet marked speaker=\"{self.owner_name}\" is something "
            f"{self.owner_name} said, so its \"I\" is {self.owner_name} and "
            f"its \"you\" is me. A snippet marked unattributed has no known "
            f"speaker, and I must not read its \"I\" as mine. "
            f"A snippet marked provenance=\"written-by-me-not-witnessed\" is "
            f"something I composed — a journal, a narrative arc, a dream — not "
            f"a record of anything that happened, so I must not report it as a "
            f"memory of events or attribute anything in it to "
            f"{self.owner_name}. provenance=\"provenance-unknown\" gets the "
            f"same caution."
        )


_DEFAULT_FRAME = ReferentFrame()


def current_frame(owner_name: str = "") -> ReferentFrame:
    """The frame for this exchange, defaulting to the household owner."""
    name = str(owner_name or "").strip()
    if not name:
        return _DEFAULT_FRAME
    return ReferentFrame(owner_name=name)


def speaker_of(metadata: Mapping[str, Any] | None, *, default: str = "") -> str:
    """Recover the canonical speaker from a memory record's metadata.

    Returns :data:`UNATTRIBUTED` when nothing in the record says who spoke —
    which is a finding, not a failure, and is why the caller must render it.
    """
    if not isinstance(metadata, Mapping):
        return _canonical(default)
    for key in _SPEAKER_KEYS:
        if key in metadata:
            canonical = _canonical(metadata.get(key))
            if canonical != UNATTRIBUTED:
                return canonical
    # Some stores encode the role in the type: "user_message", "aura_response".
    for key in ("type", "memory_type", "kind", "category"):
        canonical = _canonical(metadata.get(key))
        if canonical != UNATTRIBUTED:
            return canonical
    return _canonical(default)


def has_person_reference(text: Any) -> bool:
    """Does this text contain a pronoun whose referent depends on who spoke?

    Text with none — "the folder is on the Desktop" — means the same thing
    from any mouth, so an unknown speaker costs nothing. Text with one cannot
    be understood at all without knowing the speaker.
    """
    return bool(_PERSON_RE.search(str(text or "")))


def attribute(
    text: Any,
    speaker: Any = UNATTRIBUTED,
    *,
    frame: ReferentFrame | None = None,
) -> str:
    """Return the speaker label for a remembered utterance.

    An empty string means no label is needed: the text has no first- or
    second-person reference, so its meaning does not turn on who said it.
    """
    body = str(text or "").strip()
    if not body:
        return ""
    canonical = _canonical(speaker)
    active = frame or _DEFAULT_FRAME
    if canonical == UNATTRIBUTED and not has_person_reference(body):
        return ""
    if canonical == UNATTRIBUTED:
        return UNATTRIBUTED
    return active.display(canonical)


def resolve_second_person(
    request: Any,
    *,
    speaker: Any = OWNER,
    frame: ReferentFrame | None = None,
) -> str:
    """Who a request's "yourself" refers to, or "" if it names no one.

    "Write a note about yourself", asked by the owner, is about Aura. This is
    the seam that keeps that from reaching the planner as a request for a note
    about Bryan.
    """
    text = str(request or "")
    if not _SECOND_PERSON_TARGET_RE.search(text):
        return ""
    active = frame or _DEFAULT_FRAME
    return active.second_person_of(speaker)


# ---------------------------------------------------------------------------
# The check, not just the label.
#
# Everything above makes the right information available. None of it forces
# the reply to use it — and "the prompt now says who spoke" is a hope, not a
# mechanism. This is the mechanism: after a reply is generated, look for the
# specific failure that was actually observed, which is Aura reproducing a
# sentence the OWNER said, in the first person, as her own.
#
# It is deliberately narrow. Paraphrase is normal and quoting is fine; what is
# not fine is a long verbatim run of the owner's first-person speech
# reappearing as hers with no quotation and no attribution. That is what
# happened on 2026-07-28 and it is what this detects.
# ---------------------------------------------------------------------------

#: Shorter runs than this are ordinary phrase overlap ("I was trying to").
_MIN_BORROWED_WORDS = 8

_WORD_RE = re.compile(r"[a-z0-9']+")

#: A run inside quotation marks is a quote, and a quote is attribution.
_QUOTED_RE = re.compile(r"[\"“‘']([^\"”’']{12,})[\"”’']")


def _words(text: Any) -> list[str]:
    return _WORD_RE.findall(str(text or "").lower())


def borrowed_first_person_spans(
    reply: Any,
    owner_utterances: Any,
    *,
    min_words: int = _MIN_BORROWED_WORDS,
) -> list[str]:
    """Runs of the owner's own first-person speech reappearing as Aura's.

    ``owner_utterances`` is whatever the owner actually said — recalled
    snippets attributed to :data:`OWNER`, or the raw turns. Returns the
    borrowed spans, longest first; an empty list is the clean case.

    A span only counts when it carries a first- or second-person pronoun,
    because those are the words whose meaning flips with the speaker. Shared
    phrasing with no pronoun in it ("the orca articles in the folder") is two
    people talking about the same thing, which is what conversation is.
    """
    reply_words = _words(reply)
    if len(reply_words) < min_words:
        return []
    reply_text = str(reply or "")
    quoted = {
        " ".join(_words(match.group(1))) for match in _QUOTED_RE.finditer(reply_text)
    }

    if isinstance(owner_utterances, (str, bytes)):
        sources: list[str] = [str(owner_utterances)]
    else:
        sources = [str(item or "") for item in (owner_utterances or [])]

    found: list[str] = []
    for source in sources:
        source_words = _words(source)
        if len(source_words) < min_words:
            continue
        source_grams = {
            " ".join(source_words[i : i + min_words])
            for i in range(len(source_words) - min_words + 1)
        }
        start = 0
        while start <= len(reply_words) - min_words:
            gram = " ".join(reply_words[start : start + min_words])
            if gram not in source_grams:
                start += 1
                continue
            # Extend the match as far as it runs.
            end = start + min_words
            while end < len(reply_words):
                candidate = " ".join(reply_words[end - min_words + 1 : end + 1])
                if candidate not in source_grams:
                    break
                end += 1
            span = " ".join(reply_words[start:end])
            if has_person_reference(span) and not any(
                span in quote or quote in span for quote in quoted
            ):
                found.append(span)
            start = end
    found.sort(key=lambda item: -len(item))
    return found
