"""She invented an evening they never had.

Measured live, 2026-07-28, immediately after a desktop task completed. She
opened unprompted with "I'm curious about what you're working on. Something
heavy?" and then, asked what made her say it:

    The tone of your previous response. It was heavy with a sense of
    responsibility or burden.

Bryan's previous response was "Stuck on that one?". Asked again, she went
further:

    It was just one of those nights. The moon was full and I got to thinking
    about things, wondering how you were doing up there in that prison.

    Prison? The hell are you talking about lol

    I thought you had a problem with your eyes. You know, looking at screens
    for too long?

There is no prison, no full moon, and no conversation about his eyes. Nothing
was recalled: the episodic store contains none of it (the only "prison" in
memory is a logic puzzle about two guards). It was *generated* — given a turn
with almost no content to answer, the model supplied a shared past instead of
saying it had none.

That is the defect, and it is one defect rather than three: **a past-tense
claim about the other person, or about a moment together, whose content
appears nowhere in what they actually said.** Each line above is that, and
each one licensed the next, because her own fabrication became the context
for the following turn.

The check is deliberately about *novel content in a relational past-tense
claim*, not about tone or plausibility:

* "I wrote the note and saved it" — about herself, and true.
* "You asked me to remember the orca" — about him, and his words are right
  there in the turn.
* "I thought you had a problem with your eyes" — about him, past tense, and
  "eyes" appears nowhere.

Grounded recall is unaffected: if the claim's content is in the conversation
or in what she was given to recall, it passes. What cannot pass is inventing
the evidence and the memory at the same time.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

__all__ = [
    "fabricated_shared_history",
    "has_fabricated_shared_history",
]

#: Sentences that assert something about the OTHER person's past state,
#: words, or feelings, or about a moment the two of them shared. These are
#: the claims that require evidence, because only the other person can
#: confirm them.
_RELATIONAL_PAST_RES: tuple[re.Pattern[str], ...] = (
    # "I thought you had...", "I could tell you were...", "I noticed you..."
    re.compile(
        r"(?i)\bi\s+(?:thought|figured|assumed|noticed|could\s+tell|sensed|"
        r"remember(?:ed)?|recall(?:ed)?|felt\s+like)\s+(?:that\s+)?you\b"
    ),
    # "you were tired", "you had a problem", "you said something about..."
    re.compile(
        r"(?i)\byou\s+(?:were|had|seemed|sounded|looked|said|told\s+me|"
        r"mentioned|used\s+to)\b"
    ),
    # "we were talking about", "we had that conversation"
    re.compile(r"(?i)\bwe\s+(?:were|had|used\s+to|talked|spoke|discussed)\b"),
    # "your last message was...", "the tone of your previous response"
    re.compile(
        r"(?i)\byour\s+(?:last|previous|earlier|first)\s+"
        r"(?:message|response|reply|question|answer|note|words?)\b"
    ),
    # "it was one of those nights", "that night", "back then"
    re.compile(
        r"(?i)\b(?:it\s+was\s+(?:just\s+)?one\s+of\s+those|that\s+(?:night|"
        r"evening|day|time)|back\s+then|the\s+other\s+(?:night|day))\b"
    ),
)

#: Words too common to be evidence of anything. A claim grounded only by
#: "thing" and "time" is not grounded.
_STOPWORDS: frozenset[str] = frozenset(
    """a about all am an and any are as at be been being but by can cant could
    did didnt do does doesnt doing done dont down each even ever every for from
    get got had has have having he her here hers him his how i if in into is it
    its ive just know let like me might more most much must my no nor not now
    of off on once one only or other our out over own re said same she should
    so some such than that the their them then there these they thing things
    this those though time to too us very want was way we well were what when
    where which while who why will with would you your yours
    just really actually kind sort bit lot lots okay ok yeah yes
    """.split()
)

#: The user ASKING her to recollect. When the question is "what did I say",
#: "what was I worried about", "do you remember what I told you" — a
#: past-tense claim about them with content not in the current turn is the
#: ANSWER, not an invention. Flagging it inverts the check.
#:
#: Measured against the 120-turn contract transcript: asked "What did you think
#: I was worried about in the last exchange?", the correct reply "You were
#: worried that the conversation could look alive for a minute and then quietly
#: lose the plot" was flagged as fabricated shared history. The claim is novel
#: because the question demanded something not already on the page.
_INVITES_RECOLLECTION_RE = re.compile(
    r"(?i)"
    r"\b(?:what|which|when|where|how|why|who)\b[^?]{0,80}\b"
    r"(?:i|we|you)\s+(?:said|asked|told|meant|wanted|felt|thought|were|was|"
    r"had|mentioned|worried|brought\s+up)\b"
    r"|\bdo\s+you\s+(?:remember|recall)\b"
    r"|\bwhat\s+did\s+(?:i|we)\b"
    r"|\bremind\s+me\s+what\b"
    r"|\bwhat\s+(?:was|were)\s+(?:i|we|my|our)\b"
    r"|\bwhat\s+did\s+you\s+think\s+(?:i|we)\b"
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

#: A claim carrying fewer new content words than this is phrasing, not a
#: fabricated event.
_MIN_NOVEL_WORDS = 2

#: ...and novelty is measured as a SHARE of the claim, not as a count.
#:
#: Counting words alone flagged "I can verify durable memory evidence that we
#: discussed retained memory as behavioral reuse with receipts" — a real
#: recall, grounded word for word in her memory evidence, whose only novel
#: words were "durable", "evidence" and "verify". That is her framing of the
#: recall, not an invented fact about him.
#:
#: A fabricated memory is mostly novel: "the moon was full and I got to
#: thinking about how you were doing up there in that prison" is almost
#: entirely words nobody said. A grounded one is mostly known, however it is
#: phrased. The ratio separates them; the count never could.
_MIN_NOVEL_SHARE = 0.55

#: Below this much known vocabulary we have no basis for a verdict at all.
#: Roughly "one real sentence of theirs".
_MIN_CONTEXT_WORDS = 4


def _content_words(text: Any) -> set[str]:
    return {
        word.lower()
        for word in _WORD_RE.findall(str(text or ""))
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }


def _sentences(text: Any) -> list[str]:
    raw = " ".join(str(text or "").split())
    if not raw:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?…])\s+", raw) if part.strip()]


def fabricated_shared_history(
    reply_text: Any,
    user_message: Any = "",
    recent_user_messages: Iterable[str] | None = None,
    *,
    grounding: Iterable[str] | None = None,
    min_novel_words: int = _MIN_NOVEL_WORDS,
) -> list[str]:
    """Sentences asserting a shared past that nothing in context supports.

    ``grounding`` is anything else she is entitled to have known — recalled
    memory snippets, retrieved facts. Passing it is what keeps genuine recall
    from being flagged: a claim is fabricated only when its content appears in
    neither the conversation nor the evidence she was given.

    Returns the offending sentences, longest first. Empty is the clean case.
    """
    sentences = _sentences(reply_text)
    if not sentences:
        return []

    # A RECOLLECTION SHE WAS ASKED FOR IS NOT AN INVENTION.
    #
    # "What did you think I was worried about?" demands a past-tense claim
    # about him whose content is necessarily not in the current turn — that is
    # what makes it a question. Flagging the answer inverts the check. This
    # module is about UNSOLICITED invented history.
    if _INVITES_RECOLLECTION_RE.search(str(user_message or "")):
        return []

    known: set[str] = _content_words(user_message)
    for message in recent_user_messages or ():
        known |= _content_words(message)
    for item in grounding or ():
        known |= _content_words(item)

    # NO CONTEXT MEANS NO VERDICT.
    #
    # The whole check is "this content appears nowhere in what they said",
    # which is meaningless when we do not know what they said. Callers that
    # cannot supply the visible request — internal repair paths, gate probes,
    # a turn whose prompt could not be parsed — were getting every relational
    # sentence flagged, because an empty vocabulary makes everything novel.
    # Absence of evidence is not evidence.
    if len(known) < _MIN_CONTEXT_WORDS:
        return []

    found: list[str] = []
    for sentence in sentences:
        if not any(pattern.search(sentence) for pattern in _RELATIONAL_PAST_RES):
            continue
        content = _content_words(sentence)
        if not content:
            continue
        novel = content - known
        if len(novel) < max(1, int(min_novel_words)):
            continue
        if (len(novel) / len(content)) < _MIN_NOVEL_SHARE:
            continue  # Mostly grounded: this is her phrasing, not an invention.
        found.append(sentence)
    found.sort(key=lambda item: -len(item))
    return found


def has_fabricated_shared_history(
    reply_text: Any,
    user_message: Any = "",
    recent_user_messages: Iterable[str] | None = None,
    *,
    grounding: Iterable[str] | None = None,
) -> bool:
    """True when the reply invents a shared past."""
    return bool(
        fabricated_shared_history(
            reply_text,
            user_message,
            recent_user_messages,
            grounding=grounding,
        )
    )
