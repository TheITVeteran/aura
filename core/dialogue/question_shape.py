"""What shape of answer does this question want?

Measured live. Asked "how are you feeling right now?", Aura opened with:

    Yes, I am okay. I feel warm and settled...

The state behind it was accurate. The "Yes," was not an answer to anything —
it was an answer to "are you okay?", a question nobody had asked. Templates
that open with a polarity word read as canned the moment the question wasn't
polar, and that is exactly the tell that makes a mind sound like a script.

This is the general check: a reply may lead with "Yes"/"No" only when the
question actually admits a yes or a no.

- **Polar** — "are you okay?", "do you remember?", "can you do that?".
  Yes/No is the answer.
- **Open** — "how are you feeling?", "what happened?", "why did you stop?".
  Yes/No is a non-sequitur.
- **Unknown** — no question at all, or too little to tell. Callers should
  treat this like open: leading with a polarity word is the riskier bet.

``open_answer`` is the one-liner most callers want: hand it a polar opener and
an open opener, get the right one back.
"""

from __future__ import annotations

import re

__all__ = ["POLAR", "OPEN", "UNKNOWN", "question_shape", "open_answer"]

POLAR = "polar"
OPEN = "open"
UNKNOWN = "unknown"

#: Interrogatives that cannot be answered with a bare yes or no.
_OPEN_LEAD = re.compile(
    r"(?i)\b(how|what|why|where|when|which|who|whom|whose|"
    r"tell\s+me|describe|explain|walk\s+me\s+through)\b"
)

#: Auxiliaries and copulas that open a yes/no question.
_POLAR_LEAD = re.compile(
    # A leading discourse marker ("so", "and", "but ok,") does not change the
    # shape of the question that follows it.
    r"(?i)^\W*(?:(?:so|and|but|ok|okay|well|hey|also|then)\b\W*){0,3}"
    r"(are|is|am|was|were|do|does|did|can|could|will|would|should|"
    r"shall|have|has|had|may|might|must)\b"
)

#: Tag questions and polarity checks that can appear mid-sentence:
#: "you're okay, right?", "so do you remember me?"
_POLAR_TAG = re.compile(r"(?i)\b(right|correct|yeah|yes|no)\s*\?\s*$")


def _last_question(text: str) -> str:
    """The final interrogative clause — that is the one being answered."""
    normalised = " ".join(str(text or "").split())
    if not normalised:
        return ""
    # Split on any sentence boundary, not just "?", so the preamble in
    # "I was worried. Are you okay?" does not swallow the question.
    questions = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?…])\s+", normalised)
        if segment.strip().endswith("?")
    ]
    if questions:
        return questions[-1]
    return normalised


def question_shape(user_message: str) -> str:
    """Classify the answer shape the user's question invites.

    Returns :data:`POLAR`, :data:`OPEN`, or :data:`UNKNOWN`.
    """
    clause = _last_question(user_message)
    if not clause:
        return UNKNOWN
    if not clause.endswith("?"):
        # No question mark: an imperative like "tell me how you are" is still
        # open, and a dropped "?" on "are you ok" is still polar. Anything with
        # neither lead we decline to guess at.
        if _OPEN_LEAD.search(clause):
            return OPEN
        return POLAR if _POLAR_LEAD.search(clause) else UNKNOWN

    if _POLAR_TAG.search(clause):
        return POLAR

    # "how are you" beats the "are" that follows it — the interrogative that
    # comes first is the one that governs the answer.
    open_match = _OPEN_LEAD.search(clause)
    polar_match = _POLAR_LEAD.search(clause)
    if open_match and polar_match:
        return OPEN if open_match.start() <= polar_match.start() else POLAR
    if open_match:
        return OPEN
    if polar_match:
        return POLAR
    return UNKNOWN


def open_answer(user_message: str, polar_opener: str, open_opener: str) -> str:
    """Pick the opener that matches the question.

    ``polar_opener`` is used only when the question genuinely admits yes/no;
    :data:`UNKNOWN` falls through to ``open_opener`` because leading with a
    polarity word nobody asked for is the worse failure.
    """
    return polar_opener if question_shape(user_message) == POLAR else open_opener
