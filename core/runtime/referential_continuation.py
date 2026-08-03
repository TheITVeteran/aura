"""Messages that mean nothing on their own, and the turn that gives them meaning.

Two live failures on 2026-08-03, ten minutes apart, both this:

    Bryan: Hey, Aura can you tell me what you see on my screen currently?
    Aura:  I can't see your screen...
    Bryan: Can you do it now?          <- "it" is the screen read
    Bryan: Yes you can lol             <- asserts the capability she denied

    Aura:  Tell me the good news first. Response from who?
    Bryan: From the grant research funds manager
    Aura:  The good news first.        <- repeated her own question

Every router in the system classifies ONE message in isolation. "Can you do it
now?" contains no screen noun, so the screen-observation router says no and the
capability is never offered. "From the grant research funds manager" contains no
question, so nothing connects it to the question Aura had just asked.

The shared defect is not the classifiers. It is that a conversational turn is
being read as though it were the whole conversation. Some utterances are
*syntactically incomplete by design* — that is what makes them natural — and
their content lives in the previous turn.

This resolves them, in both directions:

* **Continuation** — a short retry/affirmation/pro-form ("do it now", "yes you
  can", "try again", "please") carries the last request Bryan made. His intent
  is the antecedent.
* **Answer** — a fragment that follows a question Aura asked is the missing
  piece of that question. Her question is the antecedent.

What it returns is an EFFECTIVE message: the current text joined with its
antecedent, for routing and reasoning only. It never rewrites what Bryan said,
never becomes the visible transcript, and when nothing plausibly refers back it
returns the message untouched.

Deliberately conservative. Resolving a continuation that is not one attaches
stale intent to a fresh request, which is worse than missing a follow-up — so a
message with any standalone content of its own is left exactly as it came.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Resolution",
    "answers_a_question",
    "effective_message",
    "is_referential_continuation",
]

#: Retry / assent / pro-form utterances. Each is a whole message that asks for
#: the previous request again, or asserts it can be done.
_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"(?:yes|yeah|yep|yup|sure|ok|okay|please|go\s+ahead|go\s+on|continue)"
    r"|(?:can|could|will|would)\s+you\s+(?:do|try)\s+(?:it|that|this)"
    r"|(?:do|try)\s+(?:it|that|this)"
    r"|try\s+again|again|once\s+more|retry"
    r"|(?:you|u)\s+can|yes\s+you\s+can"
    r"|now|right\s+now|do\s+it\s+now"
    r")\b[\s\S]{0,40}$",
    re.IGNORECASE,
)

#: A pro-form standing in for something already named ("do that", "show me it").
_PRO_FORM_RE = re.compile(r"\b(?:it|that|this|those|them|the\s+same)\b", re.IGNORECASE)

#: Content that makes a message stand on its own. If a message has a real verb
#: phrase plus a noun of its own, it is a new request, not a continuation.
_STANDALONE_HINT_RE = re.compile(
    r"\b(?:what|who|when|where|why|how|tell|explain|show|open|close|write|read|"
    r"send|search|find|make|build|check|remember|remind)\b",
    re.IGNORECASE,
)

#: Words that end a sentence with an interrogative, used to spot Aura's
#: outstanding question.
_QUESTION_RE = re.compile(r"\?\s*$|^\s*(?:who|what|when|where|why|which|how)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Resolution:
    """What this message means once the previous turn is taken into account."""

    text: str
    antecedent: str = ""
    kind: str = "standalone"

    @property
    def resolved(self) -> bool:
        return bool(self.antecedent)


def is_referential_continuation(message: str) -> bool:
    """True when the message asks for the previous request again.

    Length-bounded on purpose: "yes" is a continuation, "yes, and also please
    summarise the last three papers on corrigibility" is a new request that
    happens to start with yes.
    """
    text = str(message or "").strip()
    if not text or len(text) > 60:
        return False
    if not _CONTINUATION_RE.match(text):
        return False
    # "Can you do it now?" is a continuation. "Can you open Chrome now?" is not
    # — it names its own object, so it stands alone.
    without_proforms = _PRO_FORM_RE.sub(" ", text)
    return not _STANDALONE_HINT_RE.search(without_proforms)


def answers_a_question(message: str, previous_assistant_message: str) -> bool:
    """True when this message is the missing piece of a question just asked.

    A fragment — no verb of its own, no question — landing right after Aura
    asked something is almost always the answer to it. "From the grant research
    funds manager" is not a new topic; it is the object of "Response from who?".
    """
    text = str(message or "").strip()
    prior = str(previous_assistant_message or "").strip()
    if not text or not prior:
        return False
    if not _QUESTION_RE.search(prior):
        return False
    if len(text) > 200 or text.endswith("?"):
        return False
    # A fragment: no independent request of its own.
    return not _STANDALONE_HINT_RE.search(text)


def effective_message(
    message: str,
    *,
    previous_user_request: str = "",
    previous_assistant_message: str = "",
) -> Resolution:
    """The message as it should be ROUTED, with its antecedent restored.

    Returns the original text untouched whenever the message stands on its own,
    which is the common case. The joined form exists for classifiers and
    reasoning; it is never what gets shown back to the person.
    """
    text = str(message or "").strip()
    if not text:
        return Resolution(text=text)

    if previous_user_request and is_referential_continuation(text):
        return Resolution(
            text=f"{previous_user_request.strip()} — {text}",
            antecedent=previous_user_request.strip(),
            kind="continuation",
        )

    if previous_assistant_message and answers_a_question(
        text, previous_assistant_message
    ):
        return Resolution(
            text=f"{previous_assistant_message.strip()} {text}",
            antecedent=previous_assistant_message.strip(),
            kind="answer",
        )

    return Resolution(text=text)
