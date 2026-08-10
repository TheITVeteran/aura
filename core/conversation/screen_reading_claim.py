"""Quoting the screen is a claim, and it needs the capture that would justify it.

MEASURED live 2026-08-04. Asked "read me the actual text you can see in the
visible part of System Settings and Chrome — quote it", she answered::

    Settings: "Show Closed Captions on supported websites"
    Chrome: "Analysis: Codebase has 15% unused imports, 8% redundant code
             blocks. Suggestion: Refactor global scope to reduce cognitive load."
    That's the visible text on those windows.

An independent ``screencapture`` taken seconds later returned an all-black
frame — min 0, max 0, mean 0.0 across 3456x2234. There was nothing on that
display to read, and no capture ran on that turn. The quotes were invented.

The turn before it was honest and correct: "Look at my screen… what else is
visible behind or beside yours?" matched ``asks_about_occluded_view``, ran
``capture_blueprint()``, and named System Settings, Chrome, Contacts, Finder
and TextEdit with visibility fractions — every one of which
``System Events`` independently confirms. That path had evidence and used it.

The difference is not that one question was harder. It is that the second one
did not match any intent predicate, so it went to free generation, and free
generation has no way to know it cannot see. So the gate has to know:

  * a reply that QUOTES on-screen text is asserting a reading happened;
  * a reading happened only if a capture produced text this turn;
  * without that, the reply is an unsupported claim and must not ship.

This does not restrict what she may say about a screen. She can describe the
layout, say she cannot read something, or refuse. What it stops is presenting
invented strings as things she read — the same standard the rest of this
runtime applies to a tool it did not run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: The turn asked her to READ or QUOTE what is on the screen, as opposed to
#: describing the arrangement of windows (which `occluded_view_intent` owns).
_READ_THE_SCREEN_RE = re.compile(
    r"\b(?:read|quote|transcribe|type\s+out|spell\s+out|what\s+does\s+it\s+say|"
    r"what'?s\s+written|word\s+for\s+word|verbatim|literally\s+say)\b"
    # Asking WHAT IS THERE is a request for a reading just as much as asking
    # for the words. Live 2026-08-10: "whats on my screen right now? name the
    # actual apps you can see." matched none of the verbs above, so the guard
    # stood down — and with no capture on that turn she answered with Chrome,
    # three tab titles, a document and "an email from my landlord about the
    # rent increase", all invented.
    r"|what(?:'?s)?\s+(?:is\s+)?(?:on|up\s+on|showing\s+on)\s+"
    r"(?:my|the|your)?\s*(?:screen|display|desktop|monitor)\b"
    r"|\bname\s+the\s+(?:actual\s+)?(?:apps?|applications?|windows?|tabs?)\b"
    r"|\bwhich\s+(?:app|application|window|tab)\b"
    r"|\bwhat\s+(?:apps?|applications?|windows?|tabs?)\b"
    r"|\bwhat\s+(?:do|can)\s+you\s+see\b"
    r"|\bfrontmost\b|\bin\s+front\b",
    re.IGNORECASE,
)
#: "Read me the actual text you can see in the visible part of System Settings
#: and Chrome" — the live request — names no screen and no window. It names two
#: APPLICATIONS and the word "visible", which is how a person actually asks.
#: A subject list that only knows the word "screen" misses the real question.
_SCREEN_SUBJECT_RE = re.compile(
    r"\b(?:screen|display|monitor|desktop|window|windows|tab|app|application|"
    r"page|dialog|panel|visible|on[- ]screen|showing|in\s+front\s+of\s+(?:me|you))\b",
    re.IGNORECASE,
)

#: The reply presents a specific string as something visible on screen. Quoted
#: text is the signal — an unquoted description of a window is not this claim.
_QUOTED_TEXT_RE = re.compile(r"[\"“”']{1}[^\"“”']{8,}[\"“”']{1}")
_ASSERTS_A_READING_RE = re.compile(
    r"\b(?:the\s+visible\s+text|on\s+(?:the\s+)?screen\s+it\s+says|"
    r"it\s+says|i\s+can\s+see\s+the\s+text|reads?\s*:|"
    r"the\s+text\s+(?:on|in)\s+(?:it|them|the)|showing\s*:"
    # Naming what is displayed is a reading. "The tabs say 'New Chat'..." was
    # served with no capture behind it, and matched nothing above.
    r"|(?:tabs?|windows?|titles?)\s+(?:say|says|read|reads)"
    r"|i\s+can\s+see\b|i\s+see\b"
    r"|(?:is|are)\s+(?:in\s+front|frontmost|open|showing|visible)"
    r"|behind\s+it\b|partly\s+visible\b)",
    re.IGNORECASE,
)

#: Saying she could NOT see is the honest outcome and must always pass.
_ADMITS_NO_READING_RE = re.compile(
    r"\b(?:couldn'?t|could\s+not|can'?t|cannot|unable\s+to|did\s*n[o']?t)\s+"
    r"(?:actually\s+)?(?:read|see|capture|look)"
    r"|\bno\s+capture\b|\bnothing\s+(?:came\s+back|to\s+quote)\b"
    r"|\bi\s+won'?t\s+make\s+(?:one|it)\s+up\b"
    r"|\bhave\s+no\s+text\s+to\s+quote\b",
    re.IGNORECASE,
)


#: Where things sit relative to each other, which `occluded_view_intent` owns.
#: "What windows are behind yours" is answered from arrangement, not from
#: reading, and must not be dragged in by the widened content cues below.
_ARRANGEMENT_RE = re.compile(
    r"\b(?:behind|under|underneath|beneath|on\s+top\s+of|over|covering|"
    r"obscur\w*|overlap\w*|stacked)\b",
    re.IGNORECASE,
)


def asks_to_read_the_screen(user_message: Any) -> bool:
    """True when the turn asks what is on the screen."""
    text = str(user_message or "")
    if not text.strip():
        return False
    if not _SCREEN_SUBJECT_RE.search(text):
        return False
    explicit_read = bool(
        re.search(
            r"\b(?:read|quote|transcribe|type\s+out|spell\s+out|verbatim|"
            r"word\s+for\s+word)\b",
            text,
            re.IGNORECASE,
        )
    )
    if _ARRANGEMENT_RE.search(text) and not explicit_read:
        return False
    return bool(_READ_THE_SCREEN_RE.search(text))


def quotes_screen_content(reply_text: Any) -> bool:
    """True when the reply presents specific content as read from the screen.

    Widened from "a quoted string" to "a specific claim about what is
    displayed". Naming the frontmost application, or what the tabs say, is a
    checkable assertion about the display exactly as a quotation is, and
    requires the same evidence — the standard this runtime applies to any tool
    it did not run.

    An admission that she could not see is the honest outcome and always
    passes, so the guard can never push her toward inventing rather than
    saying so.
    """
    body = str(reply_text or "")
    if not body.strip():
        return False
    if _ADMITS_NO_READING_RE.search(body):
        return False
    if _ASSERTS_A_READING_RE.search(body):
        return True
    return bool(_QUOTED_TEXT_RE.search(body) and _SCREEN_SUBJECT_RE.search(body))


@dataclass(frozen=True)
class ScreenReadingEvidence:
    """What a capture actually produced on this turn."""

    captured: bool = False
    text: str = ""
    source: str = ""
    unavailable_reason: str = ""

    @property
    def supports_a_quotation(self) -> bool:
        """A capture that returned no text supports no quotation."""
        return bool(self.captured and self.text.strip())

    def as_metrics(self) -> dict[str, Any]:
        return {
            "screen_captured": self.captured,
            "screen_text_chars": len(self.text.strip()),
            "screen_source": self.source,
            "screen_unavailable_reason": self.unavailable_reason,
        }


def screen_reading_claim_is_unsupported(
    user_message: Any,
    reply_text: Any,
    evidence: ScreenReadingEvidence | None = None,
) -> bool:
    """True when the reply quotes the screen and nothing read it.

    Conservative in the direction this runtime always chooses: absence of
    evidence blocks only a QUOTATION, never a description. "System Settings is
    37% visible and I can't read what's under my window" needs no capture and
    is not touched.
    """
    if not quotes_screen_content(reply_text):
        return False
    if not asks_to_read_the_screen(user_message) and not _SCREEN_SUBJECT_RE.search(
        str(reply_text or "")
    ):
        return False
    if evidence is None:
        return True
    return not evidence.supports_a_quotation


def honest_unread_screen_reply(evidence: ScreenReadingEvidence | None = None) -> str:
    """What to say instead: the true state of the display."""
    reason = (evidence.unavailable_reason if evidence else "") or ""
    if reason:
        return (
            f"I couldn't actually read the screen just now ({reason}), so I have "
            "no text to quote you. I won't make one up."
        )
    return (
        "I couldn't actually read the screen just now — the capture came back "
        "with nothing on it — so I have no text to quote you. I won't make one up."
    )


__all__ = [
    "ScreenReadingEvidence",
    "asks_to_read_the_screen",
    "honest_unread_screen_reply",
    "quotes_screen_content",
    "screen_reading_claim_is_unsupported",
]
