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
    r"what'?s\s+written|word\s+for\s+word|verbatim|literally\s+say)\b",
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
    r"the\s+text\s+(?:on|in)\s+(?:it|them|the)|showing\s*:)\b",
    re.IGNORECASE,
)


def asks_to_read_the_screen(user_message: Any) -> bool:
    """True when the turn asks for the words that are on the screen."""
    text = str(user_message or "")
    if not text.strip():
        return False
    return bool(_READ_THE_SCREEN_RE.search(text) and _SCREEN_SUBJECT_RE.search(text))


def quotes_screen_content(reply_text: Any) -> bool:
    """True when the reply presents specific text as read from the screen."""
    body = str(reply_text or "")
    if not body.strip():
        return False
    if not _QUOTED_TEXT_RE.search(body):
        return False
    return bool(_ASSERTS_A_READING_RE.search(body))


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
