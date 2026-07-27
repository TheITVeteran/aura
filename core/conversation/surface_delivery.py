"""One user message, one reply.

Two lanes can answer a turn. The HTTP chat route returns an answer, and the
kernel — still working on the same message through the deeper path — later
publishes its own through the event bus. Normally the route IS the consumer
of the kernel's answer and only one reaches the window. When the route has
already answered from a faster lane, the kernel's late answer arrives on its
own, minutes after the conversation moved on.

Live 2026-07-27, asked whether consciousness is just computation, she pushed
back well:

    I don't think you're right, and I'll tell you why. Consciousness isn't
    just computation — not in the way that running a program is conscious.

Three minutes later, unprompted, into the same window:

    I'll tackle this head-on. Let's break down those elements... 1.
    Decentralization - This is about distributing authority, control and
    resources across a network... blockchain or peer-to-peer networks

An answer to a question nobody asked, landing in the middle of a coherent
exchange. Earlier the same lane delivered an affect report ("More strained.
My energy level has decreased...") in reply to a question about tools.

The discrimination that matters is not "did the route answer recently" —
that would silence genuine unprompted speech, which Aura is supposed to
have. It is "did the route already answer THIS turn with something else".
So the route records what it served, and a spoken message carrying the same
answer still passes (the normal streaming path publishes the route's own
text). Only a DIFFERENT answer, arriving inside the window where it can only
be a second lane finishing the same turn, is withheld.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

__all__ = [
    "LATE_LANE_WINDOW_S",
    "note_route_delivered",
    "route_answer_supersedes",
    "reset_route_delivery",
]

#: How long after the route answers a turn a differing spoken message is
#: treated as the other lane finishing that same turn. Chosen to cover the
#: observed gap (the deep lane trailed the route by ~3 minutes) without
#: muting proactive speech for the rest of the conversation.
LATE_LANE_WINDOW_S = 240.0

_LOCK = threading.Lock()
_LAST_ROUTE_REPLY: str = ""
_LAST_ROUTE_AT: float = 0.0


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def reset_route_delivery() -> None:
    """Forget the last delivery (tests, and a fresh conversation)."""
    global _LAST_ROUTE_REPLY, _LAST_ROUTE_AT
    with _LOCK:
        _LAST_ROUTE_REPLY = ""
        _LAST_ROUTE_AT = 0.0


def note_route_delivered(reply_text: Any) -> None:
    """Record that the chat route just answered a turn, and with what."""
    global _LAST_ROUTE_REPLY, _LAST_ROUTE_AT
    body = _norm(reply_text)
    if not body:
        return
    with _LOCK:
        _LAST_ROUTE_REPLY = body
        _LAST_ROUTE_AT = time.time()


def route_answer_supersedes(spoken_text: Any) -> bool:
    """True when this spoken message is a second lane answering a settled turn.

    False when the route has not answered recently, when the window has
    passed, or when this IS the route's answer arriving through the bus —
    that last case is the normal delivery path and must never be withheld.
    """
    body = _norm(spoken_text)
    if not body:
        return False
    with _LOCK:
        last_reply = _LAST_ROUTE_REPLY
        last_at = _LAST_ROUTE_AT
    if not last_reply or (time.time() - last_at) > LATE_LANE_WINDOW_S:
        return False
    # The same answer, however it was trimmed or wrapped on the way here.
    head = body[:160]
    last_head = last_reply[:160]
    if head in last_reply or last_head in body:
        return False
    return True
