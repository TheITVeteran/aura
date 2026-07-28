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
    "TURN_IN_FLIGHT_CEILING_S",
    "note_route_delivered",
    "note_turn_started",
    "route_answer_supersedes",
    "reset_route_delivery",
]

#: How long after the route answers a turn a differing spoken message is
#: treated as the other lane finishing that same turn. Chosen to cover the
#: observed gap (the deep lane trailed the route by ~3 minutes) without
#: muting proactive speech for the rest of the conversation.
LATE_LANE_WINDOW_S = 240.0

#: A turn cannot be "in flight" forever — a route that dies without answering
#: must not mute her for the rest of the session. Comfortably longer than the
#: slowest real turn (a reconstruction runs into the minutes) and far short of
#: a conversation.
TURN_IN_FLIGHT_CEILING_S = 1_800.0

_LOCK = threading.Lock()
_LAST_ROUTE_REPLY: str = ""
_LAST_ROUTE_AT: float = 0.0
_TURN_STARTED_AT: float = 0.0


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def reset_route_delivery() -> None:
    """Forget the last delivery (tests, and a fresh conversation)."""
    global _LAST_ROUTE_REPLY, _LAST_ROUTE_AT, _TURN_STARTED_AT
    with _LOCK:
        _LAST_ROUTE_REPLY = ""
        _LAST_ROUTE_AT = 0.0
        _TURN_STARTED_AT = 0.0


def note_turn_started() -> None:
    """The person just said something and is waiting for the answer.

    Protection used to begin only when the route ANSWERED, which leaves the
    whole of a turn unguarded — and the longer the turn, the wider the hole.
    Measured live 2026-07-28: asked to reverse-engineer 2048, the window filled
    with "Bryan, you mentioned her being your favorite person in the world..."
    at the same second as the real reply, and "I've been reading up on swarm
    protocols" four minutes later. Neither answered anything he asked; both are
    her own idle interests, which she is supposed to have — just not in the
    middle of someone waiting on an answer.
    """
    global _TURN_STARTED_AT
    with _LOCK:
        _TURN_STARTED_AT = time.time()


def note_route_delivered(reply_text: Any) -> None:
    """Record that the chat route just answered a turn, and with what."""
    global _LAST_ROUTE_REPLY, _LAST_ROUTE_AT
    body = _norm(reply_text)
    if not body:
        return
    with _LOCK:
        _LAST_ROUTE_REPLY = body
        _LAST_ROUTE_AT = time.time()


def route_answer_supersedes(spoken_text: Any, *, unprompted: bool = True) -> bool:
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
        started_at = _TURN_STARTED_AT
    now = time.time()

    # A turn is open: the person asked and has not been answered yet. Anything
    # arriving now is either that answer (let it through, matched below) or
    # something else entirely, which is an interruption.
    # Only UNPROMPTED speech waits. The route's own answer travels this same
    # bridge, and withholding it would trade a noisy window for a silent one —
    # a reply lost is a certain loss, which is the more expensive mistake.
    turn_open = (
        unprompted
        and started_at > 0.0
        and started_at > last_at
        and (now - started_at) <= TURN_IN_FLIGHT_CEILING_S
    )
    if turn_open:
        return True

    if not last_reply or (now - last_at) > LATE_LANE_WINDOW_S:
        return False
    # The same answer, however it was trimmed or wrapped on the way here.
    head = body[:160]
    last_head = last_reply[:160]
    if head in last_reply or last_head in body:
        return False
    return True
