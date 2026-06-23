"""Grounded conversational recall for positional/temporal questions.

Fixes the confabulation failure mode where a question like *"do you remember what
I first asked?"* was answered by *generating a plausible-but-false memory*
(\"you asked about my neural network\") instead of retrieving the actual earliest
turn (\"you with me, Aura?\").

Content-similarity recall (and the engram competition field) cannot solve this on
its own, because *\"first\"* / *\"last\"* are **positional** keys, not content cues —
the earliest turn rarely shares words with the question that asks about it.  This
resolver detects positional/temporal recall intent and pulls the actual turn from
the live :class:`UnifiedTranscript`, then hands it to the model as an authoritative
grounding fact so the answer is *retrieved*, not invented.

It is the positional counterpart to the plasticity competition in
``core.memory.engram_plasticity`` (content cues → winner) — together they cover
both retrieval keys a real episodic memory needs: *what* was said and *when*.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("Aura.Conversation.GroundedRecall")

RecallPosition = Literal["first", "last"]

# Self/relationship reference — the question is about *our* conversation.
_SELF_REF = r"\b(i|we|my|me|us|our)\b"
# Recall framing — asking about a past utterance.
_RECALL_VERB = (
    r"(ask|asked|say|said|saying|question|questions|message|messages|"
    r"talk|talked|talking|bring up|brought up|mention|mentioned|start|started|"
    r"begin|began|tell|told|wanted|request)"
)
_FIRST = r"\b(first|initially|originally|very first|the start|the beginning)\b"
_LAST = r"\b(last|previous|recent|recently|just|earlier|a moment ago)\b"

_FIRST_RE = re.compile(
    rf"(?=.*{_SELF_REF})(?=.*{_FIRST})(?=.*\b{_RECALL_VERB}\b)",
    re.IGNORECASE,
)
_LAST_RE = re.compile(
    rf"(?=.*{_SELF_REF})(?=.*{_LAST})(?=.*\b{_RECALL_VERB}\b)",
    re.IGNORECASE,
)
# Direct idioms that don't need all three components.
_FIRST_IDIOM_RE = re.compile(
    r"(what did i (first|initially) (ask|say)|"
    r"my first (message|question|thing)|"
    r"the first thing i (asked|said)|"
    r"what was my first|how did (this|our|the) (conversation|chat) (start|begin)|"
    r"what did we (start|begin) (with|talking about))",
    re.IGNORECASE,
)

_MAX_GROUNDED_CHARS = 400


def detect_positional_recall(user_message: str) -> RecallPosition | None:
    """Return ``"first"``/``"last"`` if the turn asks a positional-recall question."""
    text = (user_message or "").strip()
    if not text or len(text) > 240:
        return None
    if _FIRST_IDIOM_RE.search(text):
        return "first"
    if _FIRST_RE.search(text):
        return "first"
    if _LAST_RE.search(text):
        return "last"
    return None


def _user_turns(exclude: str) -> list[str]:
    """Earliest→latest user utterances from the live transcript, current excluded."""
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        transcript = UnifiedTranscript.get_instance()
        entries = list(getattr(transcript, "_entries", []) or [])
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Grounded recall: transcript unavailable: %s", exc)
        return []

    norm_exclude = (exclude or "").strip().lower()
    turns: list[str] = []
    for e in entries:
        if getattr(e, "role", "") != "user":
            continue
        content = str(getattr(e, "content", "") or "").strip()
        if not content:
            continue
        if content.lower() == norm_exclude:
            continue
        turns.append(content)
    return turns


def resolve_positional_turn(user_message: str, position: RecallPosition) -> str | None:
    """Retrieve the actual first/last user turn this session (excluding current)."""
    turns = _user_turns(exclude=user_message)
    if not turns:
        return None
    chosen = turns[0] if position == "first" else turns[-1]
    chosen = chosen.strip()
    if len(chosen) > _MAX_GROUNDED_CHARS:
        chosen = chosen[: _MAX_GROUNDED_CHARS - 1].rstrip() + "…"
    return chosen or None


def build_grounded_recall_context(user_message: str) -> str | None:
    """Authoritative grounding block for a positional-recall turn, or None.

    The block states the retrieved fact and instructs the model to answer from it
    in its own voice — so the reply is grounded in what actually happened instead
    of a confident confabulation.  Returns ``None`` when there is no positional
    intent or no prior turn to ground on (e.g. brand-new conversation).
    """
    position = detect_positional_recall(user_message)
    if position is None:
        return None
    turn = resolve_positional_turn(user_message, position)
    if not turn:
        return None
    which = "first thing" if position == "first" else "most recent thing (before this turn)"
    logger.info("🧠 [GroundedRecall] positional=%s resolved actual turn for grounding.", position)
    return (
        f"[GROUNDED RECALL — this is the verbatim fact; answer from it, do not guess]\n"
        f"The {which} the user actually said to you in this conversation was:\n"
        f"“{turn}”\n"
        f"Answer their question using this real quote, naturally and in your own voice.\n\n"
    )
