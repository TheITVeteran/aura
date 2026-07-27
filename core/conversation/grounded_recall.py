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
from typing import Any, Literal

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


# "This conversation" means what a person means by it: the run of turns since
# the last long silence. The buffer this resolver reads is the live working
# memory — global, trimmed to a fixed length, and shared by every origin the
# runtime has — so without a boundary "the first thing I asked you in this
# conversation" reaches back into whatever happened to survive in it.
#
# Measured live 2026-07-27, on the first battery turn of a fresh conversation:
#
#     Q: "What was the very first thing I asked me in this conversation?"
#     A: "The first thing you asked me was: 'If I had a whole Saturday with no
#         obligations, what would I do?'"
#
# The grounding block fired and reported success; the turn it grounded on was
# simply not from this conversation. A confident quote of the wrong turn is
# worse than an admission, because it is indistinguishable from memory.
_CONVERSATION_GAP_S = 45 * 60


def _entry_is_from_the_human(entry: dict) -> bool:
    """Did a person type this, or did the runtime write it to itself?

    Working memory carries entries from many origins, and ``role`` alone does
    not separate them — several writers append ``role="user"`` directly without
    going through ``role_for_origin``. An origin that is not user-anchored, or
    an entry marked ephemeral, is the runtime talking to itself.
    """
    if entry.get("ephemeral"):
        return False
    origin = entry.get("origin")
    if origin is None:
        # No origin recorded: trust the chat surface's own marker when present,
        # and otherwise accept it — the chat route appends without an origin.
        source = str((entry.get("metadata") or {}).get("source", "") or "")
        return source in {"", "chat_api", "desktop-ui", "desktop_ui"}
    try:
        from core.state.aura_state import _origin_is_user_anchored

        return bool(_origin_is_user_anchored(origin))
    except (ImportError, AttributeError, TypeError, ValueError):
        return True


def _within_current_conversation(history: Any) -> list[dict]:
    """Trailing run of entries with no gap longer than a long silence."""
    entries = [entry for entry in (history or []) if isinstance(entry, dict)]
    if not entries:
        return []
    stamped = [entry for entry in entries if _entry_timestamp(entry) is not None]
    if len(stamped) < 2:
        return entries
    start_index = 0
    for index in range(len(entries) - 1, 0, -1):
        current = _entry_timestamp(entries[index])
        previous = _entry_timestamp(entries[index - 1])
        if current is None or previous is None:
            continue
        if current - previous > _CONVERSATION_GAP_S:
            start_index = index
            break
    return entries[start_index:]


def _entry_timestamp(entry: dict) -> float | None:
    try:
        value = entry.get("timestamp")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _history_user_turns(history: Any, exclude_norm: str) -> list[str]:
    """What the human actually said, in this conversation, oldest first."""
    turns: list[str] = []
    for entry in _within_current_conversation(history):
        if entry.get("role") != "user" or not _entry_is_from_the_human(entry):
            continue
        content = str(entry.get("content", "") or "").strip()
        if content and content.lower() != exclude_norm:
            turns.append(content)
    return turns


def _working_memory_user_turns(exclude_norm: str) -> list[str]:
    """User utterances from the live AuraState working memory (chat's own buffer)."""
    for getter in (
        lambda: __import__("core.container", fromlist=["ServiceContainer"]).ServiceContainer.get("aura_state", default=None),
        lambda: __import__("core.runtime.service_access", fromlist=["resolve_state_repository"]).resolve_state_repository(default=None),
    ):
        try:
            obj = getter()
        except (ImportError, AttributeError, RuntimeError, TypeError):
            continue
        wm = getattr(getattr(obj, "cognition", None), "working_memory", None)
        if not isinstance(wm, list):
            wm = getattr(getattr(getattr(obj, "_current", None), "cognition", None), "working_memory", None)
        if isinstance(wm, list) and wm:
            return _history_user_turns(wm, exclude_norm)
    return []


def _transcript_user_turns(exclude_norm: str) -> list[str]:
    """Fallback: user utterances from the UnifiedTranscript singleton."""
    try:
        from core.conversation.unified_transcript import UnifiedTranscript

        entries = list(getattr(UnifiedTranscript.get_instance(), "_entries", []) or [])
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("Grounded recall: transcript unavailable: %s", exc)
        return []

    turns: list[str] = []
    for e in entries:
        if getattr(e, "role", "") != "user":
            continue
        content = str(getattr(e, "content", "") or "").strip()
        if content and content.lower() != exclude_norm:
            turns.append(content)
    return turns


def _user_turns(exclude: str, history: Any = None) -> list[str]:
    """Earliest→latest user utterances this session, current turn excluded.

    A caller-supplied ``history`` (the chat route's live working memory) is the
    most reliable source; otherwise resolve the live AuraState working memory,
    then fall back to the UnifiedTranscript.
    """
    exclude_norm = (exclude or "").strip().lower()
    turns = _history_user_turns(history, exclude_norm)
    if not turns:
        turns = _working_memory_user_turns(exclude_norm)
    if not turns:
        turns = _transcript_user_turns(exclude_norm)
    return turns


def resolve_positional_turn(
    user_message: str, position: RecallPosition, history: Any = None
) -> str | None:
    """Retrieve the actual first/last user turn this session (excluding current)."""
    turns = _user_turns(exclude=user_message, history=history)
    if not turns:
        return None
    chosen = turns[0] if position == "first" else turns[-1]
    chosen = chosen.strip()
    if len(chosen) > _MAX_GROUNDED_CHARS:
        chosen = chosen[: _MAX_GROUNDED_CHARS - 1].rstrip() + "…"
    return chosen or None


def build_grounded_recall_context(user_message: str, history: Any = None) -> str | None:
    """Authoritative grounding block for a positional-recall turn, or None.

    ``history`` is the caller's live conversation buffer (list of {role, content});
    when omitted the resolver falls back to the live working memory / transcript.
    The block states the retrieved fact and instructs the model to answer from it
    in its own voice — so the reply is grounded in what actually happened instead
    of a confident confabulation.  Returns ``None`` when there is no positional
    intent or no prior turn to ground on (e.g. brand-new conversation).
    """
    position = detect_positional_recall(user_message)
    if position is None:
        return None
    turn = resolve_positional_turn(user_message, position, history=history)
    if not turn:
        logger.info(
            "🧠 [GroundedRecall] positional=%s detected but no prior turn found "
            "(history_len=%s) — cannot ground, letting model answer.",
            position,
            len(history) if isinstance(history, list) else "n/a",
        )
        return None
    which = "first thing" if position == "first" else "most recent thing (before this turn)"
    logger.info("🧠 [GroundedRecall] positional=%s resolved actual turn for grounding.", position)
    return (
        f"[GROUNDED RECALL — this is the verbatim fact; answer from it, do not guess]\n"
        f"The {which} the user actually said to you in this conversation was:\n"
        f"“{turn}”\n"
        "The quoted speaker is the user, not you. Preserve that role boundary: refer to "
        "it as what they or 'you' said, never as something you said.\n"
        f"Answer their question using this real quote, naturally and in your own voice.\n\n"
    )


def repair_grounded_recall_speaker_attribution(
    user_message: str,
    response_text: str,
) -> tuple[str, bool]:
    """Correct first-person adoption of a retrieved user utterance.

    This only applies to positional recall questions where the user explicitly
    asks about what they said. It does not ban first-person language elsewhere.
    """

    response = str(response_text or "").strip()
    if not response or detect_positional_recall(user_message) is None:
        return response, False
    if not re.search(r"\b(?:i|my|me)\b", str(user_message or ""), re.IGNORECASE):
        return response, False

    sentence_end = re.search(r"(?<=[.!?])(?:\s|$)", response)
    boundary = sentence_end.start() if sentence_end else len(response)
    first_sentence = response[:boundary]
    remainder = response[boundary:]
    leading = re.match(r"^(\s*[\"'“‘]?)", first_sentence)
    prefix = leading.group(1) if leading else ""
    body = first_sentence[len(prefix) :]
    if not re.match(r"^(?:I\b|I'm\b|I've\b|I'd\b|My\b|Mine\b)", body, re.IGNORECASE):
        return response, False

    shifted = re.sub(r"\bI am\b", "you are", body, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI'm\b", "you're", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI have\b", "you have", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI've\b", "you've", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI was\b", "you were", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI'd\b", "you'd", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bI\b", "you", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bmy\b", "your", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bmine\b", "yours", shifted, flags=re.IGNORECASE)
    shifted = re.sub(r"\bme\b", "you", shifted, flags=re.IGNORECASE)
    shifted = shifted.lstrip()
    if shifted.lower().startswith("you said "):
        corrected = shifted
    else:
        corrected = f"You said {shifted}"
    corrected = corrected[:1].upper() + corrected[1:]
    return f"{prefix}{corrected}{remainder}", True
