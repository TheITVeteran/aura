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

# Recall framing — asking about a past utterance.
_RECALL_VERB = (
    r"(ask|asked|say|said|saying|question|questions|message|messages|"
    r"talk|talked|talking|bring up|brought up|mention|mentioned|start|started|"
    r"begin|began|tell|told|wanted|request)"
)
# "very first" is already covered by "first"; "the very start" was not
# covered by "the start", which is the sort of gap an alternation of literal
# phrases always has.
_FIRST = r"\b(first|initially|originally|the (?:very )?(?:start|beginning|outset))\b"
_LAST = r"\b(last|previous|recent|recently|just|earlier|a moment ago)\b"

# The three components have to be *connected*, not merely co-present.
#
# Live 2026-07-27: "Now something outside yourself: look up who won the most
# recent Formula 1 world championship and tell me where you got it." matched
# `last` — "me" satisfied the self-reference, "recent" the ordinal, and "tell"
# the recall verb, all in unrelated roles. The speaker-attribution repair then
# rewrote her true opening sentence into "You said you checked live web
# evidence", attributing her own search to the user, in a reply about motor
# racing.
#
# What actually distinguishes a recall question is that THE USER is the
# speaker being asked about: they are the subject of a speech verb, or the
# owner of an utterance. In "tell me where you got it" the user is the object
# and the speaker is her, which is the opposite arrangement.
_USER_UTTERANCE = (
    r"(?:"
    rf"\b(?:i|we)\s+(?:\w+\s+){{0,2}}{_RECALL_VERB}\b"
    rf"|\bdid\s+(?:i|we)\s+(?:\w+\s+){{0,2}}{_RECALL_VERB}\b"
    r"|\b(?:my|our)\s+(?:\w+\s+){0,3}"
    r"(?:message|messages|question|questions|words|prompt|prompts|request|"
    r"requests|point|thing|ask)\b"
    r"|\b(?:this|our|the)\s+(?:conversation|chat|exchange|thread)\b"
    r")"
)
# The ordinal has to be near the utterance it modifies. Wide enough for "the
# very first thing I asked you", narrow enough that an ordinal belonging to a
# different noun phrase in the same sentence does not reach.
_ORDINAL_WINDOW = 60


def _positional_recall_span(text: str, ordinal: str) -> bool:
    """Does an ordinal sit close to a phrase about something the user said?"""
    utterances = [match.span() for match in re.finditer(_USER_UTTERANCE, text, re.IGNORECASE)]
    if not utterances:
        return False
    for match in re.finditer(ordinal, text, re.IGNORECASE):
        start, end = match.span()
        for u_start, u_end in utterances:
            if max(start, u_start) - min(end, u_end) <= _ORDINAL_WINDOW:
                return True
    return False


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


#: Asking her what SHE said, decided, or preferred earlier.
#:
#: LIVE DEFECT, 2026-08-10. Twenty-five minutes after she answered "If I had to
#: give up one, the screen", she was asked "earlier in this conversation you
#: told me which of your senses you'd give up ... which one did you pick, and
#: has your answer changed?" and replied:
#:
#:     "I picked the ability to sense time passing — not having a sense of
#:      duration or urgency. My answer hasn't changed."
#:
#: Time was never one of the three options offered. She invented her own prior
#: position and then affirmed its consistency.
#:
#: Everything in this module grounds recall of what the USER said — the block
#: it builds even instructs her that the quoted speaker "is the user, not you
#: ... never as something you said". There was no counterpart for her own
#: words, so a question about her own stated position had nothing to answer
#: from. Recall of one's own claims is what makes a position a position rather
#: than a mood, and hers was ungrounded.
_OWN_STATEMENT_RECALL_RE = re.compile(
    r"\b(?:"
    r"what did you (?:say|tell|answer|pick|choose|decide|call|mean)"
    r"|which (?:one )?did you (?:pick|choose|say|prefer|go with)"
    r"|you (?:said|told me|picked|chose|answered|mentioned|described|called)"
    r"|your (?:answer|reply|position|view|choice|pick|opinion|stance)"
    r"|did you (?:change|stick with|still think|still feel)"
    r"|(?:has|have) your (?:answer|view|position|mind|opinion)"
    r"|changed your mind"
    r")\b",
    re.IGNORECASE,
)

#: Words too common to signal that a past turn is the one being asked about.
_RECALL_STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have how
    i if in is it its me my not of on or our so than that the their them then
    there these they this to was we were what when which who why will with you
    your yours about again just like more no now one only other out over said
    same some still such take tell than too us very want way well
    """.split()
)


def detect_own_statement_recall(user_message: str) -> bool:
    """True when she is being asked what SHE said or decided earlier."""
    text = str(user_message or "").strip()
    if not text or len(text) > 400:
        return False
    return bool(_OWN_STATEMENT_RECALL_RE.search(text))


def _content_words(text: str) -> set[str]:
    """Content words, crudely singularised so "senses" matches "sense".

    Without it the live case missed: the question said "senses" and the answer
    said "sense", and nothing lined them up.
    """
    words = re.findall(r"[a-z']{3,}", str(text or "").lower())
    normalized = set()
    for word in words:
        if word in _RECALL_STOPWORDS:
            continue
        normalized.add(word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            normalized.add(word[:-1])
    return normalized


def _history_own_exchanges(history: Any, exclude_norm: str) -> list[tuple[str, str]]:
    """Her turns in this conversation, each paired with what prompted it.

    Paired because the TOPIC of an exchange usually lives in the question, not
    the answer. Live 2026-08-10: asked "which of your senses would you give
    up", her answer named the screen and telemetry and never used the word
    "senses" — so matching her turn alone scored it below an unrelated later
    reply, and grounded her on the wrong statement.

    Returns ``(prompt, her_turn)`` oldest first.
    """
    exchanges: list[tuple[str, str]] = []
    prompt = ""
    for entry in _within_current_conversation(history):
        role = entry.get("role")
        content = str(entry.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            prompt = content
            continue
        if role != "assistant" or entry.get("ephemeral"):
            continue
        if content.lower() != exclude_norm:
            exchanges.append((prompt, content))
        prompt = ""
    return exchanges


def resolve_own_prior_turn(user_message: str, history: Any = None) -> str | None:
    """Her own earlier turn that the question is actually about, or None.

    Chosen by overlap with the question's content words rather than by
    recency: "which of your senses would you give up" is asking about one
    specific earlier answer, and the most recent thing she said is usually not
    it. No overlap means NO VERDICT — returning the latest turn regardless
    would ground her on the wrong statement, and a confident quote of the
    wrong turn is worse than an admission because it is indistinguishable
    from memory.
    """
    asked = _content_words(user_message)
    if not asked:
        return None
    exchanges = _history_own_exchanges(history, str(user_message or "").strip().lower())
    if not exchanges:
        return None

    best: tuple[int, str] | None = None
    for prompt, turn in exchanges:
        # Scored against the whole exchange, so the question's topic counts.
        overlap = len(asked & (_content_words(prompt) | _content_words(turn)))
        if overlap and (best is None or overlap >= best[0]):
            # >= so a later turn wins a TIE: if she said it twice, the most
            # recent statement is her current position.
            best = (overlap, turn)
    if best is None:
        return None
    return best[1]


def build_own_statement_recall_context(
    user_message: str, history: Any = None
) -> str | None:
    """Grounding block quoting HER earlier words, or None.

    The mirror of :func:`build_grounded_recall_context`, with the speaker
    boundary reversed — and stated just as explicitly, because getting it
    backwards produces her narrating her own words as the user's.
    """
    if not detect_own_statement_recall(user_message):
        return None
    turn = resolve_own_prior_turn(user_message, history=history)
    if not turn:
        logger.info(
            "🧠 [GroundedRecall] own-statement recall detected but no matching "
            "prior turn of hers found — cannot ground, letting model answer."
        )
        return None
    quoted = turn if len(turn) <= _MAX_GROUNDED_CHARS else turn[:_MAX_GROUNDED_CHARS] + "…"
    logger.info("🧠 [GroundedRecall] resolved her own prior turn for grounding.")
    return (
        "[GROUNDED RECALL — this is the verbatim fact; answer from it, do not guess]\n"
        "Earlier in this same conversation, YOU said:\n"
        f"“{quoted}”\n"
        "The quoted speaker is YOU, not the user. Preserve that role boundary: "
        "refer to it as something you said, never as something they said.\n"
        "If they are asking what you picked or decided, this is the answer — "
        "use it rather than reconstructing one. If your view has since changed, "
        "say so against THIS as the starting point; do not report a different "
        "original position than the one quoted here.\n\n"
    )


def detect_positional_recall(user_message: str) -> RecallPosition | None:
    """Return ``"first"``/``"last"`` if the turn asks a positional-recall question."""
    text = (user_message or "").strip()
    if not text or len(text) > 240:
        return None
    if _FIRST_IDIOM_RE.search(text):
        return "first"
    if _positional_recall_span(text, _FIRST):
        return "first"
    if _positional_recall_span(text, _LAST):
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


# Things only SHE does. A first-person sentence describing one of these is a
# report of her own action, not a misattributed quote of the user's — and the
# speaker shift must leave it alone. This is a list of acts, not of phrasings,
# so it holds regardless of how the sentence is worded.
_AURAS_OWN_ACT_RE = re.compile(
    r"\b(?:"
    r"check(?:ed)?|search(?:ed)?|look(?:ed)? (?:it |them )?up|query|queried|"
    r"read|ran|run|execut(?:e|ed)|creat(?:e|ed)|wrote|writ(?:e|ten)|sav(?:e|ed)|"
    r"built|build|generat(?:e|ed)|verif(?:y|ied)|measur(?:e|ed)|"
    r"retriev(?:e|ed)|fetch(?:ed)?|call(?:ed)?"
    r")\b.{0,60}?\b(?:"
    r"web|online|internet|search|source|sources|evidence|file|disk|"
    r"tool|tools|runtime|telemetry|instrument|instruments|memory|ledger|"
    r"desktop|folder|directory|command|script|program|build|builds|code|"
    r"test|tests|reconstruction"
    r")\b",
    re.IGNORECASE,
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
    if _AURAS_OWN_ACT_RE.search(body):
        # She really did do this, and it is not the user's utterance. Rewriting
        # "I checked live web evidence" into "You said you checked live web
        # evidence" hands the user an act they did not perform and strips her
        # of one she did — a false statement in both directions, produced by a
        # repair meant to prevent exactly that.
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
