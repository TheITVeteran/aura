"""Close the taste loop — learn Bryan's preference from his reactions.

The conversational amplifier stashes the features of the response it sent. When Bryan's
next message arrives, we read its tone toward that response (delight vs flat vs
correction) and nudge the TasteModel accordingly. Over time the amplifier selects more
of what actually lands with him — personalized inference-time alignment, no labels asked.

Reaction signal is deliberately conservative: only clear positive/negative reactions
move the weights; neutral turns do nothing (no spurious drift).
"""
from __future__ import annotations

import logging
import re
import threading

from core.brain.taste_model import get_taste_model
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ConversationOutcome")

_POSITIVE = re.compile(
    r"\b(thanks?|thank you|lol|lmao|haha+|love (?:it|that|this)|exactly|perfect|nice|"
    r"great|brilliant|yes!+|that's (?:it|right|good|perfect)|well said|good (?:point|call)|"
    r"underrated|so true|real|facts)\b",
    re.I,
)
_NEGATIVE = re.compile(
    r"\b(no[.,!]?$|wrong|that's wrong|not (?:what|right|true)|nope|huh\??|that's not|"
    r"makes no sense|incoherent|generic|boring|stop (?:saying|doing)|you (?:missed|didn't)|"
    r"that doesn't|come on|ugh)\b",
    re.I,
)

_lock = threading.RLock()
_pending: tuple[str, dict[str, float]] | None = None
_stats = {"recorded": 0, "positive": 0, "negative": 0, "neutral": 0}


def record_pending_response(text: str, features: dict[str, float]) -> None:
    """Remember the response just sent + its features, awaiting Bryan's reaction."""
    global _pending
    with _lock:
        _pending = (str(text or ""), dict(features or {}))
        _stats["recorded"] += 1


def _classify_reward(user_message: str) -> float:
    msg = str(user_message or "").strip()
    if not msg:
        return 0.0
    pos = bool(_POSITIVE.search(msg))
    neg = bool(_NEGATIVE.search(msg))
    if pos and not neg:
        return 1.0
    if neg and not pos:
        return -1.0
    return 0.0


def register_reaction(user_message: str) -> float | None:
    """Judge the reaction to the pending response and update the taste model.

    Returns the reward applied (+1/-1), or None if there was nothing to learn from.
    """
    global _pending
    with _lock:
        pending = _pending
        _pending = None
    if pending is None:
        return None
    _text, features = pending
    if not features:
        return None
    reward = _classify_reward(user_message)
    if reward == 0.0:
        _stats["neutral"] += 1
        return None
    try:
        get_taste_model().update(features, reward)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conversation_outcome_update", exc)
        return None
    _stats["positive" if reward > 0 else "negative"] += 1
    logger.info("🗣️ [TasteLoop] reaction reward=%+.0f applied to taste model.", reward)
    return reward


def stats() -> dict[str, int]:
    with _lock:
        return dict(_stats)


def reset() -> None:
    global _pending
    with _lock:
        _pending = None
