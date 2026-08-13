"""Request-local conversation identity shared by chat, memory, and speech."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

MAX_CONVERSATION_ID_CHARS: Final = 128
MAX_CONVERSATION_TURN_ID_CHARS: Final = 128

conversation_session_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_conversation_session",
    default="",
)
conversation_turn_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "aura_conversation_turn",
    default="",
)

LOCAL_CONVERSATION_ID: Final = "local"
# Native windows are an owner surface, not anonymous internal cognition.  The
# HTTP desktop and voice routes already derive this principal key for the same
# machine, so non-HTTP owner surfaces must join it rather than minting a third
# local conversation.
LOCAL_OWNER_CONVERSATION_ID: Final = "127.0.0.1"


def normalize_conversation_id(value: object) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if any(ord(character) < 32 for character in normalized):
        return ""
    return normalized[:MAX_CONVERSATION_ID_CHARS]


def current_conversation_session(default: str = "") -> str:
    return normalize_conversation_id(conversation_session_var.get()) or normalize_conversation_id(
        default
    )


def normalize_conversation_turn_id(value: object) -> str:
    normalized = " ".join(str(value or "").strip().split())
    if any(ord(character) < 32 for character in normalized):
        return ""
    return normalized[:MAX_CONVERSATION_TURN_ID_CHARS]


def current_conversation_turn(default: str = "") -> str:
    return normalize_conversation_turn_id(
        conversation_turn_var.get()
    ) or normalize_conversation_turn_id(default)


@contextmanager
def conversation_session_scope(session_id: str) -> Iterator[str]:
    normalized = normalize_conversation_id(session_id)
    if not normalized:
        raise ValueError("conversation session identity is required")
    token = conversation_session_var.set(normalized)
    try:
        yield normalized
    finally:
        conversation_session_var.reset(token)


__all__ = [
    "LOCAL_CONVERSATION_ID",
    "LOCAL_OWNER_CONVERSATION_ID",
    "MAX_CONVERSATION_ID_CHARS",
    "MAX_CONVERSATION_TURN_ID_CHARS",
    "conversation_session_scope",
    "conversation_session_var",
    "conversation_turn_var",
    "current_conversation_session",
    "current_conversation_turn",
    "normalize_conversation_id",
    "normalize_conversation_turn_id",
]
