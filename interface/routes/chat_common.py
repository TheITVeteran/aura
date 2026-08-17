"""Primitives every chat-lane module shares.

`interface/routes/chat.py` was one 30,000-line module with 462 top-level
functions. Splitting it means several modules need the same logger, the same
recoverable-error tuple and the same request-scoped context variables. They
live here so the import graph runs one way — this module, then the lane
modules, then chat.py — and never back into chat.py.
"""

from __future__ import annotations

from typing import Any

import time

import collections

import dataclasses

from contextvars import ContextVar
from fastapi import APIRouter, Depends, HTTPException, Request
import asyncio
import json
import logging
from core.runtime import resource_psutil as psutil

_CHAT_BLOCKING_PREFLIGHT_TIMEOUT_S = 2.0

_CHAT_RECOVERABLE_ERRORS = (
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
    ImportError,
    LookupError,
    json.JSONDecodeError,
    asyncio.InvalidStateError,
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    HTTPException,
    psutil.Error,
)

_CHAT_REQUEST_PRINCIPAL: ContextVar[str] = ContextVar(
    "aura_chat_request_principal",
    default="",
)

_CHAT_REQUEST_SURFACE: ContextVar[str] = ContextVar(
    "aura_chat_request_surface",
    default="",
)

_MAX_CONVERSATION_LOG_EXCHANGES = 500

_conversation_log: list[dict] = []  # In-memory session log for current runtime

_locks = {}

logger = logging.getLogger("Aura.Server.Chat")

from contextvars import ContextVar

_CHAT_DELIVERY_IDEMPOTENCY_KEY: ContextVar[str] = ContextVar(
    "aura_chat_delivery_idempotency_key",
    default="",
)

_CHAT_PENDING_DELIVERY_CLAIM: ContextVar[tuple[str, tuple[str, ...]]] = ContextVar(
    "aura_chat_pending_delivery_claim",
    default=("", ()),
)

_CHAT_SESSION_ID_MAX_CHARS = 64

_INTERNAL_SURFACE_CONTEXT: ContextVar[str] = ContextVar(
    "aura_internal_surface_context",
    default="",
)

_UNSET = object()

import re

_EXPLICIT_NON_EXECUTION_RE = re.compile(
    r"\b(?:do not execute|don't execute|without executing|before executing|"
    r"do not use tools|don't use tools|no tool use|no tools?|"
    r"do not run|don't run|do not open|don't open)\b",
    re.IGNORECASE,
)

_INCOMPLETE_TAIL_WORDS = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "called",
    "create",
    "for",
    "from",
    "if",
    "into",
    "make",
    "named",
    "open",
    "of",
    "or",
    "save",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "th",
    "to",
    "when",
    "where",
    "while",
    "write",
    "with",
}

_INTERNAL_STATE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:cognitive baseline tick\s*\d+)"
    r"|(?:monitoring internal state)"
    r"|(?:baseline_continuity)"
    r"|(?:In the [\d.]+ (?:seconds|minutes) just passed)"
    r"|(?:Pending initiatives:)"
    r"|(?:Reconcile continuity gap)"
    r"|(?:Drive alert:.*depleted)"
    r"|(?:Phenomenal Surge:)"
    r"|(?:Winner:.*Content:)"
)

_LOCAL_CHOICE_REFERENCE_RE = re.compile(r"\b(?:what|which)\s+one\b", re.IGNORECASE)

_ORGAN_INERT_STREAKS: dict[str, int] = {}

MAX_CHAT_MESSAGE_BYTES = 64 * 1024  # 64KB
