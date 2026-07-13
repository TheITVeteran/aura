"""Memory consent / privacy controls.

Audit-driven mode set: remember_always / ask_before_remembering /
session_only / private_mode / forget. Explicit user commands like
"forget this", "remember this part", "private mode" are honored at
write time.
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MemoryConsentMode(StrEnum):
    REMEMBER_ALWAYS = "remember_always"
    ASK_BEFORE_REMEMBERING = "ask_before_remembering"
    SESSION_ONLY = "session_only"
    PRIVATE_MODE = "private_mode"


@dataclass
class StoredRecordRef:
    record_id: str
    family: str
    stored_at: float


class MemoryConsentPolicy:
    def __init__(self, *, default_mode: MemoryConsentMode = MemoryConsentMode.ASK_BEFORE_REMEMBERING):
        self.mode = default_mode
        self._session_only_records: list[StoredRecordRef] = []
        self._lock = threading.RLock()

    def set_mode(self, mode: MemoryConsentMode) -> None:
        self.mode = mode

    def may_persist_long_term(self) -> bool:
        return self.mode == MemoryConsentMode.REMEMBER_ALWAYS

    def needs_user_approval(self) -> bool:
        return self.mode == MemoryConsentMode.ASK_BEFORE_REMEMBERING

    def is_session_only(self) -> bool:
        return self.mode == MemoryConsentMode.SESSION_ONLY

    def is_private(self) -> bool:
        return self.mode == MemoryConsentMode.PRIVATE_MODE

    def register_session_record(self, ref: StoredRecordRef) -> None:
        if self.mode == MemoryConsentMode.SESSION_ONLY:
            with self._lock:
                self._session_only_records.append(ref)

    def session_only_records(self) -> list[StoredRecordRef]:
        with self._lock:
            return list(self._session_only_records)

    def clear_session_records(self) -> list[StoredRecordRef]:
        with self._lock:
            cleared = list(self._session_only_records)
            self._session_only_records.clear()
            return cleared


_global: MemoryConsentPolicy | None = None


def get_memory_consent_policy() -> MemoryConsentPolicy:
    global _global
    if _global is None:
        _global = MemoryConsentPolicy()
    return _global


def reset_memory_consent_policy() -> None:
    global _global
    _global = None


# --- user command parser ---------------------------------------------------


CONSENT_COMMANDS = {
    "remember always": MemoryConsentMode.REMEMBER_ALWAYS,
    "always remember": MemoryConsentMode.REMEMBER_ALWAYS,
    "ask before remembering": MemoryConsentMode.ASK_BEFORE_REMEMBERING,
    "session only": MemoryConsentMode.SESSION_ONLY,
    "private mode": MemoryConsentMode.PRIVATE_MODE,
    "go private": MemoryConsentMode.PRIVATE_MODE,
    "go private mode": MemoryConsentMode.PRIVATE_MODE,
}

_COMMAND_PREFIX = re.compile(
    r"^(?:aura[,:]?\s*)?(?:(?:can|could|would) you\s+)?(?:please\s+)?",
    re.IGNORECASE,
)
_COMMAND_SUFFIX = re.compile(r"(?:\s+please)?(?:\s+now)?[.!?]*$", re.IGNORECASE)
_DELETE_ALL_COMMANDS = frozenset(
    {
        "forget everything about me",
        "delete all relational memory",
        "erase all relationship memory",
    }
)


def _normalized_command_body(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    normalized = _COMMAND_PREFIX.sub("", normalized, count=1)
    normalized = _COMMAND_SUFFIX.sub("", normalized, count=1)
    return normalized.strip()


def parse_consent_command(text: str) -> MemoryConsentMode | None:
    body = _normalized_command_body(text)
    for command, mode in CONSENT_COMMANDS.items():
        if body == command:
            return mode
    return None


def is_delete_all_relational_memory_command(text: str) -> bool:
    return _normalized_command_body(text) in _DELETE_ALL_COMMANDS


def apply_relational_memory_command(
    authority: Any,
    agent_id: str,
    message: str,
    *,
    receipt_id: str = "",
) -> dict[str, Any] | None:
    """Apply one explicit exact-agent relational-memory control command."""
    exact_id = " ".join(str(agent_id or "").strip().split())[:160]
    if not exact_id:
        raise ValueError("relational memory control requires an exact agent_id")
    normalized = " ".join(str(message or "").strip().lower().split())
    mode = parse_consent_command(normalized)
    delete_all = is_delete_all_relational_memory_command(normalized)
    if mode is None and not delete_all:
        return None
    evidence = hashlib.sha256(
        f"{exact_id}\n{normalized}".encode("utf-8", errors="replace")
    ).hexdigest()
    command_receipt_id = str(
        receipt_id or f"user-command-evidence-{evidence}"
    ).strip()[:200]
    if not command_receipt_id:
        raise ValueError("relational memory control requires command evidence")

    if delete_all:
        receipt = authority.delete_agent(
            exact_id,
            authorization_receipt_id=command_receipt_id,
        )
        return {
            "mode": "deleted",
            "receipt_id": receipt.receipt_id,
            "deleted_records": len(receipt.record_ids),
        }

    if mode is None:
        raise RuntimeError("relational memory command classification lost its mode")
    grant = None
    if mode == MemoryConsentMode.REMEMBER_ALWAYS:
        grant = authority.replace_consent(
            exact_id,
            kinds=authority.supported_kinds(),
            operations=["persist", "recall", "prompt"],
            receipt_id=command_receipt_id,
            source="explicit_user_command",
        )
    elif mode == MemoryConsentMode.SESSION_ONLY:
        grant = authority.replace_consent(
            exact_id,
            kinds=authority.supported_kinds(),
            operations=["recall", "prompt"],
            receipt_id=command_receipt_id,
            source="explicit_user_command",
        )
    else:
        authority.revoke_consent(
            exact_id,
            receipt_id=f"{command_receipt_id}:replace"[:200],
            delete_records=False,
        )
    return {
        "mode": mode.value,
        "grant_id": grant.grant_id if grant is not None else "",
        "persistence_requested": bool(
            grant is not None and "persist" in grant.operations
        ),
        "persistence_allowed": bool(
            grant is not None
            and "persist" in grant.operations
            and authority.persistence_available
        ),
        "persistence_available": bool(authority.persistence_available),
        "prompt_use_allowed": bool(
            grant is not None and "prompt" in grant.operations
        ),
    }


def is_forget_command(text: str) -> bool:
    lower = text.lower().strip()
    return any(
        cmd in lower
        for cmd in (
            "forget this",
            "delete this memory",
            "erase that",
            "forget the session",
            "delete the movie session",
        )
    )
