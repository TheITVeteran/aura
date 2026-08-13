"""Exact ownership for evidence produced while serving one conversation turn.

``ContextVar`` answers *which turn is this code running under?* It does not
answer *may this task write evidence for that turn?* Async tasks inherit a
copy of their parent's context, so putting a mutable collector in a ContextVar
lets every background task spawned during a request mutate the foreground
reply's evidence. Making the value immutable fixes contamination but also
makes legitimate child-task writes invisible to the parent.

This module separates the two questions. A custody object is shared by the
turn, but only the owner execution may use it. A deliberate child receives a
one-use lease from the owner and joins explicitly. Incidental background tasks,
thread-pool hops, and other turns inherit no authority merely because Python
copied their context.
"""

from __future__ import annotations

import asyncio
import contextvars
import secrets
import threading
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from core.conversation.session_scope import (
    conversation_session_var,
    conversation_turn_var,
    current_conversation_session,
    current_conversation_turn,
    normalize_conversation_id,
    normalize_conversation_turn_id,
)

__all__ = [
    "EvidenceParticipantLease",
    "TurnEvidenceCustody",
    "bind_turn_evidence_custody",
    "current_turn_evidence_custody",
    "join_turn_evidence_custody",
    "run_as_turn_evidence_participant",
]


def _execution_identity() -> tuple[int, int]:
    """Stable identity for one thread/task execution locus."""

    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return (threading.get_ident(), id(task) if task is not None else 0)


@dataclass(frozen=True, slots=True)
class EvidenceParticipantLease:
    """One-use capability allowing an intentional child to join a turn."""

    token: str
    session_id: str
    turn_id: str
    purpose: str
    issued_at: float


class TurnEvidenceCustody:
    """Synchronized evidence owned by one exact session/turn/task tree."""

    def __init__(self, *, session_id: str, turn_id: str) -> None:
        session = normalize_conversation_id(session_id)
        turn = normalize_conversation_turn_id(turn_id)
        if not session or not turn:
            raise ValueError("turn evidence custody requires exact session and turn identities")
        self.session_id = session
        self.turn_id = turn
        self.started_at = time.time()
        self._lock = threading.RLock()
        self._owner = _execution_identity()
        self._participants: set[tuple[int, int]] = {self._owner}
        self._leases: dict[str, EvidenceParticipantLease] = {}
        self._receipts: list[dict[str, Any]] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _identity_matches(self) -> bool:
        return (
            current_conversation_session() == self.session_id
            and current_conversation_turn() == self.turn_id
        )

    def admits_current_execution(self) -> bool:
        with self._lock:
            return (
                not self._closed
                and self._identity_matches()
                and _execution_identity() in self._participants
            )

    def issue_child_lease(self, purpose: str) -> EvidenceParticipantLease:
        """Issue a one-use lease; ambient child tasks receive no authority."""

        if not self.admits_current_execution():
            raise PermissionError("only an admitted turn participant may issue an evidence lease")
        lease = EvidenceParticipantLease(
            token=secrets.token_urlsafe(24),
            session_id=self.session_id,
            turn_id=self.turn_id,
            purpose=" ".join(str(purpose or "turn child").split())[:120],
            issued_at=time.time(),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("turn evidence custody is closed")
            self._leases[lease.token] = lease
        return lease

    @contextmanager
    def join(self, lease: EvidenceParticipantLease) -> Iterator["TurnEvidenceCustody"]:
        """Consume ``lease`` and admit this exact child for the block."""

        participant = _execution_identity()
        with self._lock:
            issued = self._leases.pop(str(getattr(lease, "token", "") or ""), None)
            if (
                self._closed
                or issued is None
                or issued != lease
                or lease.session_id != self.session_id
                or lease.turn_id != self.turn_id
                or not self._identity_matches()
            ):
                raise PermissionError("invalid, stale, or cross-turn evidence participant lease")
            self._participants.add(participant)
        try:
            yield self
        finally:
            with self._lock:
                if participant != self._owner:
                    self._participants.discard(participant)

    def clear_receipts(self) -> bool:
        if not self.admits_current_execution():
            return False
        with self._lock:
            self._receipts.clear()
        return True

    def append_receipt(self, receipt: dict[str, Any]) -> bool:
        if not self.admits_current_execution():
            return False
        row = dict(receipt)
        row["session_id"] = self.session_id
        row["turn_id"] = self.turn_id
        with self._lock:
            if self._closed:
                return False
            if len(self._receipts) < 64:
                self._receipts.append(row)
                return True
        return False

    def receipts(self) -> tuple[dict[str, Any], ...]:
        if not self.admits_current_execution():
            return ()
        with self._lock:
            return tuple(dict(item) for item in self._receipts)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._leases.clear()
            self._participants.clear()


_ACTIVE_CUSTODY: contextvars.ContextVar[TurnEvidenceCustody | None] = contextvars.ContextVar(
    "aura_turn_evidence_custody",
    default=None,
)


def current_turn_evidence_custody() -> TurnEvidenceCustody | None:
    return _ACTIVE_CUSTODY.get()


@contextmanager
def bind_turn_evidence_custody(
    *,
    session_id: str,
    turn_id: str,
) -> Iterator[TurnEvidenceCustody]:
    """Own evidence for exactly one conversation turn and close it on exit."""

    custody = TurnEvidenceCustody(session_id=session_id, turn_id=turn_id)
    session_token = conversation_session_var.set(custody.session_id)
    turn_token = conversation_turn_var.set(custody.turn_id)
    custody_token = _ACTIVE_CUSTODY.set(custody)
    try:
        yield custody
    finally:
        custody.close()
        _ACTIVE_CUSTODY.reset(custody_token)
        conversation_turn_var.reset(turn_token)
        conversation_session_var.reset(session_token)


@contextmanager
def join_turn_evidence_custody(
    lease: EvidenceParticipantLease,
) -> Iterator[TurnEvidenceCustody]:
    """Join the inherited custody only with an explicit parent-issued lease."""

    custody = current_turn_evidence_custody()
    if custody is None:
        raise RuntimeError("no turn evidence custody is active")
    with custody.join(lease):
        yield custody


def run_as_turn_evidence_participant(
    awaitable: Awaitable[Any],
    *,
    purpose: str,
) -> Awaitable[Any]:
    """Wrap a deliberate child coroutine with a one-use turn evidence lease."""

    custody = current_turn_evidence_custody()
    if custody is None or not custody.admits_current_execution():
        return awaitable
    lease = custody.issue_child_lease(purpose)

    async def _run() -> Any:
        with join_turn_evidence_custody(lease):
            return await awaitable

    return _run()
