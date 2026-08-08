"""Private, bounded persistence owner for encrypted session-memory pins."""
from __future__ import annotations

import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.memory.session_pin_cipher import SESSION_PIN_ENVELOPE_SCHEMA
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.lockdep import checked_lock

SESSION_PIN_LEDGER_FILENAME = "session_memory_pins.jsonl"
SESSION_PIN_LEDGER_MAX_RECORDS = 500
SESSION_PIN_LEDGER_MAX_BYTES = 2 * 1024 * 1024
_REQUIRED_ENVELOPE_FIELDS = frozenset(
    {"schema", "key_id", "record_id", "nonce_b64", "ciphertext_b64"}
)
_PROCESS_LOCK = checked_lock("memory.session_pin_ledger", reentrant=True)


class SessionPinLedgerError(RuntimeError):
    """The encrypted pin ledger could not be read or committed safely."""


@dataclass(frozen=True, slots=True)
class SessionPinLedgerSnapshot:
    lines: tuple[str, ...]
    truncated: bool
    permissions_repair_required: bool


class SessionPinLedger:
    """Own one fixed-name encrypted pin ledger.

    The caller may inject a directory for a hermetic test, but cannot turn this
    owner into a general file-write surface: the filename and record schema are
    fixed, reads are bounded, and every replacement traverses FileWriteGateway.
    """

    def __init__(self, path: Path | str) -> None:
        candidate = Path(path).expanduser()
        if candidate.name != SESSION_PIN_LEDGER_FILENAME:
            raise ValueError(
                f"session pin ledger filename must be {SESSION_PIN_LEDGER_FILENAME!r}"
            )
        self.path = candidate
        self._lock_path = candidate.with_name(f".{candidate.name}.lock")

    @contextmanager
    def transaction(self) -> Iterator[SessionPinLedger]:
        """Serialize one read/transform/replace across threads and processes."""

        with local_internal_governed_scope(
            "memory.session_pin_ledger",
            domain="memory_write",
            constraints={
                "fixed_filename": SESSION_PIN_LEDGER_FILENAME,
                "encrypted_records_only": True,
            },
        ):
            with _PROCESS_LOCK:
                with interprocess_file_lock(self._lock_path):
                    yield self

    def read_snapshot(self) -> SessionPinLedgerSnapshot:
        """Read only the bounded tail while proving a stable regular inode."""

        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            if not nofollow:
                raise SessionPinLedgerError("session_pin_ledger_requires_o_nofollow")
            descriptor = os.open(self.path, flags | nofollow)
        except FileNotFoundError:
            return SessionPinLedgerSnapshot((), False, False)
        except OSError as exc:
            raise SessionPinLedgerError(
                f"session_pin_ledger_open_failed:{type(exc).__qualname__}"
            ) from exc

        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
            ):
                raise SessionPinLedgerError("session_pin_ledger_identity_invalid")
            truncated = before.st_size > SESSION_PIN_LEDGER_MAX_BYTES
            if truncated:
                os.lseek(descriptor, -SESSION_PIN_LEDGER_MAX_BYTES, os.SEEK_END)
            remaining = min(before.st_size, SESSION_PIN_LEDGER_MAX_BYTES) + 1
            chunks: list[bytes] = []
            while remaining > 0:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise SessionPinLedgerError("session_pin_ledger_changed_during_read")
            payload = b"".join(chunks)
            if len(payload) > SESSION_PIN_LEDGER_MAX_BYTES:
                raise SessionPinLedgerError("session_pin_ledger_read_bound_exceeded")
        finally:
            os.close(descriptor)

        lines = payload.decode("utf-8", errors="replace").splitlines()
        if truncated and lines:
            # A tail read can begin in the middle of a JSONL row.
            lines = lines[1:]
        if len(lines) > SESSION_PIN_LEDGER_MAX_RECORDS:
            truncated = True
            lines = lines[-SESSION_PIN_LEDGER_MAX_RECORDS:]
        return SessionPinLedgerSnapshot(
            lines=tuple(lines),
            truncated=truncated,
            permissions_repair_required=stat.S_IMODE(before.st_mode) != 0o600,
        )

    def commit_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Atomically publish a canonical, bounded encrypted record set."""

        if len(records) > SESSION_PIN_LEDGER_MAX_RECORDS:
            raise SessionPinLedgerError("session_pin_ledger_record_bound_exceeded")
        import json

        rows: list[str] = []
        for record in records:
            normalized = {str(key): str(value or "") for key, value in record.items()}
            if (
                frozenset(normalized) != _REQUIRED_ENVELOPE_FIELDS
                or normalized.get("schema") != SESSION_PIN_ENVELOPE_SCHEMA
            ):
                raise SessionPinLedgerError("session_pin_ledger_record_schema_invalid")
            rows.append(json.dumps(normalized, ensure_ascii=True, sort_keys=True))
        payload = "".join(f"{row}\n" for row in rows)
        if len(payload.encode("utf-8")) > SESSION_PIN_LEDGER_MAX_BYTES:
            raise SessionPinLedgerError("session_pin_ledger_payload_bound_exceeded")
        get_file_write_gateway().write_text(
            self.path,
            payload,
            source="memory.session_pin_ledger",
        )


__all__ = [
    "SESSION_PIN_LEDGER_FILENAME",
    "SESSION_PIN_LEDGER_MAX_BYTES",
    "SESSION_PIN_LEDGER_MAX_RECORDS",
    "SessionPinLedger",
    "SessionPinLedgerError",
    "SessionPinLedgerSnapshot",
]
