"""Transactional, versioned persistence for Aura runtime settings."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import (
    atomic_append_text,
    atomic_write_text,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.settings_schema import (
    DEFAULT_VALUES,
    SCHEMA_BY_KEY,
    SETTINGS_APPLICATION_AUDIT_SCHEMA,
    SETTINGS_AUDIT_SCHEMA,
    SETTINGS_SCHEMA_NAME,
    SETTINGS_SCHEMA_VERSION,
    migrated_settings_snapshot,
    validate_setting_value,
    validate_settings_patch,
    validated_settings_snapshot,
)

logger = logging.getLogger("Aura.RuntimeSettings.ControlPlane")

_HISTORY_LIMIT = 16
_GENESIS_HASH = "sha256:" + "0" * 64
_APPLICATION_STATUSES = frozenset(
    {
        "applied",
        "awaiting_frontend",
        "deferred",
        "failed",
        "ready",
        "superseded",
        "unconfirmed",
        "unchanged",
    }
)


class SettingsControlPlaneError(RuntimeError):
    """Base class for settings persistence and integrity failures."""


class SettingsConflictError(SettingsControlPlaneError):
    def __init__(self, expected_revision: int, current_revision: int):
        super().__init__(
            f"settings_revision_conflict:expected={expected_revision}:current={current_revision}"
        )
        self.expected_revision = expected_revision
        self.current_revision = current_revision


class SettingsIntegrityError(SettingsControlPlaneError):
    """Stored state or its audit chain failed validation."""


class SettingsVersionError(SettingsControlPlaneError):
    """Stored state was written by an incompatible schema version."""


class SettingsIdempotencyError(SettingsControlPlaneError):
    """A request identifier was reused for a different transaction."""

    def __init__(self, request_id: str):
        super().__init__(f"settings_request_id_reused:{request_id}")
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    revision: int
    values: dict[str, Any]
    updated_at: float
    last_receipt_hash: str = ""
    history: tuple[dict[str, Any], ...] = ()
    migrated_from: str = ""
    unknown_keys: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "schema": SETTINGS_SCHEMA_NAME,
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "last_receipt_hash": self.last_receipt_hash,
            "migrated_from": self.migrated_from or None,
            "unknown_keys": list(self.unknown_keys),
            "values": dict(self.values),
        }


@dataclass(frozen=True, slots=True)
class SettingsMutationResult:
    snapshot: SettingsSnapshot
    changed: dict[str, dict[str, Any]]
    receipt: dict[str, Any] | None
    application: dict[str, dict[str, Any]] = field(default_factory=dict)
    application_receipt: dict[str, Any] | None = None
    application_journal_error: str = ""
    no_op: bool = False
    replayed: bool = False
    superseded: bool = False
    superseded_by_revision: int | None = None

    def public(self) -> dict[str, Any]:
        return {
            **self.snapshot.public(),
            "changed": dict(self.changed),
            "receipt": dict(self.receipt) if self.receipt else None,
            "application": dict(self.application),
            "application_receipt": (
                dict(self.application_receipt)
                if self.application_receipt
                else None
            ),
            "application_journal_error": self.application_journal_error or None,
            "no_op": self.no_op,
            "replayed": self.replayed,
            "superseded": self.superseded,
            "superseded_by_revision": self.superseded_by_revision,
        }


@dataclass(frozen=True, slots=True)
class _Subscriber:
    callback: Callable[[str, Any, Any], Any]
    owner: str
    keys: frozenset[str] | None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json_loads(raw: str) -> Any:
    def _reject_constant(value: str) -> None:
        message = f"non-finite JSON number:{value}"
        raise ValueError(message)

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key:{key}")
            result[key] = value
        return result

    return json.loads(
        raw,
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _is_sha256(value: Any) -> bool:
    normalized = str(value or "")
    return (
        len(normalized) == 71
        and normalized.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in normalized[7:])
    )


def _coerce_revision(value: Any, *, field_name: str = "expected_revision") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _coerce_timestamp(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite timestamp")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError(f"{field_name} must be a finite timestamp")
    return timestamp


class RuntimeSettingsStore:
    """One atomic settings owner with CAS, rollback, and chained receipts."""

    def __init__(
        self,
        path: str | Path,
        *,
        audit_path: str | Path | None = None,
        application_audit_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.audit_path = (
            Path(audit_path).expanduser()
            if audit_path is not None
            else self.path.with_name(f"{self.path.stem}.audit.jsonl")
        )
        self.application_audit_path = (
            Path(application_audit_path).expanduser()
            if application_audit_path is not None
            else self.path.with_name(f"{self.path.stem}.application.jsonl")
        )
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._thread_lock = threading.RLock()
        self._subscribers: list[_Subscriber] = []
        self._last_application: dict[str, dict[str, Any]] = {}
        self._load_error = ""
        try:
            with self._thread_lock, interprocess_file_lock(self.lock_path):
                self._snapshot = self._load_verified_snapshot_locked()
                application_entries = self._load_application_entries_locked(
                    self._load_audit_entries_locked()
                )
                self._last_application = self._application_for_snapshot(
                    self._snapshot,
                    application_entries,
                )
        except (
            OSError,
            SettingsControlPlaneError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self._load_error = f"{type(exc).__name__}:{exc}"
            self._snapshot = SettingsSnapshot(
                revision=0,
                values=dict(DEFAULT_VALUES),
                updated_at=0.0,
            )
            logger.warning("Runtime settings load failed closed: %s", self._load_error)
        if not self._load_error:
            self._reconcile_protected_invariants()

    def _reconcile_protected_invariants(self) -> None:
        """Audit-repair legacy values that predate protected settings."""

        for _attempt in range(4):
            current = self.snapshot()
            repairs = {
                key: definition.default
                for key, definition in SCHEMA_BY_KEY.items()
                if not definition.mutable
                and current.values.get(key) != definition.default
            }
            if not repairs:
                return
            try:
                self._commit(
                    repairs,
                    expected_revision=current.revision,
                    operation="reconcile_protected_invariants",
                    actor="runtime_settings_invariant_reconciler",
                    request_id=f"protected-invariants-v1-r{current.revision}",
                )
                logger.warning(
                    "Reconciled protected runtime settings at revision %s: %s",
                    current.revision + 1,
                    sorted(repairs),
                )
                return
            except SettingsConflictError:
                continue
        raise SettingsControlPlaneError(
            "protected runtime invariant reconciliation exhausted its conflict budget"
        )

    @property
    def revision(self) -> int:
        with self._thread_lock:
            return self._snapshot.revision

    def get(self, key: str) -> Any:
        return self.snapshot().values.get(key)

    def all(self) -> dict[str, Any]:
        return dict(self.snapshot().values)

    def snapshot(self, *, refresh: bool = True) -> SettingsSnapshot:
        with self._thread_lock:
            if refresh:
                with interprocess_file_lock(self.lock_path):
                    self._snapshot = self._load_verified_snapshot_locked()
                    self._load_error = ""
            return self._snapshot

    def subscribe(
        self,
        callback: Callable[[str, Any, Any], Any],
        *,
        owner: str | None = None,
        keys: Iterable[str] | None = None,
    ) -> None:
        if not callable(callback):
            raise TypeError("settings subscriber must be callable")
        normalized_keys = frozenset(str(key) for key in keys) if keys is not None else None
        subscriber = _Subscriber(
            callback=callback,
            owner=str(owner or getattr(callback, "__name__", "subscriber"))[:120],
            keys=normalized_keys,
        )
        with self._thread_lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def set(self, key: str, value: Any) -> Any:
        """Compatibility helper for internal callers; still commits through CAS."""

        for _attempt in range(8):
            expected = self.snapshot().revision
            try:
                result = self.patch(
                    {key: value},
                    expected_revision=expected,
                    actor="internal_compatibility_set",
                )
                return result.snapshot.values[key]
            except SettingsConflictError:
                continue
        raise SettingsControlPlaneError(
            "settings compatibility set exhausted its bounded conflict retry budget"
        )

    def patch(
        self,
        changes: dict[str, Any],
        *,
        expected_revision: int,
        actor: str = "internal",
        request_id: str | None = None,
    ) -> SettingsMutationResult:
        validated = validate_settings_patch(changes)
        return self._commit(
            validated,
            expected_revision=_coerce_revision(expected_revision),
            operation="patch",
            actor=actor,
            request_id=request_id,
        )

    def reset_section(
        self,
        section: str,
        *,
        expected_revision: int | None = None,
        actor: str = "internal",
        request_id: str | None = None,
    ) -> SettingsMutationResult:
        normalized = str(section or "").strip()
        keys = [key for key, definition in SCHEMA_BY_KEY.items() if definition.section == normalized]
        if not keys:
            raise KeyError(f"unknown_settings_section:{normalized}")
        expected = self.snapshot().revision if expected_revision is None else expected_revision
        return self._commit(
            {key: SCHEMA_BY_KEY[key].default for key in keys},
            expected_revision=_coerce_revision(expected),
            operation="reset_section",
            actor=actor,
            request_id=request_id,
            metadata={"section": normalized},
        )

    def rollback(
        self,
        target_revision: int,
        *,
        expected_revision: int,
        actor: str = "internal",
        request_id: str | None = None,
    ) -> SettingsMutationResult:
        target = _coerce_revision(target_revision, field_name="target_revision")
        expected = _coerce_revision(expected_revision)
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            current = self._load_verified_snapshot_locked()
            if current.revision != expected:
                raise SettingsConflictError(expected, current.revision)
            candidates = [
                {
                    "revision": current.revision,
                    "values": current.values,
                },
                *list(current.history),
            ]
            target_entry = next(
                (entry for entry in candidates if int(entry.get("revision", -1)) == target),
                None,
            )
            if target_entry is None:
                raise KeyError(f"settings_revision_not_retained:{target}")
            target_values, _unknown = validated_settings_snapshot(target_entry.get("values"))
        changes = {
            key: value
            for key, value in target_values.items()
            if current.values.get(key) != value
        }
        if not changes:
            return SettingsMutationResult(
                snapshot=current,
                changed={},
                receipt=None,
                application={},
                no_op=True,
            )
        return self._commit(
            changes,
            expected_revision=expected,
            operation="rollback",
            actor=actor,
            request_id=request_id,
            metadata={"target_revision": target},
        )

    def acknowledge_application(
        self,
        settings_receipt_hash: str,
        acknowledgements: dict[str, Any],
        *,
        actor: str = "runtime_owner",
    ) -> dict[str, Any]:
        """Append an owner acknowledgement without rewriting mutation history."""

        mutation_hash = str(settings_receipt_hash or "").strip()
        if not mutation_hash:
            raise ValueError("settings_receipt_hash is required")
        if not isinstance(acknowledgements, dict) or not acknowledgements:
            raise ValueError("acknowledgements must be a non-empty object")
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            current = self._load_verified_snapshot_locked()
            audit_entries = self._load_audit_entries_locked()
            mutation = next(
                (
                    entry
                    for entry in audit_entries
                    if entry.get("receipt_hash") == mutation_hash
                ),
                None,
            )
            if mutation is None:
                raise KeyError("settings_mutation_receipt_not_found")
            application_entries = self._load_application_entries_locked(
                audit_entries
            )
            previous = next(
                (
                    dict(entry.get("application") or {})
                    for entry in reversed(application_entries)
                    if entry.get("settings_receipt_hash") == mutation_hash
                ),
                self._unconfirmed_application(mutation),
            )
            changed_keys = set(mutation.get("changed") or {})
            application = dict(previous)
            for key, raw in acknowledgements.items():
                if key not in changed_keys:
                    raise KeyError(f"setting_not_changed_by_receipt:{key}")
                if not isinstance(raw, dict):
                    raise TypeError(f"acknowledgement for {key} must be an object")
                status = str(raw.get("status") or "").strip().lower()
                if status not in {"applied", "deferred", "failed"}:
                    raise ValueError(
                        f"acknowledgement status for {key} must be applied, deferred, or failed"
                    )
                application[key] = {
                    "owner": SCHEMA_BY_KEY[key].owner,
                    "status": status,
                    "detail": str(raw.get("detail") or "owner acknowledged setting")[:240],
                }
            receipt = self._append_application_receipt_locked(
                mutation=mutation,
                application=application,
                application_entries=application_entries,
                kind="owner_acknowledgement",
                actor=actor,
            )
            if current.last_receipt_hash == mutation_hash:
                self._last_application = dict(application)
            return {
                "ok": True,
                "revision": int(mutation["revision"]),
                "settings_receipt_hash": mutation_hash,
                "application": application,
                "application_receipt": receipt,
            }

    def describe(self) -> dict[str, Any]:
        try:
            snapshot = self.snapshot(refresh=True)
            integrity = self.verify_integrity()
            with self._thread_lock, interprocess_file_lock(self.lock_path):
                application = self._application_for_snapshot(
                    snapshot,
                    self._load_application_entries_locked(
                        self._load_audit_entries_locked()
                    ),
                )
                self._last_application = dict(application)
        except (
            OSError,
            SettingsControlPlaneError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self._load_error = f"{type(exc).__name__}:{exc}"
            snapshot = self._snapshot
            integrity = {
                "ok": False,
                "error": self._load_error,
                "audit_entries": 0,
            }
            application = self._application_snapshot()
        return {
            **snapshot.public(),
            "integrity": integrity,
            "application": application,
        }

    def _application_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._thread_lock:
            return dict(self._last_application)

    @staticmethod
    def _application_for_snapshot(
        snapshot: SettingsSnapshot,
        entries: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not snapshot.last_receipt_hash:
            return {}
        match = next(
            (
                entry
                for entry in reversed(entries)
                if entry.get("settings_receipt_hash")
                == snapshot.last_receipt_hash
            ),
            None,
        )
        return dict(match.get("application") or {}) if match else {}

    def verify_integrity(self) -> dict[str, Any]:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            snapshot = self._load_verified_snapshot_locked()
            audit_entries = self._load_audit_entries_locked()
            application_entries = self._load_application_entries_locked(
                audit_entries
            )
            acknowledged = {
                str(entry.get("settings_receipt_hash") or "")
                for entry in application_entries
            }
            unacknowledged = [
                entry
                for entry in audit_entries
                if str(entry.get("receipt_hash") or "") not in acknowledged
            ]
            return {
                "ok": True,
                "error": None,
                "audit_entries": len(audit_entries),
                "audit_head": (
                    str(audit_entries[-1]["receipt_hash"])
                    if audit_entries
                    else _GENESIS_HASH
                ),
                "state_receipt_hash": snapshot.last_receipt_hash or None,
                "unapplied_audit_tail": 0,
                "application_entries": len(application_entries),
                "application_head": (
                    str(application_entries[-1]["application_hash"])
                    if application_entries
                    else _GENESIS_HASH
                ),
                "unacknowledged_application_receipts": len(unacknowledged),
                "unacknowledged_revisions": [
                    int(entry["revision"])
                    for entry in unacknowledged[-16:]
                ],
            }

    def _commit(
        self,
        changes: dict[str, Any],
        *,
        expected_revision: int,
        operation: str,
        actor: str,
        request_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> SettingsMutationResult:
        for key, value in changes.items():
            definition = SCHEMA_BY_KEY[key]
            if not definition.mutable and value != definition.default:
                raise ValueError(f"protected_runtime_invariant:{key}")
        normalized_request_id = self._normalize_request_id(request_id)
        request_fingerprint = _sha256(
            {
                "operation": operation,
                "changes": changes,
                "metadata": dict(metadata or {}),
            }
        )
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            current = self._load_verified_snapshot_locked()
            audit_entries = self._load_audit_entries_locked()
            application_entries = self._load_application_entries_locked(
                audit_entries
            )
            replay = next(
                (
                    entry
                    for entry in audit_entries
                    if entry.get("request_id") == normalized_request_id
                ),
                None,
            )
            if replay is not None:
                if replay.get("request_fingerprint") != request_fingerprint:
                    raise SettingsIdempotencyError(normalized_request_id)
                application_receipt = next(
                    (
                        entry
                        for entry in reversed(application_entries)
                        if entry.get("settings_receipt_hash")
                        == replay.get("receipt_hash")
                    ),
                    None,
                )
                application = (
                    dict(application_receipt.get("application") or {})
                    if application_receipt
                    else self._unconfirmed_application(replay)
                )
                self._snapshot = current
                replay_revision = int(replay["revision"])
                superseded = current.revision > replay_revision
                return SettingsMutationResult(
                    snapshot=current,
                    changed=dict(replay.get("changed") or {}),
                    receipt=replay,
                    application=application,
                    application_receipt=application_receipt,
                    no_op=False,
                    replayed=True,
                    superseded=superseded,
                    superseded_by_revision=(
                        current.revision if superseded else None
                    ),
                )
            if current.revision != expected_revision:
                raise SettingsConflictError(expected_revision, current.revision)

            changed = {
                key: {"previous": current.values.get(key), "value": value}
                for key, value in changes.items()
                if current.values.get(key) != value
            }
            if not changed:
                self._snapshot = current
                return SettingsMutationResult(
                    snapshot=current,
                    changed={},
                    receipt=None,
                    application={
                        key: {
                            "owner": SCHEMA_BY_KEY[key].owner,
                            "status": "unchanged",
                            "detail": "requested value already active",
                        }
                        for key in changes
                    },
                    no_op=True,
                )

            next_values = dict(current.values)
            for key, change in changed.items():
                next_values[key] = change["value"]
            revision = current.revision + 1
            timestamp = time.time()
            receipt = self._build_receipt(
                revision=revision,
                operation=operation,
                actor=actor,
                request_id=normalized_request_id,
                request_fingerprint=request_fingerprint,
                changed=changed,
                values=next_values,
                previous_receipt_hash=(
                    str(audit_entries[-1]["receipt_hash"])
                    if audit_entries
                    else _GENESIS_HASH
                ),
                metadata={
                    **dict(metadata or {}),
                    **(
                        {"migrated_from": current.migrated_from}
                        if current.migrated_from
                        else {}
                    ),
                    **(
                        {"dropped_unknown_keys": list(current.unknown_keys)}
                        if current.unknown_keys
                        else {}
                    ),
                },
                timestamp=timestamp,
            )

            self._assert_owned_paths()
            ensure_private_directory(self.path.parent)
            atomic_append_text(
                self.audit_path,
                json.dumps(
                    receipt,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )

            history = list(current.history)
            history.append(self._history_entry(current))
            history = history[-_HISTORY_LIMIT:]
            snapshot = SettingsSnapshot(
                revision=revision,
                values=next_values,
                updated_at=timestamp,
                last_receipt_hash=str(receipt["receipt_hash"]),
                history=tuple(history),
                migrated_from=current.migrated_from,
            )
            self._write_snapshot_locked(snapshot)
            self._snapshot = snapshot
            self._load_error = ""

        application = self._notify_subscribers(changed)
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            response_snapshot = self._load_verified_snapshot_locked()
            self._snapshot = response_snapshot
        superseded = response_snapshot.revision > snapshot.revision
        if superseded:
            for key, change in changed.items():
                if response_snapshot.values.get(key) == change["value"]:
                    continue
                application[key] = {
                    "owner": SCHEMA_BY_KEY[key].owner,
                    "status": "superseded",
                    "detail": (
                        "a newer durable revision replaced this value during "
                        "live-owner dispatch"
                    ),
                }
        application_receipt = None
        application_journal_error = ""
        try:
            application_receipt = self._record_application_receipt(
                settings_receipt=receipt,
                application=application,
            )
        except (
            OSError,
            SettingsControlPlaneError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            application_journal_error = f"{type(exc).__name__}:{exc}"
            logger.error(
                "Settings revision %s is durable but application receipt failed: %s",
                snapshot.revision,
                application_journal_error,
            )
        with self._thread_lock:
            if self._snapshot.last_receipt_hash == receipt["receipt_hash"]:
                self._last_application = dict(application)
        return SettingsMutationResult(
            snapshot=response_snapshot,
            changed=changed,
            receipt=receipt,
            application=application,
            application_receipt=application_receipt,
            application_journal_error=application_journal_error,
            superseded=superseded,
            superseded_by_revision=(
                response_snapshot.revision if superseded else None
            ),
        )

    def _notify_subscribers(
        self,
        changed: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for key in changed:
            definition = SCHEMA_BY_KEY[key]
            status = "ready"
            detail = "durable revision visible at the next owner gate"
            if definition.apply_mode == "frontend_runtime":
                status = "awaiting_frontend"
                detail = "desktop shell acknowledgement required"
            elif definition.apply_mode == "live_bridge":
                status = "deferred"
                detail = "live owner not registered"
            result[key] = {
                "owner": definition.owner,
                "status": status,
                "detail": detail,
            }

        with self._thread_lock:
            with interprocess_file_lock(self.lock_path):
                current = self._load_verified_snapshot_locked()
                self._snapshot = current
            subscribers = tuple(self._subscribers)
            for subscriber in subscribers:
                for key, change in changed.items():
                    if subscriber.keys is not None and key not in subscriber.keys:
                        continue
                    if current.values.get(key) != change["value"]:
                        result[key] = {
                            "owner": subscriber.owner,
                            "status": "superseded",
                            "detail": (
                                "a newer durable revision replaced this value "
                                "before live-owner dispatch"
                            ),
                        }
                        continue
                    try:
                        raw = subscriber.callback(
                            key,
                            change["previous"],
                            change["value"],
                        )
                        if isinstance(raw, dict):
                            raw_status = str(raw.get("status") or "applied")[:40]
                            if raw_status not in _APPLICATION_STATUSES:
                                raise ValueError(
                                    f"invalid settings application status:{raw_status}"
                                )
                            result[key] = {
                                "owner": str(raw.get("owner") or subscriber.owner)[:120],
                                "status": raw_status,
                                "detail": str(
                                    raw.get("detail")
                                    or "live owner applied revision"
                                )[:240],
                            }
                        elif isinstance(raw, str) and raw:
                            if raw not in _APPLICATION_STATUSES:
                                raise ValueError(
                                    f"invalid settings application status:{raw}"
                                )
                            result[key] = {
                                "owner": subscriber.owner,
                                "status": raw[:40],
                                "detail": "subscriber returned application status",
                            }
                        elif SCHEMA_BY_KEY[key].apply_mode == "live_bridge":
                            result[key] = {
                                "owner": subscriber.owner,
                                "status": "applied",
                                "detail": "live owner applied revision",
                            }
                    except (
                        ImportError,
                        AttributeError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        logger.warning(
                            "Settings subscriber %s failed for %s: %s",
                            subscriber.owner,
                            key,
                            exc,
                        )
                        result[key] = {
                            "owner": subscriber.owner,
                            "status": "failed",
                            "detail": f"{type(exc).__name__}:{str(exc)[:180]}",
                        }
        return result

    def _load_verified_snapshot_locked(self) -> SettingsSnapshot:
        snapshot = self._load_state_locked()
        audit_entries = self._load_audit_entries_locked()
        self._verify_state_receipt_locked(snapshot, audit_entries)
        return self._recover_audit_tail_locked(snapshot, audit_entries)

    def _recover_audit_tail_locked(
        self,
        snapshot: SettingsSnapshot,
        audit_entries: list[dict[str, Any]],
    ) -> SettingsSnapshot:
        """Finish commits interrupted after durable audit append.

        Mutation receipts are written before state replacement. A receipt past
        the state's acknowledged hash is therefore a prepared transaction, not
        permission to reuse its revision. Replay is accepted only when every
        old value and the resulting complete-state hash match the receipt.
        """

        if not audit_entries:
            return snapshot
        if snapshot.last_receipt_hash:
            state_index = next(
                index
                for index, entry in enumerate(audit_entries)
                if entry.get("receipt_hash") == snapshot.last_receipt_hash
            )
            tail = audit_entries[state_index + 1 :]
        else:
            tail = audit_entries
        if not tail:
            return snapshot

        recovered = snapshot
        for receipt in tail:
            revision = int(receipt["revision"])
            if revision != recovered.revision + 1:
                raise SettingsIntegrityError(
                    "prepared settings receipt does not follow state revision"
                )
            changed = receipt.get("changed")
            if not isinstance(changed, dict) or not changed:
                raise SettingsIntegrityError(
                    "prepared settings receipt has no changed values"
                )
            next_values = dict(recovered.values)
            for key, transition in changed.items():
                if not isinstance(transition, dict):
                    raise SettingsIntegrityError(
                        f"prepared transition for {key} must be an object"
                    )
                if recovered.values.get(key) != transition.get("previous"):
                    raise SettingsIntegrityError(
                        f"prepared transition previous value mismatch for {key}"
                    )
                next_values[key] = transition.get("value")
            next_values, unknown = validated_settings_snapshot(next_values)
            if unknown or receipt.get("values_sha256") != _sha256(next_values):
                raise SettingsIntegrityError(
                    "prepared settings receipt does not reconstruct its state hash"
                )
            history = list(recovered.history)
            history.append(self._history_entry(recovered))
            recovered = SettingsSnapshot(
                revision=revision,
                values=next_values,
                updated_at=_coerce_timestamp(
                    receipt.get("timestamp"),
                    field_name="prepared.timestamp",
                ),
                last_receipt_hash=str(receipt["receipt_hash"]),
                history=tuple(history[-_HISTORY_LIMIT:]),
                migrated_from=recovered.migrated_from,
            )

        self._assert_owned_paths()
        ensure_private_directory(self.path.parent)
        self._write_snapshot_locked(recovered)
        logger.warning(
            "Recovered %s prepared settings transaction(s) through revision %s.",
            len(tail),
            recovered.revision,
        )
        return recovered

    def _load_state_locked(self) -> SettingsSnapshot:
        if not self.path.exists():
            return SettingsSnapshot(
                revision=0,
                values=dict(DEFAULT_VALUES),
                updated_at=0.0,
            )
        if self.path.is_symlink():
            raise SettingsIntegrityError(f"settings state may not be a symlink: {self.path}")
        raw = _strict_json_loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SettingsIntegrityError("settings state must be a JSON object")

        if "schema" not in raw and "schema_version" not in raw:
            values, unknown = migrated_settings_snapshot(raw)
            return SettingsSnapshot(
                revision=0,
                values=values,
                updated_at=_coerce_timestamp(
                    self.path.stat().st_mtime,
                    field_name="legacy.updated_at",
                ),
                migrated_from="legacy_flat_v1",
                unknown_keys=unknown,
            )

        schema = str(raw.get("schema") or raw.get("schema_name") or "")
        version = raw.get("schema_version")
        if schema != SETTINGS_SCHEMA_NAME:
            raise SettingsVersionError(f"unsupported settings schema:{schema or 'missing'}")
        if isinstance(version, bool) or not isinstance(version, int):
            raise SettingsVersionError("settings schema_version must be an integer")
        if version > SETTINGS_SCHEMA_VERSION:
            raise SettingsVersionError(
                f"settings schema v{version} is newer than supported v{SETTINGS_SCHEMA_VERSION}"
            )
        if version < SETTINGS_SCHEMA_VERSION:
            payload = raw.get("payload")
            values, unknown = migrated_settings_snapshot(payload)
            return SettingsSnapshot(
                revision=0,
                values=values,
                updated_at=_coerce_timestamp(
                    self.path.stat().st_mtime,
                    field_name="migration.updated_at",
                ),
                migrated_from=f"envelope_v{version}",
                unknown_keys=unknown,
            )

        revision = _coerce_revision(raw.get("revision"), field_name="revision")
        values, unknown = validated_settings_snapshot(raw.get("values"))
        history = self._validated_history(raw.get("history", []))
        if revision == 0 and history:
            raise SettingsIntegrityError("revision zero settings state has history")
        if history and int(history[-1]["revision"]) != revision - 1:
            raise SettingsIntegrityError(
                "settings history does not end at the previous revision"
            )
        return SettingsSnapshot(
            revision=revision,
            values=values,
            updated_at=_coerce_timestamp(
                raw.get("updated_at", 0.0),
                field_name="updated_at",
            ),
            last_receipt_hash=str(raw.get("last_receipt_hash") or ""),
            history=history,
            migrated_from=str(raw.get("migrated_from") or "")[:120],
            unknown_keys=unknown,
        )

    @staticmethod
    def _validated_history(value: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(value, list):
            raise SettingsIntegrityError("settings history must be an array")
        if len(value) > _HISTORY_LIMIT:
            raise SettingsIntegrityError(
                f"settings history exceeds retained limit {_HISTORY_LIMIT}"
            )
        history: list[dict[str, Any]] = []
        previous_revision = -1
        for entry in value[-_HISTORY_LIMIT:]:
            if not isinstance(entry, dict):
                raise SettingsIntegrityError("settings history entry must be an object")
            revision = _coerce_revision(entry.get("revision"), field_name="history.revision")
            if previous_revision >= 0 and revision != previous_revision + 1:
                raise SettingsIntegrityError(
                    "settings history revisions must be contiguous"
                )
            values, _unknown = validated_settings_snapshot(entry.get("values"))
            history.append(
                {
                    "revision": revision,
                    "updated_at": _coerce_timestamp(
                        entry.get("updated_at", 0.0),
                        field_name="history.updated_at",
                    ),
                    "last_receipt_hash": str(entry.get("last_receipt_hash") or ""),
                    "values": values,
                }
            )
            previous_revision = revision
        return tuple(history)

    def _load_audit_entries_locked(self) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        if self.audit_path.is_symlink():
            raise SettingsIntegrityError(f"settings audit may not be a symlink: {self.audit_path}")
        entries: list[dict[str, Any]] = []
        previous_hash = _GENESIS_HASH
        previous_revision = 0
        receipt_ids: set[str] = set()
        request_ids: set[str] = set()
        with self.audit_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = _strict_json_loads(line)
                except json.JSONDecodeError as exc:
                    raise SettingsIntegrityError(
                        f"settings audit line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(entry, dict):
                    raise SettingsIntegrityError(
                        f"settings audit line {line_number} must be an object"
                    )
                if entry.get("schema") != SETTINGS_AUDIT_SCHEMA:
                    raise SettingsIntegrityError(
                        f"settings audit line {line_number} has incompatible schema"
                    )
                if entry.get("previous_receipt_hash") != previous_hash:
                    raise SettingsIntegrityError(
                        f"settings audit chain break at line {line_number}"
                    )
                revision = _coerce_revision(
                    entry.get("revision"),
                    field_name=f"audit[{line_number}].revision",
                )
                if revision != previous_revision + 1:
                    raise SettingsIntegrityError(
                        f"settings audit revision break at line {line_number}"
                    )
                receipt_id = str(entry.get("receipt_id") or "")
                request_id = str(entry.get("request_id") or "")
                if not receipt_id or receipt_id in receipt_ids:
                    raise SettingsIntegrityError(
                        f"settings audit receipt id invalid at line {line_number}"
                    )
                if not request_id or request_id in request_ids:
                    raise SettingsIntegrityError(
                        f"settings audit request id invalid at line {line_number}"
                    )
                request_fingerprint = str(
                    entry.get("request_fingerprint") or ""
                )
                values_hash = str(entry.get("values_sha256") or "")
                if not _is_sha256(request_fingerprint):
                    raise SettingsIntegrityError(
                        f"settings audit request hash invalid at line {line_number}"
                    )
                if not _is_sha256(values_hash):
                    raise SettingsIntegrityError(
                        f"settings audit values hash invalid at line {line_number}"
                    )
                changed = entry.get("changed")
                if not isinstance(changed, dict) or not changed:
                    raise SettingsIntegrityError(
                        f"settings audit changes invalid at line {line_number}"
                    )
                for key, transition in changed.items():
                    if key not in SCHEMA_BY_KEY or not isinstance(transition, dict):
                        raise SettingsIntegrityError(
                            f"settings audit transition invalid for {key}"
                        )
                    if set(transition) != {"previous", "value"}:
                        raise SettingsIntegrityError(
                            f"settings audit transition fields invalid for {key}"
                        )
                    try:
                        validate_setting_value(key, transition["previous"])
                        validate_setting_value(key, transition["value"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise SettingsIntegrityError(
                            f"settings audit transition value invalid for {key}"
                        ) from exc
                _coerce_timestamp(
                    entry.get("timestamp"),
                    field_name=f"audit[{line_number}].timestamp",
                )
                claimed_hash = str(entry.get("receipt_hash") or "")
                body = dict(entry)
                body.pop("receipt_hash", None)
                expected_hash = _sha256(body)
                if claimed_hash != expected_hash:
                    raise SettingsIntegrityError(
                        f"settings audit hash mismatch at line {line_number}"
                    )
                previous_hash = claimed_hash
                previous_revision = revision
                receipt_ids.add(receipt_id)
                request_ids.add(request_id)
                entries.append(entry)
        return entries

    def _load_application_entries_locked(
        self,
        audit_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.application_audit_path.exists():
            return []
        if self.application_audit_path.is_symlink():
            raise SettingsIntegrityError(
                "settings application audit may not be a symlink: "
                f"{self.application_audit_path}"
            )
        mutation_by_hash = {
            str(entry["receipt_hash"]): entry for entry in audit_entries
        }
        entries: list[dict[str, Any]] = []
        previous_hash = _GENESIS_HASH
        receipt_ids: set[str] = set()
        with self.application_audit_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = _strict_json_loads(line)
                except json.JSONDecodeError as exc:
                    raise SettingsIntegrityError(
                        f"settings application line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(entry, dict):
                    raise SettingsIntegrityError(
                        f"settings application line {line_number} must be an object"
                    )
                if entry.get("schema") != SETTINGS_APPLICATION_AUDIT_SCHEMA:
                    raise SettingsIntegrityError(
                        f"settings application line {line_number} has incompatible schema"
                    )
                if entry.get("previous_application_hash") != previous_hash:
                    raise SettingsIntegrityError(
                        f"settings application chain break at line {line_number}"
                    )
                receipt_id = str(entry.get("receipt_id") or "")
                mutation_hash = str(entry.get("settings_receipt_hash") or "")
                mutation = mutation_by_hash.get(mutation_hash)
                if not receipt_id or receipt_id in receipt_ids:
                    raise SettingsIntegrityError(
                        f"settings application receipt id invalid at line {line_number}"
                    )
                if mutation is None:
                    raise SettingsIntegrityError(
                        f"settings application mutation reference invalid at line {line_number}"
                    )
                if entry.get("revision") != mutation.get("revision"):
                    raise SettingsIntegrityError(
                        f"settings application revision mismatch at line {line_number}"
                    )
                _coerce_timestamp(
                    entry.get("timestamp"),
                    field_name=f"application[{line_number}].timestamp",
                )
                application = entry.get("application")
                if not isinstance(application, dict) or set(application) != set(
                    mutation.get("changed") or {}
                ):
                    raise SettingsIntegrityError(
                        f"settings application coverage mismatch at line {line_number}"
                    )
                for key, status in application.items():
                    if not isinstance(status, dict):
                        raise SettingsIntegrityError(
                            f"settings application status invalid for {key}"
                        )
                    if not str(status.get("owner") or "") or not str(
                        status.get("status") or ""
                    ):
                        raise SettingsIntegrityError(
                            f"settings application status incomplete for {key}"
                        )
                claimed_hash = str(entry.get("application_hash") or "")
                body = dict(entry)
                body.pop("application_hash", None)
                if claimed_hash != _sha256(body):
                    raise SettingsIntegrityError(
                        f"settings application hash mismatch at line {line_number}"
                    )
                previous_hash = claimed_hash
                receipt_ids.add(receipt_id)
                entries.append(entry)
        return entries

    @staticmethod
    def _verify_state_receipt_locked(
        snapshot: SettingsSnapshot,
        audit_entries: list[dict[str, Any]],
    ) -> None:
        audit_by_hash = {
            str(entry.get("receipt_hash") or ""): entry
            for entry in audit_entries
        }
        for history_entry in snapshot.history:
            revision = int(history_entry["revision"])
            receipt_hash = str(history_entry.get("last_receipt_hash") or "")
            if revision == 0 and not receipt_hash:
                continue
            receipt = audit_by_hash.get(receipt_hash)
            if receipt is None or int(receipt.get("revision", -1)) != revision:
                raise SettingsIntegrityError(
                    f"settings history receipt mismatch at revision {revision}"
                )
            if receipt.get("values_sha256") != _sha256(history_entry["values"]):
                raise SettingsIntegrityError(
                    f"settings history value mismatch at revision {revision}"
                )
        if not snapshot.last_receipt_hash:
            if snapshot.revision != 0:
                raise SettingsIntegrityError(
                    "versioned settings state has no audit receipt"
                )
            return
        matching = next(
            (
                entry
                for entry in audit_entries
                if entry.get("receipt_hash") == snapshot.last_receipt_hash
            ),
            None,
        )
        if matching is None:
            raise SettingsIntegrityError("settings state receipt is missing from audit chain")
        if int(matching.get("revision", -1)) != snapshot.revision:
            raise SettingsIntegrityError("settings state revision does not match its receipt")
        if matching.get("values_sha256") != _sha256(snapshot.values):
            raise SettingsIntegrityError("settings state values do not match their receipt")

    @staticmethod
    def _build_receipt(
        *,
        revision: int,
        operation: str,
        actor: str,
        request_id: str,
        request_fingerprint: str,
        changed: dict[str, dict[str, Any]],
        values: dict[str, Any],
        previous_receipt_hash: str,
        metadata: dict[str, Any] | None,
        timestamp: float,
    ) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "schema": SETTINGS_AUDIT_SCHEMA,
            "receipt_id": f"settings-{uuid.uuid4()}",
            "request_id": request_id,
            "request_fingerprint": request_fingerprint,
            "revision": revision,
            "operation": str(operation)[:80],
            "actor": str(actor or "unknown")[:120],
            "timestamp": timestamp,
            "changed": changed,
            "values_sha256": _sha256(values),
            "previous_receipt_hash": previous_receipt_hash,
            "metadata": dict(metadata or {}),
        }
        receipt["receipt_hash"] = _sha256(receipt)
        return receipt

    @staticmethod
    def _normalize_request_id(request_id: str | None) -> str:
        if request_id is None:
            return str(uuid.uuid4())
        normalized = str(request_id).strip()
        if not normalized:
            raise ValueError("request_id may not be empty")
        if len(normalized) > 160:
            raise ValueError("request_id exceeds 160 characters")
        if any(ord(character) < 32 for character in normalized):
            raise ValueError("request_id may not contain control characters")
        return normalized

    @staticmethod
    def _history_entry(snapshot: SettingsSnapshot) -> dict[str, Any]:
        return {
            "revision": snapshot.revision,
            "updated_at": snapshot.updated_at,
            "last_receipt_hash": snapshot.last_receipt_hash,
            "values": dict(snapshot.values),
        }

    def _write_snapshot_locked(self, snapshot: SettingsSnapshot) -> None:
        envelope = {
            "schema": SETTINGS_SCHEMA_NAME,
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "revision": snapshot.revision,
            "updated_at": snapshot.updated_at,
            "last_receipt_hash": snapshot.last_receipt_hash,
            "history": list(snapshot.history),
            "values": snapshot.values,
            **(
                {"migrated_from": snapshot.migrated_from}
                if snapshot.migrated_from
                else {}
            ),
        }
        atomic_write_text(
            self.path,
            json.dumps(
                envelope,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    @staticmethod
    def _unconfirmed_application(
        settings_receipt: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "owner": SCHEMA_BY_KEY[key].owner,
                "status": "unconfirmed",
                "detail": (
                    "mutation is durable but owner acknowledgement was not "
                    "retained"
                ),
            }
            for key in settings_receipt.get("changed") or {}
        }

    def _record_application_receipt(
        self,
        *,
        settings_receipt: dict[str, Any],
        application: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        with self._thread_lock, interprocess_file_lock(self.lock_path):
            audit_entries = self._load_audit_entries_locked()
            mutation_hash = str(settings_receipt.get("receipt_hash") or "")
            if not any(
                entry.get("receipt_hash") == mutation_hash
                for entry in audit_entries
            ):
                raise SettingsIntegrityError(
                    "cannot acknowledge a missing settings mutation receipt"
                )
            application_entries = self._load_application_entries_locked(
                audit_entries
            )
            existing = next(
                (
                    entry
                    for entry in application_entries
                    if entry.get("settings_receipt_hash") == mutation_hash
                ),
                None,
            )
            if existing is not None:
                return existing
            return self._append_application_receipt_locked(
                mutation=settings_receipt,
                application=application,
                application_entries=application_entries,
                kind="control_plane_dispatch",
                actor="settings_control_plane",
            )

    def _append_application_receipt_locked(
        self,
        *,
        mutation: dict[str, Any],
        application: dict[str, dict[str, Any]],
        application_entries: list[dict[str, Any]],
        kind: str,
        actor: str,
    ) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "schema": SETTINGS_APPLICATION_AUDIT_SCHEMA,
            "receipt_id": f"settings-application-{uuid.uuid4()}",
            "revision": int(mutation["revision"]),
            "settings_receipt_hash": str(mutation["receipt_hash"]),
            "timestamp": time.time(),
            "application": application,
            "kind": str(kind or "owner_acknowledgement")[:80],
            "actor": str(actor or "unknown")[:120],
            "previous_application_hash": (
                str(application_entries[-1]["application_hash"])
                if application_entries
                else _GENESIS_HASH
            ),
        }
        receipt["application_hash"] = _sha256(receipt)
        self._assert_owned_paths()
        ensure_private_directory(self.application_audit_path.parent)
        atomic_append_text(
            self.application_audit_path,
            json.dumps(
                receipt,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
        return receipt

    def _assert_owned_paths(self) -> None:
        for path in (
            self.path,
            self.audit_path,
            self.application_audit_path,
            self.lock_path,
        ):
            if path.is_symlink():
                raise SettingsIntegrityError(
                    f"settings control-plane path may not be a symlink: {path}"
                )


__all__ = [
    "RuntimeSettingsStore",
    "SettingsConflictError",
    "SettingsControlPlaneError",
    "SettingsIdempotencyError",
    "SettingsIntegrityError",
    "SettingsMutationResult",
    "SettingsSnapshot",
    "SettingsVersionError",
]
