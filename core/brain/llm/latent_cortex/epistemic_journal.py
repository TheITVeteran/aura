"""Crash-consistent write-ahead journal for the RLC epistemic state."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Never

from core.brain.llm.latent_cortex.epistemic_state import (
    EpistemicState,
    EpistemicStateError,
    OperationOutcome,
)
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import FileWriteGateway, get_file_write_gateway

EPISTEMIC_JOURNAL_SCHEMA = "aura.rlc.epistemic_journal.v1"
ZERO_SHA256 = "0" * 64
MAX_JOURNAL_BYTES = 128 * 1024 * 1024
MAX_JOURNAL_ENTRY_BYTES = 16 * 1024 * 1024
MAX_JOURNAL_ENTRIES = 8_192

_ENTRY_FIELDS = {
    "schema",
    "sequence",
    "previous_entry_sha256",
    "state_sha256",
    "state",
    "entry_sha256",
}


class EpistemicJournalError(EpistemicStateError):
    """Stable fail-closed error for journal integrity and durability failures."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class EpistemicRecoveryReceipt:
    state: EpistemicState
    entry_count: int
    head_sha256: str
    size_bytes: int
    repaired_torn_tail_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_sha256": self.state.state_sha256,
            "state_version": self.state.version,
            "entry_count": self.entry_count,
            "head_sha256": self.head_sha256,
            "size_bytes": self.size_bytes,
            "repaired_torn_tail_bytes": self.repaired_torn_tail_bytes,
        }


def _fail(code: str) -> Never:
    raise EpistemicJournalError(code)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("journal_value_not_canonical_json")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_loads(payload: bytes) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _fail("journal_entry_not_utf8")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("journal_entry_duplicate_json_key")
            result[key] = value
        return result

    def parse_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            _fail("journal_entry_nonfinite_number")
        return value

    def reject_constant(_raw: str) -> None:
        _fail("journal_entry_nonfinite_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except EpistemicJournalError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail("journal_entry_json_invalid")


def _entry_bytes(
    *, sequence: int, previous_entry_sha256: str, state: EpistemicState
) -> bytes:
    base = {
        "schema": EPISTEMIC_JOURNAL_SCHEMA,
        "sequence": sequence,
        "previous_entry_sha256": previous_entry_sha256,
        "state_sha256": state.state_sha256,
        "state": state.to_dict(),
    }
    record = {**base, "entry_sha256": _sha256(_canonical_bytes(base))}
    payload = _canonical_bytes(record) + b"\n"
    if len(payload) > MAX_JOURNAL_ENTRY_BYTES:
        _fail("journal_entry_too_large")
    return payload


def _validate_transition(previous: EpistemicState, current: EpistemicState) -> None:
    if current.version != previous.version + 1:
        _fail("journal_state_version_drift")
    if current.parent_sha256 != previous.state_sha256:
        _fail("journal_state_parent_mismatch")
    if current.episode_id != previous.episode_id or current.problem != previous.problem:
        _fail("journal_episode_identity_drift")
    if (
        current.budget.total != previous.budget.total
        or current.budget.tool_calls_total != previous.budget.tool_calls_total
        or current.budget.used < previous.budget.used
        or current.budget.tool_calls_used < previous.budget.tool_calls_used
    ):
        _fail("journal_budget_history_invalid")

    def by_id(items: tuple[Any, ...], attribute: str) -> dict[str, Any]:
        return {getattr(item, attribute): item for item in items}

    previous_evidence = by_id(previous.evidence, "evidence_id")
    current_evidence = by_id(current.evidence, "evidence_id")
    previous_operations = by_id(previous.operations, "operation_id")
    current_operations = by_id(current.operations, "operation_id")
    previous_claims = by_id(previous.claims, "claim_id")
    current_claims = by_id(current.claims, "claim_id")
    previous_hypotheses = by_id(previous.hypotheses, "hypothesis_id")
    current_hypotheses = by_id(current.hypotheses, "hypothesis_id")

    for old, new, code in (
        (previous_evidence, current_evidence, "journal_evidence_history_rewritten"),
        (previous_operations, current_operations, "journal_operation_history_rewritten"),
    ):
        if any(identifier not in new or new[identifier] != item for identifier, item in old.items()):
            _fail(code)
    if not set(previous_claims) <= set(current_claims):
        _fail("journal_claim_history_deleted")
    if not set(previous_hypotheses) <= set(current_hypotheses):
        _fail("journal_hypothesis_history_deleted")

    new_operations = {
        operation_id: operation
        for operation_id, operation in current_operations.items()
        if operation_id not in previous_operations
    }
    if any(
        operation.input_state_sha256 != previous.state_sha256
        for operation in new_operations.values()
    ):
        _fail("journal_operation_base_mismatch")
    revised_claims = {
        claim_id
        for claim_id, claim in previous_claims.items()
        if current_claims[claim_id] != claim
    }
    covered_claims = {
        claim_id
        for operation in new_operations.values()
        if operation.outcome is OperationOutcome.SUCCEEDED
        for claim_id in operation.affected_claim_ids
    }
    if not revised_claims <= covered_claims:
        _fail("journal_claim_revision_without_operation")


class EpistemicStateJournal:
    """Hash-chained, fsync-sealed full-state journal for one RLC episode."""

    def __init__(
        self,
        path: str | Path,
        *,
        gateway: FileWriteGateway | None = None,
        repair_torn_tail: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.gateway = gateway or get_file_write_gateway()
        self.repair_torn_tail = bool(repair_torn_tail)
        self.last_recovery: EpistemicRecoveryReceipt | None = None
        self._trusted_genesis: EpistemicState | None = None

    @contextmanager
    def _lock(self) -> Iterator[None]:
        stack = ExitStack()
        try:
            stack.enter_context(interprocess_file_lock(self.lock_path))
        except OSError as exc:
            stack.close()
            raise EpistemicJournalError("journal_lock_failed") from exc
        with stack:
            yield

    def _reject_symlink_components(self) -> None:
        if any(candidate.is_symlink() for candidate in (self.path, *self.path.parents)):
            _fail("journal_path_symlink_rejected")

    def _initialize_if_absent(self, genesis: EpistemicState) -> None:
        self._reject_symlink_components()
        genesis_line = _entry_bytes(
            sequence=0,
            previous_entry_sha256=ZERO_SHA256,
            state=genesis,
        )
        try:
            self.gateway.write_bytes_if_absent(
                self.path,
                genesis_line,
                source="rlc.epistemic_journal.initialize",
            )
        except (OSError, RuntimeError) as exc:
            raise EpistemicJournalError("journal_initialize_failed") from exc

    def _open(self) -> BinaryIO:
        try:
            return self.gateway.open_owned_binary(
                self.path,
                mode="a+b",
                permissions=0o600,
                source="rlc.epistemic_journal",
            )
        except OSError as exc:
            raise EpistemicJournalError("journal_open_failed") from exc

    def _read_and_repair(self, handle: BinaryIO) -> tuple[bytes, int]:
        handle.seek(0)
        payload = handle.read(MAX_JOURNAL_BYTES + 1)
        if len(payload) > MAX_JOURNAL_BYTES:
            _fail("journal_too_large")
        if not payload:
            _fail("journal_empty")
        if payload.endswith(b"\n"):
            return payload, 0
        boundary = payload.rfind(b"\n")
        if boundary < 0:
            _fail("journal_has_no_complete_entry")
        torn_bytes = len(payload) - boundary - 1
        if not self.repair_torn_tail:
            _fail("journal_torn_tail")
        valid = payload[: boundary + 1]
        return valid, torn_bytes

    @staticmethod
    def _truncate_verified_tail(handle: BinaryIO, valid_bytes: int) -> None:
        try:
            handle.seek(0)
            handle.truncate(valid_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        except OSError as exc:
            raise EpistemicJournalError("journal_torn_tail_repair_failed") from exc

    @staticmethod
    def _decode_entry(
        payload: bytes,
        *,
        expected_sequence: int,
        previous_entry_sha256: str,
        previous_state: EpistemicState | None,
        expected_genesis: EpistemicState,
    ) -> tuple[str, EpistemicState]:
        record = _strict_json_loads(payload)
        if not isinstance(record, Mapping) or set(record) != _ENTRY_FIELDS:
            _fail("journal_entry_shape_invalid")
        if _canonical_bytes(record) != payload:
            _fail("journal_entry_noncanonical")
        if record["schema"] != EPISTEMIC_JOURNAL_SCHEMA:
            _fail("journal_entry_schema_invalid")
        if type(record["sequence"]) is not int or record["sequence"] != expected_sequence:
            _fail("journal_sequence_drift")
        if record["previous_entry_sha256"] != previous_entry_sha256:
            _fail("journal_hash_chain_invalid")
        entry_sha256 = record["entry_sha256"]
        if not _is_sha256(entry_sha256):
            _fail("journal_entry_hash_invalid")
        base = {key: value for key, value in record.items() if key != "entry_sha256"}
        if _sha256(_canonical_bytes(base)) != entry_sha256:
            _fail("journal_entry_hash_mismatch")
        if not isinstance(record["state"], Mapping):
            _fail("journal_state_shape_invalid")
        try:
            state = EpistemicState.from_dict(record["state"])
        except EpistemicStateError as exc:
            raise EpistemicJournalError("journal_state_invalid") from exc
        if record["state_sha256"] != state.state_sha256:
            _fail("journal_state_identity_mismatch")
        if expected_sequence == 0:
            if state != expected_genesis:
                _fail("journal_genesis_mismatch")
        else:
            if previous_state is None:
                _fail("journal_previous_state_missing")
            _validate_transition(previous_state, state)
        return entry_sha256, state

    def _replay(
        self, handle: BinaryIO, expected_genesis: EpistemicState
    ) -> EpistemicRecoveryReceipt:
        payload, torn_bytes = self._read_and_repair(handle)
        lines = payload.split(b"\n")[:-1]
        if len(lines) > MAX_JOURNAL_ENTRIES:
            _fail("journal_has_too_many_entries")
        previous_entry_sha256 = ZERO_SHA256
        previous_state: EpistemicState | None = None
        for sequence, line in enumerate(lines):
            if not line or len(line) + 1 > MAX_JOURNAL_ENTRY_BYTES:
                _fail("journal_entry_size_invalid")
            previous_entry_sha256, previous_state = self._decode_entry(
                line,
                expected_sequence=sequence,
                previous_entry_sha256=previous_entry_sha256,
                previous_state=previous_state,
                expected_genesis=expected_genesis,
            )
        if previous_state is None:
            _fail("journal_has_no_state")
        if torn_bytes:
            self._truncate_verified_tail(handle, len(payload))
        return EpistemicRecoveryReceipt(
            state=previous_state,
            entry_count=len(lines),
            head_sha256=previous_entry_sha256,
            size_bytes=len(payload),
            repaired_torn_tail_bytes=torn_bytes,
        )

    @staticmethod
    def _write_record(handle: BinaryIO, payload: bytes) -> None:
        try:
            written = handle.write(payload)
            if written != len(payload):
                _fail("journal_append_short_write")
            handle.flush()
            os.fsync(handle.fileno())
        except EpistemicJournalError:
            raise
        except OSError as exc:
            raise EpistemicJournalError("journal_append_failed") from exc

    def bootstrap(self, genesis: EpistemicState) -> EpistemicState:
        if not isinstance(genesis, EpistemicState) or genesis.version != 0:
            _fail("journal_genesis_invalid")
        self._reject_symlink_components()
        self.gateway.ensure_directory(
            self.path.parent,
            source="rlc.epistemic_journal.parent",
        )
        with self._lock():
            self._initialize_if_absent(genesis)
            with self._open() as handle:
                receipt = self._replay(handle, genesis)
        self._trusted_genesis = genesis
        self.last_recovery = receipt
        return receipt.state

    def append(
        self,
        *,
        expected_base: EpistemicState,
        candidate: EpistemicState,
    ) -> None:
        if not isinstance(expected_base, EpistemicState) or not isinstance(
            candidate, EpistemicState
        ):
            _fail("journal_append_state_type_invalid")
        if self._trusted_genesis is None:
            _fail("journal_not_bootstrapped")
        self._reject_symlink_components()
        with self._lock():
            with self._open() as handle:
                current = self._replay(handle, self._trusted_genesis)
                if current.state.state_sha256 != expected_base.state_sha256:
                    _fail("journal_base_is_not_head")
                _validate_transition(expected_base, candidate)
                line = _entry_bytes(
                    sequence=current.entry_count,
                    previous_entry_sha256=current.head_sha256,
                    state=candidate,
                )
                if current.size_bytes + len(line) > MAX_JOURNAL_BYTES:
                    _fail("journal_too_large")
                self._write_record(handle, line)
                record = _strict_json_loads(line[:-1])
                self.last_recovery = EpistemicRecoveryReceipt(
                    state=candidate,
                    entry_count=current.entry_count + 1,
                    head_sha256=record["entry_sha256"],
                    size_bytes=current.size_bytes + len(line),
                    repaired_torn_tail_bytes=current.repaired_torn_tail_bytes,
                )

__all__ = [
    "EPISTEMIC_JOURNAL_SCHEMA",
    "EpistemicJournalError",
    "EpistemicRecoveryReceipt",
    "EpistemicStateJournal",
]
