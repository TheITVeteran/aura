"""Crash-consistent custody for verified groups rejected before mutation.

Rejected groups do not have an update admission or changed tensors, but they
still cross two independently durable authorities: the causal campaign ledger
and the trainer checkpoint.  This store publishes an immutable rejection
intent before the campaign terminal, then chains the exact terminal and trainer
checkpoint.  Recovery may finish those publications only while the live policy
still equals the intent's unchanged policy anchor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.learning.grpo_training_state import (
    GRPOCheckpointError,
    validate_grpo_checkpoint_state,
)
from core.learning.grpo_training_state import (
    canonical_json_bytes as checkpoint_json_bytes,
)
from core.learning.verified_transition_transaction import (
    TrainerCheckpointEvidence,
    load_trainer_checkpoint_evidence,
    validate_trainer_step_static,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import read_stable_bytes

REJECTION_INTENT_SCHEMA = "aura.verified_transition.rejection_intent.v1"
REJECTION_EVENT_SCHEMA = "aura.verified_transition.rejection_event.v1"
VERIFIED_TRANSITION_STEP_SCHEMA = "aura.verified_transition.trainer_step.v1"
CAMPAIGN_TERMINAL_SCHEMAS = frozenset(
    {
        "aura.verified_transition.campaign_group_terminal.v2",
        "aura.verified_transition.causal_group_terminal.v1",
    }
)
_LEGACY_TERMINAL_KEYS = frozenset(
    {
        "schema",
        "campaign_manifest_sha256",
        "sequence",
        "group_id",
        "group_manifest_sha256",
        "group_start_sha256",
        "status",
        "reward_receipt_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "terminal_reason",
        "finished_at_unix_ns",
        "receipt_sha256",
    }
)
_CAUSAL_TERMINAL_KEYS = _LEGACY_TERMINAL_KEYS | {
    "campaign_schedule_root_sha256",
    "policy_before_sha256",
    "policy_after_sha256",
}

_INTENT_FILE = "00000000-intent.json"
_TERMINAL_FILE = "00000001-campaign-terminal.json"
_CHECKPOINT_FILE = "00000002-trainer-checkpoint.json"
_ORDERED_FILES = (_INTENT_FILE, _TERMINAL_FILE, _CHECKPOINT_FILE)
_TRANSACTION_RE = re.compile(r"^seq-(?P<sequence>[0-9]{8})-(?P<reward>[0-9a-f]{64})$")
_INTENT_KEYS = frozenset(
    {
        "schema",
        "sequence",
        "trainer_step",
        "task_id",
        "trainer_sample_seed",
        "execution_spec_sha256",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "group_manifest_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "trainer_step_static",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)
_EVENT_KEYS = frozenset(
    {
        "schema",
        "kind",
        "sequence",
        "reward_receipt_sha256",
        "previous_receipt_sha256",
        "evidence_sha256",
        "checkpoint_artifact_sha256",
        "evidence",
        "receipt_sha256",
    }
)


class VerifiedTransitionRejectionTransactionError(RuntimeError):
    """Stable fail-closed rejected-group transaction error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionRejectionTransactionError(code)


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise VerifiedTransitionRejectionTransactionError(
            "rejection_document_not_canonicalizable"
        ) from exc


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    document["receipt_sha256"] = _digest(document)
    return document


def _validate_seal(value: Mapping[str, Any], *, role: str) -> str:
    observed = _sha256(value.get("receipt_sha256"), role=f"{role}_receipt")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if observed != _digest(unsigned):
        _fail(f"{role}_digest_mismatch")
    return observed


def _private_directory(path: Path, *, role: str) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            _fail(f"{role}_symlink_rejected")
    directory = ensure_private_directory(lexical).resolve(strict=True)
    _assert_private_owned_directory(directory, role=role)
    return directory


def _assert_private_owned_directory(path: Path, *, role: str) -> None:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise VerifiedTransitionRejectionTransactionError(
            f"{role}_unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail(f"{role}_not_private_owned_directory")


def _read_document(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise VerifiedTransitionRejectionTransactionError(
            f"{role}_unavailable"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        _fail(f"{role}_not_private_owned_file")
    raw = read_stable_bytes(path, max_bytes=64 * 1024 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionRejectionTransactionError(
            f"{role}_json_invalid"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        _fail(f"{role}_json_noncanonical")
    return value


def build_rejection_intent(
    *,
    sequence: int,
    trainer_step: int,
    task_id: str,
    trainer_sample_seed: int,
    execution_spec_sha256: str,
    campaign_manifest_sha256: str,
    campaign_schedule_root_sha256: str,
    group_manifest_sha256: str,
    reward_receipt_sha256: str,
    policy_sha256: str,
    trainer_step_static: Mapping[str, Any],
    created_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    created = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
    return validate_rejection_intent(
        _seal(
            {
                "schema": REJECTION_INTENT_SCHEMA,
                "sequence": sequence,
                "trainer_step": trainer_step,
                "task_id": task_id,
                "trainer_sample_seed": trainer_sample_seed,
                "execution_spec_sha256": execution_spec_sha256,
                "campaign_manifest_sha256": campaign_manifest_sha256,
                "campaign_schedule_root_sha256": campaign_schedule_root_sha256,
                "group_manifest_sha256": group_manifest_sha256,
                "reward_receipt_sha256": reward_receipt_sha256,
                "policy_sha256": policy_sha256,
                "trainer_step_static": validate_trainer_step_static(
                    trainer_step_static
                ),
                "created_at_unix_ns": created,
            }
        )
    )


def validate_rejection_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _INTENT_KEYS:
        _fail("rejection_intent_schema_invalid")
    intent = dict(value)
    if intent.get("schema") != REJECTION_INTENT_SCHEMA:
        _fail("rejection_intent_version_invalid")
    _validate_seal(intent, role="rejection_intent")
    sequence = _integer(intent.get("sequence"), role="rejection_sequence")
    if _integer(
        intent.get("trainer_step"), role="rejection_trainer_step", minimum=1
    ) != sequence + 1:
        _fail("rejection_trainer_step_sequence_mismatch")
    task_id = intent.get("task_id")
    if (
        not isinstance(task_id, str)
        or not task_id
        or task_id != task_id.strip()
        or "\x00" in task_id
        or len(task_id.encode("utf-8")) > 4096
    ):
        _fail("rejection_task_id_invalid")
    _integer(intent.get("trainer_sample_seed"), role="rejection_trainer_seed")
    for field in (
        "execution_spec_sha256",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "group_manifest_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
    ):
        _sha256(intent.get(field), role=f"rejection_{field}")
    static = validate_trainer_step_static(intent.get("trainer_step_static"))
    reason = static.get("optimizer_admission_reason")
    if not isinstance(reason, str) or not reason:
        _fail("rejection_terminal_reason_invalid")
    intent["trainer_step_static"] = static
    _integer(intent.get("created_at_unix_ns"), role="rejection_created", minimum=1)
    return intent


def _validate_terminal(
    terminal: Mapping[str, Any], intent: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("schema") not in CAMPAIGN_TERMINAL_SCHEMAS
    ):
        _fail("rejection_terminal_schema_invalid")
    document = dict(terminal)
    expected_keys = (
        _CAUSAL_TERMINAL_KEYS
        if document.get("schema")
        == "aura.verified_transition.causal_group_terminal.v1"
        else _LEGACY_TERMINAL_KEYS
    )
    if set(document) != expected_keys:
        _fail("rejection_terminal_schema_invalid")
    _validate_seal(document, role="rejection_terminal")
    group_id = document.get("group_id")
    if (
        not isinstance(group_id, str)
        or not group_id
        or group_id != group_id.strip()
        or "\x00" in group_id
        or len(group_id) > 512
    ):
        _fail("rejection_terminal_group_id_invalid")
    _sha256(document.get("group_start_sha256"), role="rejection_group_start")
    if _integer(
        document.get("finished_at_unix_ns"),
        role="rejection_finished_at",
        minimum=1,
    ) < intent["created_at_unix_ns"]:
        _fail("rejection_terminal_precedes_intent")
    static = intent["trainer_step_static"]
    if (
        document.get("sequence") != intent["sequence"]
        or document.get("campaign_manifest_sha256")
        != intent["campaign_manifest_sha256"]
        or document.get("group_manifest_sha256")
        != intent["group_manifest_sha256"]
        or document.get("reward_receipt_sha256")
        != intent["reward_receipt_sha256"]
        or document.get("group_admission_sha256") is not None
        or document.get("update_receipt_sha256") is not None
        or document.get("status") != "rejected"
        or document.get("terminal_reason")
        != static["optimizer_admission_reason"]
        or (
            document.get("schema")
            == "aura.verified_transition.causal_group_terminal.v1"
            and (
                document.get("campaign_schedule_root_sha256")
                != intent["campaign_schedule_root_sha256"]
                or document.get("policy_before_sha256")
                != intent["policy_sha256"]
                or document.get("policy_after_sha256")
                != intent["policy_sha256"]
            )
        )
    ):
        _fail("rejection_terminal_binding_mismatch")
    return document


def _build_rejected_trainer_step(
    *, intent: Mapping[str, Any], terminal: Mapping[str, Any]
) -> dict[str, Any]:
    intent = validate_rejection_intent(intent)
    static = validate_trainer_step_static(intent["trainer_step_static"])
    terminal = _validate_terminal(terminal, intent)
    step = {
        "schema": VERIFIED_TRANSITION_STEP_SCHEMA,
        "step": intent["trainer_step"],
        "campaign_sequence": intent["sequence"],
        "task_id": intent["task_id"],
        "sample_seed": intent["trainer_sample_seed"],
        "execution_spec_sha256": intent["execution_spec_sha256"],
        "samples": static["samples"],
        "structured_rewards": static["structured_rewards"],
        "reward_receipt_sha256": intent["reward_receipt_sha256"],
        "group_manifest_sha256": intent["group_manifest_sha256"],
        "group_admission_sha256": None,
        "update_receipt_sha256": None,
        "optimizer_admission_reason": static["optimizer_admission_reason"],
        "answer_channel": static["answer_channel"],
        "advantage_report": static["advantage_report"],
        "step_kind": "verified_rejected_group",
        "update": None,
        "terminal": terminal,
        "policy_before_sha256": intent["policy_sha256"],
        "policy_after_sha256": intent["policy_sha256"],
    }
    step["receipt_sha256"] = hashlib.sha256(checkpoint_json_bytes(step)).hexdigest()
    return step


def build_rejected_transaction_trainer_step(
    transaction: LoadedRejectedGroupTransaction,
) -> dict[str, Any]:
    if not isinstance(transaction, LoadedRejectedGroupTransaction):
        _fail("rejection_trainer_step_source_invalid")
    if not transaction.events or transaction.events[0].get("kind") != "campaign_terminal":
        _fail("rejection_trainer_step_terminal_missing")
    return _build_rejected_trainer_step(
        intent=transaction.intent,
        terminal=transaction.events[0]["evidence"],
    )


@dataclass(frozen=True, slots=True)
class LoadedRejectedGroupTransaction:
    transaction_dir: Path
    intent: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]


class VerifiedTransitionRejectionTransactionStore:
    """Append-only rejection intent, terminal, and checkpoint custody."""

    def __init__(self, root: str | Path) -> None:
        self.root = _private_directory(Path(root), role="rejection_root")
        self.transactions = _private_directory(
            self.root / "rejected-groups", role="rejection_collection"
        )

    @classmethod
    def open(
        cls, root: str | Path
    ) -> VerifiedTransitionRejectionTransactionStore:
        return cls(root)

    def _directory(self, sequence: int, reward_sha256: str) -> Path:
        return self.transactions / (
            f"seq-{_integer(sequence, role='rejection_sequence'):08d}-"
            f"{_sha256(reward_sha256, role='rejection_reward')}"
        )

    def _lock(self, sequence: int, reward_sha256: str) -> Path:
        return self.root / (
            f".rejection-{_integer(sequence, role='rejection_sequence'):08d}-"
            f"{_sha256(reward_sha256, role='rejection_reward')}.lock"
        )

    @staticmethod
    def _publish(path: Path, value: Mapping[str, Any]) -> None:
        payload = _canonical_json_bytes(value)
        if not atomic_write_bytes_if_absent(path, payload, mode=0o600):
            existing = _read_document(path, role="rejection_publication")
            if existing != dict(value):
                _fail("rejection_publication_identity_conflict")

    def stage(
        self, intent: Mapping[str, Any]
    ) -> LoadedRejectedGroupTransaction:
        normalized = validate_rejection_intent(intent)
        sequence = normalized["sequence"]
        reward = normalized["reward_receipt_sha256"]
        directory = self._directory(sequence, reward)
        with interprocess_file_lock(self.root / ".rejection-inventory.lock"):
            with interprocess_file_lock(self._lock(sequence, reward)):
                if directory.is_symlink():
                    _fail("rejection_transaction_symlink_rejected")
                try:
                    directory.mkdir(mode=0o700, exist_ok=True)
                except OSError as exc:
                    raise VerifiedTransitionRejectionTransactionError(
                        "rejection_transaction_directory_unavailable"
                    ) from exc
                _assert_private_owned_directory(
                    directory, role="rejection_transaction"
                )
                self._publish(directory / _INTENT_FILE, normalized)
        loaded = self.load(sequence=sequence, reward_sha256=reward)
        if loaded is None or loaded.intent != normalized:
            _fail("rejection_intent_publication_mismatch")
        return loaded

    def _event(
        self,
        *,
        intent: Mapping[str, Any],
        kind: str,
        evidence: Mapping[str, Any],
        previous_receipt_sha256: str,
        checkpoint_artifact_sha256: str | None,
    ) -> dict[str, Any]:
        return _seal(
            {
                "schema": REJECTION_EVENT_SCHEMA,
                "kind": kind,
                "sequence": intent["sequence"],
                "reward_receipt_sha256": intent["reward_receipt_sha256"],
                "previous_receipt_sha256": previous_receipt_sha256,
                "evidence_sha256": _digest(evidence),
                "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
                "evidence": dict(evidence),
            }
        )

    def _validate_event(
        self,
        event: Mapping[str, Any],
        *,
        kind: str,
        intent: Mapping[str, Any],
        previous_receipt_sha256: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(event, Mapping)
            or set(event) != _EVENT_KEYS
            or event.get("schema") != REJECTION_EVENT_SCHEMA
            or event.get("kind") != kind
            or event.get("sequence") != intent["sequence"]
            or event.get("reward_receipt_sha256")
            != intent["reward_receipt_sha256"]
            or event.get("previous_receipt_sha256")
            != previous_receipt_sha256
        ):
            _fail("rejection_event_schema_or_lineage_invalid")
        document = dict(event)
        _validate_seal(document, role="rejection_event")
        evidence = document.get("evidence")
        if not isinstance(evidence, Mapping) or document.get("evidence_sha256") != _digest(
            evidence
        ):
            _fail("rejection_event_evidence_digest_mismatch")
        if kind == "campaign_terminal":
            if document.get("checkpoint_artifact_sha256") is not None:
                _fail("rejection_terminal_checkpoint_digest_forbidden")
            document["evidence"] = _validate_terminal(evidence, intent)
        else:
            _sha256(
                document.get("checkpoint_artifact_sha256"),
                role="rejection_checkpoint_artifact",
            )
        return document

    def load(
        self, *, sequence: int, reward_sha256: str
    ) -> LoadedRejectedGroupTransaction | None:
        directory = self._directory(sequence, reward_sha256)
        if not directory.exists():
            return None
        if directory.is_symlink():
            _fail("rejection_transaction_symlink_rejected")
        _assert_private_owned_directory(
            directory, role="rejection_transaction"
        )
        names = {path.name for path in directory.iterdir()}
        if _INTENT_FILE not in names or names - set(_ORDERED_FILES):
            _fail("rejection_transaction_file_set_invalid")
        if _CHECKPOINT_FILE in names and _TERMINAL_FILE not in names:
            _fail("rejection_transaction_event_gap")
        intent = validate_rejection_intent(
            _read_document(directory / _INTENT_FILE, role="rejection_intent")
        )
        if (
            intent["sequence"] != sequence
            or intent["reward_receipt_sha256"] != reward_sha256
        ):
            _fail("rejection_transaction_directory_binding_mismatch")
        events: list[Mapping[str, Any]] = []
        previous = intent["receipt_sha256"]
        if _TERMINAL_FILE in names:
            terminal = self._validate_event(
                _read_document(
                    directory / _TERMINAL_FILE, role="rejection_terminal_event"
                ),
                kind="campaign_terminal",
                intent=intent,
                previous_receipt_sha256=previous,
            )
            events.append(terminal)
            previous = terminal["receipt_sha256"]
        if _CHECKPOINT_FILE in names:
            checkpoint = self._validate_event(
                _read_document(
                    directory / _CHECKPOINT_FILE, role="rejection_checkpoint_event"
                ),
                kind="trainer_checkpoint",
                intent=intent,
                previous_receipt_sha256=previous,
            )
            self._validate_checkpoint(
                TrainerCheckpointEvidence(
                    checkpoint["evidence"], checkpoint["checkpoint_artifact_sha256"]
                ),
                intent=intent,
                terminal_event=events[0],
            )
            events.append(checkpoint)
        return LoadedRejectedGroupTransaction(directory, intent, tuple(events))

    def inventory(self) -> tuple[LoadedRejectedGroupTransaction, ...]:
        loaded: list[LoadedRejectedGroupTransaction] = []
        with interprocess_file_lock(self.root / ".rejection-inventory.lock"):
            for directory in sorted(self.transactions.iterdir()):
                match = _TRANSACTION_RE.fullmatch(directory.name)
                if match is None or directory.is_symlink():
                    _fail("rejection_transaction_directory_name_invalid")
                names = tuple(directory.iterdir())
                if not names:
                    directory.rmdir()
                    continue
                transaction = self.load(
                    sequence=int(match.group("sequence")),
                    reward_sha256=match.group("reward"),
                )
                assert transaction is not None
                loaded.append(transaction)
        return tuple(loaded)

    def record_campaign_terminal(
        self,
        *,
        sequence: int,
        reward_sha256: str,
        terminal_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with interprocess_file_lock(self._lock(sequence, reward_sha256)):
            loaded = self.load(sequence=sequence, reward_sha256=reward_sha256)
            if loaded is None:
                _fail("rejection_intent_missing")
            terminal = _validate_terminal(terminal_receipt, loaded.intent)
            event = self._event(
                intent=loaded.intent,
                kind="campaign_terminal",
                evidence=terminal,
                previous_receipt_sha256=loaded.intent["receipt_sha256"],
                checkpoint_artifact_sha256=None,
            )
            self._publish(loaded.transaction_dir / _TERMINAL_FILE, event)
        current = self.load(sequence=sequence, reward_sha256=reward_sha256)
        assert current is not None
        return current.events[0]

    def _validate_checkpoint(
        self,
        evidence: TrainerCheckpointEvidence,
        *,
        intent: Mapping[str, Any],
        terminal_event: Mapping[str, Any] | None,
    ) -> None:
        document = evidence.document
        try:
            validate_grpo_checkpoint_state(
                document,
                require_verified_transition=True,
                complete_document=True,
            )
        except GRPOCheckpointError as exc:
            raise VerifiedTransitionRejectionTransactionError(
                "rejection_checkpoint_state_invalid"
            ) from exc
        if hashlib.sha256(checkpoint_json_bytes(document)).hexdigest() != _sha256(
            evidence.artifact_sha256, role="rejection_checkpoint_artifact"
        ):
            _fail("rejection_checkpoint_artifact_digest_mismatch")
        steps = document.get("step_receipts")
        if (
            document.get("step") != intent["trainer_step"]
            or document.get("last_step_kind") != "verified_rejected_group"
            or not isinstance(steps, list)
            or not steps
            or not isinstance(steps[-1], Mapping)
        ):
            _fail("rejection_checkpoint_step_binding_mismatch")
        step = steps[-1]
        if terminal_event is None or step != _build_rejected_trainer_step(
            intent=intent,
            terminal=terminal_event["evidence"],
        ):
            _fail("rejection_checkpoint_step_receipt_invalid")

    def record_trainer_checkpoint(
        self,
        *,
        sequence: int,
        reward_sha256: str,
        checkpoint_dir: str | Path,
    ) -> Mapping[str, Any]:
        checkpoint = load_trainer_checkpoint_evidence(checkpoint_dir)
        with interprocess_file_lock(self._lock(sequence, reward_sha256)):
            loaded = self.load(sequence=sequence, reward_sha256=reward_sha256)
            if loaded is None or len(loaded.events) < 1:
                _fail("rejection_terminal_event_missing")
            self._validate_checkpoint(
                checkpoint,
                intent=loaded.intent,
                terminal_event=loaded.events[0],
            )
            event = self._event(
                intent=loaded.intent,
                kind="trainer_checkpoint",
                evidence=checkpoint.document,
                previous_receipt_sha256=loaded.events[0]["receipt_sha256"],
                checkpoint_artifact_sha256=checkpoint.artifact_sha256,
            )
            self._publish(loaded.transaction_dir / _CHECKPOINT_FILE, event)
        current = self.load(sequence=sequence, reward_sha256=reward_sha256)
        assert current is not None
        return current.events[1]


@dataclass(slots=True)
class VerifiedTransitionRejectionTransactionCoordinator:
    """Bind one rejected trainer group to its durable intent chain."""

    store: VerifiedTransitionRejectionTransactionStore
    sequence: int
    trainer_step: int
    task_id: str
    trainer_sample_seed: int
    execution_spec_sha256: str
    campaign_manifest_sha256: str
    campaign_schedule_root_sha256: str
    group_manifest_sha256: str
    reward_receipt_sha256: str
    trainer_step_static: Mapping[str, Any]
    _staged: bool = False

    def stage_rejection(self, *, policy_sha256: str) -> None:
        self.store.stage(
            build_rejection_intent(
                sequence=self.sequence,
                trainer_step=self.trainer_step,
                task_id=self.task_id,
                trainer_sample_seed=self.trainer_sample_seed,
                execution_spec_sha256=self.execution_spec_sha256,
                campaign_manifest_sha256=self.campaign_manifest_sha256,
                campaign_schedule_root_sha256=self.campaign_schedule_root_sha256,
                group_manifest_sha256=self.group_manifest_sha256,
                reward_receipt_sha256=self.reward_receipt_sha256,
                policy_sha256=policy_sha256,
                trainer_step_static=self.trainer_step_static,
            )
        )
        self._staged = True

    def record_campaign_terminal(self, receipt: Mapping[str, Any]) -> None:
        if not self._staged:
            _fail("rejection_coordinator_not_staged")
        self.store.record_campaign_terminal(
            sequence=self.sequence,
            reward_sha256=self.reward_receipt_sha256,
            terminal_receipt=receipt,
        )

    def record_trainer_checkpoint(self, checkpoint_dir: str | Path) -> None:
        if not self._staged:
            _fail("rejection_coordinator_not_staged")
        self.store.record_trainer_checkpoint(
            sequence=self.sequence,
            reward_sha256=self.reward_receipt_sha256,
            checkpoint_dir=checkpoint_dir,
        )


__all__ = [
    "LoadedRejectedGroupTransaction",
    "REJECTION_EVENT_SCHEMA",
    "REJECTION_INTENT_SCHEMA",
    "VerifiedTransitionRejectionTransactionCoordinator",
    "VerifiedTransitionRejectionTransactionError",
    "VerifiedTransitionRejectionTransactionStore",
    "build_rejected_transaction_trainer_step",
    "build_rejection_intent",
    "validate_rejection_intent",
]
