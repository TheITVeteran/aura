"""Crash-consistent transaction custody for verified recurrent training.

The update path has two independently durable authorities: the verified-
transition campaign ledger and the trainer checkpoint.  This module bridges
their commit gap without making either authority mutable.  A transaction is an
append-only sequence of immutable directory generations:

0. post-update adapter/optimizer tensors plus the pre-sealed trainer step;
1. the externally published verified update receipt;
2. the externally published campaign terminal receipt;
3. the trainer checkpoint that contains the exact pre-sealed step.

Generation zero must be durable before callers publish generation-one
evidence.  Subsequent records are idempotent only when their bytes are exactly
the same.  There is no pickle or executable state in this format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never

from core.learning.grpo_training_state import (
    GRPOCheckpointError,
    validate_grpo_checkpoint_state,
)
from core.runtime.atomic_writer import (
    durable_replace,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.file_read_gateway import (
    StableFileIdentity,
    open_stable_readonly_binary,
    read_stable_bytes,
)

logger = logging.getLogger("Aura.VerifiedTransition.Transaction")

PENDING_TRAINER_STEP_SCHEMA = "aura.verified_transition.pending_trainer_step.v3"
PENDING_TRAINER_STEP_SCHEMA_V4 = (
    "aura.verified_transition.pending_trainer_step.v4"
)
TRAINER_STEP_STATIC_SCHEMA = "aura.verified_transition.trainer_step_static.v1"
TRANSACTION_STAGE_SCHEMA = "aura.verified_transition.transaction_stage.v1"
TRANSACTION_EVENT_SCHEMA = "aura.verified_transition.transaction_event.v1"
TRANSACTION_RECONCILIATION_SCHEMA = (
    "aura.verified_transition.transaction_reconciliation.v1"
)

UPDATE_RECEIPT_SCHEMA = "aura.verified_transition.update_receipt.v1"
CAMPAIGN_TERMINAL_SCHEMA = "aura.verified_transition.campaign_group_terminal.v2"
CAUSAL_CAMPAIGN_TERMINAL_SCHEMA = (
    "aura.verified_transition.causal_group_terminal.v1"
)
TRAINER_CHECKPOINT_SCHEMA = "aura.grpo_checkpoint.v2"

_STAGE_DIRECTORY = "00000000-staged"
_EVENT_DIRECTORIES = {
    "update_commit": "00000001-update-commit",
    "campaign_terminal": "00000002-campaign-terminal",
    "trainer_checkpoint": "00000003-trainer-checkpoint",
}
_EVENT_ORDER = tuple(_EVENT_DIRECTORIES)
_STAGE_FILES = frozenset(
    {"adapter.safetensors", "optimizer.safetensors", "pending_step.json", "generation.json"}
)
_EVENT_FILES = frozenset({"evidence.json", "generation.json"})
_UPDATE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "group_admission_sha256",
        "reservation_sha256",
        "commit_sha256",
        "objective_record_sha256",
        "objective_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "optimizer_update_count",
        "reserved_at_unix_ns",
        "committed_at_unix_ns",
        "receipt_sha256",
    }
)
_CAMPAIGN_TERMINAL_KEYS = frozenset(
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
_CAUSAL_CAMPAIGN_TERMINAL_KEYS = frozenset(
    {
        "schema",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "sequence",
        "group_id",
        "group_manifest_sha256",
        "group_start_sha256",
        "status",
        "reward_receipt_sha256",
        "group_admission_sha256",
        "update_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "terminal_reason",
        "finished_at_unix_ns",
        "receipt_sha256",
    }
)
_PENDING_KEYS_V3 = frozenset(
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
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "trainer_step_static",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)
_PENDING_KEYS_V4 = _PENDING_KEYS_V3 | {
    "pre_measurement_sha256",
    "reservation_sha256",
}
_TRAINER_STEP_STATIC_KEYS = frozenset(
    {
        "schema",
        "samples",
        "structured_rewards",
        "optimizer_admission_reason",
        "answer_channel",
        "advantage_report",
    }
)
_TRANSACTION_DIRECTORY_RE = re.compile(r"^seq-(?P<sequence>[0-9]{8})-(?P<admission>[0-9a-f]{64})$")
_TEMPORARY_GENERATION_RE = re.compile(
    r"^\.tmp-(?:00000000-staged|00000001-update-commit|"
    r"00000002-campaign-terminal|00000003-trainer-checkpoint)-[0-9a-f]{32}$"
)

ReconciliationClass = Literal[
    "before_stage",
    "after_stage",
    "after_update_commit",
    "after_campaign_terminal",
    "after_trainer_checkpoint",
]


class VerifiedTransitionTransactionError(RuntimeError):
    """A transaction artifact is incomplete, unsafe, or not cross-bound."""


def _fail(code: str) -> Never:
    raise VerifiedTransitionTransactionError(code)


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise VerifiedTransitionTransactionError(
            "transaction_document_not_canonicalizable"
        ) from exc
    return payload + (b"\n" if newline else b"")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path, *, max_bytes: int) -> tuple[str, int, StableFileIdentity]:
    digest = hashlib.sha256()
    observed_size = 0
    with open_stable_readonly_binary(path, max_bytes=max_bytes) as (handle, identity):
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            observed_size += len(chunk)
        if observed_size != identity.size:
            _fail("transaction_artifact_read_length_mismatch")
    return digest.hexdigest(), observed_size, identity


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        _fail(f"{role}_invalid")
    return value


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed["receipt_sha256"] = _digest_bytes(_canonical_json_bytes(sealed))
    return sealed


def _validate_seal(
    document: Mapping[str, Any],
    *,
    role: str,
    newline: bool = False,
) -> str:
    observed = _sha256(document.get("receipt_sha256"), role=f"{role}_receipt")
    unsigned = dict(document)
    unsigned.pop("receipt_sha256", None)
    if observed != _digest_bytes(_canonical_json_bytes(unsigned, newline=newline)):
        _fail(f"{role}_digest_mismatch")
    return observed


def _flat_tensor_keys(tensors: Mapping[str, Any], *, role: str) -> tuple[str, ...]:
    if not isinstance(tensors, Mapping) or not tensors:
        _fail(f"{role}_tensor_map_empty")
    keys: list[str] = []
    for key, value in tensors.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 2048
            or any(ord(character) < 32 or ord(character) > 126 for character in key)
        ):
            _fail(f"{role}_tensor_key_invalid")
        if isinstance(value, Mapping):
            _fail(f"{role}_tensor_map_not_flat")
        keys.append(key)
    if len(keys) != len(set(keys)):
        _fail(f"{role}_tensor_key_duplicate")
    return tuple(sorted(keys))


def _tensor_key_digest(keys: tuple[str, ...]) -> str:
    return _digest_bytes(_canonical_json_bytes(list(keys)))


def _tensor_maps_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    try:
        import mlx.core as mx

        for key in sorted(left):
            if (
                tuple(left[key].shape) != tuple(right[key].shape)
                or str(left[key].dtype) != str(right[key].dtype)
            ):
                return False
        comparisons = [mx.array_equal(left[key], right[key]) for key in left]
        mx.eval(*comparisons)
        return all(bool(value) for value in comparisons)
    except Exception as exc:
        logger.debug("Transaction tensor comparison failed closed: %s", exc)
        return False


def _private_metadata(path: Path, *, directory: bool, role: str) -> os.stat_result:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise VerifiedTransitionTransactionError(f"{role}_unreadable") from exc
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_type(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or (not directory and metadata.st_nlink != 1)
    ):
        _fail(f"{role}_not_private_owned_{'directory' if directory else 'file'}")
    return metadata


def _assert_immutable_generation(path: Path, *, role: str) -> None:
    metadata = _private_metadata(path, directory=True, role=role)
    if stat.S_IMODE(metadata.st_mode) & 0o222:
        _fail(f"{role}_directory_is_writable")
    for child in path.iterdir():
        child_metadata = _private_metadata(child, directory=False, role=f"{role}_artifact")
        if stat.S_IMODE(child_metadata.st_mode) & 0o222:
            _fail(f"{role}_artifact_is_writable")


def _ensure_root(path: str | Path) -> Path:
    lexical = Path(path).expanduser().absolute()
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            _fail("transaction_root_symlink_component_rejected")
    root = ensure_private_directory(lexical)
    _private_metadata(root, directory=True, role="transaction_root")
    return root.resolve(strict=True)


def _ensure_private_child(parent: Path, name: str, *, role: str) -> Path:
    _validate_flat_name(name, role=role)
    _private_metadata(parent, directory=True, role=f"{role}_parent")
    child = parent / name
    if os.path.lexists(child):
        if child.is_symlink():
            _fail(f"{role}_symlink_rejected")
    else:
        child.mkdir(mode=0o700)
        _fsync_directory(parent)
    os.chmod(child, 0o700)
    _private_metadata(child, directory=True, role=role)
    return child


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_safetensors(path: Path, tensors: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    import mlx.core as mx

    keys = _flat_tensor_keys(tensors, role=role)
    try:
        mx.eval(*tensors.values())
        mx.save_safetensors(str(path), dict(tensors))
    except Exception as exc:
        raise VerifiedTransitionTransactionError(f"{role}_safetensors_write_failed") from exc
    os.chmod(path, 0o600)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    digest, size, _identity = _digest_file(path, max_bytes=(1 << 63) - 1)
    if size <= 0:
        _fail(f"{role}_safetensors_empty")
    return {
        "path": path.name,
        "sha256": digest,
        "size_bytes": size,
        "tensor_count": len(keys),
        "tensor_keys_sha256": _tensor_key_digest(keys),
    }


def _read_canonical_json(path: Path, *, role: str, newline: bool = False) -> dict[str, Any]:
    _private_metadata(path, directory=False, role=role)
    try:
        payload = read_stable_bytes(path, max_bytes=64 * 1024 * 1024)
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionTransactionError(f"{role}_json_invalid") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value, newline=newline):
        _fail(f"{role}_json_noncanonical")
    return value


def _validate_flat_name(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        _fail(f"{role}_path_invalid")
    return value


def build_trainer_step_static(
    *,
    samples: list[Mapping[str, Any]],
    structured_rewards: list[float],
    optimizer_admission_reason: str,
    answer_channel: Mapping[str, Any],
    advantage_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze trainer-step facts that exist before the policy mutation."""

    return validate_trainer_step_static(
        {
            "schema": TRAINER_STEP_STATIC_SCHEMA,
            "samples": [dict(sample) for sample in samples],
            "structured_rewards": list(structured_rewards),
            "optimizer_admission_reason": optimizer_admission_reason,
            "answer_channel": dict(answer_channel),
            "advantage_report": dict(advantage_report),
        }
    )


def validate_trainer_step_static(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRAINER_STEP_STATIC_KEYS:
        _fail("trainer_step_static_schema_invalid")
    static = dict(value)
    samples = static.get("samples")
    rewards = static.get("structured_rewards")
    if (
        static.get("schema") != TRAINER_STEP_STATIC_SCHEMA
        or not isinstance(samples, list)
        or len(samples) < 2
        or any(not isinstance(sample, Mapping) or not sample for sample in samples)
        or not isinstance(rewards, list)
        or len(rewards) != len(samples)
        or any(
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            for reward in rewards
        )
        or not isinstance(static.get("optimizer_admission_reason"), str)
        or not static["optimizer_admission_reason"]
        or not isinstance(static.get("answer_channel"), Mapping)
        or not isinstance(static.get("advantage_report"), Mapping)
    ):
        _fail("trainer_step_static_invalid")
    from core.learning.grpo import group_advantages

    normalized_rewards = [float(reward) for reward in rewards]
    if static["advantage_report"] != group_advantages(normalized_rewards):
        _fail("trainer_step_static_advantage_mismatch")
    try:
        normalized = json.loads(_canonical_json_bytes(static))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionTransactionError(
            "trainer_step_static_not_json_roundtrippable"
        ) from exc
    if normalized != static:
        _fail("trainer_step_static_not_json_roundtrippable")
    return normalized


def build_pending_trainer_step(
    *,
    sequence: int,
    trainer_step: int,
    task_id: str,
    trainer_sample_seed: int,
    execution_spec_sha256: str,
    campaign_manifest_sha256: str,
    campaign_schedule_root_sha256: str,
    group_manifest_sha256: str,
    group_admission_sha256: str,
    reward_receipt_sha256: str,
    policy_before_sha256: str,
    policy_after_sha256: str,
    trainer_step_static: Mapping[str, Any],
    pre_measurement_sha256: str | None = None,
    reservation_sha256: str | None = None,
    created_at_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Seal all facts knowable before any update evidence is published."""

    sequence = _integer(sequence, role="pending_sequence")
    trainer_step = _integer(trainer_step, role="pending_trainer_step", minimum=1)
    if trainer_step != sequence + 1:
        _fail("pending_trainer_step_sequence_mismatch")
    if not isinstance(task_id, str) or not task_id or len(task_id.encode("utf-8")) > 4096:
        _fail("pending_task_id_invalid")
    trainer_sample_seed = _integer(
        trainer_sample_seed, role="pending_trainer_sample_seed"
    )
    before = _sha256(policy_before_sha256, role="pending_policy_before")
    after = _sha256(policy_after_sha256, role="pending_policy_after")
    if before == after:
        _fail("pending_policy_unchanged")
    if (pre_measurement_sha256 is None) is not (reservation_sha256 is None):
        _fail("pending_pre_measurement_scope_incomplete")
    created = time.time_ns() if created_at_unix_ns is None else created_at_unix_ns
    material = {
            "schema": (
                PENDING_TRAINER_STEP_SCHEMA_V4
                if pre_measurement_sha256 is not None
                else PENDING_TRAINER_STEP_SCHEMA
            ),
            "sequence": sequence,
            "trainer_step": trainer_step,
            "task_id": task_id,
            "trainer_sample_seed": trainer_sample_seed,
            "execution_spec_sha256": _sha256(
                execution_spec_sha256, role="pending_execution_spec"
            ),
            "campaign_manifest_sha256": _sha256(
                campaign_manifest_sha256, role="pending_campaign_manifest"
            ),
            "campaign_schedule_root_sha256": _sha256(
                campaign_schedule_root_sha256,
                role="pending_campaign_schedule_root",
            ),
            "group_manifest_sha256": _sha256(
                group_manifest_sha256, role="pending_group_manifest"
            ),
            "group_admission_sha256": _sha256(
                group_admission_sha256, role="pending_group_admission"
            ),
            "reward_receipt_sha256": _sha256(
                reward_receipt_sha256, role="pending_reward"
            ),
            "policy_before_sha256": before,
            "policy_after_sha256": after,
            "trainer_step_static": validate_trainer_step_static(
                trainer_step_static
            ),
            "created_at_unix_ns": _integer(
                created, role="pending_created_at", minimum=1
            ),
        }
    if pre_measurement_sha256 is not None:
        material["pre_measurement_sha256"] = _sha256(
            pre_measurement_sha256,
            role="pending_pre_measurement",
        )
        material["reservation_sha256"] = _sha256(
            reservation_sha256,
            role="pending_reservation",
        )
    return _seal(material)


def validate_pending_trainer_step(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("pending_trainer_step_schema_invalid")
    pending = dict(value)
    schema = pending.get("schema")
    expected_keys = (
        _PENDING_KEYS_V3
        if schema == PENDING_TRAINER_STEP_SCHEMA
        else _PENDING_KEYS_V4
        if schema == PENDING_TRAINER_STEP_SCHEMA_V4
        else None
    )
    if expected_keys is None or set(pending) != expected_keys:
        _fail("pending_trainer_step_version_invalid")
    _validate_seal(pending, role="pending_trainer_step")
    sequence = _integer(pending.get("sequence"), role="pending_sequence")
    step = _integer(pending.get("trainer_step"), role="pending_trainer_step", minimum=1)
    if step != sequence + 1:
        _fail("pending_trainer_step_sequence_mismatch")
    if (
        not isinstance(pending.get("task_id"), str)
        or not pending["task_id"]
        or len(pending["task_id"].encode("utf-8")) > 4096
    ):
        _fail("pending_task_id_invalid")
    _integer(
        pending.get("trainer_sample_seed"), role="pending_trainer_sample_seed"
    )
    for field in (
        "execution_spec_sha256",
        "campaign_manifest_sha256",
        "campaign_schedule_root_sha256",
        "group_manifest_sha256",
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
    ):
        _sha256(pending.get(field), role=f"pending_{field}")
    if schema == PENDING_TRAINER_STEP_SCHEMA_V4:
        _sha256(
            pending.get("pre_measurement_sha256"),
            role="pending_pre_measurement",
        )
        _sha256(
            pending.get("reservation_sha256"),
            role="pending_reservation",
        )
    if pending["policy_before_sha256"] == pending["policy_after_sha256"]:
        _fail("pending_policy_unchanged")
    validate_trainer_step_static(pending.get("trainer_step_static"))
    _integer(pending.get("created_at_unix_ns"), role="pending_created_at", minimum=1)
    return pending


@dataclass(frozen=True, slots=True)
class TrainerCheckpointEvidence:
    """The exact trainer completion document and its durable artifact digest."""

    document: Mapping[str, Any]
    artifact_sha256: str


def load_trainer_checkpoint_evidence(
    checkpoint_dir: str | Path,
) -> TrainerCheckpointEvidence:
    directory = Path(checkpoint_dir).resolve(strict=True)
    path = directory / "complete.json"
    payload = read_stable_bytes(path, max_bytes=64 * 1024 * 1024)
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionTransactionError(
            "transaction_checkpoint_json_invalid"
        ) from exc
    if (
        not isinstance(document, dict)
        or payload != _canonical_json_bytes(document, newline=True)
    ):
        _fail("transaction_checkpoint_json_noncanonical")
    if document.get("checkpoint_id") != directory.name:
        _fail("transaction_checkpoint_directory_identity_mismatch")
    try:
        validate_grpo_checkpoint_state(
            document,
            require_verified_transition=True,
            complete_document=True,
        )
    except GRPOCheckpointError as exc:
        raise VerifiedTransitionTransactionError(
            "transaction_trainer_checkpoint_state_invalid"
        ) from exc
    return TrainerCheckpointEvidence(
        document=document,
        artifact_sha256=_digest_bytes(payload),
    )


@dataclass(frozen=True, slots=True)
class LoadedVerifiedTransitionTransaction:
    transaction_dir: Path
    stage: dict[str, Any]
    pending_step: dict[str, Any]
    adapter_tensors: dict[str, Any] | None
    optimizer_tensors: dict[str, Any] | None
    events: tuple[dict[str, Any], ...]


def build_transaction_trainer_step(
    transaction: LoadedVerifiedTransitionTransaction,
) -> dict[str, Any]:
    """Complete the exact trainer receipt from a staged update and terminal."""

    if not isinstance(transaction, LoadedVerifiedTransitionTransaction):
        _fail("transaction_trainer_step_source_invalid")
    if len(transaction.events) < 2 or tuple(
        event.get("kind") for event in transaction.events[:2]
    ) != ("update_commit", "campaign_terminal"):
        _fail("transaction_trainer_step_evidence_incomplete")
    pending = validate_pending_trainer_step(transaction.pending_step)
    static = validate_trainer_step_static(pending["trainer_step_static"])
    update = transaction.events[0]["evidence"]
    terminal = transaction.events[1]["evidence"]
    document = {
        "schema": "aura.verified_transition.trainer_step.v1",
        "step": pending["trainer_step"],
        "campaign_sequence": pending["sequence"],
        "task_id": pending["task_id"],
        "sample_seed": pending["trainer_sample_seed"],
        "execution_spec_sha256": pending["execution_spec_sha256"],
        "samples": static["samples"],
        "structured_rewards": static["structured_rewards"],
        "reward_receipt_sha256": pending["reward_receipt_sha256"],
        "group_manifest_sha256": pending["group_manifest_sha256"],
        "group_admission_sha256": pending["group_admission_sha256"],
        "update_receipt_sha256": update["receipt_sha256"],
        "optimizer_admission_reason": static["optimizer_admission_reason"],
        "answer_channel": static["answer_channel"],
        "advantage_report": static["advantage_report"],
        "step_kind": "verified_optimizer_update",
        "update": dict(update),
        "terminal": dict(terminal),
        "policy_before_sha256": pending["policy_before_sha256"],
        "policy_after_sha256": pending["policy_after_sha256"],
    }
    document["receipt_sha256"] = _digest_bytes(
        _canonical_json_bytes(document, newline=True)
    )
    return document


@dataclass(frozen=True, slots=True)
class VerifiedTransitionReconciliation:
    schema: str
    classification: ReconciliationClass
    sequence: int
    admission_sha256: str
    restore_staged_tensors: bool
    next_action: str
    stage_generation_sha256: str | None
    event_generation_sha256: str | None

    def receipt(self) -> dict[str, Any]:
        return _seal(
            {
                "schema": self.schema,
                "classification": self.classification,
                "sequence": self.sequence,
                "admission_sha256": self.admission_sha256,
                "restore_staged_tensors": self.restore_staged_tensors,
                "next_action": self.next_action,
                "stage_generation_sha256": self.stage_generation_sha256,
                "event_generation_sha256": self.event_generation_sha256,
            }
        )


class VerifiedTransitionTransactionStore:
    """Append-only custody for the trainer/campaign crash boundary."""

    def __init__(self, root: Path) -> None:
        self.root = _ensure_root(root)
        self.transactions = _ensure_private_child(
            self.root, "transactions", role="transaction_collection"
        )

    @classmethod
    def open(cls, root: str | Path) -> VerifiedTransitionTransactionStore:
        return cls(Path(root))

    def _transaction_dir(self, sequence: int, admission_sha256: str) -> Path:
        sequence = _integer(sequence, role="transaction_sequence")
        admission = _sha256(admission_sha256, role="transaction_admission")
        return self.transactions / f"seq-{sequence:08d}-{admission}"

    def _lock_path(self, sequence: int, admission_sha256: str) -> Path:
        return self.root / (
            f".seq-{_integer(sequence, role='transaction_sequence'):08d}-"
            f"{_sha256(admission_sha256, role='transaction_admission')}.lock"
        )

    def _generation_root(self, transaction_dir: Path) -> Path:
        return transaction_dir / "generations"

    def _visible_generations(self, generations: Path) -> dict[str, Path]:
        visible: dict[str, Path] = {}
        for path in generations.iterdir():
            if path.name.startswith(".tmp-"):
                if not _TEMPORARY_GENERATION_RE.fullmatch(path.name):
                    _fail("transaction_temporary_generation_name_invalid")
                if path.is_symlink():
                    _fail("transaction_temporary_generation_symlink_rejected")
                _private_metadata(
                    path, directory=True, role="transaction_temporary_generation"
                )
                continue
            visible[path.name] = path
        return visible

    @staticmethod
    def _remove_abandoned_tree(path: Path) -> None:
        """Delete one unpublished generation without following nested links."""

        for root, directories, files in os.walk(path, topdown=False, followlinks=False):
            root_path = Path(root)
            for filename in files:
                child = root_path / filename
                if child.is_symlink():
                    child.unlink()
                else:
                    os.chmod(child, 0o600)
            for directory in directories:
                child = root_path / directory
                if child.is_symlink():
                    child.unlink()
                else:
                    os.chmod(child, 0o700)
            os.chmod(root_path, 0o700)
        shutil.rmtree(path)

    def _cleanup_temporaries_locked(self, generations: Path) -> None:
        removed = False
        for path in tuple(generations.iterdir()):
            if not path.name.startswith(".tmp-"):
                continue
            if not _TEMPORARY_GENERATION_RE.fullmatch(path.name):
                _fail("transaction_temporary_generation_name_invalid")
            if path.is_symlink():
                _fail("transaction_temporary_generation_symlink_rejected")
            _private_metadata(
                path, directory=True, role="transaction_temporary_generation"
            )
            self._remove_abandoned_tree(path)
            removed = True
        if removed:
            _fsync_directory(generations)

    def _cleanup_empty_transaction_locked(self, transaction_dir: Path) -> bool:
        """Remove a transaction that never published generation zero."""

        entries = tuple(transaction_dir.iterdir())
        if not entries:
            transaction_dir.rmdir()
            _fsync_directory(self.transactions)
            return True
        if len(entries) != 1 or entries[0].name != "generations":
            _fail("transaction_file_set_invalid")
        generations = entries[0]
        if generations.is_symlink():
            _fail("transaction_generations_symlink_rejected")
        _private_metadata(generations, directory=True, role="transaction_generations")
        self._cleanup_temporaries_locked(generations)
        if any(generations.iterdir()):
            return False
        generations.rmdir()
        transaction_dir.rmdir()
        _fsync_directory(self.transactions)
        return True

    def _publish_generation(
        self,
        *,
        generations: Path,
        name: str,
        writer: Callable[[Path], None],
    ) -> Path:
        _validate_flat_name(name, role="generation")
        target = generations / name
        if os.path.lexists(target):
            _private_metadata(target, directory=True, role="generation")
            return target
        temporary = generations / f".tmp-{name}-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            writer(temporary)
            for child in temporary.iterdir():
                if child.is_symlink() or not child.is_file():
                    _fail("transaction_generation_artifact_invalid")
                os.chmod(child, 0o400)
            os.chmod(temporary, 0o500)
            _fsync_directory(temporary)
            if os.path.lexists(target):
                _fail("transaction_generation_race")
            durable_replace(temporary, target)
            return target
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def stage(
        self,
        *,
        adapter_tensors: Mapping[str, Any],
        optimizer_tensors: Mapping[str, Any],
        pending_trainer_step: Mapping[str, Any],
    ) -> LoadedVerifiedTransitionTransaction:
        """Durably stage post-update state before publishing the update commit."""

        pending = validate_pending_trainer_step(pending_trainer_step)
        sequence = int(pending["sequence"])
        admission = str(pending["group_admission_sha256"])
        transaction_dir = self._transaction_dir(sequence, admission)
        with interprocess_file_lock(self.root / ".inventory.lock"):
            with interprocess_file_lock(self._lock_path(sequence, admission)):
                if os.path.lexists(transaction_dir) and transaction_dir.is_symlink():
                    _fail("transaction_symlink_rejected")
                if not os.path.lexists(transaction_dir):
                    transaction_dir.mkdir(mode=0o700)
                    _fsync_directory(self.transactions)
                _private_metadata(transaction_dir, directory=True, role="transaction")
                generations = _ensure_private_child(
                    transaction_dir, "generations", role="transaction_generations"
                )
                self._cleanup_temporaries_locked(generations)

                def write_stage(temporary: Path) -> None:
                    adapter = _write_safetensors(
                        temporary / "adapter.safetensors", adapter_tensors, role="adapter"
                    )
                    optimizer = _write_safetensors(
                        temporary / "optimizer.safetensors",
                        optimizer_tensors,
                        role="optimizer",
                    )
                    pending_bytes = _canonical_json_bytes(pending)
                    _write_file(temporary / "pending_step.json", pending_bytes)
                    generation = _seal(
                        {
                            "schema": TRANSACTION_STAGE_SCHEMA,
                            "generation": 0,
                            "kind": "staged",
                            "sequence": sequence,
                            "group_admission_sha256": admission,
                            "pending_step": {
                                "path": "pending_step.json",
                                "sha256": _digest_bytes(pending_bytes),
                                "size_bytes": len(pending_bytes),
                                "receipt_sha256": pending["receipt_sha256"],
                            },
                            "adapter": adapter,
                            "optimizer": optimizer,
                        }
                    )
                    _write_file(
                        temporary / "generation.json", _canonical_json_bytes(generation)
                    )

                target = self._publish_generation(
                    generations=generations,
                    name=_STAGE_DIRECTORY,
                    writer=write_stage,
                )
                existing = self._load_locked(
                    transaction_dir, load_tensors=True, expected_events=()
                )
                if target != existing.transaction_dir / "generations" / _STAGE_DIRECTORY:
                    _fail("transaction_stage_path_mismatch")
                if existing.pending_step != pending:
                    _fail("transaction_stage_identity_conflict")
                if existing.adapter_tensors is None or existing.optimizer_tensors is None:
                    _fail("transaction_stage_tensors_missing")
                if not _tensor_maps_equal(existing.adapter_tensors, adapter_tensors):
                    _fail("transaction_stage_adapter_identity_conflict")
                if not _tensor_maps_equal(existing.optimizer_tensors, optimizer_tensors):
                    _fail("transaction_stage_optimizer_identity_conflict")
                return existing

    def _validate_tensor_binding(
        self,
        generation_dir: Path,
        binding: Any,
        *,
        role: str,
        load_tensors: bool,
    ) -> dict[str, Any] | None:
        required = {
            "path",
            "sha256",
            "size_bytes",
            "tensor_count",
            "tensor_keys_sha256",
        }
        if not isinstance(binding, Mapping) or set(binding) != required:
            _fail(f"{role}_binding_schema_invalid")
        filename = _validate_flat_name(binding.get("path"), role=role)
        if filename != f"{role}.safetensors":
            _fail(f"{role}_binding_path_invalid")
        path = generation_dir / filename
        metadata = _private_metadata(path, directory=False, role=role)
        size = _integer(binding.get("size_bytes"), role=f"{role}_size", minimum=1)
        if metadata.st_size != size:
            _fail(f"{role}_size_mismatch")
        expected_digest = _sha256(
            binding.get("sha256"), role=f"{role}_sha256"
        )
        count = _integer(binding.get("tensor_count"), role=f"{role}_count", minimum=1)
        keys_digest = _sha256(
            binding.get("tensor_keys_sha256"), role=f"{role}_keys"
        )
        if not load_tensors:
            digest, observed_size, _identity = _digest_file(
                path, max_bytes=size
            )
            if observed_size != size or digest != expected_digest:
                _fail(f"{role}_digest_mismatch")
            return None
        try:
            import mlx.core as mx

            digest = hashlib.sha256()
            observed_size = 0
            with open_stable_readonly_binary(
                path,
                max_bytes=size,
            ) as (handle, identity):
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
                if (
                    observed_size != size
                    or observed_size != identity.size
                    or digest.hexdigest() != expected_digest
                ):
                    _fail(f"{role}_digest_mismatch")
                handle.seek(0)
                tensors = mx.load(handle, format="safetensors")
                if isinstance(tensors, Mapping):
                    mx.eval(*tensors.values())
        except Exception as exc:
            if isinstance(exc, VerifiedTransitionTransactionError):
                raise
            raise VerifiedTransitionTransactionError(
                f"{role}_safetensors_load_failed"
            ) from exc
        if not isinstance(tensors, dict):
            _fail(f"{role}_safetensors_container_invalid")
        keys = _flat_tensor_keys(tensors, role=role)
        if len(keys) != count or _tensor_key_digest(keys) != keys_digest:
            _fail(f"{role}_tensor_inventory_mismatch")
        return dict(tensors)

    def _load_stage(
        self, generation_dir: Path, *, load_tensors: bool
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        _private_metadata(generation_dir, directory=True, role="stage_generation")
        _assert_immutable_generation(generation_dir, role="stage_generation")
        if {path.name for path in generation_dir.iterdir()} != _STAGE_FILES:
            _fail("stage_generation_file_set_invalid")
        generation = _read_canonical_json(
            generation_dir / "generation.json", role="stage_generation"
        )
        required = {
            "schema",
            "generation",
            "kind",
            "sequence",
            "group_admission_sha256",
            "pending_step",
            "adapter",
            "optimizer",
            "receipt_sha256",
        }
        if set(generation) != required or generation.get("schema") != TRANSACTION_STAGE_SCHEMA:
            _fail("stage_generation_schema_invalid")
        _validate_seal(generation, role="stage_generation")
        if generation.get("generation") != 0 or generation.get("kind") != "staged":
            _fail("stage_generation_identity_invalid")
        pending_binding = generation.get("pending_step")
        if not isinstance(pending_binding, Mapping) or set(pending_binding) != {
            "path",
            "sha256",
            "size_bytes",
            "receipt_sha256",
        }:
            _fail("pending_step_binding_schema_invalid")
        if _validate_flat_name(pending_binding.get("path"), role="pending_step") != "pending_step.json":
            _fail("pending_step_binding_path_invalid")
        pending_path = generation_dir / "pending_step.json"
        pending = validate_pending_trainer_step(
            _read_canonical_json(pending_path, role="pending_step")
        )
        pending_digest, pending_size, _pending_identity = _digest_file(
            pending_path,
            max_bytes=_integer(
                pending_binding.get("size_bytes"),
                role="pending_step_size",
                minimum=1,
            ),
        )
        if (
            pending_digest != _sha256(pending_binding.get("sha256"), role="pending_step")
            or pending_size
            != _integer(pending_binding.get("size_bytes"), role="pending_step_size", minimum=1)
            or pending.get("receipt_sha256")
            != _sha256(pending_binding.get("receipt_sha256"), role="pending_step_receipt")
            or generation.get("sequence") != pending.get("sequence")
            or generation.get("group_admission_sha256")
            != pending.get("group_admission_sha256")
        ):
            _fail("stage_pending_step_binding_mismatch")
        if pending.get("schema") == PENDING_TRAINER_STEP_SCHEMA_V4:
            from core.learning.verified_transition_measurement_chain import (
                VerifiedTransitionMeasurementChainError,
                load_pre_measurement_for_transaction,
            )

            try:
                intent = load_pre_measurement_for_transaction(
                    self.root,
                    sequence=int(pending["sequence"]),
                    admission_sha256=str(
                        pending["group_admission_sha256"]
                    ),
                    expected_receipt_sha256=str(
                        pending["pre_measurement_sha256"]
                    ),
                )
            except VerifiedTransitionMeasurementChainError as exc:
                raise VerifiedTransitionTransactionError(
                    "stage_pre_measurement_unavailable"
                ) from exc
            if (
                intent["policy_before_sha256"]
                != pending["policy_before_sha256"]
                or intent["execution_spec_sha256"]
                != pending["execution_spec_sha256"]
                or intent["campaign_manifest_sha256"]
                != pending["campaign_manifest_sha256"]
                or intent["campaign_schedule_root_sha256"]
                != pending["campaign_schedule_root_sha256"]
                or intent["group_manifest_sha256"]
                != pending["group_manifest_sha256"]
                or intent["trainer_step_static_sha256"]
                != _digest_bytes(
                    _canonical_json_bytes(
                        pending["trainer_step_static"]
                    )
                )
                or intent["reservation_sha256"]
                != pending["reservation_sha256"]
            ):
                _fail("stage_pre_measurement_binding_mismatch")
        adapter = self._validate_tensor_binding(
            generation_dir, generation.get("adapter"), role="adapter", load_tensors=load_tensors
        )
        optimizer = self._validate_tensor_binding(
            generation_dir,
            generation.get("optimizer"),
            role="optimizer",
            load_tensors=load_tensors,
        )
        return generation, pending, adapter, optimizer

    def _validate_update_receipt(
        self, receipt: Mapping[str, Any], pending: Mapping[str, Any]
    ) -> None:
        if set(receipt) != _UPDATE_RECEIPT_KEYS or receipt.get("schema") != UPDATE_RECEIPT_SCHEMA:
            _fail("transaction_update_receipt_schema_invalid")
        _validate_seal(receipt, role="transaction_update_receipt")
        if (
            receipt.get("group_admission_sha256")
            != pending["group_admission_sha256"]
            or receipt.get("policy_before_sha256") != pending["policy_before_sha256"]
            or receipt.get("policy_after_sha256") != pending["policy_after_sha256"]
            or receipt.get("optimizer_update_count") != 1
        ):
            _fail("transaction_update_receipt_binding_mismatch")

    def _validate_campaign_terminal(
        self,
        receipt: Mapping[str, Any],
        pending: Mapping[str, Any],
        update_receipt: Mapping[str, Any],
    ) -> None:
        schema = receipt.get("schema")
        expected_keys = (
            _CAUSAL_CAMPAIGN_TERMINAL_KEYS
            if schema == CAUSAL_CAMPAIGN_TERMINAL_SCHEMA
            else _CAMPAIGN_TERMINAL_KEYS
        )
        if set(receipt) != expected_keys or schema not in {
            CAMPAIGN_TERMINAL_SCHEMA,
            CAUSAL_CAMPAIGN_TERMINAL_SCHEMA,
        }:
            _fail("transaction_campaign_terminal_schema_invalid")
        _validate_seal(receipt, role="transaction_campaign_terminal")
        if (
            receipt.get("sequence") != pending["sequence"]
            or receipt.get("campaign_manifest_sha256")
            != pending["campaign_manifest_sha256"]
            or receipt.get("group_manifest_sha256")
            != pending["group_manifest_sha256"]
            or receipt.get("group_admission_sha256")
            != pending["group_admission_sha256"]
            or receipt.get("reward_receipt_sha256")
            != pending["reward_receipt_sha256"]
            or receipt.get("update_receipt_sha256")
            != update_receipt.get("receipt_sha256")
            or receipt.get("status") != "updated"
            or (
                schema == CAUSAL_CAMPAIGN_TERMINAL_SCHEMA
                and (
                    receipt.get("campaign_schedule_root_sha256")
                    != pending["campaign_schedule_root_sha256"]
                    or receipt.get("policy_before_sha256")
                    != pending["policy_before_sha256"]
                    or receipt.get("policy_after_sha256")
                    != pending["policy_after_sha256"]
                )
            )
        ):
            _fail("transaction_campaign_terminal_binding_mismatch")

    def _validate_checkpoint(
        self,
        evidence: TrainerCheckpointEvidence,
        pending: Mapping[str, Any],
        stage: Mapping[str, Any],
        update_receipt: Mapping[str, Any],
        terminal_receipt: Mapping[str, Any],
    ) -> None:
        document = evidence.document
        if not isinstance(document, Mapping) or document.get("schema") != TRAINER_CHECKPOINT_SCHEMA:
            _fail("transaction_trainer_checkpoint_schema_invalid")
        try:
            validate_grpo_checkpoint_state(
                document,
                require_verified_transition=True,
                complete_document=True,
            )
        except GRPOCheckpointError as exc:
            raise VerifiedTransitionTransactionError(
                "transaction_trainer_checkpoint_state_invalid"
            ) from exc
        artifact_sha256 = _sha256(
            evidence.artifact_sha256, role="transaction_checkpoint_artifact"
        )
        if _digest_bytes(_canonical_json_bytes(document, newline=True)) != artifact_sha256:
            _fail("transaction_trainer_checkpoint_artifact_digest_mismatch")
        step_receipts = document.get("step_receipts")
        if (
            document.get("step") != pending["trainer_step"]
            or document.get("last_step_committed") is not True
            or not isinstance(step_receipts, list)
            or not step_receipts
            or not isinstance(step_receipts[-1], Mapping)
        ):
            _fail("transaction_trainer_checkpoint_step_binding_mismatch")
        trainer_step = step_receipts[-1]
        if (
            trainer_step.get("schema")
            != "aura.verified_transition.trainer_step.v1"
            or trainer_step.get("step") != pending["trainer_step"]
            or trainer_step.get("campaign_sequence") != pending["sequence"]
            or trainer_step.get("task_id") != pending["task_id"]
            or trainer_step.get("sample_seed") != pending["trainer_sample_seed"]
            or trainer_step.get("execution_spec_sha256")
            != pending["execution_spec_sha256"]
            or trainer_step.get("group_manifest_sha256")
            != pending["group_manifest_sha256"]
            or trainer_step.get("reward_receipt_sha256")
            != pending["reward_receipt_sha256"]
            or trainer_step.get("group_admission_sha256")
            != pending["group_admission_sha256"]
            or trainer_step.get("update_receipt_sha256")
            != update_receipt.get("receipt_sha256")
            or trainer_step.get("policy_before_sha256")
            != pending["policy_before_sha256"]
            or trainer_step.get("policy_after_sha256")
            != pending["policy_after_sha256"]
            or trainer_step.get("update") != dict(update_receipt)
            or trainer_step.get("terminal") != dict(terminal_receipt)
        ):
            _fail("transaction_trainer_checkpoint_step_receipt_invalid")
        _validate_seal(
            trainer_step, role="transaction_trainer_step", newline=True
        )
        for role in ("adapter", "optimizer"):
            checkpoint_binding = document.get(role)
            stage_binding = stage.get(role)
            if (
                not isinstance(checkpoint_binding, Mapping)
                or not isinstance(stage_binding, Mapping)
                or _validate_flat_name(
                    checkpoint_binding.get("path"), role=f"checkpoint_{role}"
                )
                != f"{role}.safetensors"
                or checkpoint_binding.get("sha256") != stage_binding.get("sha256")
                or checkpoint_binding.get("size_bytes") != stage_binding.get("size_bytes")
            ):
                _fail(f"transaction_trainer_checkpoint_{role}_binding_mismatch")

    def _publish_event(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        kind: str,
        evidence: Mapping[str, Any],
        checkpoint_artifact_sha256: str | None = None,
    ) -> dict[str, Any]:
        if kind not in _EVENT_DIRECTORIES:
            _fail("transaction_event_kind_invalid")
        transaction_dir = self._transaction_dir(sequence, admission_sha256)
        with interprocess_file_lock(self._lock_path(sequence, admission_sha256)):
            loaded = self._load_locked(transaction_dir, load_tensors=False)
            expected_index = _EVENT_ORDER.index(kind)
            if len(loaded.events) < expected_index:
                _fail("transaction_event_predecessor_missing")
            if len(loaded.events) > expected_index:
                existing = loaded.events[expected_index]
                if existing.get("evidence") != dict(evidence):
                    _fail("transaction_event_identity_conflict")
                return existing
            pending = loaded.pending_step
            if kind == "update_commit":
                self._validate_update_receipt(evidence, pending)
            elif kind == "campaign_terminal":
                self._validate_campaign_terminal(
                    evidence,
                    pending,
                    loaded.events[0]["evidence"],
                )
            else:
                if checkpoint_artifact_sha256 is None:
                    _fail("transaction_checkpoint_artifact_digest_missing")
                self._validate_checkpoint(
                    TrainerCheckpointEvidence(evidence, checkpoint_artifact_sha256),
                    pending,
                    loaded.stage,
                    loaded.events[0]["evidence"],
                    loaded.events[1]["evidence"],
                )
            previous = (
                loaded.events[-1]["receipt_sha256"]
                if loaded.events
                else loaded.stage["receipt_sha256"]
            )
            event = _seal(
                {
                    "schema": TRANSACTION_EVENT_SCHEMA,
                    "generation": expected_index + 1,
                    "kind": kind,
                    "sequence": sequence,
                    "group_admission_sha256": admission_sha256,
                    "stage_generation_sha256": loaded.stage["receipt_sha256"],
                    "previous_generation_sha256": previous,
                    "evidence_sha256": _digest_bytes(
                        _canonical_json_bytes(
                            evidence, newline=kind == "trainer_checkpoint"
                        )
                    ),
                    "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
                    "evidence": dict(evidence),
                }
            )

            def write_event(temporary: Path) -> None:
                _write_file(
                    temporary / "evidence.json",
                    _canonical_json_bytes(
                        evidence, newline=kind == "trainer_checkpoint"
                    ),
                )
                _write_file(
                    temporary / "generation.json", _canonical_json_bytes(event)
                )

            generations = self._generation_root(transaction_dir)
            self._publish_generation(
                generations=generations,
                name=_EVENT_DIRECTORIES[kind],
                writer=write_event,
            )
            replayed = self._load_locked(transaction_dir, load_tensors=False)
            return replayed.events[expected_index]

    def record_update_commit(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        update_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._publish_event(
            sequence=sequence,
            admission_sha256=admission_sha256,
            kind="update_commit",
            evidence=update_receipt,
        )

    def record_campaign_terminal(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        terminal_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._publish_event(
            sequence=sequence,
            admission_sha256=admission_sha256,
            kind="campaign_terminal",
            evidence=terminal_receipt,
        )

    def record_trainer_checkpoint(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        checkpoint: TrainerCheckpointEvidence,
    ) -> dict[str, Any]:
        return self._publish_event(
            sequence=sequence,
            admission_sha256=admission_sha256,
            kind="trainer_checkpoint",
            evidence=checkpoint.document,
            checkpoint_artifact_sha256=checkpoint.artifact_sha256,
        )

    def publish_update_commit(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        publish: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Invoke an external commit callback only after a valid stage exists."""

        self.load(sequence=sequence, admission_sha256=admission_sha256, load_tensors=False)
        receipt = publish()
        return self.record_update_commit(
            sequence=sequence,
            admission_sha256=admission_sha256,
            update_receipt=receipt,
        )

    def publish_campaign_terminal(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        publish: Callable[[], Mapping[str, Any]],
    ) -> dict[str, Any]:
        receipt = publish()
        return self.record_campaign_terminal(
            sequence=sequence,
            admission_sha256=admission_sha256,
            terminal_receipt=receipt,
        )

    def publish_trainer_checkpoint(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        publish: Callable[[], TrainerCheckpointEvidence],
    ) -> dict[str, Any]:
        checkpoint = publish()
        return self.record_trainer_checkpoint(
            sequence=sequence,
            admission_sha256=admission_sha256,
            checkpoint=checkpoint,
        )

    def _load_event(
        self,
        generation_dir: Path,
        *,
        expected_kind: str,
        generation_number: int,
        pending: Mapping[str, Any],
        stage: Mapping[str, Any],
        previous_sha256: str,
        prior_events: tuple[Mapping[str, Any], ...],
    ) -> dict[str, Any]:
        _private_metadata(generation_dir, directory=True, role="event_generation")
        _assert_immutable_generation(generation_dir, role="event_generation")
        if {path.name for path in generation_dir.iterdir()} != _EVENT_FILES:
            _fail("event_generation_file_set_invalid")
        event = _read_canonical_json(
            generation_dir / "generation.json", role="transaction_event"
        )
        required = {
            "schema",
            "generation",
            "kind",
            "sequence",
            "group_admission_sha256",
            "stage_generation_sha256",
            "previous_generation_sha256",
            "evidence_sha256",
            "checkpoint_artifact_sha256",
            "evidence",
            "receipt_sha256",
        }
        if set(event) != required or event.get("schema") != TRANSACTION_EVENT_SCHEMA:
            _fail("transaction_event_schema_invalid")
        _validate_seal(event, role="transaction_event")
        if (
            event.get("generation") != generation_number
            or event.get("kind") != expected_kind
            or event.get("sequence") != pending["sequence"]
            or event.get("group_admission_sha256")
            != pending["group_admission_sha256"]
            or event.get("stage_generation_sha256") != stage["receipt_sha256"]
            or event.get("previous_generation_sha256") != previous_sha256
        ):
            _fail("transaction_event_chain_mismatch")
        evidence_path = generation_dir / "evidence.json"
        evidence = _read_canonical_json(
            evidence_path,
            role="transaction_event_evidence",
            newline=expected_kind == "trainer_checkpoint",
        )
        if evidence != event.get("evidence") or _digest_bytes(
            _canonical_json_bytes(
                evidence, newline=expected_kind == "trainer_checkpoint"
            )
        ) != _sha256(event.get("evidence_sha256"), role="event_evidence"):
            _fail("transaction_event_evidence_binding_mismatch")
        if expected_kind == "update_commit":
            if event.get("checkpoint_artifact_sha256") is not None:
                _fail("transaction_noncheckpoint_has_checkpoint_digest")
            self._validate_update_receipt(evidence, pending)
        elif expected_kind == "campaign_terminal":
            if event.get("checkpoint_artifact_sha256") is not None:
                _fail("transaction_noncheckpoint_has_checkpoint_digest")
            if len(prior_events) != 1:
                _fail("transaction_campaign_terminal_predecessor_invalid")
            self._validate_campaign_terminal(
                evidence,
                pending,
                prior_events[0]["evidence"],
            )
        else:
            if len(prior_events) != 2:
                _fail("transaction_checkpoint_predecessors_invalid")
            self._validate_checkpoint(
                TrainerCheckpointEvidence(
                    evidence,
                    _sha256(
                        event.get("checkpoint_artifact_sha256"),
                        role="transaction_checkpoint_artifact",
                    ),
                ),
                pending,
                stage,
                prior_events[0]["evidence"],
                prior_events[1]["evidence"],
            )
        return event

    def _load_locked(
        self,
        transaction_dir: Path,
        *,
        load_tensors: bool,
        expected_events: tuple[str, ...] | None = None,
    ) -> LoadedVerifiedTransitionTransaction:
        _private_metadata(transaction_dir, directory=True, role="transaction")
        if {path.name for path in transaction_dir.iterdir()} != {"generations"}:
            _fail("transaction_file_set_invalid")
        generations = self._generation_root(transaction_dir)
        _private_metadata(generations, directory=True, role="transaction_generations")
        visible = self._visible_generations(generations)
        allowed = {_STAGE_DIRECTORY, *_EVENT_DIRECTORIES.values()}
        if not set(visible).issubset(allowed):
            _fail("transaction_generation_name_invalid")
        if _STAGE_DIRECTORY not in visible:
            _fail("transaction_stage_missing")
        stage, pending, adapter, optimizer = self._load_stage(
            visible[_STAGE_DIRECTORY], load_tensors=load_tensors
        )
        expected_transaction = self._transaction_dir(
            int(pending["sequence"]), str(pending["group_admission_sha256"])
        )
        if transaction_dir.resolve(strict=True) != expected_transaction.resolve(strict=True):
            _fail("transaction_directory_identity_mismatch")
        events: list[dict[str, Any]] = []
        previous = str(stage["receipt_sha256"])
        missing_seen = False
        for index, kind in enumerate(_EVENT_ORDER, start=1):
            name = _EVENT_DIRECTORIES[kind]
            if name not in visible:
                missing_seen = True
                continue
            if missing_seen:
                _fail("transaction_event_chain_has_gap")
            event = self._load_event(
                visible[name],
                expected_kind=kind,
                generation_number=index,
                pending=pending,
                stage=stage,
                previous_sha256=previous,
                prior_events=tuple(events),
            )
            events.append(event)
            previous = str(event["receipt_sha256"])
        if expected_events is not None and tuple(
            str(event["kind"]) for event in events
        ) != expected_events:
            _fail("transaction_stage_already_advanced")
        return LoadedVerifiedTransitionTransaction(
            transaction_dir=transaction_dir,
            stage=stage,
            pending_step=pending,
            adapter_tensors=adapter,
            optimizer_tensors=optimizer,
            events=tuple(events),
        )

    def load(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        load_tensors: bool = True,
    ) -> LoadedVerifiedTransitionTransaction | None:
        transaction_dir = self._transaction_dir(sequence, admission_sha256)
        with interprocess_file_lock(self._lock_path(sequence, admission_sha256)):
            if not os.path.lexists(transaction_dir):
                return None
            if transaction_dir.is_symlink():
                _fail("transaction_symlink_rejected")
            generations = transaction_dir / "generations"
            _private_metadata(transaction_dir, directory=True, role="transaction")
            if os.path.lexists(generations) and generations.is_symlink():
                _fail("transaction_generations_symlink_rejected")
            if not os.path.lexists(generations):
                self._cleanup_empty_transaction_locked(transaction_dir)
                return None
            _private_metadata(
                generations, directory=True, role="transaction_generations"
            )
            self._cleanup_temporaries_locked(generations)
            visible = self._visible_generations(generations)
            if not visible:
                if not self._cleanup_empty_transaction_locked(transaction_dir):
                    _fail("transaction_unpublished_generation_cleanup_failed")
                return None
            return self._load_locked(transaction_dir, load_tensors=load_tensors)

    def inventory(
        self, *, load_tensors: bool = False
    ) -> tuple[LoadedVerifiedTransitionTransaction, ...]:
        """Return every transaction in strict sequence order.

        Unknown directory names, duplicate sequences, and symlinks fail
        closed. Sequence gaps are valid because rejected groups do not mutate
        tensors and therefore do not create update transactions.
        """

        with interprocess_file_lock(self.root / ".inventory.lock"):
            observed: list[tuple[int, str]] = []
            for path in self.transactions.iterdir():
                if path.is_symlink():
                    _fail("transaction_inventory_symlink_rejected")
                match = _TRANSACTION_DIRECTORY_RE.fullmatch(path.name)
                if match is None:
                    _fail("transaction_inventory_name_invalid")
                _private_metadata(
                    path, directory=True, role="transaction_inventory_entry"
                )
                observed.append(
                    (int(match.group("sequence")), match.group("admission"))
                )
            observed.sort()
            sequences = [sequence for sequence, _admission in observed]
            if len(sequences) != len(set(sequences)):
                _fail("transaction_inventory_duplicate_sequence")
            loaded: list[LoadedVerifiedTransitionTransaction] = []
            for sequence, admission in observed:
                transaction = self.load(
                    sequence=sequence,
                    admission_sha256=admission,
                    load_tensors=load_tensors,
                )
                if transaction is not None:
                    loaded.append(transaction)
            return tuple(loaded)

    def reconcile(
        self,
        *,
        sequence: int,
        admission_sha256: str,
        load_update_receipt: Callable[[], Mapping[str, Any] | None] | None = None,
        load_campaign_terminal: Callable[[], Mapping[str, Any] | None] | None = None,
        load_trainer_checkpoint: Callable[[], TrainerCheckpointEvidence | None]
        | None = None,
    ) -> VerifiedTransitionReconciliation:
        """Classify a restart, importing exact external evidence when supplied."""

        admission = _sha256(admission_sha256, role="transaction_admission")
        sequence = _integer(sequence, role="transaction_sequence")
        loaded = self.load(
            sequence=sequence, admission_sha256=admission, load_tensors=False
        )
        if loaded is None:
            external = (
                load_update_receipt() if load_update_receipt is not None else None,
                load_campaign_terminal() if load_campaign_terminal is not None else None,
                load_trainer_checkpoint() if load_trainer_checkpoint is not None else None,
            )
            if any(item is not None for item in external):
                _fail("transaction_external_evidence_without_stage")
            return VerifiedTransitionReconciliation(
                schema=TRANSACTION_RECONCILIATION_SCHEMA,
                classification="before_stage",
                sequence=sequence,
                admission_sha256=admission,
                restore_staged_tensors=False,
                next_action="prepare_fresh_verified_transition_group",
                stage_generation_sha256=None,
                event_generation_sha256=None,
            )

        observed_update = (
            load_update_receipt() if load_update_receipt is not None else None
        )
        observed_terminal = (
            load_campaign_terminal() if load_campaign_terminal is not None else None
        )
        observed_checkpoint = (
            load_trainer_checkpoint() if load_trainer_checkpoint is not None else None
        )
        if (
            observed_terminal is not None
            and observed_update is None
            and len(loaded.events) < 1
        ):
            _fail("transaction_campaign_terminal_without_update_commit")
        if observed_checkpoint is not None and (
            observed_terminal is None and len(loaded.events) < 2
        ):
            _fail("transaction_trainer_checkpoint_without_campaign_terminal")

        if len(loaded.events) == 0 and observed_update is not None:
            self.record_update_commit(
                sequence=sequence,
                admission_sha256=admission,
                update_receipt=observed_update,
            )
        loaded = self.load(sequence=sequence, admission_sha256=admission, load_tensors=False)
        assert loaded is not None
        if len(loaded.events) == 1 and observed_terminal is not None:
            self.record_campaign_terminal(
                sequence=sequence,
                admission_sha256=admission,
                terminal_receipt=observed_terminal,
            )
        loaded = self.load(sequence=sequence, admission_sha256=admission, load_tensors=False)
        assert loaded is not None
        if len(loaded.events) == 2 and observed_checkpoint is not None:
            self.record_trainer_checkpoint(
                sequence=sequence,
                admission_sha256=admission,
                checkpoint=observed_checkpoint,
            )
        loaded = self.load(sequence=sequence, admission_sha256=admission, load_tensors=False)
        assert loaded is not None

        classifications: tuple[ReconciliationClass, ...] = (
            "after_stage",
            "after_update_commit",
            "after_campaign_terminal",
            "after_trainer_checkpoint",
        )
        next_actions = (
            "restore_staged_tensors_then_publish_update_commit",
            "restore_staged_tensors_then_publish_campaign_terminal",
            "restore_staged_tensors_then_publish_trainer_checkpoint",
            "resume_from_trainer_checkpoint",
        )
        event_sha = (
            str(loaded.events[-1]["receipt_sha256"]) if loaded.events else None
        )
        return VerifiedTransitionReconciliation(
            schema=TRANSACTION_RECONCILIATION_SCHEMA,
            classification=classifications[len(loaded.events)],
            sequence=sequence,
            admission_sha256=admission,
            restore_staged_tensors=len(loaded.events) < 3,
            next_action=next_actions[len(loaded.events)],
            stage_generation_sha256=str(loaded.stage["receipt_sha256"]),
            event_generation_sha256=event_sha,
        )


@dataclass
class VerifiedTransitionTransactionCoordinator:
    """Bridge one trainer step to the append-only transaction store."""

    store: VerifiedTransitionTransactionStore
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
    adapter_tensors: Callable[[], Mapping[str, Any]]
    optimizer_tensors: Callable[[], Mapping[str, Any]]
    measurement_chain: Any | None = None
    _admission_sha256: str | None = None
    _pre_measurement_sha256: str | None = None
    _reservation_sha256: str | None = None

    def record_pre_measurement(
        self,
        *,
        group_admission_sha256: str,
        reservation_sha256: str,
        policy_before_sha256: str,
        trajectory_source_binding: Mapping[str, Any],
        recurrent_grpo_config: Any,
        bridge_tokens: tuple[int, ...],
        recorded_at_unix_ns: int,
    ) -> dict[str, Any]:
        if self.measurement_chain is None:
            _fail("transaction_pre_measurement_chain_missing")
        admission = _sha256(
            group_admission_sha256,
            role="coordinator_group_admission",
        )
        intent = self.measurement_chain.begin(
            sequence=self.sequence,
            trainer_step=self.trainer_step,
            group_admission_sha256=admission,
            reservation_sha256=reservation_sha256,
            policy_before_sha256=policy_before_sha256,
            campaign_manifest_sha256=self.campaign_manifest_sha256,
            campaign_schedule_root_sha256=(
                self.campaign_schedule_root_sha256
            ),
            group_manifest_sha256=self.group_manifest_sha256,
            execution_spec_sha256=self.execution_spec_sha256,
            trainer_step_static=self.trainer_step_static,
            trajectory_source_binding=trajectory_source_binding,
            recurrent_grpo_config=recurrent_grpo_config,
            bridge_tokens=bridge_tokens,
            live_adapter_tensors=self.adapter_tensors(),
            live_optimizer_tensors=self.optimizer_tensors(),
            recorded_at_unix_ns=recorded_at_unix_ns,
        )
        self._admission_sha256 = admission
        self._pre_measurement_sha256 = str(
            intent["receipt_sha256"]
        )
        self._reservation_sha256 = _sha256(
            reservation_sha256,
            role="coordinator_reservation",
        )
        return intent

    def stage_post_update(
        self,
        *,
        policy_before_sha256: str,
        policy_after_sha256: str,
        group_admission_sha256: str,
    ) -> LoadedVerifiedTransitionTransaction:
        admission = _sha256(
            group_admission_sha256, role="coordinator_group_admission"
        )
        if (
            self.measurement_chain is not None
            and (
                self._pre_measurement_sha256 is None
                or self._reservation_sha256 is None
                or self._admission_sha256 != admission
            )
        ):
            _fail("transaction_pre_measurement_not_recorded")
        pending = build_pending_trainer_step(
            sequence=self.sequence,
            trainer_step=self.trainer_step,
            task_id=self.task_id,
            trainer_sample_seed=self.trainer_sample_seed,
            execution_spec_sha256=self.execution_spec_sha256,
            campaign_manifest_sha256=self.campaign_manifest_sha256,
            campaign_schedule_root_sha256=self.campaign_schedule_root_sha256,
            group_manifest_sha256=self.group_manifest_sha256,
            group_admission_sha256=admission,
            reward_receipt_sha256=self.reward_receipt_sha256,
            policy_before_sha256=policy_before_sha256,
            policy_after_sha256=policy_after_sha256,
            trainer_step_static=self.trainer_step_static,
            pre_measurement_sha256=self._pre_measurement_sha256,
            reservation_sha256=self._reservation_sha256,
        )
        loaded = self.store.stage(
            adapter_tensors=self.adapter_tensors(),
            optimizer_tensors=self.optimizer_tensors(),
            pending_trainer_step=pending,
        )
        self._admission_sha256 = admission
        return loaded

    def _admission(self) -> str:
        if self._admission_sha256 is None:
            _fail("transaction_coordinator_not_staged")
        return self._admission_sha256

    def record_update_commit(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        return self.store.record_update_commit(
            sequence=self.sequence,
            admission_sha256=self._admission(),
            update_receipt=receipt,
        )

    def record_campaign_terminal(
        self, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.store.record_campaign_terminal(
            sequence=self.sequence,
            admission_sha256=self._admission(),
            terminal_receipt=receipt,
        )

    def record_trainer_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        evidence = load_trainer_checkpoint_evidence(checkpoint_dir)
        return self.store.record_trainer_checkpoint(
            sequence=self.sequence,
            admission_sha256=self._admission(),
            checkpoint=TrainerCheckpointEvidence(
                document=evidence.document,
                artifact_sha256=evidence.artifact_sha256,
            ),
        )


__all__ = [
    "CAUSAL_CAMPAIGN_TERMINAL_SCHEMA",
    "CAMPAIGN_TERMINAL_SCHEMA",
    "PENDING_TRAINER_STEP_SCHEMA",
    "PENDING_TRAINER_STEP_SCHEMA_V4",
    "TRAINER_CHECKPOINT_SCHEMA",
    "TRANSACTION_EVENT_SCHEMA",
    "TRANSACTION_RECONCILIATION_SCHEMA",
    "TRANSACTION_STAGE_SCHEMA",
    "TRAINER_STEP_STATIC_SCHEMA",
    "UPDATE_RECEIPT_SCHEMA",
    "LoadedVerifiedTransitionTransaction",
    "TrainerCheckpointEvidence",
    "VerifiedTransitionReconciliation",
    "VerifiedTransitionTransactionError",
    "VerifiedTransitionTransactionCoordinator",
    "VerifiedTransitionTransactionStore",
    "build_pending_trainer_step",
    "build_trainer_step_static",
    "build_transaction_trainer_step",
    "load_trainer_checkpoint_evidence",
    "validate_pending_trainer_step",
    "validate_trainer_step_static",
]
