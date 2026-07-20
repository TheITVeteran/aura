"""Certified recurrence checkpoint migration after a resource-envelope abort."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never

from core.runtime.atomic_writer import atomic_write_bytes, ensure_private_directory
from core.runtime.file_read_gateway import read_stable_bytes

MIGRATION_SCHEMA = "aura.recurrence_checkpoint_migration.v1"
CHECKPOINT_SCHEMA = "aura.recurrence_native_checkpoint.v3"
POINTER_SCHEMA = "aura.recurrence_native_checkpoint_pointer.v1"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_TENSOR_BYTES = 1 << 40


class RecurrenceCheckpointMigrationError(RuntimeError):
    """Stable fail-closed checkpoint migration error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrenceCheckpointMigrationError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    raw = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecurrenceCheckpointMigrationError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return raw, value


def _binding(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        _fail("migration_artifact_storage_invalid")
    raw = read_stable_bytes(resolved, max_bytes=max_bytes)
    return {"path": str(resolved), "sha256": _sha(raw), "size_bytes": len(raw)}


def _binding_matches(binding: Mapping[str, Any], *, max_bytes: int) -> bool:
    try:
        observed = _binding(Path(str(binding["path"])), max_bytes=max_bytes)
    except (KeyError, OSError, RecurrenceCheckpointMigrationError):
        return False
    return observed == dict(binding)


def _copy_bound(source: Path, destination: Path, *, max_bytes: int) -> dict[str, Any]:
    payload = read_stable_bytes(source.resolve(strict=True), max_bytes=max_bytes)
    atomic_write_bytes(destination, payload, mode=0o600)
    return _binding(destination, max_bytes=max_bytes)


def migration_identity_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, path-free migration identity embedded in a bundle."""

    source = document.get("source")
    destination = document.get("destination")
    failure = document.get("failure")
    change = document.get("required_execution_change")
    trainer = document.get("new_trainer")
    if not all(
        isinstance(value, Mapping)
        for value in (source, destination, failure, change, trainer)
    ):
        _fail("migration_evidence_invalid")
    complete = destination.get("complete")
    adapter = destination.get("adapter")
    optimizer = destination.get("optimizer")
    if not all(isinstance(value, Mapping) for value in (complete, adapter, optimizer)):
        _fail("migration_evidence_invalid")
    return {
        "schema": MIGRATION_SCHEMA,
        "migration_sha256": document.get("migration_sha256"),
        "source_checkpoint": source.get("checkpoint"),
        "source_step": source.get("step"),
        "source_config_sha256": source.get("config_sha256"),
        "dataset_sha256": source.get("dataset_sha256"),
        "execution_spec_sha256": source.get("execution_spec_sha256"),
        "checkpoint_complete_sha256": complete.get("sha256"),
        "adapter_sha256": adapter.get("sha256"),
        "optimizer_sha256": optimizer.get("sha256"),
        "failure_tombstone_sha256": failure.get("tombstone", {}).get("sha256"),
        "activation_rematerialization": change.get("activation_rematerialization"),
        "new_trainer_sha256": trainer.get("sha256"),
    }


def prepare_migration(
    *,
    source_root: Path,
    destination_root: Path,
    failed_trainer_receipt: Path,
    sentinel_receipt: Path,
    tombstone: Path,
    protocol: Path,
    amendment: Path,
    new_trainer: Path,
    output: Path,
) -> dict[str, Any]:
    source = source_root.expanduser().resolve(strict=True)
    destination = destination_root.expanduser().resolve(strict=False)
    if source == destination or destination.exists() or destination.is_symlink():
        _fail("migration_destination_invalid")
    latest_raw, latest = _read_json(source / "latest.json", role="source_latest")
    if latest.get("schema") != POINTER_SCHEMA:
        _fail("source_latest_invalid")
    checkpoint_relative = latest.get("checkpoint")
    if not isinstance(checkpoint_relative, str) or not checkpoint_relative.startswith(
        "checkpoints/"
    ):
        _fail("source_checkpoint_path_invalid")
    checkpoint = (source / checkpoint_relative).resolve(strict=True)
    if checkpoint.parent != (source / "checkpoints").resolve(strict=True):
        _fail("source_checkpoint_path_invalid")
    complete_raw, complete = _read_json(checkpoint / "complete.json", role="source_complete")
    if (
        complete.get("schema") != CHECKPOINT_SCHEMA
        or latest.get("complete_sha256") != _sha(complete_raw)
        or complete.get("step") is None
    ):
        _fail("source_checkpoint_invalid")
    training_config_raw, training_config = _read_json(
        source / "training_config.json",
        role="source_training_config",
    )
    if complete.get("config_sha256") != _sha(training_config_raw):
        _fail("source_training_config_mismatch")
    dataset_raw, dataset_document = _read_json(
        source / "dataset_manifest.json",
        role="source_dataset_manifest",
    )
    execution_spec_raw, execution_spec_document = _read_json(
        source / "execution_spec.json",
        role="source_execution_spec",
    )
    if (
        complete.get("dataset_sha256") != _sha(dataset_raw)
        or complete.get("execution_spec_sha256") != _sha(
            _canonical(execution_spec_document)
        )
        or training_config.get("dataset_sha256") != _sha(dataset_raw)
        or training_config.get("execution_spec_sha256")
        != complete.get("execution_spec_sha256")
    ):
        _fail("source_scientific_inputs_mismatch")
    trainer_raw, trainer_receipt = _read_json(
        failed_trainer_receipt,
        role="failed_trainer_receipt",
    )
    sentinel_raw, sentinel_document = _read_json(
        sentinel_receipt,
        role="sentinel_receipt",
    )
    tombstone_raw, tombstone_document = _read_json(tombstone, role="sentinel_tombstone")
    final_sample = tombstone_document.get("final_sample")
    if (
        trainer_receipt.get("returncode") != -9
        or trainer_receipt.get("containment_verified") is not True
        or trainer_receipt.get("process_group_empty") is not True
        or trainer_receipt.get("lineage_empty") is not True
        or sentinel_document.get("returncode") != 0
        or sentinel_document.get("containment_verified") is not True
        or tombstone_document.get("schema") != "aura.memory_sentinel.tombstone.v1"
        or tombstone_document.get("guard_stage") != "compute"
        or tombstone_document.get("reason")
        != "external sentinel killed process tree at lethal ceiling"
        or not isinstance(final_sample, Mapping)
        or final_sample.get("managed_mb", 0) < final_sample.get("active_lethal_mb", 1)
    ):
        _fail("resource_abort_evidence_invalid")
    destination_checkpoint = ensure_private_directory(
        destination / "checkpoints" / checkpoint.name
    )
    copied = {
        "complete": _copy_bound(
            checkpoint / "complete.json",
            destination_checkpoint / "complete.json",
            max_bytes=_MAX_JSON_BYTES,
        ),
        "adapter": _copy_bound(
            checkpoint / str(complete["adapter"]["path"]),
            destination_checkpoint / str(complete["adapter"]["path"]),
            max_bytes=_MAX_TENSOR_BYTES,
        ),
        "optimizer": _copy_bound(
            checkpoint / str(complete["optimizer"]["path"]),
            destination_checkpoint / str(complete["optimizer"]["path"]),
            max_bytes=_MAX_TENSOR_BYTES,
        ),
    }
    latest_copy = _copy_bound(
        source / "latest.json",
        destination / "latest.json",
        max_bytes=_MAX_JSON_BYTES,
    )
    material = {
        "schema": MIGRATION_SCHEMA,
        "reason": "compute_memory_envelope_exceeded_requires_exact_rematerialization",
        "prepared_at": time.time(),
        "source": {
            "root": str(source),
            "latest": _binding(source / "latest.json"),
            "checkpoint": checkpoint_relative,
            "complete": _binding(checkpoint / "complete.json"),
            "adapter": _binding(
                checkpoint / str(complete["adapter"]["path"]),
                max_bytes=_MAX_TENSOR_BYTES,
            ),
            "optimizer": _binding(
                checkpoint / str(complete["optimizer"]["path"]),
                max_bytes=_MAX_TENSOR_BYTES,
            ),
            "training_config": _binding(source / "training_config.json"),
            "training_config_document": training_config,
            "dataset_manifest": _binding(source / "dataset_manifest.json"),
            "dataset_manifest_document": dataset_document,
            "execution_spec": _binding(source / "execution_spec.json"),
            "execution_spec_document": execution_spec_document,
            "step": complete["step"],
            "config_sha256": complete["config_sha256"],
            "dataset_sha256": complete["dataset_sha256"],
            "execution_spec_sha256": complete["execution_spec_sha256"],
        },
        "destination": {
            "root": str(destination),
            "checkpoint": checkpoint_relative,
            "latest": latest_copy,
            **copied,
        },
        "failure": {
            "trainer_receipt": _binding(failed_trainer_receipt),
            "trainer_receipt_claimed_sha256": trainer_receipt.get("receipt_sha256"),
            "sentinel_receipt": _binding(sentinel_receipt),
            "sentinel_receipt_claimed_sha256": sentinel_document.get("receipt_sha256"),
            "tombstone": _binding(tombstone),
            "terminal_managed_mb": final_sample["managed_mb"],
            "terminal_lethal_mb": final_sample["active_lethal_mb"],
            "trainer_receipt_payload_sha256": _sha(trainer_raw),
            "sentinel_receipt_payload_sha256": _sha(sentinel_raw),
            "tombstone_payload_sha256": _sha(tombstone_raw),
        },
        "protocol": _binding(protocol),
        "amendment": _binding(amendment),
        "new_trainer": _binding(new_trainer),
        "required_execution_change": {
            "gradient_schema": "aura.recurrence_streamed_depth_gradient.v2",
            "activation_rematerialization": "full_depth_graph_checkpoint",
            "optimizer_state_reset": False,
            "adapter_state_reset": False,
            "sample_cursor_reset": False,
        },
    }
    document = {**material, "migration_sha256": _sha(_canonical(material))}
    output = output.expanduser().resolve(strict=False)
    if output != destination / "checkpoint_migration.json":
        _fail("migration_output_path_invalid")
    atomic_write_bytes(output, _canonical(document) + b"\n", mode=0o600)
    return document


def verify_migration(
    path: Path,
    *,
    expected_destination_root: Path,
    expected_trainer_sha256: str,
) -> dict[str, Any]:
    _raw, document = _read_json(path, role="checkpoint_migration")
    claimed = document.get("migration_sha256")
    material = dict(document)
    material.pop("migration_sha256", None)
    source = document.get("source")
    destination = document.get("destination")
    change = document.get("required_execution_change")
    trainer = document.get("new_trainer")
    failure = document.get("failure")
    if (
        document.get("schema") != MIGRATION_SCHEMA
        or claimed != _sha(_canonical(material))
        or document.get("reason")
        != "compute_memory_envelope_exceeded_requires_exact_rematerialization"
        or not all(
            isinstance(value, Mapping)
            for value in (source, destination, change, trainer, failure)
        )
        or Path(str(destination.get("root"))).resolve(strict=True)
        != expected_destination_root.resolve(strict=True)
        or trainer.get("sha256") != expected_trainer_sha256
        or change
        != {
            "gradient_schema": "aura.recurrence_streamed_depth_gradient.v2",
            "activation_rematerialization": "full_depth_graph_checkpoint",
            "optimizer_state_reset": False,
            "adapter_state_reset": False,
            "sample_cursor_reset": False,
        }
    ):
        _fail("migration_evidence_invalid")
    source_training_config = source.get("training_config")
    source_training_document = source.get("training_config_document")
    source_dataset_manifest = source.get("dataset_manifest")
    source_dataset_document = source.get("dataset_manifest_document")
    source_execution_spec = source.get("execution_spec")
    source_execution_document = source.get("execution_spec_document")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_training_config,
            source_training_document,
            source_dataset_manifest,
            source_dataset_document,
            source_execution_spec,
            source_execution_document,
        )
    ):
        _fail("migration_evidence_invalid")
    bindings = [
        (source.get("latest"), _MAX_JSON_BYTES),
        (source.get("complete"), _MAX_JSON_BYTES),
        (source.get("adapter"), _MAX_TENSOR_BYTES),
        (source.get("optimizer"), _MAX_TENSOR_BYTES),
        (source.get("training_config"), _MAX_JSON_BYTES),
        (source.get("dataset_manifest"), _MAX_JSON_BYTES),
        (source.get("execution_spec"), _MAX_JSON_BYTES),
        (destination.get("latest"), _MAX_JSON_BYTES),
        (destination.get("complete"), _MAX_JSON_BYTES),
        (destination.get("adapter"), _MAX_TENSOR_BYTES),
        (destination.get("optimizer"), _MAX_TENSOR_BYTES),
        (document.get("protocol"), _MAX_JSON_BYTES),
        (document.get("amendment"), _MAX_JSON_BYTES),
        (trainer, _MAX_JSON_BYTES),
        (failure.get("trainer_receipt"), _MAX_JSON_BYTES),
        (failure.get("sentinel_receipt"), _MAX_JSON_BYTES),
        (failure.get("tombstone"), _MAX_JSON_BYTES),
    ]
    if any(
        not isinstance(binding, Mapping)
        or not _binding_matches(binding, max_bytes=max_bytes)
        for binding, max_bytes in bindings
    ):
        _fail("migration_artifact_binding_changed")
    training_config_raw, observed_training_config = _read_json(
        Path(str(source_training_config["path"])),
        role="source_training_config",
    )
    if (
        observed_training_config != source_training_document
        or _sha(training_config_raw) != source.get("config_sha256")
    ):
        _fail("migration_source_config_changed")
    dataset_raw, observed_dataset = _read_json(
        Path(str(source_dataset_manifest["path"])),
        role="source_dataset_manifest",
    )
    execution_raw, observed_execution = _read_json(
        Path(str(source_execution_spec["path"])),
        role="source_execution_spec",
    )
    if (
        observed_dataset != source_dataset_document
        or _sha(dataset_raw) != source.get("dataset_sha256")
        or observed_execution != source_execution_document
        or _sha(_canonical(observed_execution)) != source.get("execution_spec_sha256")
        or not execution_raw
    ):
        _fail("migration_source_scientific_inputs_changed")
    for role in ("complete", "adapter", "optimizer"):
        source_binding = source.get(role)
        destination_binding = destination.get(role)
        if (
            not isinstance(source_binding, Mapping)
            or not isinstance(destination_binding, Mapping)
            or source_binding.get("sha256") != destination_binding.get("sha256")
            or source_binding.get("size_bytes") != destination_binding.get("size_bytes")
        ):
            _fail("migration_checkpoint_copy_mismatch")
    return migration_identity_summary(document)


__all__ = [
    "MIGRATION_SCHEMA",
    "RecurrenceCheckpointMigrationError",
    "migration_identity_summary",
    "prepare_migration",
    "verify_migration",
]
