"""Exact, fail-closed recurrence checkpoint migration contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.learning.recurrence_checkpoint_migration import (
    RecoveryAttemptEvidence,
    RecurrenceCheckpointMigrationError,
    prepare_migration,
    verify_migration,
)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object, *, newline: bool = False) -> bytes:
    payload = _canonical(value, newline=newline)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    checkpoint_name = "step-00000010-0123456789abcdef"
    checkpoint = source / "checkpoints" / checkpoint_name
    checkpoint.mkdir(parents=True)
    adapter = b"exact-adapter-state"
    optimizer = b"exact-optimizer-state"
    (checkpoint / "adapter.safetensors").write_bytes(adapter)
    (checkpoint / "optimizer.safetensors").write_bytes(optimizer)
    dataset_raw = _write_json(
        source / "dataset_manifest.json",
        {
            "schema": "aura.recurrence_native_dataset.v2",
            "train_seed": 17,
            "families": ["khop"],
            "task_depths": [2],
            "per_cell": 2,
        },
        newline=True,
    )
    execution_spec = {
        "n_slots": 4,
        "branch_roles": ["constructive_solution", "counterexample_search"],
    }
    _write_json(source / "execution_spec.json", execution_spec, newline=True)
    execution_spec_sha256 = _sha(_canonical(execution_spec))
    config = {
        "schema": "aura.recurrence_native_training_config.v2",
        "dataset_sha256": _sha(dataset_raw),
        "execution_spec_sha256": execution_spec_sha256,
    }
    config_raw = _write_json(source / "training_config.json", config)
    complete = {
        "schema": "aura.recurrence_native_checkpoint.v3",
        "step": 10,
        "config_sha256": _sha(config_raw),
        "dataset_sha256": _sha(dataset_raw),
        "execution_spec_sha256": execution_spec_sha256,
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": _sha(adapter),
            "size_bytes": len(adapter),
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": _sha(optimizer),
            "size_bytes": len(optimizer),
        },
    }
    complete_raw = _write_json(checkpoint / "complete.json", complete, newline=True)
    _write_json(
        source / "latest.json",
        {
            "schema": "aura.recurrence_native_checkpoint_pointer.v1",
            "checkpoint": f"checkpoints/{checkpoint_name}",
            "complete_sha256": _sha(complete_raw),
        },
        newline=True,
    )
    failed_trainer = tmp_path / "failed_trainer.json"
    _write_json(
        failed_trainer,
        {
            "returncode": -9,
            "containment_verified": True,
            "process_group_empty": True,
            "lineage_empty": True,
            "receipt_sha256": "1" * 64,
            "started_at": 1.0,
        },
    )
    sentinel = tmp_path / "sentinel.json"
    _write_json(
        sentinel,
        {
            "returncode": 0,
            "containment_verified": True,
            "receipt_sha256": "2" * 64,
        },
    )
    tombstone = tmp_path / "tombstone.json"
    _write_json(
        tombstone,
        {
            "schema": "aura.memory_sentinel.tombstone.v1",
            "guard_stage": "compute",
            "reason": "external sentinel killed process tree at lethal ceiling",
            "final_sample": {"managed_mb": 74_846.3, "active_lethal_mb": 73_728.0},
        },
    )
    protocol = tmp_path / "protocol.json"
    amendment = tmp_path / "amendment.json"
    trainer = tmp_path / "trainer.py"
    _write_json(protocol, {"schema": "test.protocol.v1"})
    _write_json(amendment, {"schema": "test.amendment.v1"})
    trainer.write_text("# exact recovery trainer\n", encoding="ascii")
    return {
        "source": source,
        "destination": destination,
        "failed_trainer": failed_trainer,
        "sentinel": sentinel,
        "tombstone": tombstone,
        "protocol": protocol,
        "amendment": amendment,
        "trainer": trainer,
        "trainer_sha256": _sha(trainer.read_bytes()),
    }


def _recovery_attempt(tmp_path: Path) -> RecoveryAttemptEvidence:
    previous_root = tmp_path / "previous" / "adapter"
    previous_root.mkdir(parents=True)
    previous_migration = tmp_path / "previous_migration.json"
    migration_material = {
        "schema": "aura.recurrence_checkpoint_migration.v1",
        "destination": {"root": str(previous_root)},
    }
    _write_json(
        previous_migration,
        {
            **migration_material,
            "migration_sha256": _sha(_canonical(migration_material)),
        },
    )
    trainer = tmp_path / "recovery_trainer.json"
    _write_json(
        trainer,
        {
            "returncode": -9,
            "containment_verified": True,
            "process_group_empty": True,
            "lineage_empty": True,
            "receipt_sha256": "6" * 64,
            "started_at": 2.0,
            "command": [
                "trainer.py",
                "--out-dir",
                str(previous_root),
                "--resume-migration-evidence",
                str(previous_migration),
            ],
        },
    )
    sentinel = tmp_path / "recovery_sentinel.json"
    _write_json(
        sentinel,
        {
            "returncode": 0,
            "containment_verified": True,
            "receipt_sha256": "7" * 64,
        },
    )
    sample = {
        "managed_mb": 75_192.0,
        "active_lethal_mb": 73_728.0,
        "guard_stage": "compute",
    }
    tombstone = tmp_path / "recovery_tombstone.json"
    _write_json(
        tombstone,
        {
            "schema": "aura.memory_sentinel.tombstone.v1",
            "guard_stage": "compute",
            "reason": "external sentinel killed process tree at lethal ceiling",
            "final_sample": sample,
        },
    )
    ring = tmp_path / "recovery_ring.jsonl"
    _write_json(ring, sample, newline=True)
    return RecoveryAttemptEvidence(
        checkpoint_migration=previous_migration,
        trainer_receipt=trainer,
        sentinel_receipt=sentinel,
        tombstone=tombstone,
        footprint_ring=ring,
    )


def _prepare(
    paths: dict[str, Path | str],
    *,
    recovery_attempts: tuple[RecoveryAttemptEvidence, ...] = (),
) -> Path:
    destination = Path(paths["destination"])
    output = destination / "checkpoint_migration.json"
    prepare_migration(
        source_root=Path(paths["source"]),
        destination_root=destination,
        failed_trainer_receipt=Path(paths["failed_trainer"]),
        sentinel_receipt=Path(paths["sentinel"]),
        tombstone=Path(paths["tombstone"]),
        protocol=Path(paths["protocol"]),
        amendment=Path(paths["amendment"]),
        new_trainer=Path(paths["trainer"]),
        output=output,
        recovery_attempts=recovery_attempts,
    )
    return output


def test_migration_copies_exact_state_and_verifies_identity(tmp_path: Path):
    paths = _fixture(tmp_path)
    migration = _prepare(paths)

    result = verify_migration(
        migration,
        expected_destination_root=Path(paths["destination"]),
        expected_trainer_sha256=str(paths["trainer_sha256"]),
    )

    assert result["source_step"] == 10
    assert result["source_checkpoint"].startswith("checkpoints/step-00000010-")
    assert result["activation_rematerialization"] == "transformer_layer_group_checkpoint"
    assert result["layer_group_size"] == 4
    assert result["new_trainer_sha256"] == paths["trainer_sha256"]
    document = json.loads(migration.read_text(encoding="ascii"))
    for role in ("complete", "adapter", "optimizer"):
        assert document["source"][role]["sha256"] == document["destination"][role]["sha256"]


@pytest.mark.parametrize("target", ["adapter", "optimizer", "complete"])
def test_migration_rejects_destination_checkpoint_tamper(tmp_path: Path, target: str):
    paths = _fixture(tmp_path)
    migration = _prepare(paths)
    document = json.loads(migration.read_text(encoding="ascii"))
    bound = Path(document["destination"][target]["path"])
    bound.write_bytes(bound.read_bytes() + b"tamper")

    with pytest.raises(
        RecurrenceCheckpointMigrationError,
        match="migration_artifact_binding_changed",
    ):
        verify_migration(
            migration,
            expected_destination_root=Path(paths["destination"]),
            expected_trainer_sha256=str(paths["trainer_sha256"]),
        )


def test_migration_rejects_source_config_tamper(tmp_path: Path):
    paths = _fixture(tmp_path)
    migration = _prepare(paths)
    config = Path(paths["source"]) / "training_config.json"
    config.write_bytes(config.read_bytes() + b" ")

    with pytest.raises(
        RecurrenceCheckpointMigrationError,
        match="migration_artifact_binding_changed",
    ):
        verify_migration(
            migration,
            expected_destination_root=Path(paths["destination"]),
            expected_trainer_sha256=str(paths["trainer_sha256"]),
        )


def test_migration_requires_the_bound_recovery_trainer(tmp_path: Path):
    paths = _fixture(tmp_path)
    migration = _prepare(paths)

    with pytest.raises(RecurrenceCheckpointMigrationError, match="migration_evidence_invalid"):
        verify_migration(
            migration,
            expected_destination_root=Path(paths["destination"]),
            expected_trainer_sha256="f" * 64,
        )


def test_migration_binds_and_verifies_ordered_recovery_failure_chain(tmp_path: Path):
    paths = _fixture(tmp_path)
    attempt = _recovery_attempt(tmp_path)
    migration = _prepare(paths, recovery_attempts=(attempt,))

    result = verify_migration(
        migration,
        expected_destination_root=Path(paths["destination"]),
        expected_trainer_sha256=str(paths["trainer_sha256"]),
    )

    document = json.loads(migration.read_text(encoding="ascii"))
    assert result["recovery_attempt_count"] == 1
    assert result["recovery_attempts_sha256"] == _sha(
        _canonical(document["recovery_attempts"])
    )

    attempt.footprint_ring.write_bytes(attempt.footprint_ring.read_bytes() + b" ")
    with pytest.raises(
        RecurrenceCheckpointMigrationError,
        match="recovery_attempt_evidence_changed|migration_artifact_binding_changed",
    ):
        verify_migration(
            migration,
            expected_destination_root=Path(paths["destination"]),
            expected_trainer_sha256=str(paths["trainer_sha256"]),
        )
