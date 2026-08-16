"""Custody and crash-recovery contracts for immutable recurrence checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.unified_intrinsic_checkpoint import (
    CHECKPOINT_GENERATION_SCHEMA,
    CHECKPOINT_POINTER_SCHEMA,
    CHECKPOINT_RETENTION_SCHEMA,
    TRAINING_SCHEMA,
    UnifiedCheckpointError,
    checkpoint_retention_plan,
    prune_checkpoint_generations,
    resolve_checkpoint_generation,
    unpointed_checkpoint_inventory,
)
from tools.unified_intrinsic_resident_identity import canonical_bytes, canonical_sha256


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


def _generation(
    output: Path,
    *,
    stem: str,
    step: int,
    nonce: int,
) -> tuple[Path, dict[str, object], bytes]:
    root = output / "checkpoint_generations"
    root.mkdir(mode=0o700, exist_ok=True)
    checkpoint_id = f"{stem}-step-{step:08d}-{nonce:032x}"
    generation = root / checkpoint_id
    generation.mkdir(mode=0o700)
    payload = f"weights:{stem}:{step}:{nonce}".encode("ascii")
    weights = generation / "bundle.safetensors"
    weights.write_bytes(payload)
    body: dict[str, object] = {
        "schema": TRAINING_SCHEMA,
        "checkpoint_generation_schema": CHECKPOINT_GENERATION_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "stem": stem,
        "step": step,
        "optimization_phase": "test",
        "history": [],
        "training_state": {},
        "identity": {"identity_sha256": f"{nonce:064x}"[-64:]},
        "checkpoint_file": weights.name,
        "checkpoint_size_bytes": len(payload),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    receipt_raw = canonical_bytes(receipt) + b"\n"
    complete = generation / "complete.json"
    complete.write_bytes(receipt_raw)
    os.chmod(weights, 0o400)
    os.chmod(complete, 0o400)
    os.chmod(generation, 0o500)
    return generation, receipt, receipt_raw


def _point(
    output: Path,
    *,
    stem: str,
    generation: Path,
    receipt: dict[str, object],
    receipt_raw: bytes,
) -> None:
    identity = receipt["identity"]
    assert isinstance(identity, dict)
    pointer = {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "checkpoint": f"checkpoint_generations/{generation.name}",
        "complete_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "identity_sha256": identity["identity_sha256"],
        "step": receipt["step"],
        "stem": stem,
    }
    (output / f"{stem}_pointer.json").write_bytes(
        canonical_bytes(pointer) + b"\n"
    )


def test_checkpoint_output_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    output = _private(tmp_path / "output")
    linked = tmp_path / "linked-output"
    linked.symlink_to(output, target_is_directory=True)

    with pytest.raises(UnifiedCheckpointError, match="output is a symlink"):
        resolve_checkpoint_generation(linked, required=False)


def test_generation_root_symlink_cannot_escape_output_custody(tmp_path: Path) -> None:
    output = _private(tmp_path / "output")
    external = _private(tmp_path / "external")
    (output / "checkpoint_generations").symlink_to(
        external,
        target_is_directory=True,
    )
    checkpoint_id = f"checkpoint_latest-step-00000001-{'a' * 32}"
    pointer = {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "checkpoint": f"checkpoint_generations/{checkpoint_id}",
        "complete_sha256": "b" * 64,
        "identity_sha256": "c" * 64,
        "step": 1,
        "stem": "checkpoint_latest",
    }
    (output / "checkpoint_latest_pointer.json").write_text(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(UnifiedCheckpointError, match="generation is a symlink"):
        resolve_checkpoint_generation(output)


def test_unpointed_inventory_accepts_only_owned_checkpoint_shapes(
    tmp_path: Path,
) -> None:
    output = _private(tmp_path / "output")
    generations = _private(output / "checkpoint_generations")
    orphan = generations / f"checkpoint_latest-step-00000001-{'d' * 32}"
    orphan.mkdir(mode=0o500)
    staging = generations / f".checkpoint-stage-{'e' * 32}"
    staging.mkdir(mode=0o700)

    assert unpointed_checkpoint_inventory(output) == {
        "orphan_generations": 1,
        "staged_generations": 1,
    }

    unsafe = generations / "unexpected"
    unsafe.mkdir(mode=0o700)
    with pytest.raises(UnifiedCheckpointError, match="name differs"):
        unpointed_checkpoint_inventory(output)


def test_retention_preserves_every_pointer_and_two_rollbacks_per_stem(
    tmp_path: Path,
) -> None:
    output = _private(tmp_path / "output")
    latest = [
        _generation(output, stem="checkpoint_latest", step=step, nonce=step)
        for step in range(1, 6)
    ]
    best = [
        _generation(output, stem="checkpoint_best", step=step, nonce=100 + step)
        for step in range(1, 3)
    ]
    heldout = _generation(
        output,
        stem="checkpoint_heldout",
        step=1,
        nonce=201,
    )
    _point(
        output,
        stem="checkpoint_latest",
        generation=latest[-1][0],
        receipt=latest[-1][1],
        receipt_raw=latest[-1][2],
    )
    _point(
        output,
        stem="checkpoint_best",
        generation=best[-1][0],
        receipt=best[-1][1],
        receipt_raw=best[-1][2],
    )
    _point(
        output,
        stem="checkpoint_heldout",
        generation=heldout[0],
        receipt=heldout[1],
        receipt_raw=heldout[2],
    )
    staging = output / "checkpoint_generations" / f".checkpoint-stage-{'f' * 32}"
    staging.mkdir(mode=0o700)

    plan = checkpoint_retention_plan(output, rollback_generations_per_stem=2)

    assert set(plan.protected_generations) == {
        latest[-1][0].name,
        best[-1][0].name,
        heldout[0].name,
    }
    assert set(plan.rollback_generations) == {
        latest[-2][0].name,
        latest[-3][0].name,
        best[0][0].name,
    }
    assert {candidate.name for candidate in plan.candidates} == {
        latest[0][0].name,
        latest[1][0].name,
    }
    assert plan.staged_generations == (staging.name,)

    receipt = prune_checkpoint_generations(
        output,
        rollback_generations_per_stem=2,
    )

    assert receipt["schema"] == CHECKPOINT_RETENTION_SCHEMA
    assert receipt["state"] == "complete"
    assert set(receipt["deleted_generations"]) == {
        latest[0][0].name,
        latest[1][0].name,
    }
    assert not latest[0][0].exists()
    assert not latest[1][0].exists()
    assert all(row[0].exists() for row in latest[2:] + best + [heldout])
    assert staging.exists()
    receipt_path = Path(str(receipt["receipt_path"]))
    stored = json.loads(receipt_path.read_text(encoding="ascii"))
    stored_body = {key: value for key, value in stored.items() if key != "receipt_sha256"}
    assert stored["receipt_sha256"] == canonical_sha256(stored_body)


def test_retention_dry_run_reports_without_removing(tmp_path: Path) -> None:
    output = _private(tmp_path / "output")
    generations = [
        _generation(output, stem="checkpoint_latest", step=step, nonce=step)
        for step in range(1, 5)
    ]
    _point(
        output,
        stem="checkpoint_latest",
        generation=generations[-1][0],
        receipt=generations[-1][1],
        receipt_raw=generations[-1][2],
    )

    receipt = prune_checkpoint_generations(
        output,
        rollback_generations_per_stem=1,
        dry_run=True,
    )

    assert receipt["state"] == "dry_run"
    assert receipt["candidate_count"] == 2
    assert all(generation[0].exists() for generation in generations)
    assert not (output / "checkpoint_retention_receipts").exists()


def test_incomplete_generation_refuses_retention_before_any_delete(
    tmp_path: Path,
) -> None:
    output = _private(tmp_path / "output")
    generations = [
        _generation(output, stem="checkpoint_latest", step=step, nonce=step)
        for step in range(1, 4)
    ]
    _point(
        output,
        stem="checkpoint_latest",
        generation=generations[-1][0],
        receipt=generations[-1][1],
        receipt_raw=generations[-1][2],
    )
    incomplete = (
        output
        / "checkpoint_generations"
        / f"checkpoint_latest-step-00000004-{'e' * 32}"
    )
    incomplete.mkdir(mode=0o500)

    with pytest.raises(UnifiedCheckpointError, match="JSON|unavailable|unreadable"):
        prune_checkpoint_generations(output, rollback_generations_per_stem=0)

    assert all(generation[0].exists() for generation in generations)
    assert incomplete.exists()


def test_symlink_generation_refuses_retention_without_touching_target(
    tmp_path: Path,
) -> None:
    output = _private(tmp_path / "output")
    generation = _generation(
        output,
        stem="checkpoint_latest",
        step=1,
        nonce=1,
    )
    _point(
        output,
        stem="checkpoint_latest",
        generation=generation[0],
        receipt=generation[1],
        receipt_raw=generation[2],
    )
    outside = _private(tmp_path / "outside")
    linked_name = f"checkpoint_latest-step-00000002-{'f' * 32}"
    (output / "checkpoint_generations" / linked_name).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(UnifiedCheckpointError, match="symlink"):
        prune_checkpoint_generations(output, rollback_generations_per_stem=0)

    assert outside.exists()
    assert generation[0].exists()


@pytest.mark.parametrize("mutable_name", ["bundle.safetensors", "complete.json"])
def test_mutable_generation_refuses_retention_before_any_delete(
    tmp_path: Path,
    mutable_name: str,
) -> None:
    output = _private(tmp_path / "output")
    generations = [
        _generation(output, stem="checkpoint_latest", step=step, nonce=step)
        for step in range(1, 3)
    ]
    _point(
        output,
        stem="checkpoint_latest",
        generation=generations[-1][0],
        receipt=generations[-1][1],
        receipt_raw=generations[-1][2],
    )
    os.chmod(generations[0][0], 0o700)
    os.chmod(generations[0][0] / mutable_name, 0o600)
    os.chmod(generations[0][0], 0o500)

    with pytest.raises(UnifiedCheckpointError, match="weights differ"):
        prune_checkpoint_generations(output, rollback_generations_per_stem=0)

    assert all(generation[0].exists() for generation in generations)
