"""Custody and crash-recovery contracts for immutable recurrence checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.unified_intrinsic_checkpoint import (
    CHECKPOINT_POINTER_SCHEMA,
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
    unpointed_checkpoint_inventory,
)


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


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
