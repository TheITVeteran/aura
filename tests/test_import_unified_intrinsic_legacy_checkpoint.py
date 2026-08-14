"""Custody contracts for historical portable recurrent-controller tissue."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from tools.import_unified_intrinsic_legacy_checkpoint import (  # noqa: E402
    import_legacy_checkpoint,
)
from tools.train_unified_intrinsic_recurrence import _canonical_sha256  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    resolve_checkpoint_generation,
)


def _legacy_checkpoint(root: Path) -> tuple[Path, Path, str, str]:
    root.mkdir(mode=0o700)
    weights_path = root / "checkpoint_best_trained.safetensors"
    mx.save_safetensors(
        str(weights_path),
        {
            "bundle.controller.weight": mx.arange(8, dtype=mx.float32),
            "optimizer.moment": mx.zeros((8,), dtype=mx.float32),
        },
    )
    checkpoint_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    identity_body = {
        "schema": "aura.unified_intrinsic_training.v1",
        "window_tissue_mode": "controller_only",
    }
    identity = {
        **identity_body,
        "identity_sha256": _canonical_sha256(identity_body),
    }
    body = {
        "schema": "aura.unified_intrinsic_training.v1",
        "step": 60,
        "optimization_phase": "answer_bridge",
        "history": [{"step": 60}],
        "training_state": {},
        "identity": identity,
        "checkpoint_sha256": checkpoint_sha256,
    }
    receipt = {**body, "receipt_sha256": _canonical_sha256(body)}
    receipt_path = root / "checkpoint_best_trained.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return (
        receipt_path,
        weights_path,
        checkpoint_sha256,
        identity["identity_sha256"],
    )


def test_legacy_controller_import_is_hash_pinned_and_immutable(tmp_path: Path) -> None:
    receipt, weights, checkpoint_sha256, identity_sha256 = _legacy_checkpoint(
        tmp_path / "legacy"
    )
    destination = tmp_path / "imported"
    result = import_legacy_checkpoint(
        receipt_path=receipt,
        weights_path=weights,
        output_dir=destination,
        stem="checkpoint_best_trained",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_identity_sha256=identity_sha256,
    )
    resolved = resolve_checkpoint_generation(
        destination,
        stem="checkpoint_best_trained",
        required=True,
    )
    assert resolved is not None
    assert result["bytes_preserved_exactly"] is True
    assert result["optimizer_bootstrap_authority"] is False
    assert resolved.receipt["checkpoint_sha256"] == checkpoint_sha256
    assert resolved.weights_path.stat().st_mode & 0o222 == 0
    assert resolved.generation_dir.stat().st_mode & 0o777 == 0o500
    assert (destination / "legacy_import_receipt.json").stat().st_mode & 0o222 == 0


def test_legacy_controller_import_rejects_an_unexpected_commitment(
    tmp_path: Path,
) -> None:
    receipt, weights, _checkpoint_sha256, identity_sha256 = _legacy_checkpoint(
        tmp_path / "legacy"
    )
    with pytest.raises(RuntimeError, match="commitment differs"):
        import_legacy_checkpoint(
            receipt_path=receipt,
            weights_path=weights,
            output_dir=tmp_path / "rejected",
            stem="checkpoint_best_trained",
            expected_checkpoint_sha256="0" * 64,
            expected_identity_sha256=identity_sha256,
        )


def test_legacy_import_cli_loads_from_a_direct_script_invocation() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/import_unified_intrinsic_legacy_checkpoint.py",
            "--help",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--expected-checkpoint-sha256" in result.stdout
