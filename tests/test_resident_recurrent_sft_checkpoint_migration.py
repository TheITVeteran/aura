from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from core.learning import resident_recurrent_sft_checkpoint_migration as migration
from core.learning.resident_recurrent_sft_bootstrap_authority import SAMPLER_NAME
from core.learning.resident_recurrent_sft_bootstrap_state import (
    BINDING_ROLES,
    inspect_checkpoint,
    load_checkpoint,
    order_sha256,
    save_checkpoint,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _bindings(prefix: str) -> dict[str, str]:
    return {role: _sha(f"{prefix}:{role}") for role in BINDING_ROLES}


def _state(
    bindings: dict[str, str], *, sequence: int = 3, step: int = 2
) -> dict[str, Any]:
    order = [2, 0, 1]
    losses = [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}]
    return {
        **bindings,
        "checkpoint_sequence": sequence,
        "step": step,
        "optimizer_updates": step,
        "epoch": 0,
        "cursor": step,
        "order": order,
        "order_sha256": order_sha256(order=order, seed=17, epoch=0),
        "sampler": SAMPLER_NAME,
        "seed": 17,
        "train_example_count": 3,
        "validation_example_count": 2,
        "elapsed_training_s": 12.5,
        "invocation_count": 1,
        "sample_history_sha256": _sha("history"),
        "initial_adapter_sha256": _sha("initial"),
        "adapter_topology_sha256": _sha("topology"),
        "loss_trail": losses[:step],
        "validation_trail": [],
        "pending_losses": [],
        "baseline_validation": {"examples": 2, "mean_loss": 2.5},
        "last_step_committed": True,
        "terminal": False,
        "halt_reason": None,
    }


def _seed_checkpoint(root: Path, bindings: dict[str, str]) -> None:
    for sequence, step in ((1, 0), (2, 1), (3, 2)):
        save_checkpoint(
            root,
            adapter_tensors={"adapter.weight": mx.array([[1.0, 2.0]])},
            optimizer_tensors={"state.m": mx.array([0.25, 0.5])},
            state=_state(bindings, sequence=sequence, step=step),
        )


def _authority(root: Path, *, campaign: str, trainer_sha: str) -> dict[str, Any]:
    stat = root.stat()
    trust_path = root.parent / f"{campaign}-trust.json"
    trust_document = {
        "schema": "test-trust.v1",
        "campaign_id": campaign,
        "source": {"commit": trainer_sha},
        "policy_sha256": _sha(f"policy:{campaign}"),
        "training_only": True,
        "gain_claim_allowed": False,
    }
    trust_payload = json.dumps(
        trust_document, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    trust_path.write_bytes(trust_payload)
    return {
        "authority_sha256": _sha(campaign),
        "campaign_id": campaign,
        "campaign_scope": "full_bootstrap",
        "artifact_root": root.name,
        "artifact_root_identity": {"st_dev": stat.st_dev, "st_ino": stat.st_ino},
        "dataset": {"dataset_sha256": _sha("dataset")},
        "model": {
            "base_checkpoint": {"fingerprint": _sha("model")},
            "behavior_bundle": {"bundle_sha256": _sha("behavior")},
            "personality_bundle": {"identity_sha256": _sha("personality")},
        },
        "tokenizer": {
            "identity_sha256": _sha(f"tokenizer:{campaign}"),
            "artifact_sha256": _sha("tokenizer-artifact"),
            "runtime_sha256": _sha("tokenizer-runtime"),
        },
        "execution_spec": {"semantic_sha256": _sha("spec")},
        "trainer": {"objective": "same", "max_steps": 4},
        "runtime": {"identity_sha256": _sha("runtime")},
        "trust_policy": {
            "path": trust_path.name,
            "sha256": hashlib.sha256(trust_payload).hexdigest(),
            "semantic_sha256": hashlib.sha256(trust_payload).hexdigest(),
        },
        "sources": {
            "trainer": {
                "path": "tools/train.py",
                "sha256": trainer_sha,
                "size_bytes": 100,
            },
            "objective": {
                "path": "core/objective.py",
                "sha256": _sha("objective"),
                "size_bytes": 200,
            },
        },
    }


def _write_authority(path: Path, authority: dict[str, Any]) -> None:
    path.write_bytes(
        json.dumps(authority, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def test_migration_preserves_exact_training_state_and_rebinds_source(monkeypatch, tmp_path):
    source_root = tmp_path / "source-run"
    destination_root = tmp_path / "destination-run"
    source_root.mkdir()
    destination_root.mkdir()
    source_bindings = _bindings("source")
    destination_bindings = _bindings("destination")
    _seed_checkpoint(source_root, source_bindings)
    source_authority = _authority(
        source_root, campaign="source-campaign", trainer_sha=_sha("trainer-old")
    )
    destination_authority = _authority(
        destination_root,
        campaign="destination-campaign",
        trainer_sha=_sha("trainer-fixed"),
    )
    source_authority_path = tmp_path / "source-authority.json"
    destination_authority_path = tmp_path / "destination-authority.json"
    _write_authority(source_authority_path, source_authority)
    _write_authority(destination_authority_path, destination_authority)

    monkeypatch.setattr(migration, "validate_authority", lambda value, **_kwargs: dict(value))
    def authority_bindings(authority):
        assert "_artifact_binding" not in authority
        return (
            source_bindings
            if authority["campaign_id"] == "source-campaign"
            else destination_bindings
        )

    monkeypatch.setattr(migration, "authority_state_bindings", authority_bindings)

    receipt = migration.migrate_checkpoint(
        source_repo_root=tmp_path,
        source_authority_path=source_authority_path,
        destination_repo_root=tmp_path,
        destination_authority_path=destination_authority_path,
    )
    verified = migration.verify_migration(
        destination_root / "checkpoint-migration.json",
        destination_repo_root=tmp_path,
        destination_authority=destination_authority,
    )
    source_loaded = load_checkpoint(source_root, expected_bindings=source_bindings)
    destination_loaded = load_checkpoint(
        destination_root, expected_bindings=destination_bindings
    )

    assert receipt["changed_source_roles"] == ["trainer"]
    assert verified["migration_sha256"] == receipt["migration_sha256"]
    assert receipt["preservation"] == {
        "adapter_state_reset": False,
        "optimizer_state_reset": False,
        "sample_cursor_reset": False,
        "loss_or_validation_history_reset": False,
    }
    assert {
        key: value
        for key, value in source_loaded.state.items()
        if key not in BINDING_ROLES
    } == {
        key: value
        for key, value in destination_loaded.state.items()
        if key not in BINDING_ROLES
    }
    assert all(
        bool(mx.array_equal(source_loaded.adapter_tensors[key], value))
        for key, value in destination_loaded.adapter_tensors.items()
    )
    assert all(
        bool(mx.array_equal(source_loaded.optimizer_tensors[key], value))
        for key, value in destination_loaded.optimizer_tensors.items()
    )
    assert inspect_checkpoint(
        destination_root, expected_bindings=destination_bindings
    ).state["step"] == 2

    source_optimizer = source_loaded.checkpoint_dir / "optimizer.safetensors"
    source_optimizer.write_bytes(source_optimizer.read_bytes() + b"drift")
    with pytest.raises(
        migration.ResidentSFTCheckpointMigrationError,
        match="source_binding_drift",
    ):
        migration.verify_migration(
            destination_root / "checkpoint-migration.json",
            destination_repo_root=tmp_path,
            destination_authority=destination_authority,
        )


def test_migration_refuses_scientific_config_change(monkeypatch, tmp_path):
    source_root = tmp_path / "source-run"
    destination_root = tmp_path / "destination-run"
    source_root.mkdir()
    destination_root.mkdir()
    source_bindings = _bindings("source")
    _seed_checkpoint(source_root, source_bindings)
    source = _authority(source_root, campaign="source", trainer_sha=_sha("old"))
    destination = _authority(
        destination_root, campaign="destination", trainer_sha=_sha("fixed")
    )
    destination["trainer"]["max_steps"] = 5
    source_path = tmp_path / "source.json"
    destination_path = tmp_path / "destination.json"
    _write_authority(source_path, source)
    _write_authority(destination_path, destination)
    monkeypatch.setattr(migration, "validate_authority", lambda value, **_kwargs: dict(value))

    with pytest.raises(
        migration.ResidentSFTCheckpointMigrationError,
        match="scientific_identity_changed",
    ):
        migration.migrate_checkpoint(
            source_repo_root=tmp_path,
            source_authority_path=source_path,
            destination_repo_root=tmp_path,
            destination_authority_path=destination_path,
        )
