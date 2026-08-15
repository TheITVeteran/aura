"""Exact source-repair migration contracts for unified recurrence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from tools.migrate_unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedIntrinsicMigrationError,
    _controller_sha256,
    _target_identity,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    adopt_source_migration_identity,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    CAMPAIGN_BINDING_SCHEMA,
    canonical_bytes,
    canonical_sha256,
)


def _campaign(*, training_profile_sha256: str = "a" * 64) -> dict:
    body = {
        "schema": CAMPAIGN_BINDING_SCHEMA,
        "campaign_id": "cp-test",
        "campaign_config_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "source_tree": "3" * 40,
        "source_manifest_sha256": "4" * 64,
        "model_manifest_sha256": "5" * 64,
        "runtime_identity_sha256": "6" * 64,
        "dataset_identity_sha256": "7" * 64,
        "tokenizer_identity_sha256": "8" * 64,
        "tokenized_dataset_identity_sha256": "9" * 64,
        "training_profile_sha256": training_profile_sha256,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def test_controller_digest_is_reconstructed_from_checkpoint_tensors(
    tmp_path: Path,
) -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            initialization_seed=73,
        )
    )
    mx.eval(controller.parameters())
    path = tmp_path / "bundle.safetensors"
    mx.save_safetensors(
        str(path),
        {
            f"bundle.controller.{name}": value
            for name, value in tree_flatten(controller.parameters())
        },
    )
    identity = {
        "controller_rank": 4,
        "depth_basis_size": 4,
        "init_seed": 73,
        "literal_observation_contract": {"digit_token_ids": []},
        "opcode_observation_contract": {"patterns": [], "contexts": []},
    }
    assert _controller_sha256(path, identity) == controller.parameter_sha256()


def test_controller_digest_reconstructs_semantic_frontier_topology(
    tmp_path: Path,
) -> None:
    patterns = tuple((opcode, (100 + opcode,)) for opcode in range(9, 16))
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            state_slots=11,
            initialization_seed=74,
            numeric_observation_max_value=960,
            frontier_family_token_patterns=patterns,
        )
    )
    mx.eval(controller.parameters())
    path = tmp_path / "semantic.safetensors"
    mx.save_safetensors(
        str(path),
        {
            f"bundle.controller.{name}": value
            for name, value in tree_flatten(controller.parameters())
        },
    )
    identity = {
        "controller_rank": 4,
        "state_slots": 11,
        "depth_basis_size": 4,
        "init_seed": 74,
        "literal_observation_contract": {"digit_token_ids": []},
        "numeric_observation_contract": {"max_value": 960},
        "opcode_observation_contract": {"patterns": [], "contexts": []},
        "frontier_family_observation_contract": {
            "patterns": [
                {"opcode": opcode, "token_ids": list(tokens)}
                for opcode, tokens in patterns
            ]
        },
    }
    assert _controller_sha256(path, identity) == controller.parameter_sha256()


def test_target_identity_changes_only_the_explicit_source_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import migrate_unified_intrinsic_checkpoint as migration

    training_profile_sha256 = canonical_sha256(
        {"profile": "recovery", "training": {}, "training_args": []}
    )
    source_identity = {
        "schema": "aura.unified_intrinsic_training.v1",
        "source_sha256s": {"trainer.py": "a" * 64, "objective.py": "b" * 64},
        "campaign_binding": _campaign(
            training_profile_sha256=training_profile_sha256
        ),
        "bootstrap": {"schema": "old"},
        "initial_controller_sha256": "c" * 64,
    }
    source_identity["identity_sha256"] = canonical_sha256(source_identity)
    monkeypatch.setattr(
        migration,
        "TRAINING_SOURCE_FILES",
        ("trainer.py", "objective.py"),
    )
    monkeypatch.setattr(
        migration,
        "_source_sha256s",
        lambda _root: {"trainer.py": "d" * 64, "objective.py": "b" * 64},
    )
    config = {
        "campaign_id": "cp-test",
        "config_sha256": "1" * 64,
        "profile": "recovery",
        "source": {
            "git": {"root": "/source", "commit": "2" * 40, "tree": "3" * 40},
            "manifest": {"manifest_sha256": "4" * 64},
        },
        "model": {"manifest_sha256": "5" * 64},
        "runtime": {"identity_sha256": "6" * 64},
        "dataset": {"identity_sha256": "7" * 64},
        "tokenizer": {"identity_sha256": "8" * 64},
        "tokenized_dataset": {"identity_sha256": "9" * 64},
        "training": {},
        "training_args": [],
        "bootstrap": {
            "stem": "checkpoint_latest",
            "parent_step": 34,
            "parent_checkpoint_sha256": "e" * 64,
            "parent_receipt_sha256": "f" * 64,
            "parent_identity_sha256": "0" * 64,
        },
    }
    target, differences = _target_identity(
        source_identity,
        target_config=config,
        controller_sha256="1" * 64,
        allowed_source_changes=frozenset({"trainer.py"}),
    )
    assert differences == {"trainer.py": {"source": "a" * 64, "target": "d" * 64}}
    assert target["source_sha256s"]["objective.py"] == "b" * 64
    assert target["initial_controller_sha256"] == "c" * 64
    assert target["source_migration_controller_sha256"] == "1" * 64
    assert target["bootstrap"] == {"schema": "old"}
    assert target["identity_sha256"] == canonical_sha256(
        {key: value for key, value in target.items() if key != "identity_sha256"}
    )


def test_target_identity_preserves_fresh_scientific_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import migrate_unified_intrinsic_checkpoint as migration

    training_profile_sha256 = canonical_sha256(
        {
            "profile": "process_semantic_transition_canary",
            "training": {},
            "training_args": [],
        }
    )
    source_identity = {
        "schema": "aura.unified_intrinsic_training.v1",
        "source_sha256s": {"trainer.py": "a" * 64},
        "campaign_binding": _campaign(
            training_profile_sha256=training_profile_sha256
        ),
        "bootstrap": None,
        "initial_controller_sha256": "c" * 64,
    }
    source_identity["identity_sha256"] = canonical_sha256(source_identity)
    monkeypatch.setattr(migration, "TRAINING_SOURCE_FILES", ("trainer.py",))
    monkeypatch.setattr(
        migration,
        "_source_sha256s",
        lambda _root: {"trainer.py": "d" * 64},
    )
    config = {
        "campaign_id": "cp-test",
        "config_sha256": "1" * 64,
        "profile": "process_semantic_transition_canary",
        "source": {
            "git": {"root": "/source", "commit": "2" * 40, "tree": "3" * 40},
            "manifest": {"manifest_sha256": "4" * 64},
        },
        "model": {"manifest_sha256": "5" * 64},
        "runtime": {"identity_sha256": "6" * 64},
        "dataset": {"identity_sha256": "7" * 64},
        "tokenizer": {"identity_sha256": "8" * 64},
        "tokenized_dataset": {"identity_sha256": "9" * 64},
        "training": {},
        "training_args": [],
        "bootstrap": {"stem": "checkpoint_latest"},
    }
    target, _differences = _target_identity(
        source_identity,
        target_config=config,
        controller_sha256="1" * 64,
        allowed_source_changes=frozenset({"trainer.py"}),
    )
    assert target["bootstrap"] is None
    assert target["initial_controller_sha256"] == "c" * 64

    with pytest.raises(
        UnifiedIntrinsicMigrationError,
        match="migration_source_change_set_differs",
    ):
        _target_identity(
            source_identity,
            target_config=config,
            controller_sha256="1" * 64,
            allowed_source_changes=frozenset({"objective.py"}),
        )


def test_target_identity_rejects_a_different_recovery_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import migrate_unified_intrinsic_checkpoint as migration

    source_identity = {
        "schema": "aura.unified_intrinsic_training.v1",
        "source_sha256s": {"trainer.py": "a" * 64},
        "campaign_binding": _campaign(),
        "bootstrap": {"schema": "original"},
        "initial_controller_sha256": "c" * 64,
    }
    source_identity["identity_sha256"] = canonical_sha256(source_identity)
    monkeypatch.setattr(migration, "TRAINING_SOURCE_FILES", ("trainer.py",))
    monkeypatch.setattr(
        migration,
        "_source_sha256s",
        lambda _root: {"trainer.py": "d" * 64},
    )
    config = {
        "campaign_id": "cp-test",
        "config_sha256": "1" * 64,
        "profile": "recovery",
        "source": {
            "git": {"root": "/source", "commit": "2" * 40, "tree": "3" * 40},
            "manifest": {"manifest_sha256": "4" * 64},
        },
        "model": {"manifest_sha256": "5" * 64},
        "runtime": {"identity_sha256": "6" * 64},
        "dataset": {"identity_sha256": "7" * 64},
        "tokenizer": {"identity_sha256": "8" * 64},
        "tokenized_dataset": {"identity_sha256": "9" * 64},
        "training": {"max_steps": 36},
        "training_args": ["--max-steps", "36"],
        "bootstrap": {"stem": "checkpoint_latest"},
    }
    with pytest.raises(
        UnifiedIntrinsicMigrationError,
        match="migration_training_profile_changed",
    ):
        _target_identity(
            source_identity,
            target_config=config,
            controller_sha256="1" * 64,
            allowed_source_changes=frozenset({"trainer.py"}),
        )


@pytest.mark.parametrize("original_bootstrap", ({"schema": "original"}, None))
def test_resume_adopts_only_a_fully_bound_source_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original_bootstrap: dict[str, str] | None,
) -> None:
    from tools import unified_intrinsic_checkpoint as checkpoint

    campaign = tmp_path / "campaign"
    output = campaign / "training-output"
    source_output = tmp_path / "source-output"
    output.mkdir(parents=True)
    source_output.mkdir()
    binding = _campaign()
    computed = {
        "schema": "aura.unified_intrinsic_training.v1",
        "campaign_binding": binding,
        "initial_controller_sha256": "1" * 64,
        "bootstrap": {"schema": "resume"},
        "phase_schedule": {
            "mode": "bootstrap_process_acquisition_only",
            "bootstrap_required": True,
            "state_transition": {"start": 0, "stop": 384},
        },
        "spec": {"train_depths": (1, 3, 5)},
        "source_sha256s": {"trainer.py": "2" * 64},
    }
    stored = {
        **computed,
        "initial_controller_sha256": "3" * 64,
        "bootstrap": original_bootstrap,
        "phase_schedule": {
            "mode": "process_acquisition_only",
            "bootstrap_required": False,
            "state_transition": {"start": 0, "stop": 384},
        },
        "spec": {"train_depths": [1, 3, 5]},
        "source_migration_controller_sha256": "1" * 64,
    }
    stored["identity_sha256"] = canonical_sha256(stored)
    target_receipt = {
        "step": 8,
        "checkpoint_sha256": "4" * 64,
        "receipt_sha256": "5" * 64,
        "identity": stored,
    }
    source_receipt = {
        "step": 8,
        "checkpoint_sha256": "4" * 64,
        "receipt_sha256": "6" * 64,
        "identity": {"identity_sha256": "7" * 64},
    }
    target = SimpleNamespace(
        receipt=target_receipt,
        generation_dir=Path("checkpoint_latest-step-00000008-target"),
    )
    source = SimpleNamespace(
        receipt=source_receipt,
        generation_dir=Path("checkpoint_latest-step-00000008-source"),
    )

    def resolve(path: Path, **_kwargs: object) -> SimpleNamespace:
        return target if Path(path).resolve() == output.resolve() else source

    monkeypatch.setattr(checkpoint, "resolve_checkpoint_generation", resolve)
    monkeypatch.setattr(
        checkpoint,
        "_stable_file_identity",
        lambda *_args, **_kwargs: {"sha256": "8" * 64},
    )
    body = {
        "schema": checkpoint.SOURCE_MIGRATION_SCHEMA,
        "state": "complete",
        "source": {
            "output": str(source_output),
            "generation": source.generation_dir.name,
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "6" * 64,
            "identity_sha256": "7" * 64,
        },
        "destination": {
            "campaign_id": binding["campaign_id"],
            "config_sha256": binding["campaign_config_sha256"],
            "generation": target.generation_dir.name,
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "5" * 64,
            "identity_sha256": stored["identity_sha256"],
        },
        "payload_byte_identical": True,
        "optimizer_and_bundle_bytes_preserved": True,
        "history_preserved": True,
        "training_state_preserved": True,
        "scientific_initialization_preserved": True,
        "training_profile_preserved": True,
        "migration_tool_sha256": "8" * 64,
    }
    migration = {**body, "migration_sha256": canonical_sha256(body)}
    (campaign / "checkpoint-source-migration.json").write_bytes(
        canonical_bytes(migration) + b"\n"
    )
    assert adopt_source_migration_identity(output, computed) == stored

    migration["training_profile_preserved"] = False
    material = {key: value for key, value in migration.items() if key != "migration_sha256"}
    migration["migration_sha256"] = canonical_sha256(material)
    (campaign / "checkpoint-source-migration.json").write_bytes(
        canonical_bytes(migration) + b"\n"
    )
    with pytest.raises(UnifiedCheckpointError, match="source migration differs"):
        adopt_source_migration_identity(output, computed)


def test_resume_rejects_a_changed_scientific_phase_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import unified_intrinsic_checkpoint as checkpoint

    campaign = tmp_path / "campaign"
    output = campaign / "training-output"
    source_output = tmp_path / "source-output"
    output.mkdir(parents=True)
    source_output.mkdir()
    binding = _campaign()
    computed = {
        "schema": "aura.unified_intrinsic_training.v1",
        "campaign_binding": binding,
        "initial_controller_sha256": "1" * 64,
        "bootstrap": {"schema": "resume"},
        "phase_schedule": {
            "mode": "bootstrap_process_acquisition_only",
            "bootstrap_required": True,
            "state_transition": {"start": 0, "stop": 383},
        },
        "source_sha256s": {"trainer.py": "2" * 64},
    }
    stored = {
        **computed,
        "initial_controller_sha256": "3" * 64,
        "bootstrap": None,
        "phase_schedule": {
            "mode": "process_acquisition_only",
            "bootstrap_required": False,
            "state_transition": {"start": 0, "stop": 384},
        },
        "source_migration_controller_sha256": "1" * 64,
    }
    stored["identity_sha256"] = canonical_sha256(stored)
    target = SimpleNamespace(
        receipt={
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "5" * 64,
            "identity": stored,
        },
        generation_dir=Path("checkpoint_latest-step-00000008-target"),
    )
    source = SimpleNamespace(
        receipt={
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "6" * 64,
            "identity": {"identity_sha256": "7" * 64},
        },
        generation_dir=Path("checkpoint_latest-step-00000008-source"),
    )

    monkeypatch.setattr(
        checkpoint,
        "resolve_checkpoint_generation",
        lambda path, **_kwargs: (
            target if Path(path).resolve() == output.resolve() else source
        ),
    )
    monkeypatch.setattr(
        checkpoint,
        "_stable_file_identity",
        lambda *_args, **_kwargs: {"sha256": "8" * 64},
    )
    body = {
        "schema": checkpoint.SOURCE_MIGRATION_SCHEMA,
        "state": "complete",
        "source": {
            "output": str(source_output),
            "generation": source.generation_dir.name,
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "6" * 64,
            "identity_sha256": "7" * 64,
        },
        "destination": {
            "campaign_id": binding["campaign_id"],
            "config_sha256": binding["campaign_config_sha256"],
            "generation": target.generation_dir.name,
            "step": 8,
            "checkpoint_sha256": "4" * 64,
            "receipt_sha256": "5" * 64,
            "identity_sha256": stored["identity_sha256"],
        },
        "payload_byte_identical": True,
        "optimizer_and_bundle_bytes_preserved": True,
        "history_preserved": True,
        "training_state_preserved": True,
        "scientific_initialization_preserved": True,
        "training_profile_preserved": True,
        "migration_tool_sha256": "8" * 64,
    }
    (campaign / "checkpoint-source-migration.json").write_bytes(
        canonical_bytes({**body, "migration_sha256": canonical_sha256(body)}) + b"\n"
    )

    with pytest.raises(
        UnifiedCheckpointError,
        match="source migration phase schedule differs",
    ):
        adopt_source_migration_identity(output, computed)
