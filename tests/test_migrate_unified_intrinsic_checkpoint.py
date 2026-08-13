"""Exact source-repair migration contracts for unified recurrence."""

from __future__ import annotations

from pathlib import Path

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
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    CAMPAIGN_BINDING_SCHEMA,
    canonical_sha256,
)


def _campaign() -> dict:
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
        "training_profile_sha256": "a" * 64,
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


def test_target_identity_changes_only_the_explicit_source_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import migrate_unified_intrinsic_checkpoint as migration

    source_identity = {
        "schema": "aura.unified_intrinsic_training.v1",
        "source_sha256s": {"trainer.py": "a" * 64, "objective.py": "b" * 64},
        "campaign_binding": _campaign(),
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
    assert target["bootstrap"]["parent_step"] == 34
    assert target["identity_sha256"] == canonical_sha256(
        {key: value for key, value in target.items() if key != "identity_sha256"}
    )

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
