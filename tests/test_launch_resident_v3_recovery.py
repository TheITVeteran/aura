"""Command and admission contracts for the resident-v3 recovery launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import launch_resident_v3_recovery as recovery


def _documents(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    destination = tmp_path / "resident_32b_v3_cp191" / "adapter"
    destination.mkdir(parents=True)
    source_checkpoint = "checkpoints/step-00000010-frozen"
    (destination / "latest.json").write_text(
        json.dumps({"checkpoint": source_checkpoint}),
        encoding="ascii",
    )
    config = {
        "model_path": str(model),
        "personality_adapter_path": "",
        "train_seed": 2026071901,
        "curriculum_depths": [1, 2, 4],
        "monotonicity_weight": 0.5,
        "max_steps": 540,
        "objective_options": {
            "depth_margin": 0.05,
            "diversity_weight": 0.25,
            "diversity_target_cos": 0.98,
        },
        "bridge": {"policy": "assistant_answer"},
        "holdout": {"per_cell": 1, "eval_samples": 8},
        "lora": {"rank": 8, "targets": ["o_proj", "v_proj"]},
        "optimizer": {"learning_rate": 0.0001},
    }
    dataset = {
        "families": ["khop", "boolean"],
        "task_depths": [2, 4, 8],
        "per_cell": 16,
    }
    spec = {
        "n_slots": 4,
        "branch_roles": ["constructive_solution", "counterexample_search"],
        "exchange_interval": 1,
        "alpha": 0.5,
        "alpha_schedule": "constant",
    }
    migration = {
        "source": {
            "training_config_document": config,
            "dataset_manifest_document": dataset,
            "execution_spec_document": spec,
        },
        "destination": {"root": str(destination)},
    }
    summary = {
        "source_checkpoint": source_checkpoint,
        "source_step": 10,
    }
    return migration, summary, destination


def test_calibration_command_changes_only_execution_memory_and_paths(tmp_path, monkeypatch):
    migration, summary, destination = _documents(tmp_path)
    migration_path = tmp_path / "migration.json"
    migration_path.write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        recovery,
        "_migration",
        lambda _path, **_kwargs: (migration, summary),
    )
    monkeypatch.setattr(recovery, "_validate_no_competing_model_process", lambda _model: None)

    launcher, target, paths = recovery.build_commands(
        migration_path,
        phase="calibration",
    )

    assert "--activation-checkpointing" in target
    assert "--resume-migration-evidence" in target
    assert "--resume" in target
    assert target[target.index("--max-steps") + 1] == "540"
    assert target[target.index("--max-minutes") + 1] == "0.001"
    assert target[target.index("--families") + 1] == "khop,boolean"
    assert target[target.index("--n-slots") + 1] == "4"
    assert target[target.index("--lora-targets") + 1] == "o_proj,v_proj"
    assert paths["run"] == destination.parent / "detached-calibration"
    assert launcher[-len(target) :] == target


def test_calibration_rejects_latest_pointer_drift(tmp_path, monkeypatch):
    migration, summary, destination = _documents(tmp_path)
    (destination / "latest.json").write_text(
        json.dumps({"checkpoint": "checkpoints/step-00000011-other"}),
        encoding="ascii",
    )
    monkeypatch.setattr(
        recovery,
        "_migration",
        lambda _path, **_kwargs: (migration, summary),
    )

    with pytest.raises(recovery.ResidentV3RecoveryError, match="source_checkpoint_changed"):
        recovery.build_commands(tmp_path / "migration.json", phase="calibration")


def test_resume_requires_positive_calibration_verdict(tmp_path, monkeypatch):
    migration, summary, _destination = _documents(tmp_path)
    monkeypatch.setattr(
        recovery,
        "_migration",
        lambda _path, **_kwargs: (migration, summary),
    )
    monkeypatch.setattr(recovery, "verify_calibration", lambda _path: {"passed": False})

    with pytest.raises(recovery.ResidentV3RecoveryError, match="calibration_not_admitted"):
        recovery.build_commands(tmp_path / "migration.json", phase="resume")
