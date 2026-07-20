from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import launch_resident_recurrence_training as launch


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(launch, "REPO_ROOT", tmp_path)
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    wrapper = tmp_path / "tools/run_recurrence_training_envelope.py"
    trainer = tmp_path / "tools/recurrence_native_train_v2.py"
    detached = tmp_path / "tools/run_detached_step.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes(b"# wrapper\n")
    trainer.write_bytes(b"# trainer\n")
    detached.write_bytes(b"# detached\n")
    model = tmp_path / "model"
    model.mkdir()
    adapter = tmp_path / "proof/adapter"
    adapter.mkdir(parents=True)
    _write_json(
        adapter / "receipt.json",
        {
            "complete": False,
            "halt_reason": "wall_clock",
            "steps": 1,
            "objective_schema": "aura.recurrence_native_objective.v3",
        },
    )
    protocol = {
        "schema": launch.PROTOCOL_SCHEMA,
        "why_v3": "Unicode is valid here: recurrence → depth.",
        "model": {"path": str(model)},
        "training": {
            "adapter_id": "resident-test-v3",
            "output_dir": str(adapter),
            "objective": "v3",
            "train_seed": 2026071901,
            "families": ["khop", "boolean"],
            "task_depths": [2, 4],
            "per_cell": 2,
            "curriculum_depths": [1, 2, 4],
            "n_slots": 4,
            "branch_roles": ["constructive_solution", "counterexample_search"],
            "exchange_interval": 1,
            "alpha": 0.5,
            "alpha_schedule": "constant",
            "bridge_policy": "assistant_answer",
            "depth_margin": 0.05,
            "diversity_weight": 0.25,
            "diversity_target_cos": 0.98,
            "lora_rank": 8,
            "lora_targets": ["o_proj", "v_proj"],
            "learning_rate": 0.0001,
            "monotonicity_weight": 0.5,
            "holdout_per_cell": 1,
            "holdout_eval_samples": 8,
            "max_steps": 540,
            "hard_training_minutes_after_resume": 720,
        },
    }
    protocol_path = tmp_path / "proof/protocol.json"
    protocol_raw = _write_json(protocol_path, protocol)
    failure = {
        "schema": launch.FAILURE_SCHEMA,
        "process": {
            "detached_returncode": -9,
            "timed_out": False,
            "restart_count": 0,
            "containment_verified": True,
            "process_group_empty": True,
            "lineage_empty": True,
        },
        "kernel_evidence": {
            "termination_classification": "jetsam_largest_compressed_process",
            "compressed_process_mb": 83805,
        },
    }
    failure_path = tmp_path / "proof/failure.json"
    _write_json(failure_path, failure)
    amendment = {
        "schema": launch.AMENDMENT_SCHEMA,
        "parent_protocol": {
            "path": protocol_path.name,
            "sha256": hashlib.sha256(protocol_raw).hexdigest(),
            "size_bytes": len(protocol_raw),
        },
        "triggering_failure": {
            "path": failure_path.name,
            "required_classification": "jetsam_largest_compressed_process",
            "required_compressed_process_mb": 83805,
        },
        "resource_envelope": {
            "wrapper": str(wrapper.relative_to(tmp_path)),
            "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "trainer": str(trainer.relative_to(tmp_path)),
            "trainer_sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
            "memory_limit_gb": 40,
            "cache_limit_gb": 2,
            "wired_limit_gb": 48,
            "cache_cleared_before_model_load": True,
            "envelope_out": str(adapter / "envelope.json"),
        },
        "resume": {
            "run_dir": str(tmp_path / "proof/run"),
            "name": "resident-test",
            "timeout_seconds": 46800,
            "expected_resume_step": 1,
            "checkpoint_every_steps": 5,
            "log_every_steps": 5,
        },
    }
    amendment_path = tmp_path / "proof/amendment.json"
    _write_json(amendment_path, amendment)
    return protocol_path, amendment_path, wrapper, trainer


def test_launch_command_is_enveloped_and_preserves_v3_contract(tmp_path, monkeypatch):
    protocol_path, amendment_path, wrapper, trainer = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, "_validate_no_competing_model_process", lambda _model: None)

    launcher, target = launch.build_launch_command(protocol_path, amendment_path)

    assert target[:2] == [str(tmp_path / ".venv/bin/python"), str(wrapper)]
    assert target[target.index("--trainer") + 1] == str(trainer)
    assert target[target.index("--memory-limit-gb") + 1] == "40"
    assert target[target.index("--cache-limit-gb") + 1] == "2"
    assert target[target.index("--wired-limit-gb") + 1] == "48"
    assert target[target.index("--objective") + 1] == "v3"
    assert target[target.index("--checkpoint-every") + 1] == "5"
    assert target[-1] == "--resume"
    assert launcher[0] == str(tmp_path / ".venv/bin/python")
    assert launcher[-len(target) :] == target


def test_launch_rejects_changed_resource_wrapper(tmp_path, monkeypatch):
    protocol_path, amendment_path, wrapper, _trainer = _fixture(tmp_path, monkeypatch)
    wrapper.write_bytes(b"# changed after amendment\n")

    with pytest.raises(
        launch.ResidentRecurrenceLaunchError,
        match="resource_envelope_source_mismatch",
    ):
        launch._validate_contract(protocol_path, amendment_path)


def test_launch_rejects_reused_run_directory(tmp_path, monkeypatch):
    protocol_path, amendment_path, _wrapper, _trainer = _fixture(tmp_path, monkeypatch)
    (tmp_path / "proof/run").mkdir()

    with pytest.raises(
        launch.ResidentRecurrenceLaunchError,
        match="run_dir_already_exists",
    ):
        launch._validate_contract(protocol_path, amendment_path)


def test_launch_rejects_unpreregistered_memory_limits(tmp_path, monkeypatch):
    protocol_path, amendment_path, _wrapper, _trainer = _fixture(tmp_path, monkeypatch)
    amendment = json.loads(amendment_path.read_text())
    amendment["resource_envelope"]["wired_limit_gb"] = 52
    _write_json(amendment_path, amendment)

    with pytest.raises(
        launch.ResidentRecurrenceLaunchError,
        match="resource_envelope_limits_not_preregistered",
    ):
        launch._validate_contract(protocol_path, amendment_path)
