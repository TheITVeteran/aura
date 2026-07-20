from __future__ import annotations

import hashlib
import json
import shutil
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
    sentinel_program = tmp_path / "tools/memory_sentinel.py"
    stage_guard = tmp_path / "core/runtime/resource_stage_guard.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes(b"# wrapper\n")
    trainer.write_bytes(b"# trainer\n")
    detached.write_bytes(b"# detached\n")
    sentinel_program.write_bytes(b"# sentinel\n")
    stage_guard.parent.mkdir(parents=True)
    stage_guard.write_bytes(b"# stage guard\n")
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
    failed_receipt = {
        "returncode": -9,
        "timed_out": False,
        "restart_count": 0,
        "containment_verified": True,
        "process_group_empty": True,
        "lineage_empty": True,
    }
    failed_receipt_path = tmp_path / "proof/failed_receipt.json"
    failed_receipt_raw = _write_json(failed_receipt_path, failed_receipt)
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
        "detached_receipt": {
            "path": failed_receipt_path.name,
            "sha256": hashlib.sha256(failed_receipt_raw).hexdigest(),
            "size_bytes": len(failed_receipt_raw),
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
        "sentinel": {
            "program": "tools/memory_sentinel.py",
            "program_sha256": hashlib.sha256(
                sentinel_program.read_bytes()
            ).hexdigest(),
            "stage_guard_source": "core/runtime/resource_stage_guard.py",
            "stage_guard_source_sha256": hashlib.sha256(
                stage_guard.read_bytes()
            ).hexdigest(),
            "lethal_mb": 59392,
            "startup_lethal_mb": 73728,
            "interval_seconds": 2.0,
            "required_for_each_phase": True,
            "tombstone_means_phase_failure": True,
            "partial_ring": str(adapter / "partial-ring.jsonl"),
            "resume_ring": str(adapter / "resume-ring.jsonl"),
            "partial_stage_path": str(adapter / "partial-stage.json"),
            "resume_stage_path": str(adapter / "resume-stage.json"),
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


def test_partial_phase_uses_same_envelope_without_resume(tmp_path, monkeypatch):
    protocol_path, amendment_path, wrapper, _trainer = _fixture(tmp_path, monkeypatch)
    protocol = json.loads(protocol_path.read_text())
    adapter = Path(protocol["training"]["output_dir"])
    shutil.rmtree(adapter)
    amendment = json.loads(amendment_path.read_text())
    amendment["partial"] = {
        "run_dir": str(tmp_path / "proof/partial-run"),
        "name": "resident-test-partial",
        "timeout_seconds": 1800,
        "max_minutes": 0.001,
        "checkpoint_every_steps": 5,
        "log_every_steps": 5,
    }
    _write_json(amendment_path, amendment)
    monkeypatch.setattr(launch, "_validate_no_competing_model_process", lambda _model: None)

    launcher, target = launch.build_launch_command(
        protocol_path,
        amendment_path,
        phase="partial",
    )

    assert target[1] == str(wrapper)
    assert target[target.index("--max-minutes") + 1] == "0.001"
    assert "--resume" not in target
    assert launcher[launcher.index("--run-dir") + 1].endswith("partial-run")


def test_sentinel_launcher_is_phase_bound(tmp_path, monkeypatch):
    _protocol_path, amendment_path, _wrapper, _trainer = _fixture(tmp_path, monkeypatch)
    amendment = json.loads(amendment_path.read_text())

    command = launch._sentinel_launch_command(
        amendment,
        phase="resume",
        trainer_pid=4242,
    )

    assert command[command.index("--pid") + 1] == "4242"
    assert command[command.index("--lethal-mb") + 1] == "59392.0"
    assert command[command.index("--startup-lethal-mb") + 1] == "73728.0"
    assert command[command.index("--steady-marker") + 1].endswith(
        "resume-stage.json"
    )
    assert command[command.index("--interval") + 1] == "2.0"
    assert command[command.index("--ring") + 1].endswith("resume-ring.jsonl")
    assert command[command.index("--run-dir") + 1].endswith("sentinel-resume")


def test_target_waits_for_the_same_phase_resource_marker(tmp_path, monkeypatch):
    protocol_path, amendment_path, _wrapper, _trainer = _fixture(
        tmp_path, monkeypatch
    )

    monkeypatch.setattr(
        launch,
        "_validate_no_competing_model_process",
        lambda _model: None,
    )
    _launcher, target = launch.build_launch_command(protocol_path, amendment_path)

    assert target[target.index("--resource-stage-path") + 1].endswith(
        "resume-stage.json"
    )
    assert target[target.index("--resource-startup-lethal-mb") + 1] == "73728.0"
    assert target[target.index("--resource-steady-lethal-mb") + 1] == "59392.0"


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
