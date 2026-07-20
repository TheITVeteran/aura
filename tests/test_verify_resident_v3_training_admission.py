from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from core.runtime.resource_stage_guard import (
    publish_armed_ack,
    publish_compute_lease_ack,
    publish_compute_lease_request,
    publish_ready_marker,
)
from tools import launch_resident_recurrence_training as launch
from tools import verify_resident_v3_training_admission as admission


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _checkpoint_fixture(
    adapter: Path,
    *,
    complete_run: bool,
) -> tuple[dict, dict]:
    step = 5 if complete_run else 1
    checkpoint = adapter / f"checkpoints/step-{step:08d}-test"
    checkpoint.mkdir(parents=True)
    adapter_tensor = checkpoint / "adapter.safetensors"
    optimizer_tensor = checkpoint / "optimizer.safetensors"
    adapter_tensor.write_bytes(b"adapter")
    optimizer_tensor.write_bytes(b"optimizer")
    committed_loss = [{"step": 5, "mean_loss": 1.0}] if complete_run else []
    pending_losses = [] if complete_run else [1.25]
    pending_cosines = [] if complete_run else [0.5]
    durable_holdout = (
        [{"step": 5, "mean_loss": 1.5, "examples": 8, "depth": 4}] if complete_run else []
    )
    checkpoint_complete = {
        "schema": admission.TRAINING_CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint.name,
        "step": step,
        "epoch": 0,
        "cursor": step,
        "order": list(range(5)),
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "execution_spec_sha256": "c" * 64,
        "elapsed_training_s": 10.0,
        "invocation_count": 1 if not complete_run else 2,
        "loss_trail": committed_loss,
        "pending_window_losses": pending_losses,
        "pending_window_cosines": pending_cosines,
        "holdout_trail": durable_holdout,
        "holdout_eval_count": len(durable_holdout),
        "sampler": "sha256_stateless_epoch_permutation.v1",
        "stochastic_state": "none_all_keys_explicit",
        "adapter": {
            "path": adapter_tensor.name,
            "sha256": hashlib.sha256(adapter_tensor.read_bytes()).hexdigest(),
            "size_bytes": adapter_tensor.stat().st_size,
        },
        "optimizer": {
            "path": optimizer_tensor.name,
            "sha256": hashlib.sha256(optimizer_tensor.read_bytes()).hexdigest(),
            "size_bytes": optimizer_tensor.stat().st_size,
        },
    }
    complete_raw = _write_json(checkpoint / "complete.json", checkpoint_complete)
    _write_json(
        adapter / "latest.json",
        {
            "schema": "aura.recurrence_native_checkpoint_pointer.v1",
            "checkpoint": f"checkpoints/{checkpoint.name}",
            "complete_sha256": hashlib.sha256(complete_raw).hexdigest(),
        },
    )
    receipt = {
        "complete": complete_run,
        "halt_reason": "max_steps" if complete_run else "wall_clock",
        "steps": step,
        "final_checkpoint": checkpoint.name,
        "objective_schema": "aura.recurrence_native_objective.v3",
        "loss_trail": (
            committed_loss
            if complete_run
            else [
                {
                    "step": 1,
                    "mean_loss": 1.25,
                    "window_steps": 1,
                    "partial_window": True,
                    "pairwise_cos_mean": 0.5,
                }
            ]
        ),
        "holdout_trail": (
            durable_holdout
            if complete_run
            else [{"step": 1, "mean_loss": 2.0, "examples": 8, "depth": 4}]
        ),
    }
    _write_json(adapter / "receipt.json", receipt)
    return checkpoint_complete, receipt


def _receipt(*, complete: bool, steps: int, halt_reason: str):
    return {
        "complete": complete,
        "steps": steps,
        "halt_reason": halt_reason,
        "loss_trail": [
            {"step": 100, "mean_loss": 2.0},
            {"step": steps, "mean_loss": 1.0},
        ],
        "holdout_trail": [
            {"step": 100, "mean_loss": 2.0},
            {"step": steps, "mean_loss": 1.5},
        ],
    }


def test_complete_and_bounded_partial_states_are_distinct():
    complete = admission.evaluate_training_state(
        _receipt(complete=True, steps=540, halt_reason="max_steps"),
        {"max_steps": 540},
    )
    partial = admission.evaluate_training_state(
        _receipt(complete=False, steps=200, halt_reason="wall_clock"),
        {"max_steps": 540},
    )

    assert complete["scope"] == "complete_training"
    assert complete["complete"] is True
    assert partial["scope"] == "bounded_partial_training"
    assert partial["complete"] is False
    assert partial["holdout_observed_ratio"] == 1.0


@pytest.mark.parametrize(
    ("complete", "steps", "halt_reason", "error"),
    [
        (False, 199, "wall_clock", "bounded_partial_not_admissible"),
        (False, 540, "wall_clock", "bounded_partial_not_admissible"),
        (False, 200, "interrupted", "bounded_partial_not_admissible"),
        (False, 200, "non_finite_loss", "bounded_partial_not_admissible"),
        (True, 539, "max_steps", "training_completion_state_invalid"),
        (True, 540, "wall_clock", "training_completion_state_invalid"),
    ],
)
def test_nonadmissible_terminal_training_states_fail(
    complete,
    steps,
    halt_reason,
    error,
):
    with pytest.raises(admission.ResidentV3TrainingAdmissionError, match=error):
        admission.evaluate_training_state(
            _receipt(complete=complete, steps=steps, halt_reason=halt_reason),
            {"max_steps": 540},
        )


def test_holdout_guard_rejects_terminal_overfit():
    receipt = _receipt(complete=False, steps=200, halt_reason="wall_clock")
    receipt["holdout_trail"] = [
        {"step": 100, "mean_loss": 1.0},
        {"step": 200, "mean_loss": 1.500001},
    ]

    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="holdout_overfitting_guard_failed",
    ):
        admission.evaluate_training_state(receipt, {"max_steps": 540})


@pytest.mark.parametrize(
    "trail",
    [[], [{"step": 200, "mean_loss": float("nan")}]],
)
def test_loss_trails_must_be_nonempty_and_finite(trail):
    receipt = _receipt(complete=False, steps=200, halt_reason="wall_clock")
    receipt["loss_trail"] = trail

    with pytest.raises(admission.ResidentV3TrainingAdmissionError):
        admission.evaluate_training_state(receipt, {"max_steps": 540})


def _resume_contract(tmp_path: Path):
    wrapper = tmp_path / "tools/run_recurrence_training_envelope.py"
    trainer = tmp_path / "tools/recurrence_native_train_v2.py"
    protocol = {
        "training": {
            "output_dir": str(tmp_path / "adapter"),
            "adapter_id": "resident-v3",
            "max_steps": 540,
        }
    }
    amendment = {
        "resource_envelope": {
            "wrapper": "tools/run_recurrence_training_envelope.py",
            "trainer": "tools/recurrence_native_train_v2.py",
            "envelope_out": str(tmp_path / "adapter/envelope.json"),
        },
        "partial": {"checkpoint_every_steps": 5},
        "resume": {"checkpoint_every_steps": 5},
        "sentinel": {
            "partial_stage_path": str(tmp_path / "adapter/partial-stage.json"),
            "resume_stage_path": str(tmp_path / "adapter/resume-stage.json"),
        },
    }
    command = [
        str(tmp_path / ".venv/bin/python"),
        str(wrapper),
        "--memory-limit-gb",
        "40",
        "--cache-limit-gb",
        "2",
        "--wired-limit-gb",
        "48",
        "--envelope-out",
        amendment["resource_envelope"]["envelope_out"],
        "--trainer",
        str(trainer),
        "--",
        "--out-dir",
        protocol["training"]["output_dir"],
        "--adapter-id",
        "resident-v3",
        "--objective",
        "v3",
        "--max-steps",
        "540",
        "--checkpoint-every",
        "5",
        "--resource-stage-path",
        amendment["sentinel"]["resume_stage_path"],
        "--resource-startup-lethal-mb",
        "73728",
        "--resource-steady-lethal-mb",
        "59392",
        "--resume",
    ]
    return (
        protocol,
        amendment,
        {"command": command, "plan_sha256": "a" * 64, "command_sha256": "b" * 64},
    )


def test_resume_command_requires_envelope_and_exact_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "REPO_ROOT", tmp_path)
    protocol, amendment, plan = _resume_contract(tmp_path)

    evidence = admission._verify_resume_command(plan, protocol, amendment)
    assert evidence["resource_limits_gb"] == {"active": 40, "cache": 2, "wired": 48}

    plan["command"][plan["command"].index("--wired-limit-gb") + 1] = "52"
    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="resume_command_contract_mismatch",
    ):
        admission._verify_resume_command(plan, protocol, amendment)


def test_partial_command_forbids_resume_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "REPO_ROOT", tmp_path)
    protocol, amendment, plan = _resume_contract(tmp_path)
    plan["command"].remove("--resume")
    stage_index = plan["command"].index("--resource-stage-path") + 1
    plan["command"][stage_index] = amendment["sentinel"]["partial_stage_path"]

    evidence = admission._verify_resume_command(
        plan,
        protocol,
        amendment,
        phase="partial",
    )
    assert evidence["phase"] == "partial"

    plan["command"].append("--resume")
    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="resume_command_contract_mismatch",
    ):
        admission._verify_resume_command(
            plan,
            protocol,
            amendment,
            phase="partial",
        )


def test_resume_command_rejects_direct_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "REPO_ROOT", tmp_path)
    protocol, amendment, plan = _resume_contract(tmp_path)
    plan["command"] = [
        str(tmp_path / ".venv/bin/python"),
        str(tmp_path / "tools/recurrence_native_train_v2.py"),
        "--resume",
    ]

    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="resume_command_invalid",
    ):
        admission._verify_resume_command(plan, protocol, amendment)


def test_footprint_requires_clean_sentinel_and_stays_below_lethal(tmp_path, monkeypatch):
    sentinel_dir = tmp_path / "sentinel"
    sentinel_dir.mkdir()
    ring = tmp_path / "ring.jsonl"
    stage_path = tmp_path / "stage.json"
    _marker, marker_raw = publish_ready_marker(
        stage_path,
        target_pid=123,
        trainer_sha256="c" * 64,
    )
    _initial_path, _initial, initial_ack_raw = publish_armed_ack(
        stage_path,
        marker_raw=marker_raw,
        target_pid=123,
        sentinel_pid=456,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    acquire_path, _acquire, acquire_raw = publish_compute_lease_request(
        stage_path,
        marker_raw=marker_raw,
        target_pid=123,
        sequence=1,
        workload="training_step",
        action="acquire",
        predecessor_ack_raw=initial_ack_raw,
    )
    _path, _ack, acquire_ack_raw = publish_compute_lease_ack(
        acquire_path,
        request_raw=acquire_raw,
        target_pid=123,
        sentinel_pid=456,
        sequence=1,
        workload="training_step",
        action="acquire",
        active_lethal_mb=73728.0,
    )
    release_path, _release, release_raw = publish_compute_lease_request(
        stage_path,
        marker_raw=marker_raw,
        target_pid=123,
        sequence=1,
        workload="training_step",
        action="release",
        predecessor_ack_raw=acquire_ack_raw,
    )
    publish_compute_lease_ack(
        release_path,
        request_raw=release_raw,
        target_pid=123,
        sentinel_pid=456,
        sequence=1,
        workload="training_step",
        action="release",
        active_lethal_mb=59392.0,
    )
    ring.write_text(
        "\n".join(
            json.dumps(
                {
                    "at": index,
                    "managed_mb": managed,
                    "guard_stage": stage,
                    "active_lethal_mb": lethal,
                    "marker_observed": observed,
                    "lease_sequence": sequence,
                    "lease_workload": workload,
                }
            )
            for index, (managed, stage, lethal, observed, sequence, workload) in enumerate(
                (
                    (22000.0, "startup", 73728.0, False, 1, ""),
                    (21000.0, "steady", 59392.0, True, 1, ""),
                    (65000.0, "compute", 73728.0, True, 1, "training_step"),
                    (57000.0, "draining", 73728.0, True, 1, "training_step"),
                    (41000.0, "steady", 59392.0, True, 2, ""),
                ),
                start=1,
            )
        )
        + "\n"
    )
    plan = {
        "command": [
            str(tmp_path / ".venv/bin/python"),
            str(tmp_path / "tools/memory_sentinel.py"),
            "--pid",
            "123",
            "--lethal-mb",
            "59392",
            "--startup-lethal-mb",
            "73728",
            "--steady-marker",
            str(stage_path),
            "--interval",
            "0.5",
            "--immediate-kill-overshoot",
            "1.05",
            "--ring",
            str(ring),
            "--ring-window-seconds",
            "46800",
            "--tombstone-dir",
            str(sentinel_dir),
        ],
        "plan_sha256": "a" * 64,
    }
    receipt = {
        "returncode": 0,
        "status": "passed",
        "receipt_sha256": "b" * 64,
        "child_pid": 456,
    }
    monkeypatch.setattr(admission, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(admission, "_detached_terminal", lambda *_args, **_kwargs: (plan, receipt))

    evidence = admission._verify_footprint(
        ring,
        sentinel_dir,
        trainer_pid=123,
        stage_path=stage_path,
        expected_trainer_sha256="c" * 64,
    )
    assert evidence["sample_count"] == 5
    assert evidence["stage_peak_managed_mb"]["compute"] == 65000.0
    assert evidence["compute_lease_workloads"] == {"training_step": 1}

    (sentinel_dir / "sentinel_tombstone_1.json").write_text("{}")
    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="memory_sentinel_lethal_abort",
    ):
        admission._verify_footprint(
            ring,
            sentinel_dir,
            trainer_pid=123,
            stage_path=stage_path,
            expected_trainer_sha256="c" * 64,
        )


def test_admission_output_is_create_once(tmp_path):
    output = tmp_path / "admission.json"
    first = admission._write_once(output, {"schema": admission.SCHEMA, "decision": "admit"})
    second = admission._write_once(output, {"schema": admission.SCHEMA, "decision": "admit"})
    assert first == second

    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="admission_output_exists_different",
    ):
        admission._write_once(output, {"schema": admission.SCHEMA, "decision": "reject"})


def test_terminal_checkpoint_requires_empty_pending_state_on_completion(tmp_path):
    adapter = tmp_path / "adapter"
    _checkpoint, receipt = _checkpoint_fixture(adapter, complete_run=True)

    evidence = admission._verify_terminal_checkpoint_state(
        adapter,
        receipt,
        log_every=5,
    )

    assert evidence["pending_loss_count"] == 0
    assert evidence["holdout_eval_count"] == 1

    checkpoint_path = adapter / evidence["checkpoint"] / "complete.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["pending_window_losses"] = [9.0]
    checkpoint_raw = _write_json(checkpoint_path, checkpoint)
    latest_path = adapter / "latest.json"
    latest = json.loads(latest_path.read_text())
    latest["complete_sha256"] = hashlib.sha256(checkpoint_raw).hexdigest()
    _write_json(latest_path, latest)
    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="terminal_checkpoint_exact_evidence_mismatch",
    ):
        admission._verify_terminal_checkpoint_state(
            adapter,
            receipt,
            log_every=5,
        )


def test_admission_replays_pre_resume_partial_checkpoint_attestation(tmp_path):
    adapter = tmp_path / "adapter"
    checkpoint, _receipt = _checkpoint_fixture(adapter, complete_run=False)
    protocol_path = tmp_path / "protocol.json"
    protocol_raw = _write_json(
        protocol_path,
        {"training": {"output_dir": str(adapter)}},
    )
    evidence_path = tmp_path / "partial-checkpoint-evidence.json"
    amendment_path = tmp_path / "amendment.json"
    amendment_raw = _write_json(
        amendment_path,
        {
            "resource_envelope": {"trainer_sha256": "d" * 64},
            "resume": {
                "expected_resume_step": 1,
                "partial_checkpoint_evidence_path": str(evidence_path),
            },
        },
    )
    before = time.time()
    launch.capture_partial_checkpoint_evidence(protocol_path, amendment_path)
    after = time.time()

    evidence = admission._verify_partial_checkpoint_evidence(
        evidence_path=evidence_path,
        adapter_dir=adapter,
        protocol_raw=protocol_raw,
        amendment_raw=amendment_raw,
        trainer_sha256="d" * 64,
        expected_step=1,
        partial_finished_at=before,
        resume_started_at=after,
    )

    assert evidence["pending_loss_count"] == 1
    assert evidence["durable_holdout_eval_count"] == 0
    assert (
        evidence["checkpoint_complete_sha256"]
        == hashlib.sha256(
            (adapter / "checkpoints/step-00000001-test/complete.json").read_bytes()
        ).hexdigest()
    )
    assert checkpoint["loss_trail"] == []
