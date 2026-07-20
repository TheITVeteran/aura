from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import verify_resident_v3_training_admission as admission


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
        "--resume",
    ]
    return protocol, amendment, {"command": command, "plan_sha256": "a" * 64, "command_sha256": "b" * 64}


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
    ring.write_text(
        "\n".join(
            json.dumps({"at": index, "managed_mb": managed})
            for index, managed in enumerate((22000.0, 41000.0, 57000.0), start=1)
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
            "--interval",
            "2.0",
            "--ring",
            str(ring),
            "--tombstone-dir",
            str(sentinel_dir),
        ],
        "plan_sha256": "a" * 64,
    }
    receipt = {"returncode": 0, "status": "passed", "receipt_sha256": "b" * 64}
    monkeypatch.setattr(admission, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(admission, "_detached_terminal", lambda *_args, **_kwargs: (plan, receipt))

    evidence = admission._verify_footprint(ring, sentinel_dir, trainer_pid=123)
    assert evidence["sample_count"] == 3
    assert evidence["peak_managed_mb"] == 57000.0

    (sentinel_dir / "sentinel_tombstone_1.json").write_text("{}")
    with pytest.raises(
        admission.ResidentV3TrainingAdmissionError,
        match="memory_sentinel_lethal_abort",
    ):
        admission._verify_footprint(ring, sentinel_dir, trainer_pid=123)


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
