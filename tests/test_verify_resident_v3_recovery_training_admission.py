from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import verify_resident_v3_recovery_training_admission as recovery_admission


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_hashed(
    path: Path,
    material: dict,
    *,
    hash_key: str,
) -> dict:
    value = {**material, hash_key: hashlib.sha256(_canonical(material)).hexdigest()}
    path.write_bytes(_canonical(value) + b"\n")
    return value


def _source_migration(adapter: Path, model: Path) -> dict:
    return {
        "source": {
            "training_config_document": {
                "model_path": str(model),
                "base_checkpoint": {"fingerprint": "a" * 64},
                "train_seed": 11,
                "max_steps": 540,
                "curriculum_depths": [1, 2, 4],
                "monotonicity_weight": 0.5,
                "objective_options": {
                    "depth_margin": 0.05,
                    "diversity_weight": 0.25,
                    "diversity_target_cos": 0.98,
                },
                "holdout": {"per_cell": 1, "count": 36, "eval_samples": 8},
                "lora": {"rank": 8, "targets": ["o_proj", "v_proj"]},
                "optimizer": {"learning_rate": 0.0001},
            },
            "execution_spec_document": {
                "n_slots": 4,
                "branch_roles": ["constructive_solution", "counterexample_search"],
                "exchange_interval": 1,
                "alpha": 0.5,
                "alpha_schedule": "constant",
            },
        },
        "destination": {"root": str(adapter)},
    }


def test_read_json_rejects_non_object_and_symlink(tmp_path):
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="ascii")
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="document_invalid",
    ):
        recovery_admission._read_json(scalar, role="document")

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="ascii")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="document_path_invalid",
    ):
        recovery_admission._read_json(link, role="document")


def test_hashed_document_rejects_tamper(tmp_path):
    path = tmp_path / "receipt.json"
    expected = _write_hashed(
        path,
        {"schema": "example.v1", "status": "passed"},
        hash_key="receipt_sha256",
    )
    _raw, observed = recovery_admission._verify_hashed_document(
        path,
        role="receipt",
        schema="example.v1",
        hash_key="receipt_sha256",
    )
    assert observed == expected

    observed["status"] = "failed"
    path.write_bytes(_canonical(observed))
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="receipt_invalid",
    ):
        recovery_admission._verify_hashed_document(
            path,
            role="receipt",
            schema="example.v1",
            hash_key="receipt_sha256",
        )


def test_protocol_is_derived_from_migration_source(tmp_path):
    root = tmp_path / "resident_32b_v3_cp195"
    adapter = root / "adapter"
    model = tmp_path / "model"
    migration = _source_migration(adapter, model)

    protocol = recovery_admission._protocol(migration, adapter)

    assert protocol["model"] == {
        "path": str(model),
        "expected_full_weight_sha256": "a" * 64,
    }
    assert protocol["training"]["adapter_id"] == "resident-32b-recurrence-v3-cp195"
    assert protocol["training"]["branch_roles"] == [
        "constructive_solution",
        "counterexample_search",
    ]
    assert protocol["training"]["max_steps"] == 540


def _resume_plan(
    *,
    migration: Path,
    adapter: Path,
    model: Path,
    stage: Path,
) -> dict:
    return {
        "plan_sha256": "a" * 64,
        "command_sha256": "b" * 64,
        "command": [
            "/venv/python",
            str(recovery_admission.ROOT / "tools/run_recurrence_training_envelope.py"),
            "--memory-limit-gb",
            "40",
            "--cache-limit-gb",
            "2",
            "--wired-limit-gb",
            "48",
            "--envelope-out",
            str(adapter / "envelope.json"),
            "--trainer",
            str(recovery_admission.ROOT / "tools/recurrence_native_train_v2.py"),
            "--",
            "--model",
            str(model),
            "--out-dir",
            str(adapter),
            "--max-minutes",
            "2880.0",
            "--resource-stage-path",
            str(stage),
            "--resume",
            "--resume-migration-evidence",
            str(migration),
        ],
    }


def test_resume_command_is_cross_bound_and_rejects_substitution(tmp_path):
    adapter = tmp_path / "resident_32b_v3_cp195/adapter"
    model = tmp_path / "model"
    migration = adapter / "checkpoint_migration.json"
    stage = adapter / "resource_stage_resume_resident_32b_v3_cp195.json"
    plan = _resume_plan(
        migration=migration,
        adapter=adapter,
        model=model,
        stage=stage,
    )

    evidence = recovery_admission._verify_resume_command(
        plan,
        migration_path=migration,
        adapter_root=adapter,
        model_path=model,
        stage_path=stage,
    )
    assert evidence == {"plan_sha256": "a" * 64, "command_sha256": "b" * 64}

    plan["command"][plan["command"].index("--model") + 1] = str(tmp_path / "other")
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="resume_command_invalid",
    ):
        recovery_admission._verify_resume_command(
            plan,
            migration_path=migration,
            adapter_root=adapter,
            model_path=model,
            stage_path=stage,
        )


def test_controller_binding_requires_calibration_trainer_and_sentinel():
    details = {
        "calibration_verdict_sha256": "a" * 64,
        "training_receipts": {
            "trainer": {"receipt_sha256": "b" * 64},
            "sentinel": {"receipt_sha256": "c" * 64},
        },
    }
    recovery_admission._verify_controller_terminal_binding(
        details,
        calibration_verdict_sha256="a" * 64,
        trainer_receipt_sha256="b" * 64,
        sentinel_receipt_sha256="c" * 64,
    )

    details["training_receipts"]["sentinel"]["receipt_sha256"] = "d" * 64
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="controller_terminal_binding_mismatch",
    ):
        recovery_admission._verify_controller_terminal_binding(
            details,
            calibration_verdict_sha256="a" * 64,
            trainer_receipt_sha256="b" * 64,
            sentinel_receipt_sha256="c" * 64,
        )


def test_archive_binds_detached_command_and_content(tmp_path, monkeypatch):
    operational = tmp_path / "operational.jsonl"
    archive = tmp_path / "proof/archive.jsonl"
    state = archive.parent / "archive_state.json"
    receipt_path = archive.parent / "archive_receipt.json"
    operational.write_bytes(b'{"at":1}\n')
    archive.parent.mkdir()
    archive.write_bytes(b'{"at":1}\n')
    receipt = _write_hashed(
        receipt_path,
        {
            "schema": recovery_admission.ARCHIVE_RECEIPT_SCHEMA,
            "status": "passed",
            "source": str(operational),
            "archive": str(archive),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "sample_count": 1,
            "target_pid": 123,
        },
        hash_key="receipt_sha256",
    )
    plan = {
        "plan_sha256": "a" * 64,
        "command": [
            "/venv/python",
            str(recovery_admission.ROOT / "tools/archive_memory_sentinel_ring.py"),
            "--source",
            str(operational),
            "--archive",
            str(archive),
            "--state",
            str(state),
            "--receipt",
            str(receipt_path),
            "--target-pid",
            "123",
            "--interval",
            "5",
        ],
    }
    detached_receipt = {
        "returncode": 0,
        "status": "passed",
        "receipt_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        recovery_admission.admission,
        "_detached_terminal",
        lambda *_args, **_kwargs: (plan, detached_receipt),
    )

    evidence = recovery_admission._verify_archive(
        archive_run_dir=tmp_path / "run",
        archive_ring=archive,
        archive_receipt_path=receipt_path,
        operational_ring=operational,
        trainer_pid=123,
    )
    assert evidence["receipt_sha256"] == receipt["receipt_sha256"]
    assert evidence["sample_count"] == 1

    plan["command"][plan["command"].index("--source") + 1] = str(tmp_path / "wrong")
    with pytest.raises(
        recovery_admission.ResidentV3RecoveryAdmissionError,
        match="sentinel_archive_command_invalid",
    ):
        recovery_admission._verify_archive(
            archive_run_dir=tmp_path / "run",
            archive_ring=archive,
            archive_receipt_path=receipt_path,
            operational_ring=operational,
            trainer_pid=123,
        )
