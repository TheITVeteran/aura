"""Contracts for detached resident-v3 calibration promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import run_resident_v3_recovery_controller as controller


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _migration(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "resident_32b_v3_cp195"
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    path = adapter / "checkpoint_migration.json"
    path.write_text(
        json.dumps(
            {
                "migration_sha256": "a" * 64,
                "destination": {"root": str(adapter)},
            }
        ),
        encoding="ascii",
    )
    return path, root


def _statuses(returncode: int = 0):
    return {
        role: {
            "terminal": True,
            "receipt": {
                "returncode": returncode,
                "receipt_sha256": character * 64,
                "containment_verified": True,
                "process_group_empty": True,
                "lineage_empty": True,
                "restart_count": 0,
            },
        }
        for role, character in (("trainer", "b"), ("sentinel", "c"))
    }


def test_controller_promotes_verified_calibration_and_waits_for_training(
    tmp_path,
    monkeypatch,
):
    migration, root = _migration(tmp_path)
    phases: list[str] = []
    statuses = iter((_statuses(), _statuses()))
    monkeypatch.setattr(
        controller,
        "_terminal_statuses",
        lambda _root, *, phase, timeout_s, poll_s: phases.append(phase)
        or next(statuses),
    )
    monkeypatch.setattr(
        controller.recovery,
        "verify_calibration",
        lambda _path: {"verdict_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        controller.recovery,
        "launch_phase",
        lambda _path, *, phase: {"phase": phase, "trainer_pid": 123},
    )

    verdict = controller.run_controller(migration)

    assert phases == ["calibration", "resume"]
    assert verdict["decision"] == "training_terminal_pending_strict_admission"
    assert verdict["details"]["calibration_verdict_sha256"] == "d" * 64
    material = dict(verdict)
    claimed = material.pop("verdict_sha256")
    assert claimed == hashlib.sha256(_canonical(material)).hexdigest()
    persisted = json.loads((root / "recovery_controller_verdict.json").read_text())
    assert persisted == verdict


def test_controller_fails_closed_when_calibration_verification_fails(
    tmp_path,
    monkeypatch,
):
    migration, root = _migration(tmp_path)
    monkeypatch.setattr(
        controller,
        "_terminal_statuses",
        lambda *_args, **_kwargs: _statuses(returncode=1),
    )

    def fail(_path):
        raise RuntimeError("calibration_failed")

    monkeypatch.setattr(controller.recovery, "verify_calibration", fail)

    verdict = controller.run_controller(migration)

    assert verdict["decision"] == "recovery_failed_closed"
    assert verdict["details"]["error"] == "calibration_failed"
    state = json.loads((root / "recovery_controller_state.json").read_text())
    assert state["stage"] == "failed"


def test_controller_reuses_immutable_terminal_verdict(tmp_path):
    migration, root = _migration(tmp_path)
    material = {
        "schema": controller.VERDICT_SCHEMA,
        "decision": "already_done",
        "migration_sha256": "a" * 64,
    }
    expected = {
        **material,
        "verdict_sha256": hashlib.sha256(_canonical(material)).hexdigest(),
    }
    (root / "recovery_controller_verdict.json").write_text(
        json.dumps(expected),
        encoding="ascii",
    )

    assert controller.run_controller(migration) == expected


def test_controller_rejects_tampered_terminal_verdict(tmp_path):
    migration, root = _migration(tmp_path)
    (root / "recovery_controller_verdict.json").write_text(
        json.dumps(
            {
                "schema": controller.VERDICT_SCHEMA,
                "decision": "already_done",
                "migration_sha256": "a" * 64,
                "verdict_sha256": "0" * 64,
            }
        ),
        encoding="ascii",
    )

    try:
        controller.run_controller(migration)
    except controller.ResidentV3RecoveryControllerError as exc:
        assert exc.code == "controller_verdict_invalid"
    else:
        raise AssertionError("tampered controller verdict was accepted")


def test_controller_attaches_only_to_cross_bound_resume_plans(tmp_path, monkeypatch):
    migration, root = _migration(tmp_path)
    run_dir = root / "detached-resume"
    sentinel_dir = root / "sentinel-resume"
    run_dir.mkdir()
    sentinel_dir.mkdir()
    adapter = root / "adapter"
    stage = adapter / f"resource_stage_resume_{root.name}.json"
    ring = adapter / f"physical_footprint_resume_{root.name}.jsonl"
    trainer = {
        "plan_sha256": "d" * 64,
        "command": [
            "python",
            "trainer.py",
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
    sentinel = {
        "plan_sha256": "e" * 64,
        "command": [
            "python",
            "memory_sentinel.py",
            "--pid",
            "123",
            "--steady-marker",
            str(stage),
            "--ring",
            str(ring),
        ],
    }
    (run_dir / "detached_plan.json").write_text(json.dumps(trainer), encoding="ascii")
    (sentinel_dir / "detached_plan.json").write_text(
        json.dumps(sentinel),
        encoding="ascii",
    )
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda path: {
            "completion_indeterminate": False,
            "child_pid": 123 if path == run_dir else 456,
        },
    )

    result = controller._launch_or_attach_resume(migration, root)

    assert result["attached"] is True
    assert result["trainer_pid"] == 123
    sentinel["command"][-1] = str(root / "wrong.jsonl")
    (sentinel_dir / "detached_plan.json").write_text(
        json.dumps(sentinel),
        encoding="ascii",
    )
    try:
        controller._launch_or_attach_resume(migration, root)
    except controller.ResidentV3RecoveryControllerError as exc:
        assert exc.code == "resume_attach_plan_invalid"
    else:
        raise AssertionError("cross-unbound resume plan was accepted")
