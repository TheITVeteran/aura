from __future__ import annotations

import json
import time
from pathlib import Path

from tools import artifact_consistency_validator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_artifact_consistency_rejects_failed_newer_proof_step(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "proof_steps" / "dnu_agi_battery.json",
        {
            "name": "dnu_agi_battery",
            "passed": False,
            "returncode": 1,
            "timed_out": False,
            "finished_at": time.time(),
        },
    )
    _write_json(
        tmp_path / "agi_live" / "RUN_STATUS.json",
        {
            "schema": "aura.dnu_run_status.v1",
            "status": "complete",
            "runner_completed": True,
            "tasks_completed": 100,
            "total_tasks": 100,
        },
    )

    result = artifact_consistency_validator.main(["--artifacts", str(tmp_path)])

    assert result == 1
    report = json.loads((tmp_path / "artifact_consistency.json").read_text())
    assert report["proof_steps_consistent"] is False
    assert any("dnu_agi_battery" in reason for reason in report["reasons"])


def test_artifact_consistency_rejects_incomplete_dnu_run_status(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "agi_live" / "RUN_STATUS.json",
        {
            "schema": "aura.dnu_run_status.v1",
            "status": "running",
            "runner_completed": False,
            "tasks_completed": 0,
            "total_tasks": 100,
        },
    )

    result = artifact_consistency_validator.main(["--artifacts", str(tmp_path)])

    assert result == 1
    report = json.loads((tmp_path / "artifact_consistency.json").read_text())
    assert report["dnu_status_complete"] is False
    assert "DNU run status is not complete: 'running'." in report["reasons"]
