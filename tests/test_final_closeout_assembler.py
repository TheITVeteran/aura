from __future__ import annotations

import json
from pathlib import Path

from tools.closeout.final_closeout_assembler import assemble


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_artifacts(root: Path) -> tuple[Path, Path]:
    current = root / "current"
    closeout = root / "closeout"
    _write_json(current / "live_desktop_runtime" / "LATEST_VERDICT.json", {"passed": True})
    _write_json(current / "background_autonomy" / "MANIFEST.json", {"passed": True})
    _write_json(
        current / "background_autonomy" / "BACKGROUND_AUTONOMY_REPORT.json",
        {
            "passed": True,
            "components_running": 22,
            "components_total": 22,
            "desktop_access": {
                "overall_status": "ready",
                "permission_confidence": "direct",
                "screen_capture_ready": True,
                "desktop_control_ready": True,
                "screen_text_ready": True,
                "blocking_permissions": [],
            },
        },
    )
    _write_json(
        current / "agi_live" / "RUN_STATUS.json",
        {
            "schema": "aura.dnu_run_status.v1",
            "status": "complete",
            "runner_completed": True,
            "tasks_completed": 100,
            "total_tasks": 100,
        },
    )
    _write_json(current / "agi_live" / "SCORECARD.json", {"average_score": 1.0})
    _write_json(current / "agi_live" / "DNU_AGI_PROOF.json", {"passed": True})
    _write_json(current / "aletheia_tier5_validation.json", {"passed": True})
    _write_json(current / "receipt_coverage.json", {"passed": True})
    _write_json(current / "artifact_consistency.json", {"passed": True})
    _write_json(current / "final_claim_validation.json", {"passed": True})
    _write_json(closeout / "operational_label_battery_latest.json", {"passed": True})
    _write_json(closeout / "frontier_standards_latest.json", {"passed": True})
    _write_json(
        closeout / "remaining_checkpoint_contract_latest.json",
        {"summary": {"gaps": 0, "remaining_checkpoints": 0}},
    )
    return current, closeout


def test_final_closeout_assembler_writes_bundle_for_complete_evidence(tmp_path):
    current, _closeout = _seed_artifacts(tmp_path / "artifacts")
    out_dir = current / "final_closeout"

    rc, report = assemble(artifacts_dir=current, out_dir=out_dir, skip_validators=True)

    assert rc == 0
    assert report["passed"] is True
    assert (out_dir / "FINAL_CLOSEOUT.json").exists()
    assert (out_dir / "FINAL_CLOSEOUT.md").exists()
    assert (out_dir / "SHA256SUMS.txt").exists()
    saved = json.loads((out_dir / "FINAL_CLOSEOUT.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert not saved["failed_evidence"]
    keys = {item["key"] for item in saved["evidence"]}
    assert "background_autonomy_report" in keys
    assert "remaining_checkpoint_contract" in keys


def test_final_closeout_assembler_fails_on_incomplete_dnu_status(tmp_path):
    current, _closeout = _seed_artifacts(tmp_path / "artifacts")
    _write_json(
        current / "agi_live" / "RUN_STATUS.json",
        {
            "schema": "aura.dnu_run_status.v1",
            "status": "running",
            "runner_completed": False,
            "tasks_completed": 99,
            "total_tasks": 100,
        },
    )

    rc, report = assemble(
        artifacts_dir=current,
        out_dir=current / "final_closeout",
        skip_validators=True,
    )

    assert rc == 1
    assert report["passed"] is False
    failed = {item["key"]: item for item in report["failed_evidence"]}
    assert failed["dnu_run_status"]["reason"] == "dnu run status is incomplete"


def test_final_closeout_assembler_fails_when_required_closeout_matrix_missing(tmp_path):
    current, closeout = _seed_artifacts(tmp_path / "artifacts")
    (closeout / "frontier_standards_latest.json").unlink()

    rc, report = assemble(
        artifacts_dir=current,
        out_dir=current / "final_closeout",
        skip_validators=True,
    )

    assert rc == 1
    assert report["passed"] is False
    failed_keys = {item["key"] for item in report["failed_evidence"]}
    assert "frontier_standards" in failed_keys
