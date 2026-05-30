import json

from tools.agi import run_dnu_agi_proof_battery as dnu_runner


def test_interrupted_dnu_runs_cannot_leave_stale_completion_artifacts():
    assert "MANIFEST.json" in dnu_runner.DNU_STALE_ARTIFACTS
    assert "RUN_STATUS.json" in dnu_runner.DNU_STALE_ARTIFACTS
    assert "RESOURCE_TRACE.jsonl" in dnu_runner.DNU_STALE_ARTIFACTS
    assert "LIFECYCLE_EVENTS.jsonl" in dnu_runner.DNU_STALE_ARTIFACTS


def test_dnu_standard_copy_includes_lifecycle_artifacts():
    assert "RUN_STATUS.json" in dnu_runner.DNU_STANDARD_COPY_ARTIFACTS
    assert "RESOURCE_TRACE.jsonl" in dnu_runner.DNU_STANDARD_COPY_ARTIFACTS
    assert "LIFECYCLE_EVENTS.jsonl" in dnu_runner.DNU_STANDARD_COPY_ARTIFACTS


def test_write_run_status_marks_completion_truthfully(tmp_path):
    payload = dnu_runner.write_run_status(
        tmp_path,
        status="complete",
        run_id="run-1",
        commit_sha="abc123",
        phase="complete",
        tasks_completed=100,
        total_tasks=100,
        lifecycle_events=2,
    )

    stored = json.loads((tmp_path / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert stored == payload
    assert stored["schema"] == "aura.dnu_run_status.v1"
    assert stored["runner_completed"] is True
    assert stored["tasks_completed"] == 100
    assert stored["total_tasks"] == 100
    assert stored["lifecycle_events"] == 2


def test_primary_full_dnu_defaults_to_periodic_model_recycling(monkeypatch):
    monkeypatch.delenv("AURA_DNU_MODEL_RECYCLE_INTERVAL", raising=False)

    assert (
        dnu_runner.dnu_model_recycle_interval(
            "primary",
            total_tasks=100,
            smoke=False,
        )
        == 40
    )
    assert (
        dnu_runner.dnu_model_recycle_interval(
            "tertiary",
            total_tasks=100,
            smoke=False,
        )
        == 0
    )
    assert (
        dnu_runner.dnu_model_recycle_interval(
            "primary",
            total_tasks=100,
            smoke=True,
        )
        == 0
    )
