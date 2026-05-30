import json
import sys

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


def test_dnu_claims_canonical_runtime_lock_before_boot():
    source = dnu_runner.Path(dnu_runner.__file__).read_text(encoding="utf-8")

    assert "from aura_main import bootstrap_lock" in source
    assert "bootstrap_lock(skip_lock=False)" in source
    assert "canonical_runtime_lock_claimed_by_runner_pid" in source


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


def test_dnu_orphan_cleanup_recognizes_temp_aura_checkouts(tmp_path):
    temp_checkout = tmp_path / "aura-live-example" / "repo"
    temp_checkout.mkdir(parents=True)

    assert dnu_runner._is_aura_checkout_cwd(dnu_runner.PROJECT_ROOT)
    assert dnu_runner._is_aura_checkout_cwd(temp_checkout)
    assert not dnu_runner._is_aura_checkout_cwd(tmp_path / "other")


def test_dnu_orphan_cleanup_reaps_descendant_keep_awake(monkeypatch):
    class FakeProc:
        def __init__(self, pid, ppid, cmdline, children=None):
            self.info = {"pid": pid, "ppid": ppid, "cmdline": cmdline}
            self.pid = pid
            self._children = children or []
            self.terminated = False
            self.killed = False

        def cwd(self):
            return str(dnu_runner.PROJECT_ROOT)

        def cmdline(self):
            return list(self.info["cmdline"])

        def children(self, recursive=False):
            return list(self._children)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    keep_awake = FakeProc(202, 101, ["caffeinate", "-i", "-m", "-s"])
    parent = FakeProc(
        101,
        1,
        [
            sys.executable,
            "-c",
            "from multiprocessing.spawn import spawn_main; spawn_main()",
        ],
        children=[keep_awake],
    )

    class FakePsutil:
        AccessDenied = RuntimeError
        NoSuchProcess = RuntimeError
        ZombieProcess = RuntimeError

        @staticmethod
        def process_iter(attrs):
            return [parent, keep_awake]

        @staticmethod
        def Process(pid):
            return {101: parent, 202: keep_awake}[pid]

        @staticmethod
        def wait_procs(processes, timeout):
            return list(processes), []

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)

    reaped = dnu_runner.stop_orphaned_aura_multiprocessing_children()

    assert {entry["pid"] for entry in reaped} == {101, 202}
    assert parent.terminated is True
    assert keep_awake.terminated is True
