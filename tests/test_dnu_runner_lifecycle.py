import json
import sys
import asyncio
import multiprocessing as multiprocessing_module

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


def test_dnu_artifact_manifest_never_hashes_itself(tmp_path):
    (tmp_path / "SCORECARD.json").write_text("{}", encoding="utf-8")
    (tmp_path / "MANIFEST.json").write_text('{"old": true}', encoding="utf-8")

    manifest = dnu_runner.write_artifact_manifest(
        tmp_path,
        run_id="run-1",
        commit_sha="abc123",
        include_files=["SCORECARD.json", "MANIFEST.json"],
    )
    stored = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest == stored
    assert "SCORECARD.json" in stored["files"]
    assert "MANIFEST.json" not in stored["files"]


def test_dnu_proof_health_wait_requires_actual_recovery(monkeypatch):
    snapshots = [
        {"runtime_health_contract": {"healthy": False, "status": "degraded"}},
        {"runtime_health_contract": {"healthy": True, "status": "healthy"}},
    ]

    def fake_collect(**_kwargs):
        return snapshots.pop(0)

    def fake_blockers(snapshot):
        return [] if snapshot["runtime_health_contract"]["healthy"] else ["runtime health status is degraded"]

    monkeypatch.setattr(dnu_runner, "collect_proof_resource_snapshot", fake_collect)
    monkeypatch.setattr(dnu_runner, "proof_runtime_health_blockers", fake_blockers)

    snapshot, blockers = asyncio.run(
        dnu_runner.wait_for_proof_runtime_health(
            label="after_model_lane_probe",
            timeout_s=1.0,
            interval_s=0.0,
        )
    )

    assert blockers == []
    assert snapshot["runtime_health_contract"]["healthy"] is True
    assert snapshot["runtime_health_recovery"]["initial_blockers"] == [
        "runtime health status is degraded"
    ]
    assert snapshot["runtime_health_recovery"]["recovered"] is True


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


def test_dnu_exclusivity_scanner_ignores_wrapper_shell_commands(monkeypatch):
    current_user = "runner"
    monkeypatch.setenv("USER", current_user)

    class FakeProc:
        def __init__(self, pid, cmdline, ppid=1):
            self.info = {
                "pid": pid,
                "ppid": ppid,
                "username": current_user,
                "cmdline": cmdline,
            }

        def cwd(self):
            return str(dnu_runner.PROJECT_ROOT)

    wrapper = FakeProc(
        201,
        [
            "zsh",
            "-lc",
            "AURA_AGI_MAX_TASKS=1 python tools/agi/run_dnu_agi_proof_battery.py > probe.log",
        ],
    )
    proof_runner = FakeProc(
        202,
        [sys.executable, "tools/agi/run_dnu_agi_proof_battery.py", "--full"],
    )
    aura_wrapper = FakeProc(203, ["zsh", "-lc", "python aura_main.py --desktop"])
    aura_runtime = FakeProc(204, [sys.executable, "aura_main.py", "--desktop"])

    class FakePsutil:
        AccessDenied = RuntimeError
        NoSuchProcess = RuntimeError
        ZombieProcess = RuntimeError

        @staticmethod
        def process_iter(attrs):
            return [wrapper, proof_runner, aura_wrapper, aura_runtime]

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)

    proof_matches = dnu_runner.find_existing_proof_runners()
    aura_matches = dnu_runner.find_existing_aura_runtimes()

    assert [match["pid"] for match in proof_matches] == [202]
    assert [match["pid"] for match in aura_matches] == [204]


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


def test_dnu_shutdown_reaper_preserves_python_resource_tracker(monkeypatch):
    class FakeProc:
        def __init__(self, pid, name, cmdline):
            self.pid = pid
            self._name = name
            self._cmdline = cmdline
            self.terminated = False
            self.killed = False

        def name(self):
            return self._name

        def cmdline(self):
            return list(self._cmdline)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    class FakeParent:
        def __init__(self, children):
            self._children = children

        def children(self, recursive=False):
            return list(self._children)

    resource_tracker = FakeProc(
        301,
        "resource_tracker",
        [sys.executable, "-m", "multiprocessing.resource_tracker"],
    )
    worker = FakeProc(302, "Python", [sys.executable, "-c", "worker"])

    class FakePsutil:
        AccessDenied = RuntimeError
        NoSuchProcess = RuntimeError
        ZombieProcess = RuntimeError

        @staticmethod
        def Process(pid):
            return FakeParent([resource_tracker, worker])

        @staticmethod
        def wait_procs(processes, timeout):
            return list(processes), []

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(multiprocessing_module, "active_children", lambda: [])

    asyncio.run(dnu_runner._reap_proof_child_processes("test_shutdown"))

    assert resource_tracker.terminated is False
    assert resource_tracker.killed is False
    assert worker.terminated is True
