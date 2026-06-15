import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.resilience.stability_guardian import StabilityGuardian
from core.runtime import runtime_hygiene as runtime_hygiene_module
from core.runtime.runtime_hygiene import MemorySample, RuntimeHygieneManager
from core.utils.task_tracker import TaskTracker

TMP_ROOT = Path(tempfile.gettempdir())


@pytest.mark.asyncio
async def test_task_tracker_loop_hygiene_observes_raw_asyncio_tasks():
    tracker = TaskTracker(name="RuntimeHygieneTest")
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(None)
    tracker.install_loop_hygiene(loop)
    release = asyncio.Event()

    async def _hold():
        await release.wait()

    try:
        task = loop.create_task(_hold(), name="runtime_hygiene.implicit")
        await asyncio.sleep(0)

        stats = tracker.get_stats()
        assert stats["implicit_active"] >= 1
        assert getattr(task, "_aura_task_supervision", "") == "implicit"
        assert getattr(task, "_aura_task_tracker", "") == "RuntimeHygieneTest"
    finally:
        release.set()
        await asyncio.sleep(0)
        tracker.restore_loop_hygiene(loop)
        loop.set_task_factory(previous_factory)


@pytest.mark.asyncio
async def test_task_tracker_shutdown_cancels_protected_tasks():
    tracker = TaskTracker(name="ProtectedShutdownTest")
    cancelled: list[str] = []

    async def _hold(label: str):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(label)

    ordinary = tracker.create_task(_hold("ordinary"), name="ordinary")
    protected = tracker.create_task(_hold("protected"), name="protected")
    protected._aura_protected = True
    await asyncio.sleep(0)

    await tracker.shutdown(timeout=0.2)
    await asyncio.sleep(0)

    assert ordinary.cancelled()
    assert protected.cancelled()
    assert set(cancelled) == {"ordinary", "protected"}
    assert tracker.active_count == 0


@pytest.mark.asyncio
async def test_runtime_hygiene_tracks_non_daemon_threads():
    hygiene = RuntimeHygieneManager()
    hygiene.stale_thread_age_s = 0.0
    release = threading.Event()

    def _worker():
        release.wait(0.5)

    await hygiene.start(asyncio.get_running_loop())
    try:
        thread = threading.Thread(target=_worker, name="runtime-hygiene-thread", daemon=False)
        thread.start()
        await asyncio.sleep(0.05)

        report = hygiene.audit()

        assert report["threads"]["active_non_daemon"] >= 1
        assert report["healthy"]
        assert report["threads"]["stale_non_daemon"] >= 1
    finally:
        release.set()
        thread.join(timeout=1.0)
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_runtime_hygiene_tracks_subprocesses():
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())
    proc = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "import time; time.sleep(0.25)"],
    )

    try:
        await asyncio.sleep(0.05)
        report = hygiene.audit()
        assert report["processes"]["active_registered"] >= 1
        assert report["processes"]["active_subprocesses"] >= 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        await hygiene.stop()
        hygiene.reset_state()


@pytest.mark.asyncio
async def test_runtime_hygiene_adopts_existing_subprocesses_started_before_hygiene():
    assert runtime_hygiene_module._HAS_PSUTIL, "psutil unavailable in this environment"
    try:
        runtime_hygiene_module.psutil.Process().children(recursive=True)
    except PermissionError as exc:
        pytest.fail(f"psutil child-process inspection is blocked: {exc}")

    proc = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", "import time; time.sleep(1.0)"],
    )
    hygiene = RuntimeHygieneManager()
    await hygiene.start(asyncio.get_running_loop())

    try:
        report = {}
        for _ in range(20):
            await asyncio.sleep(0.05)
            report = hygiene.audit()
            if report["processes"]["active_registered"] >= 1:
                break
        assert report["processes"]["active_registered"] >= 1
        assert report["processes"]["rogue_child_processes"] == 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        await hygiene.stop()
        hygiene.reset_state()


def test_runtime_hygiene_skips_tracemalloc_by_default(monkeypatch):
    calls = []

    monkeypatch.delenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", raising=False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "is_tracing", lambda: False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "start", lambda frames=1: calls.append(frames))

    hygiene = RuntimeHygieneManager()
    hygiene._start_tracemalloc()

    assert calls == []


def test_runtime_hygiene_can_opt_in_tracemalloc(monkeypatch):
    calls = []

    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", "1")
    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES", "3")
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "is_tracing", lambda: False)
    monkeypatch.setattr(runtime_hygiene_module.tracemalloc, "start", lambda frames=1: calls.append(frames))

    hygiene = RuntimeHygieneManager()
    hygiene._start_tracemalloc()

    assert calls == [3]


def test_runtime_hygiene_treats_active_model_growth_as_transient(monkeypatch):
    hygiene = RuntimeHygieneManager()
    now = time.monotonic()
    hygiene._samples.clear()

    for idx in range(hygiene.memory_growth_window):
        hygiene._samples.append(
            MemorySample(
                timestamp=now + idx,
                rss_bytes=int((100 + (idx * 35)) * 1024 * 1024),
                traced_bytes=0,
                task_count=0,
                thread_count=1,
                child_process_count=1,
            )
        )

    monkeypatch.setattr(
        hygiene,
        "_active_local_model_activity",
        lambda: ["Qwen2.5-32B-Instruct-8bit:warming"],
    )

    summary = hygiene._memory_summary()

    assert summary["sustained_growth"] is False
    assert summary["transient_growth"] is True
    assert "local model activity" in summary["message"].lower()


def test_runtime_hygiene_treats_recent_model_warmup_as_transient(monkeypatch):
    hygiene = RuntimeHygieneManager()
    hygiene.model_activity_grace_s = 120.0
    now = time.time()

    fake_client = SimpleNamespace(
        get_lane_status=lambda: {
            "state": "ready",
            "warmup_in_flight": False,
            "current_request_started_at": 0.0,
            "last_ready_at": now - 10.0,
            "last_progress_at": now - 12.0,
            "last_transition_at": now - 15.0,
        }
    )
    fake_mlx_module = SimpleNamespace(_CLIENTS={str(TMP_ROOT / "cortex"): fake_client})
    fake_server_module = SimpleNamespace(_SERVER_CLIENTS={})
    monkeypatch.setitem(sys.modules, "core.brain.llm.mlx_client", fake_mlx_module)
    monkeypatch.setitem(sys.modules, "core.brain.llm.local_server_client", fake_server_module)

    assert hygiene._active_local_model_activity() == ["cortex:recent"]


def test_runtime_hygiene_tolerates_model_registry_churn(monkeypatch):
    hygiene = RuntimeHygieneManager()
    now = time.time()

    fake_client = SimpleNamespace(
        get_lane_status=lambda: {
            "state": "ready",
            "warmup_in_flight": False,
            "current_request_started_at": 0.0,
            "last_ready_at": now - 10.0,
            "last_progress_at": now - 12.0,
            "last_transition_at": now - 15.0,
        }
    )

    class FlakyRegistry(dict):
        def __init__(self):
            super().__init__({str(TMP_ROOT / "cortex"): fake_client})
            self.calls = 0

        def items(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("dictionary changed size during iteration")
            return super().items()

    fake_mlx_module = SimpleNamespace(_CLIENTS=FlakyRegistry())
    fake_server_module = SimpleNamespace(_SERVER_CLIENTS={})
    monkeypatch.setitem(sys.modules, "core.brain.llm.mlx_client", fake_mlx_module)
    monkeypatch.setitem(sys.modules, "core.brain.llm.local_server_client", fake_server_module)

    assert hygiene._active_local_model_activity() == ["cortex:recent"]


def test_runtime_hygiene_adopts_late_active_children_before_flagging_rogue_processes():
    class _ChildProc:
        pid = 43210

        def cmdline(self):
            return [sys.executable, "-m", "multiprocessing.spawn"]

        def name(self):
            return "spawned-child"

        def is_running(self):
            return True

        def status(self):
            return "sleeping"

    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [_ChildProc()])

    hygiene._adopt_active_child_processes()
    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_subprocesses"] == 1
    assert summary["rogue_child_processes"] == 0


def test_runtime_hygiene_explicit_process_owner_registration_deduplicates_by_pid():
    class _OwnedProc:
        pid = 43212
        name = "MLXWorker-test"

        def is_alive(self):
            return True

    hygiene = RuntimeHygieneManager()
    proc = _OwnedProc()

    hygiene.register_process_handle(
        proc,
        kind="multiprocessing",
        name="MLXWorker-test",
        source="test.worker_owner",
        command="MLX worker for test",
    )
    hygiene.register_process_handle(
        proc,
        kind="multiprocessing",
        name="MLXWorker-test",
        source="test.worker_owner",
        command="MLX worker for test",
    )

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["active_multiprocessing"] == 1


def test_runtime_hygiene_process_iter_system_error_is_nonfatal(monkeypatch):
    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [])
    if not runtime_hygiene_module._HAS_PSUTIL:
        hygiene._adopt_active_child_processes()
        assert hygiene._process_records == {}
        return

    monkeypatch.setattr(
        runtime_hygiene_module.psutil,
        "process_iter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemError("proc_cmdline permission wrapper failed")
        ),
    )

    hygiene._adopt_active_child_processes()

    assert hygiene._process_records == {}


def test_runtime_hygiene_classifies_registered_worker_descendants_as_owned():
    class _Proc:
        def __init__(self, pid: int, ppid: int, name: str):
            self.pid = pid
            self._ppid = ppid
            self._name = name

        def ppid(self):
            return self._ppid

        def cmdline(self):
            return [sys.executable, "-m", self._name]

        def name(self):
            return self._name

        def is_running(self):
            return True

        def status(self):
            return "sleeping"

    parent = _Proc(61001, 999, "mlx-worker")
    helper = _Proc(61002, 61001, "mlx-helper")
    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [parent, helper])
    hygiene.register_process_handle(
        SimpleNamespace(pid=61001),
        kind="multiprocessing",
        name="mlx-worker",
        source="test.worker_owner",
        command="MLX worker",
    )

    summary = hygiene._process_summary()

    assert summary["active_registered"] == 1
    assert summary["owned_descendant_processes"] == 1
    assert summary["rogue_child_processes"] == 0
    assert summary["rogue_samples"] == []


def test_runtime_hygiene_keeps_unowned_child_process_fail_closed():
    class _Proc:
        pid = 62002

        def ppid(self):
            return 999

        def cmdline(self):
            return [sys.executable, "-m", "unexpected_worker"]

        def name(self):
            return "unexpected-worker"

    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [_Proc()])

    summary = hygiene._process_summary()

    assert summary["owned_descendant_processes"] == 0
    assert summary["rogue_child_processes"] == 1
    assert summary["rogue_samples"][0]["pid"] == 62002
    assert "unexpected-worker" in summary["rogue_samples"][0]["name"]


@pytest.mark.asyncio
async def test_runtime_hygiene_ignores_python_resource_tracker_children():
    class _ResourceTrackerProc:
        pid = 43211

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.waited = False

        def cmdline(self):
            return [
                sys.executable,
                "-c",
                "from multiprocessing.resource_tracker import main;main(11)",
            ]

        def name(self):
            return "Python"

        def is_running(self):
            return True

        def status(self):
            return "sleeping"

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True

        def kill(self):
            self.killed = True

    child = _ResourceTrackerProc()
    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [child])

    hygiene._adopt_active_child_processes()
    summary = hygiene._process_summary()
    await hygiene._cleanup_child_processes()

    assert summary["active_registered"] == 0
    assert summary["rogue_child_processes"] == 0
    assert child.terminated is False
    assert child.waited is False
    assert child.killed is False


def test_runtime_hygiene_thread_join_helper_skips_current_thread():
    current = threading.current_thread()

    RuntimeHygieneManager._join_thread_if_not_current(current, 0.01)


def test_runtime_hygiene_shutdown_thread_join_env_is_bounded(monkeypatch):
    monkeypatch.setenv("AURA_RUNTIME_HYGIENE_MAX_SHUTDOWN_THREAD_JOINS", "not-an-int")

    hygiene = RuntimeHygieneManager()

    assert hygiene.max_thread_joins_per_shutdown == 16


@pytest.mark.asyncio
async def test_runtime_hygiene_shutdown_thread_join_is_bounded(monkeypatch):
    recorded = []

    def fake_record_degradation(subsystem, error, **kwargs):
        recorded.append((subsystem, error, kwargs))

    monkeypatch.setattr(runtime_hygiene_module, "record_degradation", fake_record_degradation)

    class FakeThread:
        daemon = False

        def __init__(self, idx: int):
            self.ident = 10_000 + idx
            self.name = f"fake-thread-{idx}"
            self.joined = False

        def is_alive(self):
            return True

        def join(self, timeout=None):
            del timeout
            self.joined = True

    hygiene = RuntimeHygieneManager()
    hygiene.max_thread_joins_per_shutdown = 2
    threads = [FakeThread(idx) for idx in range(5)]
    hygiene._thread_refs = {idx: thread for idx, thread in enumerate(threads)}

    await hygiene._join_non_daemon_threads()

    assert [thread.joined for thread in threads] == [True, True, False, False, False]
    assert recorded
    subsystem, error, kwargs = recorded[0]
    assert subsystem == "runtime_hygiene_shutdown"
    assert "left for owner shutdown" in str(error)
    assert kwargs["severity"] == "warning"
    assert kwargs["enforce_failure_policy"] is False
    assert kwargs["extra"]["skipped_count"] == 3


@pytest.mark.asyncio
async def test_runtime_hygiene_shutdown_thread_join_errors_do_not_fail_closed(monkeypatch):
    recorded = []

    def fake_record_degradation(subsystem, error, **kwargs):
        recorded.append((subsystem, type(error).__name__, kwargs))

    monkeypatch.setattr(runtime_hygiene_module, "record_degradation", fake_record_degradation)

    class FailingThread:
        daemon = False
        ident = 20_000
        name = "failing-thread"

        def is_alive(self):
            return True

        def join(self, timeout=None):
            del timeout
            raise RuntimeError("join failed")

    hygiene = RuntimeHygieneManager()
    hygiene.max_thread_joins_per_shutdown = 1
    hygiene._thread_refs = {1: FailingThread()}

    await hygiene._join_non_daemon_threads()

    assert recorded == [
        (
            "runtime_hygiene_shutdown",
            "RuntimeError",
            {
                "severity": "warning",
                "action": "continued shutdown after a bounded thread join failed",
                "enforce_failure_policy": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_runtime_hygiene_child_cleanup_is_concurrent():
    class SlowTerminatingProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            time.sleep(0.15)
            raise subprocess.TimeoutExpired(cmd="slow", timeout=timeout or 0.0)

        def kill(self):
            self.killed = True

    hygiene = RuntimeHygieneManager()
    hygiene.process_shutdown_timeout_s = 0.2
    processes = [SlowTerminatingProcess(), SlowTerminatingProcess(), SlowTerminatingProcess()]
    hygiene._process_refs = {idx: proc for idx, proc in enumerate(processes)}

    started = time.monotonic()
    await hygiene._cleanup_child_processes()
    elapsed = time.monotonic() - started

    assert all(proc.terminated for proc in processes)
    assert all(proc.killed for proc in processes)
    assert elapsed < 0.6


@pytest.mark.asyncio
async def test_runtime_hygiene_cleans_adopted_psutil_children(monkeypatch):
    class PsutilChild:
        pid = 54321

        def __init__(self):
            self.terminated = False
            self.killed = False
            self.running = True

        def cmdline(self):
            return [sys.executable, "-m", "multiprocessing.spawn"]

        def name(self):
            return "spawned-child"

        def is_running(self):
            return self.running

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.running = False

        def kill(self):
            self.killed = True
            self.running = False

    child = PsutilChild()
    monkeypatch.setattr(runtime_hygiene_module, "_HAS_PSUTIL", True)

    hygiene = RuntimeHygieneManager()
    hygiene._proc = SimpleNamespace(children=lambda recursive=True: [child])
    hygiene.process_shutdown_timeout_s = 0.2

    hygiene._adopt_active_child_processes()
    await hygiene._cleanup_child_processes()

    assert child.terminated
    assert child.running is False


@pytest.mark.asyncio
async def test_stability_guardian_surfaces_runtime_hygiene_findings(service_container):
    service_container.register_instance(
        "runtime_hygiene",
        SimpleNamespace(
            audit=lambda: {
                "healthy": False,
                "critical": False,
                "issues": ["1 long-lived implicit task(s) still running"],
                "repair_actions": ["gc.collect()"],
            }
        ),
        required=False,
    )
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    result = await guardian._check_runtime_hygiene()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "long-lived implicit task" in result.message
    assert result.action_taken == "gc.collect()"


@pytest.mark.asyncio
async def test_stability_guardian_rejects_runtime_hygiene_report_without_health_evidence(service_container):
    service_container.register_instance(
        "runtime_hygiene",
        SimpleNamespace(
            audit=lambda: {
                "critical": False,
                "issues": ["runtime hygiene did not emit healthy"],
                "repair_actions": [],
            }
        ),
        required=False,
    )
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))

    result = await guardian._check_runtime_hygiene()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "runtime hygiene did not emit healthy" in result.message


def test_stability_guardian_treats_slow_user_facing_ticks_as_info():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    now = time.time()

    for _ in range(5):
        guardian.record_tick_health(
            SimpleNamespace(
                tick_duration_ms=22000.0,
                origin="user",
                priority=True,
                is_user_facing=True,
            )
        )
    guardian._loop_lag_samples.append((now, 40.0))

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert result.severity == "info"
    assert "Foreground turns are slow" in result.message


def test_stability_guardian_flags_actual_event_loop_lag():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time() - 1000.0))
    now = time.time()
    guardian.record_tick_health(
        SimpleNamespace(
            tick_duration_ms=450.0,
            origin="system",
            priority=False,
            is_user_facing=False,
        )
    )
    guardian._loop_lag_samples.append((now, guardian.MAX_EVENT_LOOP_LAG_MS + 250.0))

    result = guardian._check_tick_rate()

    assert result.healthy is False
    assert result.severity == "warning"
    assert "Event loop lag is elevated" in result.message


def test_stability_guardian_treats_stale_event_loop_lag_as_info():
    guardian = StabilityGuardian(SimpleNamespace(start_time=time.time()))
    guardian.record_tick_health(
        SimpleNamespace(
            tick_duration_ms=450.0,
            origin="system",
            priority=False,
            is_user_facing=False,
        )
    )
    guardian._loop_lag_samples.append(
        (
            time.time() - (guardian.EVENT_LOOP_LAG_WINDOW_S + 5.0),
            guardian.MAX_EVENT_LOOP_LAG_MS + 300.0,
        )
    )

    result = guardian._check_tick_rate()

    assert result.healthy is True
    assert result.severity in {"info", "warning"}
    assert "tick health ok" in result.message.lower()
