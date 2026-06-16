from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

try:
    import psutil

    _HAS_PSUTIL = True
    _PSUTIL_PROCESS_ERRORS = (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        psutil.ZombieProcess,
        psutil.Error,
    )
except ImportError:
    _HAS_PSUTIL = False
    _PSUTIL_PROCESS_ERRORS = ()

logger = logging.getLogger("Aura.RuntimeHygiene")
_PROCESS_INTROSPECTION_ERRORS = (
    RuntimeError,
    SystemError,
    AttributeError,
    TypeError,
    ValueError,
    OSError,
) + _PSUTIL_PROCESS_ERRORS
_THREAD_RUN_FAILURES = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SystemError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _snapshot_mapping_items(mapping: Any) -> list[tuple[Any, Any]]:
    """Return a bounded snapshot of a live mapping without crashing on churn."""
    if not mapping:
        return []
    last_error: RuntimeError | None = None
    for _attempt in range(3):
        try:
            return list(mapping.items())
        except RuntimeError as exc:
            if "changed size" not in str(exc):
                raise
            last_error = exc
            time.sleep(0)
    logger.debug("RuntimeHygiene: skipped mutating registry snapshot: %s", last_error)
    return []


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _process_cmdline(proc: Any) -> list[str]:
    try:
        return [str(part) for part in (proc.cmdline() or [])]
    except _PROCESS_INTROSPECTION_ERRORS:
        return []


def _process_name(proc: Any) -> str:
    try:
        return str(proc.name() or "")
    except _PROCESS_INTROSPECTION_ERRORS:
        return ""


def _is_python_resource_tracker_process(proc: Any) -> bool:
    """Return true for Python's internal multiprocessing tracker processes.

    The resource tracker owns semaphore/shared-memory bookkeeping for the
    current interpreter. Terminating it during runtime cleanup causes noisy
    relaunches and can corrupt its unregister cache, so it is observed but not
    adopted, flagged as rogue, or force-reaped by Aura cleanup.
    """

    name = _process_name(proc).lower()
    cmdline = " ".join(_process_cmdline(proc)).lower()
    return (
        name in {"resource_tracker", "semaphore_tracker"}
        or "multiprocessing.resource_tracker" in cmdline
        or "multiprocessing.semaphore_tracker" in cmdline
    )


def _is_python_multiprocessing_spawn_process(proc: Any) -> bool:
    """Return true for Python multiprocessing worker children owned by this runtime.

    Workers can appear between the adoption pass and the child-process summary
    scan. They are still Aura-owned if they are direct Python multiprocessing
    spawn children; adopt them instead of reporting a transient rogue child.
    """

    cmdline = " ".join(_process_cmdline(proc)).lower()
    return (
        "multiprocessing.spawn" in cmdline
        and "--multiprocessing-fork" in cmdline
    )


def _process_pid(proc: Any) -> int:
    try:
        return int(getattr(proc, "pid", 0) or 0)
    except _PROCESS_INTROSPECTION_ERRORS:
        return 0


def _process_ppid(proc: Any) -> int:
    try:
        if hasattr(proc, "ppid"):
            return int(proc.ppid() or 0)
    except _PROCESS_INTROSPECTION_ERRORS:
        return 0
    info = getattr(proc, "info", None) or {}
    try:
        return int(info.get("ppid") or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class MemorySample:
    timestamp: float
    rss_bytes: int
    traced_bytes: int
    task_count: int
    thread_count: int
    child_process_count: int


@dataclass
class ThreadRecord:
    key: int
    name: str
    daemon: bool
    source: str
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    ident: int | None = None
    exception: str | None = None

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        origin = self.started_at or self.created_at
        return max(0.0, current_time - origin)


@dataclass
class ProcessRecord:
    key: int
    kind: str
    name: str
    source: str
    command: str
    created_at: float = field(default_factory=time.monotonic)
    pid: int | None = None
    exit_code: int | None = None
    finished_at: float | None = None

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)


class RuntimeHygieneManager:
    """Tracks tasks, threads, child processes, and memory growth across the runtime."""

    def __init__(self):
        self._running = False
        self._thread_records: dict[int, ThreadRecord] = {}
        self._thread_refs: dict[int, threading.Thread] = {}
        self._process_records: dict[int, ProcessRecord] = {}
        self._process_refs: dict[int, Any] = {}
        self._samples: deque[MemorySample] = deque(maxlen=36)
        self._task_tracker = get_task_tracker()
        self._last_gc_at = 0.0

        self.memory_growth_window = 6
        self.memory_growth_min_delta_mb = 128.0
        self.memory_growth_ratio = 0.12
        self.model_activity_grace_s = max(
            0.0,
            float(os.getenv("AURA_RUNTIME_HYGIENE_MODEL_GRACE_S", "120") or 120.0),
        )
        self.stale_thread_age_s = 900.0
        self.stale_task_age_s = 900.0
        self.process_shutdown_timeout_s = 1.0
        self.thread_join_timeout_s = 0.2
        self.max_thread_joins_per_shutdown = _env_int(
            "AURA_RUNTIME_HYGIENE_MAX_SHUTDOWN_THREAD_JOINS",
            16,
            low=1,
            high=256,
        )
        self.shutdown_timeout_s = max(
            1.5,
            float(os.getenv("AURA_RUNTIME_HYGIENE_SHUTDOWN_TIMEOUT_S", "4.0") or 4.0),
        )
        self.tracemalloc_enabled = str(
            os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.tracemalloc_frames = max(
            1,
            int(os.getenv("AURA_RUNTIME_HYGIENE_TRACEMALLOC_FRAMES", "1") or 1),
        )
        self._tracemalloc_started_by_hygiene = False

        self._original_thread_start = None
        self._original_popen_init = None
        self._original_mp_start = None
        self._original_new_event_loop = None

        self._proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    async def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if self._running:
            target_loop = loop
            if target_loop is not None:
                self._task_tracker.install_loop_hygiene(target_loop)
            return

        self._running = True
        target_loop = loop or asyncio.get_running_loop()
        self._task_tracker.install_loop_hygiene(target_loop)
        self._patch_asyncio_new_event_loop()
        self._patch_threading()
        self._patch_subprocess()
        self._patch_multiprocessing()
        self._start_tracemalloc()
        self._adopt_active_child_processes()
        self.capture_sample()

    async def stop(self) -> None:
        self._task_tracker.restore_loop_hygiene()
        self._restore_patches()
        self._adopt_active_child_processes()
        await self._cleanup_child_processes()
        await self._join_non_daemon_threads()
        if self._tracemalloc_started_by_hygiene and tracemalloc.is_tracing():
            try:
                tracemalloc.stop()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: tracemalloc stop failed: %s", exc)
            finally:
                self._tracemalloc_started_by_hygiene = False
        self.capture_sample()
        self._running = False

    async def on_stop_async(self) -> None:
        await self.stop()

    def cleanup(self) -> None:
        self._restore_patches()

    def reset_state(self) -> None:
        self._thread_records.clear()
        self._thread_refs.clear()
        self._process_records.clear()
        self._process_refs.clear()
        self._samples.clear()
        self._last_gc_at = 0.0

    def capture_sample(self) -> MemorySample:
        rss_bytes = 0
        if self._proc is not None:
            try:
                rss_bytes = int(self._proc.memory_info().rss)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: failed to read RSS: %s", exc)
        traced_bytes = 0
        try:
            if tracemalloc.is_tracing():
                traced_bytes, _peak = tracemalloc.get_traced_memory()
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: tracemalloc snapshot failed: %s", exc)

        task_stats = self._task_tracker.get_stats()
        sample = MemorySample(
            timestamp=time.monotonic(),
            rss_bytes=rss_bytes,
            traced_bytes=traced_bytes,
            task_count=int(task_stats.get("active", 0)),
            thread_count=len(threading.enumerate()),
            child_process_count=self._count_child_processes(),
        )
        self._samples.append(sample)
        return sample

    def audit(self) -> dict[str, Any]:
        sample = self.capture_sample()
        self._adopt_active_child_processes()
        self._refresh_thread_records()
        self._refresh_process_records()

        task_stats = self._task_tracker.get_stats()
        stale_tasks = self._task_tracker.get_stale_tasks(self.stale_task_age_s)
        thread_summary = self._thread_summary()
        process_summary = self._process_summary()
        memory_summary = self._memory_summary()

        repair_actions: list[str] = []
        issues: list[str] = []
        critical = False

        # Stale tasks and non-daemon threads are expected for long-lived components
        # (e.g. ThreadPoolExecutor, background event loops). We track them in the
        # telemetry payload but do not flag them as active issues to avoid noise.
        if process_summary["rogue_child_processes"]:
            issues.append(f"{process_summary['rogue_child_processes']} unregistered child process(es) detected")
            critical = True
        if memory_summary["sustained_growth"]:
            issues.append(memory_summary["message"])
            if time.monotonic() - self._last_gc_at > 60.0:
                gc.collect()
                self._last_gc_at = time.monotonic()
                repair_actions.append("gc.collect()")

        summary = {
            "healthy": not issues,
            "critical": critical,
            "issues": issues,
            "repair_actions": repair_actions,
            "tasks": {
                **task_stats,
                "stale_implicit_tasks": stale_tasks[:5],
            },
            "threads": thread_summary,
            "processes": process_summary,
            "memory": memory_summary,
            "latest_sample": {
                "rss_mb": round(sample.rss_bytes / (1024 * 1024), 1),
                "traced_mb": round(sample.traced_bytes / (1024 * 1024), 1),
                "task_count": sample.task_count,
                "thread_count": sample.thread_count,
                "child_process_count": sample.child_process_count,
            },
        }
        return summary

    def get_status(self) -> dict[str, Any]:
        report = self.audit()
        report["running"] = self._running
        return report

    def _patch_asyncio_new_event_loop(self) -> None:
        if self._original_new_event_loop is not None:
            return

        self._original_new_event_loop = asyncio.new_event_loop
        tracker = self._task_tracker

        def _patched_new_event_loop():
            loop = self._original_new_event_loop()
            try:
                tracker.install_loop_hygiene(loop)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: failed to install task factory on new loop: %s", exc)
            return loop

        asyncio.new_event_loop = _patched_new_event_loop

    def _patch_threading(self) -> None:
        if self._original_thread_start is not None:
            return

        self._original_thread_start = threading.Thread.start
        manager = self

        def _patched_start(thread: threading.Thread, *args, **kwargs):
            manager._register_thread(thread, source="thread.start")
            return manager._original_thread_start(thread, *args, **kwargs)

        threading.Thread.start = _patched_start

    def _patch_subprocess(self) -> None:
        if self._original_popen_init is not None:
            return

        self._original_popen_init = subprocess.Popen.__init__
        manager = self

        def _patched_init(proc_self, *args, **kwargs):
            manager._original_popen_init(proc_self, *args, **kwargs)
            manager._register_subprocess(proc_self, args=args, kwargs=kwargs)

        subprocess.Popen.__init__ = _patched_init

    def _patch_multiprocessing(self) -> None:
        if self._original_mp_start is not None:
            return

        self._original_mp_start = mp.process.BaseProcess.start
        manager = self

        def _patched_start(proc_self, *args, **kwargs):
            result = manager._original_mp_start(proc_self, *args, **kwargs)
            manager._register_multiprocessing_process(proc_self)
            return result

        mp.process.BaseProcess.start = _patched_start

    def _restore_patches(self) -> None:
        if self._original_thread_start is not None:
            threading.Thread.start = self._original_thread_start
            self._original_thread_start = None
        if self._original_popen_init is not None:
            subprocess.Popen.__init__ = self._original_popen_init
            self._original_popen_init = None
        if self._original_mp_start is not None:
            mp.process.BaseProcess.start = self._original_mp_start
            self._original_mp_start = None
        if self._original_new_event_loop is not None:
            asyncio.new_event_loop = self._original_new_event_loop
            self._original_new_event_loop = None

    def _start_tracemalloc(self) -> None:
        if not self.tracemalloc_enabled:
            return
        if tracemalloc.is_tracing():
            return
        try:
            tracemalloc.start(self.tracemalloc_frames)
            self._tracemalloc_started_by_hygiene = True
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "runtime_hygiene",
                exc,
                severity="warning",
                action="continued runtime hygiene with tracemalloc disabled",
                extra={"tracemalloc_frames": self.tracemalloc_frames},
            )
            logger.debug("RuntimeHygiene: tracemalloc start failed: %s", exc)

    def _adopt_active_child_processes(self) -> None:
        if self._proc is None:
            return
        try:
            children = list(self._proc.children(recursive=True))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: existing child adoption skipped: %s", exc)
            return

        if not children and _HAS_PSUTIL:
            try:
                parent_pid = int(os.getpid())
                children = [
                    proc
                    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "status"])
                    if int((proc.info or {}).get("ppid") or 0) == parent_pid
                ]
            except _PROCESS_INTROSPECTION_ERRORS + (ConnectionError, TimeoutError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: process_iter child adoption skipped: %s", exc)
                children = []

        tracked_pids = {
            int(record.pid)
            for record in self._process_records.values()
            if record.finished_at is None and getattr(record, "pid", None)
        }
        for child in children:
            if _is_python_resource_tracker_process(child):
                continue
            try:
                pid = int(getattr(child, "pid", 0) or 0)
            except _PROCESS_INTROSPECTION_ERRORS:
                pid = 0
            if pid and pid in tracked_pids:
                continue
            command_parts = _process_cmdline(child)
            name = _process_name(child) or (f"pid:{pid}" if pid else "unknown_child")
            key = -(pid or len(self._process_records) + 1)
            self._process_records[key] = ProcessRecord(
                key=key,
                kind="subprocess",
                name=name,
                source="psutil.adopt_existing_child",
                command=" ".join(str(part) for part in command_parts)[:240] or name,
                pid=pid or None,
            )
            self._process_refs[key] = child
            if pid:
                tracked_pids.add(pid)

    def _register_thread(self, thread: threading.Thread, source: str) -> None:
        key = id(thread)
        record = self._thread_records.get(key)
        if record is None:
            record = ThreadRecord(
                key=key,
                name=thread.name,
                daemon=bool(thread.daemon),
                source=source,
            )
            self._thread_records[key] = record
            self._thread_refs[key] = thread
        else:
            record.name = thread.name
            record.daemon = bool(thread.daemon)

        if getattr(thread, "_aura_runtime_hygiene_wrapped", False):
            return

        original_run = thread.run

        def _wrapped_run(*args, **kwargs):
            record.started_at = time.monotonic()
            record.ident = threading.get_ident()
            try:
                return original_run(*args, **kwargs)
            except _THREAD_RUN_FAILURES as exc:
                record.exception = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record.finished_at = time.monotonic()

        thread.run = _wrapped_run
        thread._aura_runtime_hygiene_wrapped = True

    def _register_subprocess(self, proc: subprocess.Popen, *, args: tuple, kwargs: dict) -> None:
        command = kwargs.get("args")
        if command is None and args:
            command = args[0]
        if isinstance(command, (list, tuple)):
            command_text = " ".join(str(part) for part in command)
        else:
            command_text = str(command)

        key = id(proc)
        self._process_records[key] = ProcessRecord(
            key=key,
            kind="subprocess",
            name=getattr(proc, "args", command_text) if getattr(proc, "args", None) else command_text,
            source="subprocess.Popen",
            command=command_text[:240],
            pid=getattr(proc, "pid", None),
        )
        self._process_refs[key] = proc

    def register_process_handle(
        self,
        proc: Any,
        *,
        kind: str = "multiprocessing",
        name: str | None = None,
        source: str = "explicit_process_owner",
        command: str | None = None,
    ) -> None:
        """Register a child process from the subsystem that owns its lifecycle.

        Runtime hygiene patches process creation, but production model workers
        can be spawned from alternate multiprocessing contexts or after patches
        are temporarily restored during shutdown/restart edges. The owner still
        has the strongest provenance, so explicit registration is the canonical
        path for long-lived worker children.
        """

        pid = getattr(proc, "pid", None)
        for record in self._process_records.values():
            if pid is not None and record.finished_at is None and record.pid == pid:
                record.kind = kind or record.kind
                record.name = str(name or record.name or getattr(proc, "name", kind))
                record.source = str(source or record.source)
                record.command = str(command or record.command or record.name)[:240]
                return
        key = id(proc)
        self._process_records[key] = ProcessRecord(
            key=key,
            kind=str(kind or "multiprocessing"),
            name=str(name or getattr(proc, "name", kind) or kind),
            source=str(source or "explicit_process_owner"),
            command=str(command or name or getattr(proc, "name", kind) or kind)[:240],
            pid=pid,
        )
        self._process_refs[key] = proc

    def _register_multiprocessing_process(self, proc: mp.Process) -> None:
        self.register_process_handle(
            proc,
            kind="multiprocessing",
            name=getattr(proc, "name", "multiprocessing"),
            source="multiprocessing.Process.start",
            command=getattr(proc, "name", "multiprocessing"),
        )

    def _refresh_thread_records(self) -> None:
        now = time.monotonic()
        live_idents = {thread.ident for thread in threading.enumerate()}
        for key, thread in list(self._thread_refs.items()):
            record = self._thread_records.get(key)
            if record is None:
                continue
            record.name = thread.name
            if thread.ident is not None:
                record.ident = thread.ident
            if thread.ident is not None and record.started_at is None:
                record.started_at = now
            if record.ident is not None and record.ident not in live_idents and record.finished_at is None:
                record.finished_at = now

    def _refresh_process_records(self) -> None:
        now = time.monotonic()
        for key, proc in list(self._process_refs.items()):
            record = self._process_records.get(key)
            if record is None:
                continue
            if hasattr(proc, "poll"):
                try:
                    return_code = proc.poll()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: subprocess poll failed: %s", exc)
                    return_code = None
                if return_code is not None:
                    record.exit_code = int(return_code)
                    record.finished_at = record.finished_at or now
            elif hasattr(proc, "is_alive"):
                try:
                    alive = proc.is_alive()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: multiprocessing liveness failed: %s", exc)
                    alive = False
                if not alive:
                    record.exit_code = getattr(proc, "exitcode", None)
                    record.finished_at = record.finished_at or now
                else:
                    record.pid = getattr(proc, "pid", record.pid)
            elif hasattr(proc, "is_running"):
                try:
                    alive = bool(proc.is_running())
                    status = proc.status() if alive else "stopped"
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: adopted child liveness failed: %s", exc)
                    alive = False
                    status = "error"
                if not alive or status == "zombie":
                    record.finished_at = record.finished_at or now

    def _thread_summary(self) -> dict[str, Any]:
        now = time.monotonic()
        active = 0
        active_non_daemon = 0
        stale_non_daemon = 0
        sample: list[dict[str, Any]] = []
        for record in self._thread_records.values():
            if record.finished_at is not None:
                continue
            active += 1
            if not record.daemon:
                active_non_daemon += 1
                if record.age_s(now) >= self.stale_thread_age_s:
                    stale_non_daemon += 1
                    sample.append(
                        {
                            "name": record.name,
                            "age_s": round(record.age_s(now), 1),
                            "source": record.source,
                        }
                    )
        return {
            "active": active,
            "active_non_daemon": active_non_daemon,
            "stale_non_daemon": stale_non_daemon,
            "sample": sample[:5],
        }

    def _process_summary(self) -> dict[str, Any]:
        active_registered = 0
        active_subprocesses = 0
        active_multiprocessing = 0
        active_registered_pids = set()
        for record in self._process_records.values():
            if record.finished_at is not None:
                continue
            active_registered += 1
            if getattr(record, "pid", None):
                try:
                    active_registered_pids.add(int(record.pid))
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "runtime_hygiene",
                        exc,
                        severity="warning",
                        action="ignored malformed registered process pid during hygiene summary",
                    )
                    logger.debug("RuntimeHygiene: malformed registered pid %r: %s", record.pid, exc)
            if record.kind == "subprocess":
                active_subprocesses += 1
            elif record.kind == "multiprocessing":
                active_multiprocessing += 1
        rogue_children = 0
        owned_descendants = 0
        rogue_samples: list[dict[str, Any]] = []
        if self._proc is not None:
            try:
                active_children = list(self._proc.children(recursive=True))
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation('runtime_hygiene', exc)
                logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
                active_children = []
            child_by_pid = {
                pid: child
                for child in active_children
                if (pid := _process_pid(child)) > 0
            }
            for child in active_children:
                child_pid = _process_pid(child)
                if child_pid in active_registered_pids or _is_python_resource_tracker_process(child):
                    continue
                if (
                    _process_ppid(child) == int(os.getpid())
                    and _is_python_multiprocessing_spawn_process(child)
                ):
                    self.register_process_handle(
                        child,
                        kind="multiprocessing",
                        name=_process_name(child) or "multiprocessing.spawn",
                        source="psutil.adopt_during_summary",
                        command=" ".join(_process_cmdline(child))[:240],
                    )
                    active_registered += 1
                    active_multiprocessing += 1
                    if child_pid > 0:
                        active_registered_pids.add(child_pid)
                    continue
                if self._is_owned_descendant_process(
                    child,
                    active_registered_pids=active_registered_pids,
                    child_by_pid=child_by_pid,
                ):
                    owned_descendants += 1
                    continue
                rogue_children += 1
                if len(rogue_samples) < 5:
                    rogue_samples.append(
                        {
                            "pid": child_pid or None,
                            "ppid": _process_ppid(child) or None,
                            "name": _process_name(child)[:80],
                            "command": " ".join(_process_cmdline(child))[:160],
                        }
                    )
        return {
            "active_registered": max(0, active_registered),
            "active_subprocesses": max(0, active_subprocesses),
            "active_multiprocessing": max(0, active_multiprocessing),
            "owned_descendant_processes": max(0, owned_descendants),
            "rogue_child_processes": max(0, rogue_children),
            "rogue_samples": rogue_samples,
        }

    def _is_owned_descendant_process(
        self,
        proc: Any,
        *,
        active_registered_pids: set[int],
        child_by_pid: dict[int, Any],
    ) -> bool:
        """Return true when a recursive child belongs to a registered owner.

        ``psutil.children(recursive=True)`` returns grandchildren as well as
        direct children. A registered model worker can legitimately spawn a
        short-lived helper below it; that helper should be visible in telemetry
        without being misclassified as an unregistered root process. The walk is
        bounded and stops at Aura's current process so an unrelated child still
        fails the hygiene check.
        """

        current_pid = int(os.getpid())
        seen: set[int] = set()
        parent_pid = _process_ppid(proc)
        for _ in range(16):
            if parent_pid <= 0 or parent_pid == current_pid or parent_pid in seen:
                return False
            if parent_pid in active_registered_pids:
                return True
            seen.add(parent_pid)
            parent = child_by_pid.get(parent_pid)
            if parent is None:
                return False
            parent_pid = _process_ppid(parent)
        return False

    def _memory_summary(self) -> dict[str, Any]:
        if len(self._samples) < self.memory_growth_window:
            latest = self._samples[-1] if self._samples else None
            return {
                "sustained_growth": False,
                "transient_growth": False,
                "message": "warming_up",
                "rss_mb": round((latest.rss_bytes if latest else 0) / (1024 * 1024), 1),
                "delta_mb": 0.0,
            }

        window = list(self._samples)[-self.memory_growth_window:]
        first = window[0]
        last = window[-1]
        delta_bytes = last.rss_bytes - first.rss_bytes
        delta_mb = delta_bytes / (1024 * 1024)
        baseline = max(float(first.rss_bytes), 1.0)
        positive_steps = sum(1 for idx in range(1, len(window)) if window[idx].rss_bytes >= window[idx - 1].rss_bytes)
        growth_ratio = delta_bytes / baseline
        sustained_growth = (
            delta_mb >= self.memory_growth_min_delta_mb
            or (growth_ratio >= self.memory_growth_ratio and positive_steps >= len(window) - 1)
        )
        transient_model_growth = []
        if sustained_growth:
            transient_model_growth = self._active_local_model_activity()
        message = "memory_growth_stable"
        if sustained_growth and transient_model_growth:
            message = "Transient RSS growth during local model activity: " + ", ".join(transient_model_growth[:3])
            sustained_growth = False
        elif sustained_growth:
            message = f"Sustained RSS growth detected (+{delta_mb:.1f}MB over {len(window)} samples)"
        return {
            "sustained_growth": sustained_growth,
            "transient_growth": bool(transient_model_growth),
            "message": message,
            "rss_mb": round(last.rss_bytes / (1024 * 1024), 1),
            "delta_mb": round(delta_mb, 1),
        }

    def _active_local_model_activity(self) -> list[str]:
        active: list[str] = []
        now = time.time()
        registries = (
            ("core.brain.llm.mlx_client", "_CLIENTS"),
            ("core.brain.llm.local_server_client", "_SERVER_CLIENTS"),
        )
        for module_name, registry_attr in registries:
            try:
                module = __import__(module_name, fromlist=[registry_attr])
                registry_items = _snapshot_mapping_items(getattr(module, registry_attr, {}) or {})
            except (RuntimeError, AttributeError, TypeError):
                continue

            for client_path, client in registry_items:
                if client is None or not hasattr(client, "get_lane_status"):
                    continue
                try:
                    lane = client.get_lane_status()
                except (OSError, ConnectionError, TimeoutError):
                    continue
                state = str(lane.get("state", "") or "").strip().lower()
                current_request = float(lane.get("current_request_started_at", 0.0) or 0.0)
                if bool(lane.get("warmup_in_flight")) or current_request > 0.0 or state in {
                    "spawning",
                    "handshaking",
                    "warming",
                    "recovering",
                }:
                    active.append(f"{os.path.basename(str(client_path))}:{state or 'active'}")
                    continue

                recent_activity_at = max(
                    float(lane.get("last_ready_at", 0.0) or 0.0),
                    float(lane.get("last_progress_at", 0.0) or 0.0),
                    float(lane.get("last_transition_at", 0.0) or 0.0),
                )
                if (
                    self.model_activity_grace_s > 0.0
                    and recent_activity_at > 0.0
                    and (now - recent_activity_at) <= self.model_activity_grace_s
                ):
                    active.append(f"{os.path.basename(str(client_path))}:recent")
        return active

    def _count_child_processes(self) -> int:
        if self._proc is None:
            return 0
        try:
            return len(self._proc.children(recursive=True))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            record_degradation('runtime_hygiene', exc)
            logger.debug("RuntimeHygiene: child process scan failed: %s", exc)
            return 0

    async def _cleanup_child_processes(self) -> None:
        async def _cleanup_one(proc: Any) -> None:
            if _is_python_resource_tracker_process(proc):
                return
            if hasattr(proc, "poll"):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(
                                asyncio.to_thread(proc.wait, self.process_shutdown_timeout_s),
                                timeout=self.process_shutdown_timeout_s + 0.25,
                            )
                        except (RuntimeError, TimeoutError, AttributeError, subprocess.TimeoutExpired):
                            proc.kill()
                            try:
                                await asyncio.wait_for(
                                    asyncio.to_thread(proc.wait, 0.2),
                                    timeout=0.3,
                                )
                            except (RuntimeError, TimeoutError, AttributeError, subprocess.TimeoutExpired) as exc:
                                record_degradation(
                                    "runtime_hygiene",
                                    exc,
                                    severity="warning",
                                    action="subprocess did not confirm exit after kill",
                                )
                                logger.debug("RuntimeHygiene: subprocess kill wait failed: %s", exc)
                except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: subprocess cleanup failed: %s", exc)
            elif hasattr(proc, "is_alive"):
                try:
                    if proc.is_alive():
                        proc.terminate()
                        await asyncio.wait_for(
                            asyncio.to_thread(proc.join, self.process_shutdown_timeout_s),
                            timeout=self.process_shutdown_timeout_s + 0.25,
                        )
                        if proc.is_alive():
                            proc.kill()
                            try:
                                await asyncio.wait_for(asyncio.to_thread(proc.join, 0.2), timeout=0.3)
                            except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                                record_degradation(
                                    "runtime_hygiene",
                                    exc,
                                    severity="warning",
                                    action="multiprocessing child did not confirm exit after kill",
                                )
                                logger.debug("RuntimeHygiene: multiprocessing kill join failed: %s", exc)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: multiprocessing cleanup failed: %s", exc)
            elif _HAS_PSUTIL and hasattr(proc, "is_running"):
                try:
                    if proc.is_running():
                        proc.terminate()
                        try:
                            await asyncio.wait_for(
                                asyncio.to_thread(proc.wait, self.process_shutdown_timeout_s),
                                timeout=self.process_shutdown_timeout_s + 0.25,
                            )
                        except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError):
                            if proc.is_running():
                                proc.kill()
                                try:
                                    await asyncio.wait_for(
                                        asyncio.to_thread(proc.wait, 0.2),
                                        timeout=0.3,
                                    )
                                except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                                    record_degradation(
                                        "runtime_hygiene",
                                        exc,
                                        severity="warning",
                                        action="psutil child did not confirm exit after kill",
                                    )
                                    logger.debug("RuntimeHygiene: psutil kill wait failed: %s", exc)
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation('runtime_hygiene', exc)
                    logger.debug("RuntimeHygiene: psutil child cleanup failed: %s", exc)

        cleanup_coros = [_cleanup_one(proc) for proc in list(self._process_refs.values())]
        if not cleanup_coros:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*cleanup_coros, return_exceptions=True),
                timeout=max(1.0, self.process_shutdown_timeout_s + 0.75),
            )
        except TimeoutError as exc:
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="warning",
                action="continued shutdown after bounded concurrent child-process cleanup timed out",
                enforce_failure_policy=False,
            )

    async def _join_non_daemon_threads(self) -> None:
        join_candidates: list[threading.Thread] = []
        for thread in list(self._thread_refs.values()):
            if thread.daemon:
                continue
            if not thread.is_alive():
                continue
            if thread.ident == threading.get_ident():
                continue
            join_candidates.append(thread)
        if not join_candidates:
            return

        selected = join_candidates[: self.max_thread_joins_per_shutdown]
        skipped = join_candidates[self.max_thread_joins_per_shutdown :]
        if skipped:
            record_degradation(
                "runtime_hygiene_shutdown",
                RuntimeError(f"{len(skipped)} non-daemon thread(s) left for owner shutdown"),
                severity="warning",
                action=(
                    "bounded runtime hygiene shutdown thread joins; remaining live threads "
                    "are left to their owning services"
                ),
                extra={
                    "skipped_threads": [getattr(thread, "name", "unknown") for thread in skipped[:10]],
                    "selected_count": len(selected),
                    "skipped_count": len(skipped),
                },
                enforce_failure_policy=False,
            )

        join_coros = [
            asyncio.to_thread(self._join_thread_if_not_current, thread, self.thread_join_timeout_s)
            for thread in selected
        ]
        if not join_coros:
            return
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*join_coros, return_exceptions=True),
                timeout=max(0.5, self.thread_join_timeout_s + 0.3),
            )
        except TimeoutError as exc:
            record_degradation(
                "runtime_hygiene_shutdown",
                exc,
                severity="warning",
                action="continued shutdown after bounded concurrent thread join timed out",
                extra={
                    "selected_count": len(selected),
                    "skipped_count": len(skipped),
                    "selected_threads": [getattr(thread, "name", "unknown") for thread in selected[:10]],
                },
                enforce_failure_policy=False,
            )
            return
        for result in results:
            if isinstance(result, (RuntimeError, AttributeError, TypeError, ValueError)):
                record_degradation(
                    "runtime_hygiene_shutdown",
                    result,
                    severity="warning",
                    action="continued shutdown after a bounded thread join failed",
                    enforce_failure_policy=False,
                )
                logger.debug("RuntimeHygiene: thread join failed: %s", result)

    @staticmethod
    def _join_thread_if_not_current(thread: threading.Thread, timeout_s: float) -> None:
        if thread.ident == threading.get_ident():
            return
        thread.join(timeout_s)


_runtime_hygiene: RuntimeHygieneManager | None = None


def get_runtime_hygiene() -> RuntimeHygieneManager:
    global _runtime_hygiene
    if _runtime_hygiene is None:
        _runtime_hygiene = RuntimeHygieneManager()
    return _runtime_hygiene
