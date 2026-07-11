#!/usr/bin/env python3
"""Bounded external proof for Aura's monotonic shutdown contract.

Each case starts a fresh real runtime, injects SIGINT/SIGTERM at a named
lifecycle boundary, and verifies the process from outside. A pass requires the
terminal root-exit receipt, all canonical shutdown phases exactly once, no
surviving post-latch admission, no descendant process, a free port, and a
reacquirable singleton lock. The driver hard-kills only its own process group
after a bounded failure; it never uses broad process-name cleanup.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.live_boot_proof import (  # noqa: E402
    build_safe_boot_env,
    live_proof_rss_abort_mb,
    resolve_launch_python,
)

SHUTDOWN_PHASES = (
    "output_flush",
    "memory_commit",
    "state_vault",
    "actors",
    "model_runtime",
    "event_bus",
    "task_supervisor",
)
PHASE_MARKER = "ShutdownCoordinator: phase started (phase={phase} "
READY_MARKER = "Registry Locked. Aura Ready (Desktop)."
FOREGROUND_MARKER = "Foreground chat reservation acquired"
PROBE_START_MARKER = "Lifecycle probe hold started (target={target} "

FAILURE_MARKERS = (
    "Traceback (most recent call last)",
    "[DEGRADATION]",
    "RuntimeWarning:",
    "Task exception was never retrieved",
    "Exception in callback",
    "Root process exit receipt persistence failed",
    "shutdown complete (clean=False",
    "FATAL BOOT ERROR",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    trigger_marker: str
    first_signal: signal.Signals
    repeat_signal: signal.Signals | None = None
    repeat_marker: str | None = None
    repeat_delay_s: float = 0.2
    probe_target: str | None = None
    start_foreground_chat: bool = False
    trigger_timeout_s: float = 180.0
    shutdown_timeout_s: float = 140.0


CASE_SPECS: dict[str, CaseSpec] = {
    "launcher_bootstrap": CaseSpec(
        name="launcher_bootstrap",
        trigger_marker="Instance lock acquired: orchestrator",
        first_signal=signal.SIGTERM,
        trigger_timeout_s=45.0,
    ),
    "orchestrator_boot_repeated": CaseSpec(
        name="orchestrator_boot_repeated",
        trigger_marker="Orchestrator boot beginning",
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
    ),
    "ready_repeated": CaseSpec(
        name="ready_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGINT,
        repeat_signal=signal.SIGTERM,
    ),
    "state_vault_repeated": CaseSpec(
        name="state_vault_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
        repeat_marker=PROBE_START_MARKER.format(target="coordinator:state_vault"),
        probe_target="coordinator:state_vault",
    ),
    "model_runtime_repeated": CaseSpec(
        name="model_runtime_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGINT,
        repeat_signal=signal.SIGTERM,
        repeat_marker=PROBE_START_MARKER.format(target="coordinator:model_runtime"),
        probe_target="coordinator:model_runtime",
    ),
    "container_repeated": CaseSpec(
        name="container_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
        repeat_marker=PROBE_START_MARKER.format(target="container"),
        probe_target="container",
    ),
    "root_finalization_repeated": CaseSpec(
        name="root_finalization_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGINT,
        repeat_signal=signal.SIGTERM,
        repeat_marker=PROBE_START_MARKER.format(target="root_finalization"),
        probe_target="root_finalization",
    ),
    "active_foreground_repeated": CaseSpec(
        name="active_foreground_repeated",
        trigger_marker=READY_MARKER,
        first_signal=signal.SIGTERM,
        repeat_signal=signal.SIGINT,
        start_foreground_chat=True,
        repeat_delay_s=0.25,
        trigger_timeout_s=300.0,
    ),
}

DEFAULT_CASES = tuple(CASE_SPECS)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    create_time: float
    cmdline: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "create_time": self.create_time,
            "cmdline": list(self.cmdline),
        }


class LogCursor:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.text = ""

    def refresh(self) -> str:
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return ""
        if chunk:
            self.text += chunk
        return chunk


def _is_aura_main_process(cmdline: list[str] | tuple[str, ...]) -> bool:
    if not cmdline:
        return False
    executable = Path(str(cmdline[0])).name.lower()
    if "python" not in executable:
        return False
    return any(Path(str(arg)).name == "aura_main.py" for arg in cmdline[1:])


def _running_aura_main_pids() -> list[int]:
    pids: list[int] = []
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = list(proc.info.get("cmdline") or [])
        except psutil.Error:
            continue
        if _is_aura_main_process(cmdline):
            pids.append(proc.pid)
    return sorted(pids)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", int(port))) != 0


def _orchestrator_lock_available() -> tuple[bool, str]:
    lock_path = Path.home() / ".aura" / "locks" / "orchestrator.lock"
    if not lock_path.exists():
        return True, "lock file absent"
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError as exc:
        return False, f"open failed: {type(exc).__name__}: {exc}"
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False, "lock is still held"
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True, "lock reacquired"
    finally:
        os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _nested_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def evaluate_terminal_report(
    report: dict[str, Any],
    *,
    root_pid: int,
    expected_first_reason: str,
    minimum_signal_requests: int,
) -> dict[str, bool]:
    components = _nested_dict(report, "components")
    coordinator = _nested_dict(components, "coordinator")
    container = _nested_dict(components, "container")
    hygiene = _nested_dict(components, "runtime_hygiene")
    final_tasks = _nested_dict(components, "final_tasks")
    request = _nested_dict(report, "request")
    admission = _nested_dict(report, "admission")
    counts = _nested_dict(admission, "counts")
    root_exit = _nested_dict(report, "root_exit")
    root_resources = _nested_dict(root_exit, "resources")
    verdict = _nested_dict(report, "verdict")

    return {
        "schema": report.get("schema") == "aura.shutdown_verdict.v1",
        "root_pid": report.get("pid") == root_pid,
        "terminal_stage": report.get("stage") == "root_process_exit",
        "final": report.get("final") is True,
        "terminal_receipt_once": report.get("terminal_receipt_sequence") == 1,
        "verdict_clean": verdict.get("clean") is True and not verdict.get("blockers"),
        "first_reason": request.get("first_reason") == expected_first_reason,
        "signal_request_count": int(request.get("request_count", 0) or 0)
        >= minimum_signal_requests,
        "all_phases_once": coordinator.get("completed_phases") == list(SHUTDOWN_PHASES),
        "coordinator_clean": coordinator.get("clean") is True,
        "container_clean": container.get("clean") is True,
        "runtime_hygiene_clean": hygiene.get("clean") is True,
        "final_tasks_empty": final_tasks.get("count") == 0,
        "no_shutdown_resurrection": int(counts.get("survived", 0) or 0) == 0,
        "root_lock_released": root_exit.get("lock_released") is True,
        "root_finalizers_complete": root_exit.get("multiprocessing_finalizers_completed")
        is True,
        "root_logging_flushed": root_exit.get("logging_shutdown_completed") is True,
        "root_resources_clean": root_resources.get("clean") is True,
        "root_exit_zero": root_exit.get("exit_code") == 0,
    }


class SignalMatrixCase:
    def __init__(
        self,
        *,
        spec: CaseSpec,
        port: int,
        artifact_dir: Path,
        rss_abort_mb: float | None,
    ) -> None:
        self.spec = spec
        self.port = int(port)
        self.artifact_dir = artifact_dir.resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.artifact_dir / "runtime_stdout.log"
        self.transcript_path = self.artifact_dir / "events.jsonl"
        self.verdict_path = self.artifact_dir / "case_verdict.json"
        self.report_path = self.artifact_dir / "shutdown_report.json"
        self.cursor = LogCursor(self.stdout_path)
        self.proc: subprocess.Popen[bytes] | None = None
        self.stdout_handle: Any = None
        self.seen_identities: dict[tuple[int, float], ProcessIdentity] = {}
        self.peak_rss_mb = 0.0
        self.rss_abort_mb_override = rss_abort_mb
        self.rss_abort_mb = 0.0
        self.started_monotonic = time.monotonic()
        self.chat_thread: threading.Thread | None = None
        self.chat_result: dict[str, Any] = {}
        self.signal_events: list[dict[str, Any]] = []

    def record(self, event: str, **detail: Any) -> None:
        payload = {
            "at_unix": time.time(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "event": event,
            **detail,
        }
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        print(f"[{self.spec.name}] {event}: {detail.get('summary', '')}", flush=True)

    def _capture_identity(self, proc: psutil.Process) -> None:
        try:
            identity = ProcessIdentity(
                pid=proc.pid,
                create_time=float(proc.create_time()),
                cmdline=tuple(str(item) for item in proc.cmdline()),
            )
        except psutil.Error:
            return
        self.seen_identities[(identity.pid, identity.create_time)] = identity

    def sample_process_tree(self) -> float:
        if self.proc is None:
            return 0.0
        total = 0
        try:
            root = psutil.Process(self.proc.pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            processes = []
        for process in processes:
            self._capture_identity(process)
            try:
                total += int(process.memory_info().rss)
            except psutil.Error:
                continue
        rss_mb = total / (1024 * 1024)
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        if self.rss_abort_mb > 0 and rss_mb > self.rss_abort_mb:
            raise RuntimeError(
                f"process tree RSS {rss_mb:.0f}MB exceeded {self.rss_abort_mb:.0f}MB ceiling"
            )
        return rss_mb

    def wait_for_marker(self, marker: str, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            self.cursor.refresh()
            if marker in self.cursor.text:
                self.record("marker_observed", summary=marker, marker=marker)
                return True
            if self.proc is not None and self.proc.poll() is not None:
                self.cursor.refresh()
                return marker in self.cursor.text
            self.sample_process_tree()
            time.sleep(0.05)
        self.cursor.refresh()
        return marker in self.cursor.text

    def send_signal(self, sig: signal.Signals, *, role: str) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError(f"cannot send {sig.name}: root process already exited")
        os.kill(self.proc.pid, sig)
        event = {
            "role": role,
            "signal": sig.name,
            "at_unix": time.time(),
            "log_offset": self.cursor.offset,
        }
        self.signal_events.append(event)
        self.record("signal_sent", summary=f"{role} {sig.name}", **event)

    def start_foreground_chat(self) -> None:
        def _request() -> None:
            started = time.monotonic()
            try:
                with httpx.Client(
                    timeout=180.0,
                    headers={
                        "X-Aura-Surface": "desktop-ui",
                        "X-Aura-Require-CognitiveEngine": "true",
                    },
                ) as client:
                    response = client.post(
                        f"http://127.0.0.1:{self.port}/api/chat",
                        json={
                            "message": (
                                "Reason carefully about a three-stage reliability migration and "
                                "state the strongest failure invariant for each stage."
                            ),
                            "session_id": f"shutdown-matrix-{self.spec.name}",
                        },
                    )
                self.chat_result.update(
                    status_code=response.status_code,
                    body=response.text[:1000],
                )
            except (httpx.HTTPError, OSError) as exc:
                self.chat_result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self.chat_result["duration_seconds"] = round(time.monotonic() - started, 3)

        self.chat_thread = threading.Thread(
            target=_request,
            name=f"shutdown-matrix-chat-{self.spec.name}",
            daemon=True,
        )
        self.chat_thread.start()
        self.record("foreground_chat_started", summary="POST /api/chat dispatched")

    def spawn(self) -> None:
        env = build_safe_boot_env(os.environ, mode="desktop")
        env["PYTHONUNBUFFERED"] = "1"
        env["AURA_SHUTDOWN_REPORT_PATH"] = str(self.report_path)
        env["AURA_ARTIFACTS_DIR"] = str(self.artifact_dir / "runtime_artifacts")
        env["AURA_LIVENESS_HEARTBEAT_FILE"] = str(
            self.artifact_dir / "liveness_heartbeat.json"
        )
        env["AURA_REAPER_MANIFEST"] = str(self.artifact_dir / "reaper_manifest.json")
        if self.spec.probe_target:
            env["AURA_SHUTDOWN_PROBE_ENABLED"] = "1"
            env["AURA_SHUTDOWN_PROBE_TARGET"] = self.spec.probe_target
            env["AURA_SHUTDOWN_PROBE_HOLD_SECONDS"] = "0.75"
        derived_rss = live_proof_rss_abort_mb(env)
        self.rss_abort_mb = (
            min(float(self.rss_abort_mb_override), derived_rss)
            if self.rss_abort_mb_override is not None
            else derived_rss
        )
        self.stdout_handle = self.stdout_path.open("wb")
        command = [
            resolve_launch_python(),
            "aura_main.py",
            "--desktop",
            "--port",
            str(self.port),
        ]
        self.proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=self.stdout_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.record(
            "spawned",
            summary=f"pid={self.proc.pid} port={self.port}",
            pid=self.proc.pid,
            port=self.port,
            command=command,
            rss_abort_mb=self.rss_abort_mb,
        )
        self.sample_process_tree()

    def wait_for_exit(self) -> bool:
        if self.proc is None:
            return False
        deadline = time.monotonic() + self.spec.shutdown_timeout_s
        while time.monotonic() < deadline:
            self.cursor.refresh()
            self.sample_process_tree()
            if self.proc.poll() is not None:
                self.cursor.refresh()
                return True
            time.sleep(0.1)
        return self.proc.poll() is not None

    def hard_kill_owned_group(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                self.proc.kill()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass

    def residual_processes(self) -> list[ProcessIdentity]:
        residuals: list[ProcessIdentity] = []
        for identity in self.seen_identities.values():
            try:
                process = psutil.Process(identity.pid)
                if abs(float(process.create_time()) - identity.create_time) > 0.001:
                    continue
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    residuals.append(identity)
            except psutil.Error:
                continue
        return sorted(residuals, key=lambda item: item.pid)

    def _failure_lines(self) -> list[str]:
        lines: list[str] = []
        for line in self.cursor.text.splitlines():
            if any(marker.lower() in line.lower() for marker in FAILURE_MARKERS):
                lines.append(line[:1000])
        return lines[:50]

    def _probe_window(self) -> dict[str, Any]:
        target = self.spec.probe_target
        if not target:
            return {}
        escaped_target = re.escape(target)
        starts = re.findall(
            rf"Lifecycle probe hold started \(target={escaped_target} "
            rf"started_at_unix=([0-9.]+) hold_seconds=([0-9.]+)\)",
            self.cursor.text,
        )
        completions = re.findall(
            rf"Lifecycle probe hold completed \(target={escaped_target} "
            rf"completed_at_unix=([0-9.]+)\)",
            self.cursor.text,
        )
        repeat_events = [
            event for event in self.signal_events if event.get("role") == "repeat"
        ]
        started_at = float(starts[0][0]) if len(starts) == 1 else 0.0
        completed_at = float(completions[0]) if len(completions) == 1 else 0.0
        repeat_at = (
            float(repeat_events[0].get("at_unix", 0.0) or 0.0)
            if len(repeat_events) == 1
            else 0.0
        )
        return {
            "target": target,
            "started_count": len(starts),
            "completed_count": len(completions),
            "started_at_unix": started_at,
            "completed_at_unix": completed_at,
            "repeat_at_unix": repeat_at,
            "repeat_inside_hold": bool(
                started_at > 0.0
                and completed_at >= started_at
                and started_at <= repeat_at <= completed_at
            ),
        }

    def finalize_verdict(self, *, trigger_seen: bool, repeat_seen: bool, exited: bool) -> bool:
        if self.stdout_handle is not None:
            self.stdout_handle.flush()
            self.stdout_handle.close()
            self.stdout_handle = None
        self.cursor.refresh()
        if self.chat_thread is not None:
            self.chat_thread.join(timeout=5.0)
        time.sleep(1.5)

        report = _read_json(self.report_path)
        root_pid = self.proc.pid if self.proc is not None else -1
        expected_first_reason = f"desktop_signal:{self.spec.first_signal.name}"
        report_checks = evaluate_terminal_report(
            report,
            root_pid=root_pid,
            expected_first_reason=expected_first_reason,
            minimum_signal_requests=len(self.signal_events),
        )
        residuals = self.residual_processes()
        lock_available, lock_detail = _orchestrator_lock_available()
        phase_counts = {
            phase: self.cursor.text.count(PHASE_MARKER.format(phase=phase))
            for phase in SHUTDOWN_PHASES
        }
        history = list((self.report_path.parent / "shutdown_history").glob("*.json"))
        failure_lines = self._failure_lines()
        probe_window = self._probe_window()
        external_checks = {
            "trigger_seen": trigger_seen,
            "repeat_trigger_seen": repeat_seen,
            "root_exited": exited,
            "root_exit_code_zero": self.proc is not None and self.proc.returncode == 0,
            "phase_markers_once": all(count == 1 for count in phase_counts.values()),
            "history_receipt_once": len(history) == 1,
            "no_residual_processes": not residuals,
            "port_free": _port_is_free(self.port),
            "singleton_lock_available": lock_available,
            "runtime_stream_clean": not failure_lines,
        }
        if self.spec.start_foreground_chat:
            try:
                chat_body = json.loads(str(self.chat_result.get("body") or ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                chat_body = {}
            external_checks["foreground_shutdown_outcome_explicit"] = bool(
                self.chat_result.get("status_code") in {200, 503}
                and isinstance(chat_body, dict)
                and chat_body.get("status") == "runtime_shutdown"
            )
        if self.spec.probe_target:
            external_checks.update(
                probe_hold_started_once=probe_window.get("started_count") == 1,
                probe_hold_completed_once=probe_window.get("completed_count") == 1,
                repeat_inside_probe_hold=probe_window.get("repeat_inside_hold") is True,
            )
        checks = {**report_checks, **external_checks}
        passed = all(checks.values())
        verdict = {
            "schema": "aura.shutdown_signal_case.v1",
            "case": self.spec.name,
            "passed": passed,
            "root_pid": root_pid,
            "root_returncode": self.proc.returncode if self.proc is not None else None,
            "port": self.port,
            "checks": checks,
            "signal_events": self.signal_events,
            "probe_window": probe_window,
            "phase_marker_counts": phase_counts,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "rss_abort_mb": round(self.rss_abort_mb, 1),
            "residual_processes": [item.as_dict() for item in residuals],
            "lock_detail": lock_detail,
            "failure_lines": failure_lines,
            "chat_result": self.chat_result,
            "shutdown_report_path": str(self.report_path),
            "stdout_path": str(self.stdout_path),
            "history_paths": [str(path) for path in history],
        }
        self.verdict_path.write_text(
            json.dumps(verdict, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.record(
            "verdict",
            summary=f"passed={passed} failed={[key for key, ok in checks.items() if not ok]}",
            passed=passed,
            failed_checks=[key for key, ok in checks.items() if not ok],
        )
        return passed

    def run(self) -> bool:
        trigger_seen = False
        repeat_seen = self.spec.repeat_signal is None
        exited = False
        try:
            self.spawn()
            trigger_seen = self.wait_for_marker(
                self.spec.trigger_marker,
                self.spec.trigger_timeout_s,
            )
            if not trigger_seen:
                raise TimeoutError(f"trigger marker not observed: {self.spec.trigger_marker}")

            if self.spec.start_foreground_chat:
                self.start_foreground_chat()
                trigger_seen = self.wait_for_marker(FOREGROUND_MARKER, 180.0)
                if not trigger_seen:
                    raise TimeoutError(f"foreground marker not observed: {FOREGROUND_MARKER}")

            self.send_signal(self.spec.first_signal, role="first")
            if self.spec.repeat_signal is not None:
                if self.spec.repeat_marker:
                    repeat_seen = self.wait_for_marker(
                        self.spec.repeat_marker,
                        self.spec.shutdown_timeout_s,
                    )
                    if not repeat_seen:
                        raise TimeoutError(
                            f"repeat marker not observed: {self.spec.repeat_marker}"
                        )
                else:
                    time.sleep(max(0.0, self.spec.repeat_delay_s))
                    repeat_seen = True
                self.send_signal(self.spec.repeat_signal, role="repeat")

            exited = self.wait_for_exit()
            if not exited:
                raise TimeoutError(
                    f"root process did not exit within {self.spec.shutdown_timeout_s:.1f}s"
                )
        except (OSError, RuntimeError, TimeoutError, psutil.Error) as exc:
            self.record("case_error", summary=f"{type(exc).__name__}: {exc}")
            self.hard_kill_owned_group()
        finally:
            if self.proc is not None and self.proc.poll() is None:
                self.hard_kill_owned_group()
            if self.proc is not None:
                self.proc.poll()
        return self.finalize_verdict(
            trigger_seen=trigger_seen,
            repeat_seen=repeat_seen,
            exited=exited,
        )


def _parse_cases(values: list[str] | None) -> list[str]:
    if not values or values == ["all"]:
        return list(DEFAULT_CASES)
    selected: list[str] = []
    for value in values:
        for name in str(value).split(","):
            normalized = name.strip()
            if not normalized:
                continue
            if normalized == "all":
                return list(DEFAULT_CASES)
            if normalized not in CASE_SPECS:
                raise ValueError(f"unknown case '{normalized}'")
            if normalized not in selected:
                selected.append(normalized)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Case name, comma-separated names, or 'all' (default: all)",
    )
    parser.add_argument("--base-port", type=int, default=8870)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--rss-abort-mb", type=float)
    parser.add_argument("--list", action="store_true", help="List case names and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name in DEFAULT_CASES:
            print(name)
        return 0
    try:
        selected = _parse_cases(args.cases)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    existing = _running_aura_main_pids()
    if existing:
        print(
            f"Refusing shutdown matrix while another aura_main.py is running: {existing}",
            file=sys.stderr,
        )
        return 2
    occupied = [args.base_port + index for index in range(len(selected)) if not _port_is_free(args.base_port + index)]
    if occupied:
        print(f"Refusing shutdown matrix because ports are occupied: {occupied}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_root = (
        args.artifact_dir
        or ROOT / "artifacts" / "current" / f"shutdown_signal_matrix_{stamp}"
    ).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results: list[dict[str, object]] = []
    for index, name in enumerate(selected):
        case = SignalMatrixCase(
            spec=CASE_SPECS[name],
            port=args.base_port + index,
            artifact_dir=artifact_root / name,
            rss_abort_mb=args.rss_abort_mb,
        )
        passed = case.run()
        results.append(
            {
                "case": name,
                "passed": passed,
                "verdict_path": str(case.verdict_path),
            }
        )
        if _running_aura_main_pids():
            print("A case left an aura_main.py process; refusing to contaminate later cases.")
            break

    matrix_passed = len(results) == len(selected) and all(bool(item["passed"]) for item in results)
    summary = {
        "schema": "aura.shutdown_signal_matrix.v1",
        "passed": matrix_passed,
        "selected_cases": selected,
        "completed_cases": results,
        "duration_seconds": round(time.monotonic() - started, 3),
        "artifact_root": str(artifact_root),
    }
    summary_path = artifact_root / "MATRIX_VERDICT.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Shutdown signal matrix passed={matrix_passed} "
        f"duration={summary['duration_seconds']}s artifact={summary_path}",
        flush=True,
    )
    return 0 if matrix_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
