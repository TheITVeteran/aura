"""Pre-launch cleanup for stale Aura desktop runtime processes."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.process_identity import (  # noqa: E402
    process_invokes_python_script,
    select_script_process_tree,
)
from core.runtime.resource_observation import get_resource_observer  # noqa: E402
from core.utils.singleton import read_instance_lock_metadata, read_instance_lock_pid  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.IroncladCleanup")

_AURA_RUNTIME_SCRIPTS = (
    ROOT / "aura_main.py",
    ROOT / "core" / "brain" / "llm" / "mlx_worker.py",
    ROOT / "interface" / "gui_actor.py",
)
_NATIVE_LAUNCHER_SUFFIX = "Aura.app/Contents/MacOS/aura-launcher"


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _recent_grace_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("AURA_CLEANUP_RECENT_GRACE_S", "45") or 45))
    except (TypeError, ValueError):
        return 45.0


def _verified_live_runtime_pid() -> int | None:
    """Return the live runtime PID from the orchestrator lock, if trustworthy."""
    pid = read_instance_lock_pid("orchestrator")
    if not pid or pid <= 0:
        return None
    process = get_resource_observer().process(pid)
    if process is None or process.status.lower() in {"dead", "zombie"}:
        return None
    metadata = read_instance_lock_metadata("orchestrator")
    expected_cwd = str(metadata.get("cwd") or "")
    try:
        cwd_matches = not expected_cwd or Path(process.cwd).resolve() == Path(
            expected_cwd
        ).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        cwd_matches = False
    if cwd_matches and process_invokes_python_script(
        process,
        expected_script=ROOT / "aura_main.py",
    ):
        return pid
    return None


def _kill_stale_processes() -> None:
    live_pid = None if _truthy_env("AURA_CLEANUP_FORCE") else _verified_live_runtime_pid()
    if live_pid is not None:
        logger.info(
            "Verified live Aura runtime detected (PID: %s); skipping aggressive pre-launch process cleanup.",
            live_pid,
        )
        return
    observer = get_resource_observer()
    table = observer.process_table()
    if not table.available:
        logger.warning(
            "Process table unavailable; refusing broad stale-runtime cleanup: %s",
            table.error or "unknown error",
        )
        return
    observations = select_script_process_tree(
        table.processes,
        expected_scripts=_AURA_RUNTIME_SCRIPTS,
        protected_pids=(os.getpid(), os.getppid()),
    )
    if not observations:
        logger.info("No exact stale Aura runtime process tree was observed.")
        return

    try:
        import psutil
    except ImportError:
        logger.warning("psutil unavailable; refusing unverified stale-runtime signals.")
        return

    handles = []
    for observation in observations:
        try:
            observed_create_time = float(observation.create_time or 0.0)
            current_observation = observer.process(observation.pid)
            if current_observation is None or current_observation.status.lower() in {
                "dead",
                "zombie",
            }:
                continue
            current_create_time = float(current_observation.create_time or 0.0)
            if (
                observed_create_time > 0.0
                and abs(current_create_time - observed_create_time) > 0.01
            ):
                logger.warning(
                    "Skipping PID %s because creation time changed after observation.",
                    observation.pid,
                )
                continue
            process = psutil.Process(observation.pid)
            handles.append(process)
        except psutil.Error:
            continue

    for process in handles:
        try:
            logger.info("Terminating exact stale Aura process PID: %s", process.pid)
            process.terminate()
        except psutil.Error as exc:
            logger.debug("Aura process PID %s already exited: %s", process.pid, exc)
    _gone, alive = psutil.wait_procs(handles, timeout=5.0)
    for process in alive:
        try:
            logger.warning("Killing unresponsive exact Aura process PID: %s", process.pid)
            process.kill()
        except psutil.Error as exc:
            logger.debug("Aura process PID %s unavailable for kill: %s", process.pid, exc)
    if alive:
        psutil.wait_procs(alive, timeout=2.0)


def _is_native_launcher_process(proc) -> bool:
    try:
        info = getattr(proc, "info", {}) or {}
        exe = str(info.get("exe") or getattr(proc, "exe", "") or "")
        cmdline = [
            str(item)
            for item in (info.get("cmdline") or getattr(proc, "cmdline", ()) or ())
        ]
        name = str(info.get("name") or getattr(proc, "name", "") or "")
    except (AttributeError, TypeError, ValueError):
        return False
    first_arg = cmdline[0] if cmdline else ""
    return (
        exe.endswith(_NATIVE_LAUNCHER_SUFFIX)
        or first_arg.endswith(_NATIVE_LAUNCHER_SUFFIX)
        or (name == "aura-launcher" and _NATIVE_LAUNCHER_SUFFIX in " ".join([exe, *cmdline]))
    )


def _kill_stale_native_launchers() -> None:
    """Terminate stale native launchers without killing the caller's launcher UI."""

    if not _truthy_env("AURA_CLEANUP_FORCE") and _verified_live_runtime_pid() is not None:
        logger.info(
            "Verified live Aura runtime detected; preserving native Aura.app launcher bridge."
        )
        return

    current_pid = os.getpid()
    parent_pid = os.getppid()
    grace_s = _recent_grace_seconds()
    now = time.time()
    table = get_resource_observer().process_table()
    if not table.available:
        logger.warning(
            "Process table unavailable; preserving native launchers: %s",
            table.error or "unknown error",
        )
        return
    candidate_pids: list[int] = []
    for process in table.processes:
        pid = process.pid
        if pid in {current_pid, parent_pid}:
            continue
        if not _is_native_launcher_process(process):
            continue
        age_s = now - process.create_time
        if not _truthy_env("AURA_CLEANUP_FORCE") and age_s < grace_s:
            logger.info(
                "Preserving recent Aura native launcher PID %s (age %.1fs < %.1fs).",
                pid,
                age_s,
                grace_s,
            )
            continue
        candidate_pids.append(pid)

    try:
        import psutil
    except ImportError:
        return
    candidates = []
    for pid in candidate_pids:
        try:
            candidates.append(psutil.Process(pid))
        except psutil.Error:
            continue

    for proc in candidates:
        try:
            logger.info("Terminating stale Aura native launcher PID: %s", proc.pid)
            proc.terminate()
        except psutil.Error as exc:
            logger.debug("Native launcher PID %s already exited: %s", getattr(proc, "pid", "?"), exc)
    for proc in candidates:
        try:
            proc.wait(timeout=3.0)
        except psutil.TimeoutExpired:
            try:
                proc.kill()
            except psutil.Error as exc:
                logger.debug("Native launcher PID %s unavailable for kill: %s", proc.pid, exc)
        except psutil.Error:
            continue


def _reset_stale_locks() -> None:
    live_pid = None if _truthy_env("AURA_CLEANUP_FORCE") else _verified_live_runtime_pid()
    if live_pid is not None:
        logger.info(
            "Verified live Aura runtime detected (PID: %s); preserving lock directory.",
            live_pid,
        )
        return
    lock_dir = Path.home() / ".aura" / "locks"
    if not lock_dir.exists():
        lock_dir.mkdir(parents=True, exist_ok=True)
        return
    logger.info("Resetting stale lock directory: %s", lock_dir)
    try:
        shutil.rmtree(lock_dir)
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Failed to reset locks: %s", exc)


def main() -> None:
    logger.info("Starting Aura pre-launch cleanup.")
    _kill_stale_processes()
    _kill_stale_native_launchers()
    _reset_stale_locks()
    time.sleep(2)
    logger.info("Aura pre-launch cleanup complete.")


if __name__ == "__main__":
    main()
