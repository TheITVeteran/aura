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

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from core.utils.singleton import read_instance_lock_metadata, read_instance_lock_pid  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.IroncladCleanup")

_CLEANUP_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)
_AURA_PROCESS_PATTERNS = (
    "aura_main.py",
    "mlx_worker.py",
    "gui_actor.py",
    "simulate_200.py",
)
_NATIVE_LAUNCHER_SUFFIX = "Aura.app/Contents/MacOS/aura-launcher"


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _verified_live_runtime_pid() -> int | None:
    """Return the live runtime PID from the orchestrator lock, if trustworthy."""
    pid = read_instance_lock_pid("orchestrator")
    if not pid or pid <= 0:
        return None
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return None
        command = " ".join(proc.cmdline() or []).lower()
        metadata = read_instance_lock_metadata("orchestrator")
        expected_cwd = str(metadata.get("cwd") or "")
        cwd_matches = not expected_cwd or Path(proc.cwd()).resolve() == Path(expected_cwd).resolve()
        if cwd_matches and "aura_main.py" in command:
            return pid
    except (
        OSError,
        ProcessLookupError,
        RuntimeError,
        TypeError,
        ValueError,
        psutil.AccessDenied,
        psutil.NoSuchProcess,
        psutil.ZombieProcess,
    ):
        return None
    return None


def _kill_stale_processes() -> None:
    live_pid = None if _truthy_env("AURA_CLEANUP_FORCE") else _verified_live_runtime_pid()
    if live_pid is not None:
        logger.info(
            "Verified live Aura runtime detected (PID: %s); skipping aggressive pre-launch process cleanup.",
            live_pid,
        )
        return
    for pattern in _AURA_PROCESS_PATTERNS:
        try:
            logger.info("Terminating stale Aura process pattern: %s", pattern)
            result = get_subprocess_gateway().run(
                ["pkill", "-9", "-f", pattern],
                cwd=ROOT,
                timeout=10,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:aura_cleanup:pkill",
            )
            if result.returncode not in {0, 1}:
                logger.warning("pkill failed for %s: %s", pattern, result.stderr.strip())
        except _CLEANUP_RECOVERABLE_ERRORS as exc:
            logger.warning("Failed to kill %s: %s: %s", pattern, type(exc).__name__, exc)


def _is_native_launcher_process(proc) -> bool:
    try:
        info = getattr(proc, "info", {}) or {}
        exe = str(info.get("exe") or "")
        cmdline = [str(item) for item in (info.get("cmdline") or [])]
        name = str(info.get("name") or "")
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

    try:
        import psutil
    except ImportError:
        return

    current_pid = os.getpid()
    parent_pid = os.getppid()
    candidates = []
    for proc in psutil.process_iter(["pid", "exe", "cmdline", "name"]):
        try:
            pid = int(proc.info.get("pid") or proc.pid)
            if pid in {current_pid, parent_pid}:
                continue
            if _is_native_launcher_process(proc):
                candidates.append(proc)
        except (psutil.Error, OSError, TypeError, ValueError):
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
