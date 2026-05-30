from core.runtime.errors import record_degradation
from core.runtime.atomic_writer import atomic_write_text
import fcntl
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger("Aura.Utils.Singleton")

_LOCK_FD: int | None = None
_LOCK_NAME: str | None = None
T = TypeVar("T")


def singleton(cls: type[T]) -> Callable[..., T]:
    """
    Decorator to make a class a singleton.
    Usage:
    @singleton
    class MyClass: ...
    """
    instances: dict[type[T], T] = {}

    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def instance_lock_path(lock_name: str = "singleton") -> Path:
    return Path.home() / ".aura" / "locks" / f"{lock_name}.lock"


def instance_lock_metadata_path(lock_name: str = "singleton") -> Path:
    return Path.home() / ".aura" / "locks" / f"{lock_name}.lock.meta.json"


def parse_instance_lock_pid(raw: str) -> int | None:
    """Parse both legacy PID-only locks and future structured lock payloads."""
    text = str(raw or "").strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    try:
        return int(first_line)
    except ValueError:
        pass
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    try:
        return int(payload.get("pid"))
    except (AttributeError, TypeError, ValueError):
        return None


def read_instance_lock_pid(lock_name: str = "singleton") -> int | None:
    try:
        return parse_instance_lock_pid(instance_lock_path(lock_name).read_text(encoding="utf-8"))
    except OSError:
        return None


def read_instance_lock_metadata(lock_name: str = "singleton") -> dict[str, Any]:
    try:
        payload = json.loads(instance_lock_metadata_path(lock_name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_process_metadata(lock_name: str, pid: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema": "aura.instance_lock.v1",
        "lock_name": lock_name,
        "pid": int(pid),
        "written_at": time.time(),
        "cwd": os.getcwd(),
        "cmdline": list(sys.argv),
    }
    try:
        import psutil

        proc = psutil.Process(pid)
        metadata.update(
            {
                "create_time": float(proc.create_time()),
                "ppid": int(proc.ppid()),
                "username": str(proc.username()),
            }
        )
    except Exception as exc:  # pragma: no cover - optional diagnostic metadata
        metadata["identity_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _write_instance_lock_metadata(lock_name: str, pid: int) -> None:
    path = instance_lock_metadata_path(lock_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path,
            json.dumps(_current_process_metadata(lock_name, pid), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        record_degradation("singleton", exc)
        logger.warning("Failed to write instance lock metadata for '%s': %s", lock_name, exc)


def acquire_instance_lock(lock_name: str = "singleton", skip_lock: bool = False) -> None:
    """
    Ensure only one instance of a specific Aura component/process is running.
    Uses a file lock in ~/.aura/locks/.
    
    Args:
        lock_name: Name of the lock (e.g., "orchestrator", "genesis").
        skip_lock: If True, bypass the lock check.
    """
    if skip_lock:
        return

    global _LOCK_FD, _LOCK_NAME
    if _LOCK_FD is not None and _LOCK_NAME == lock_name:
        return

    # Standardize lock path
    lock_file = instance_lock_path(lock_name)
    lock_dir = lock_file.parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open with O_RDWR to allow writing the PID.
        # Add O_CLOEXEC to prevent FD inheritance by child processes.
        # This ensures the lock is freed on main process crash even if children (Reaper/GUI) survive.
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
            
        _LOCK_FD = os.open(str(lock_file), flags, 0o600)
        
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Lock is held by another process. Read the PID.
            try:
                os.lseek(_LOCK_FD, 0, os.SEEK_SET)
                pid_bytes = os.read(_LOCK_FD, 4096)
                pid = parse_instance_lock_pid(pid_bytes.decode()) or 0
                if pid <= 0:
                    raise ValueError("invalid lock pid")
                
                # Check if the process is actually running
                try:
                    os.kill(pid, 0)
                    message = f"⚠️  Aura ({lock_name}) is already running (PID: {pid})."
                    logger.error(message)
                    print(message)
                    raise SystemExit(0)
                except OSError:
                    # Process is dead. Reclaim the stale lock.
                    logger.warning(
                        "🔓 Stale lock found for dead PID %d. Reclaiming lock for %s.",
                        pid, lock_name,
                    )
                    # Close and reopen to get a fresh FD
                    try:
                        os.close(_LOCK_FD)
                    except OSError as _exc:
                        logger.debug("Suppressed %s in core.utils.singleton: %s", type(_exc).__name__, _exc)
                    flags = os.O_CREAT | os.O_RDWR
                    if hasattr(os, "O_CLOEXEC"):
                        flags |= os.O_CLOEXEC
                    _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                    try:
                        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        # If we still can't get it, a child process holds the FD.
                        # Last resort: unlink and recreate.
                        try:
                            lock_file.unlink()
                            os.close(_LOCK_FD)
                            _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                            fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            logger.info("🔓 Stale lock reclaimed via unlink+recreate.")
                        except (OSError, IOError) as reclaim_exc:
                            message = f"⚠️  Failed to reclaim stale lock for {lock_name}: {reclaim_exc}"
                            logger.error(message)
                            print(message)
                            raise SystemExit(1) from None
            except (ValueError, UnicodeDecodeError):
                # PID file is corrupt. Reclaim unconditionally.
                logger.warning("🔓 Corrupt lock file for %s. Reclaiming.", lock_name)
                try:
                    os.close(_LOCK_FD)
                    lock_file.unlink(missing_ok=True)
                    flags = os.O_CREAT | os.O_RDWR
                    if hasattr(os, "O_CLOEXEC"):
                        flags |= os.O_CLOEXEC
                    _LOCK_FD = os.open(str(lock_file), flags, 0o600)
                    fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError):
                    message = f"⚠️  Aura ({lock_name}) is already running in another window."
                    logger.error(message)
                    print(message)
                    raise SystemExit(0) from None
        
        # Lock acquired. Write current PID.
        os.ftruncate(_LOCK_FD, 0)
        os.write(_LOCK_FD, f"{os.getpid()}\n".encode())
        os.fsync(_LOCK_FD)
        _LOCK_NAME = lock_name
        _write_instance_lock_metadata(lock_name, os.getpid())
        
        logger.info("🔒 Instance lock acquired: %s (PID: %d)", lock_name, os.getpid())
        
    except (OSError, IOError) as e:
        record_degradation('singleton', e)
        logger.warning("Failed to acquire single-instance lock for '%s': %s", lock_name, e)


def release_instance_lock() -> None:
    """Explicitly release the lock (usually handled by process exit)."""
    global _LOCK_FD, _LOCK_NAME
    if _LOCK_FD is not None:
        try:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            os.close(_LOCK_FD)
            _LOCK_FD = None
            _LOCK_NAME = None
        except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
            record_degradation('singleton', _e)
            logger.debug('Ignored Exception in singleton.py: %s', _e)
