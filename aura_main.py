#!/usr/bin/env python3
"""
Aura Main Entry Point
---------------------
Standardized, single-entry launcher for CLI, Server, Desktop, and Watchdog modes.
Replaces: aura_launcher.py, aura_desktop.py, run_aura.py, run_aura_loop.py, and reboot.py.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

if sys.version_info < (3, 12):  # noqa: UP036 - boot contract asserts a clear runtime guard.
    raise SystemExit("Aura requires Python 3.12+")

import httpx

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested, request_shutdown
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.singleton import (
    acquire_instance_lock,
    instance_lock_metadata_path,
    parse_instance_lock_pid,
    read_instance_lock_metadata,
    release_instance_lock,
)
from core.utils.task_tracker import get_task_tracker

# QUAL-07: Define logger early so venv injection logging works.
logger = logging.getLogger("Aura.Main")

_RUNTIME_LOCK_CLAIMED = False
_AURA_MAIN_DEGRADATION_KEY = "aura_main"
_AURA_MAIN_BOUNDARY_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    subprocess.SubprocessError,
    httpx.HTTPError,
)

# Install global task supervision before subsystems spawn background tasks.
try:
    import core.utils.asyncio_patch  # noqa: F401
except _AURA_MAIN_BOUNDARY_ERRORS as exc:
    record_degradation(_AURA_MAIN_DEGRADATION_KEY, exc)

# Phase 31: Native Apple Silicon Resilience Fixes
# 0. Force 'spawn' on macOS to prevent Cocoa/XPC deadlocks in child actors
if sys.platform == "darwin":
    os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
    os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    # PyAV's bundled libavdevice and OpenCV's bundled libavdevice both register
    # the Objective-C classes AVFFrameReceiver / AVFAudioReceiver.  The objc
    # runtime unconditionally prints a "Class … is implemented in both …"
    # warning to stderr when this happens, which floods our logs on every boot
    # (this is the source of the recurring "homebrew" error the user reported).
    # The conflict is benign in practice — objc keeps the first registration,
    # and nothing in Aura uses the AVFoundation capture classes — so we
    # eagerly load both libraries here with stderr muted, which absorbs the
    # one-shot duplicate-class notice and leaves subsequent transitive imports
    # silent.
    native_media_preload = os.environ.get("AURA_PRELOAD_NATIVE_MEDIA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    safe_desktop_context = any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("AURA_SAFE_BOOT_DESKTOP", "AURA_LAUNCHED_FROM_APP", "AURA_HEADLESS")
    ) or any(arg in sys.argv for arg in ("--headless", "--desktop", "--gui-window"))

    if native_media_preload and not safe_desktop_context:
        try:
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            _saved_stderr = os.dup(2)
            try:
                os.dup2(_devnull_fd, 2)
                import av as _av  # noqa: F401  (ordering matters — av first)
                import cv2 as _cv2  # noqa: F401  (eager load to absorb dup warning)
            finally:
                os.dup2(_saved_stderr, 2)
                os.close(_devnull_fd)
                os.close(_saved_stderr)
        except ImportError as exc:
            logger.debug("Optional AV/OpenCV preload skipped: %s", exc)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            # Never let the dylib suppression block boot.
            record_degradation("aura_main", exc)
    else:
        logger.debug(
            "Optional AV/OpenCV preload skipped: disabled for stable desktop boot."
        )

# Early .env loading — ensures AURA_LOCAL_BACKEND and other env vars are
# available BEFORE module-level code in model_registry.py reads os.getenv().
# Without this, pydantic's env_file loading happens too late.
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv as _load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        _load_dotenv(_env_path, override=False)

# 1. Path Resolution & Environment Locking (Radical Fix)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _foreground_only_runtime() -> bool:
    return _env_flag("AURA_FOREGROUND_ONLY", False)


def _bounded_memory_ceiling_mb(
    total_mb: float,
    requested_mb: Any | None = None,
    *,
    absolute_ceiling_mb: float = 46080.0,
    ceiling_fraction: float = 0.70,
    floor_mb: float = 8192.0,
) -> float:
    """Return a host-safe memory kill ceiling.

    Environment overrides are useful for lab runs, but a stale or excessive
    value must not let the live desktop process grow until macOS kills the
    whole machine. Unsafe overrides require an explicit opt-in flag.
    """

    try:
        total = max(float(total_mb), floor_mb)
    except (TypeError, ValueError, OverflowError):
        total = 65536.0
    safe_ceiling = min(float(absolute_ceiling_mb), max(float(floor_mb), total * float(ceiling_fraction)))
    if requested_mb is None:
        return safe_ceiling
    try:
        requested = max(float(floor_mb), float(requested_mb))
    except (TypeError, ValueError, OverflowError):
        return safe_ceiling
    if _env_flag("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", False):
        return requested
    return min(requested, safe_ceiling)


def _should_start_keep_awake_controller() -> bool:
    """Start macOS keep-awake only from the root Aura process.

    Multiprocessing spawn imports this module inside child actors as
    ``__mp_main__``. Starting keep-awake at import time from those children
    leaks orphan ``caffeinate`` helpers and can keep actor processes alive after
    shutdown, so the controller is root-process only.
    """

    if any(arg in {"-h", "--help", "--stop"} for arg in sys.argv[1:]):
        return False
    try:
        process_name = multiprocessing.current_process().name
    except _AURA_MAIN_BOUNDARY_ERRORS:
        process_name = "unknown"
    return process_name == "MainProcess" and __name__ != "__mp_main__"


def _should_force_root_process_exit_after_main(args: Any) -> bool:
    """Return true for long-lived root runtimes after their shutdown completes."""

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if _env_flag("AURA_DISABLE_HARD_EXIT_AFTER_MAIN", False):
        return False
    if not _should_start_keep_awake_controller():
        return False
    return not any(
        bool(getattr(args, name, False))
        for name in ("cli", "watchdog", "gui_window", "philosophy")
    )


def _run_multiprocessing_finalizers_before_hard_exit(timeout_s: float = 3.0) -> None:
    """Give multiprocessing a bounded chance to unregister queues/semaphores."""

    done = threading.Event()

    def _run_finalizers() -> None:
        try:
            import multiprocessing.util as mp_util

            mp_util._exit_function()
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation(
                "aura_main",
                exc,
                action="continued root hard-exit after multiprocessing finalizer failed",
            )
            logger.debug("Multiprocessing finalizer failed before root hard-exit: %s", exc)
        finally:
            done.set()

    thread = threading.Thread(
        target=_run_finalizers,
        name="aura-multiprocessing-finalizers",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=max(0.0, float(timeout_s)))
    if not done.is_set():
        record_degradation(
            "aura_main",
            TimeoutError("multiprocessing finalizer timeout before hard exit"),
            action="continued root hard-exit after bounded multiprocessing cleanup timed out",
        )
        logger.warning(
            "Multiprocessing finalizers did not finish within %.1fs before root hard-exit.",
            timeout_s,
        )


def _finalize_root_runtime_process_exit(args: Any, exit_code: int = 0) -> None:
    if not _should_force_root_process_exit_after_main(args):
        return
    try:
        from core.runtime.keep_awake import get_keep_awake_controller

        get_keep_awake_controller().stop()
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            "aura_main",
            exc,
            action="continued root process exit after keep-awake stop failed",
        )
        logger.debug("Keep-awake final stop failed during root process exit: %s", exc)
    try:
        release_instance_lock()
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            "aura_main",
            exc,
            action="continued root process exit after lock release failed",
        )
        logger.debug("Instance lock release failed during root process exit: %s", exc)
    _run_multiprocessing_finalizers_before_hard_exit()
    logger.info("Root runtime shutdown complete; exiting process with code %d.", exit_code)
    os._exit(int(exit_code))


def _profile_is_proof(profile: str | None, ready_label: str | None = None) -> bool:
    """Return True for canonical proof/evaluation boot profiles."""

    tokens = {
        str(profile or "").strip().lower(),
        str(ready_label or "").strip().lower(),
    }
    return bool(tokens & {"proof", "eval", "evaluation", "validation", "benchmark"}) or any(
        any(marker in token for marker in ("proof", "validation", "benchmark"))
        for token in tokens
        if token
    )


def _activate_proof_runtime_policy(profile: str | None, ready_label: str | None = None) -> None:
    """Make proof-profile boots enforce the same runtime policy everywhere.

    Proof runners use the normal Aura boot path, but they need stricter lane
    contracts so background/autonomy work cannot silently spin up a lower local
    model while a primary-lane proof is being measured.
    """

    if not _profile_is_proof(profile, ready_label):
        return
    os.environ["AURA_PROOF_RUN"] = "1"
    os.environ.setdefault("AURA_PROOF_MODEL_TIER", "primary")
    # Proof/evaluation boots must still use the canonical Aura runtime, but
    # unsolicited background autonomy cannot compete with sealed evaluator turns.
    os.environ["AURA_ENABLE_PROACTIVE_SYSTEMS"] = "0"
    os.environ["AURA_ENABLE_RESEARCH_CYCLE"] = "0"
    os.environ["AURA_ENABLE_SENSORIMOTOR_GROUNDING"] = "0"
    os.environ["AURA_ENABLE_PROACTIVE_VISION"] = "0"


def _record_main_degradation(exc: BaseException, message: str, *args: Any) -> None:
    record_degradation(_AURA_MAIN_DEGRADATION_KEY, exc)
    logger.warning(message, *args, exc)


def _env_float(name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            _record_main_degradation(exc, "Invalid float environment value for %s=%r; using %.2f: %s", name, raw, default)
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value

# [STABILITY] Force the execution context to the absolute path of the current venv
# This prevents the "ModuleNotFoundError" when pip is in the venv but the script runs elsewhere.
VENV_PATH = PROJECT_ROOT / ".venv"
if not VENV_PATH.exists():
    VENV_PATH = PROJECT_ROOT / ".venv_aura"

if VENV_PATH.exists():
    # Scan for any python3.x directory to handle version mismatches (e.g. venv is 3.12, system is 3.14)
    lib_dir = VENV_PATH / "lib"
    if lib_dir.exists():
        curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for py_dir in lib_dir.glob("python3.*"):
            if py_dir.name != curr_ver:
                logger.debug("⏭️  Skipping venv injection for mismatched version: %s (Current: %s)", py_dir.name, curr_ver)
                continue
                
            site_packages = py_dir / "site-packages"
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
                import site
                site.addsitedir(str(site_packages))
                logger.info("📍 Injected venv site-packages: %s", site_packages)

# Desktop-safe boot should keep the main Aura process off the in-process
# MLX/Metal path. The managed LLM runtimes use their own subprocesses.
try:
    from core.runtime.desktop_boot_safety import configure_inprocess_mlx_runtime

    _mlx_runtime = configure_inprocess_mlx_runtime()
    if _mlx_runtime.get("device") == "cpu":
        logger.info(
            "🛡️ In-process MLX pinned to CPU (%s).",
            _mlx_runtime.get("reason", "guard"),
        )
    elif _mlx_runtime.get("device") == "metal":
        logger.info(
            "⚡ In-process MLX Metal retained (%s).",
            _mlx_runtime.get("reason", "enabled"),
        )
except _AURA_MAIN_BOUNDARY_ERRORS as _mlx_guard_exc:
    logger.debug("In-process MLX boot guard unavailable: %s", _mlx_guard_exc)

# Phase 31: Native Apple Silicon Resilience Fixes
# 1. Address AVFFrameReceiver conflict (cv2 vs av/PyAV) on macOS
# This prevents the "AVFFrameReceiver: ... is already established" crash
if sys.platform == "darwin":
    os.environ["OPENCV_VIDEOIO_AVFOUNDATION_USE_FRAME_RECEIVER"] = "0"
    os.environ["PYAV_SKIP_AVF_FRAME_RECEIVER"] = "1"

# Strip PyInstaller matplotlib bloat in frozen builds
if getattr(sys, 'frozen', False):
    os.environ.pop("MPLBACKEND", None)

# 2. Bootstrap Configuration & Logging
try:
    from core.config import config
    from core.logging_config import setup_logging
    # Centralized logging setup - always include log_dir for persistence
    setup_logging(log_dir=config.paths.log_dir)
    logger = logging.getLogger("Aura.Main")
    if _should_start_keep_awake_controller():
        try:
            from core.runtime.keep_awake import start_from_environment

            _keep_awake_status = start_from_environment()
            if _keep_awake_status.active:
                logger.info("Aura keep-awake assertion active: pid=%s", _keep_awake_status.pid)
        except _AURA_MAIN_BOUNDARY_ERRORS as _keep_awake_exc:
            record_degradation("aura_main", _keep_awake_exc)
            logger.warning("Aura keep-awake setup failed: %s", _keep_awake_exc)
except _AURA_MAIN_BOUNDARY_ERRORS as exc:
    # Minimal fallback logging if core is broken
    import traceback
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("Aura.Main")
    record_degradation(_AURA_MAIN_DEGRADATION_KEY, exc)
    logger.error("❌ BOOTSTRAP FAILURE: Could not load core configuration.")
    logger.error(traceback.format_exc())
    config = None # Ensure NameError is avoided

# Category 11: Reliability Hardening
_supervisor_tree: Any | None = None

def get_supervisor_tree() -> Any:
    global _supervisor_tree
    if _supervisor_tree is None:
        from core.supervisor.tree import SupervisionTree
        _supervisor_tree = SupervisionTree()
    return _supervisor_tree

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def validate_security_config():
    """Verify that we aren't exposing a public API without authentication."""
    if config is None:
        logger.warning("⚠️ Config unavailable — skipping security validation (bootstrap failure).")
        return
    internal_only = getattr(config.security, "internal_only_mode", False)
    api_token = config.api_token
    
    # If host is NOT localhost and no token is set, we are in a dangerous state
    # Note: We check this even if the user passed --host 127.0.0.1 because
    # the server.py might override it or be proxied.
    if not internal_only and not api_token:
        from core.exceptions import SecurityConfigError
        logger.critical("🚨 SECURITY VIOLATION: Public API access enabled but AURA_API_TOKEN is unset.")
        logger.critical("   To fix this: Set AURA_API_TOKEN in .env or run with AURA_INTERNAL_ONLY=1")
        raise SecurityConfigError("Public API access enabled without AURA_API_TOKEN")


def check_environment():
    """Verify system readiness."""
    logger.info("🔍 Verifying Environment Integrity...")
    logger.info("📍 RUNTIME PATH Diagnostic:")
    logger.info("   • __file__: %s", __file__)
    logger.info("   • sys.executable: %s", sys.executable)
    logger.info("   • sys.path: %s", sys.path)
    try:
        import core
        logger.info("   • core.__file__: %s", core.__file__)
    except _AURA_MAIN_BOUNDARY_ERRORS as e:
        record_degradation('aura_main', e)
        logger.error("   • core import failed: %s", e)
    
    if config is None:
        logger.error("❌ Environment check aborted: Configuration not loaded.")
        raise RuntimeError("Configuration not loaded")

    # Perplexity Audit Fix: Fail-closed security validation
    validate_security_config()

    # Validate autonomous repair registry before self-modification can resume.
    registry_file = config.paths.data_dir / "selfmod" / "pending_patch_registry.jsonl"
    if registry_file.exists():
        logger.info("🛠️  Validating self-modification repair registry...")
        try:
            from core.self_modification.repair_registry import validate_repair_registry

            validate_repair_registry(registry_file)
        except (ImportError, json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            record_degradation("aura_main", exc)
            raise RuntimeError(f"Self-modification repair registry is not trustworthy: {registry_file}") from exc

    # Ensure home directory exists
    config.paths.create_directories()

def kill_port(port: int, pattern: str = "aura"):
    """Terminate Aura-owned processes on a port.

    Port 10003 is Aura's private supervisor lane and may be force-cleared.
    Shared development ports such as 8000 stay pattern-limited so unrelated
    local servers are not silently killed.
    """
    try:
        import psutil
    except ImportError:
        logger.warning("psutil missing - skipping advanced port cleanup.")
        return

    force_all_ports = {10003}
    shared_ports = {8000}
    force_all = port in force_all_ports
    if port in shared_ports:
        logger.warning(
            "Port %s is a shared development port; cleanup is limited to processes matching pattern '%s'.",
            port,
            pattern,
        )

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    pid = proc.pid
                    name = proc.info.get("name") or ""
                    cmd_str = ""
                    try:
                        cmd_str = " ".join(proc.cmdline() or []).lower()
                    except (psutil.Error, PermissionError, SystemError, OSError) as exc:
                        logger.debug("Skipping cmdline inspection for PID %s during port cleanup: %s", pid, exc)

                    should_kill = force_all or (pattern in cmd_str or pattern in name.lower())
                    
                    if should_kill:
                        if force_all:
                            logger.warning(
                                "Force-clearing Aura-private port %s by terminating PID %s (%s): %s",
                                port,
                                pid,
                                name,
                                cmd_str[:200],
                            )
                        logger.info("Terminating process %s (%s) on port %s...", pid, name, port)
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            logger.warning("Process %s resistant to SIGTERM. Sending SIGKILL.", pid)
                            proc.kill()
                    else:
                        logger.warning(
                            "Leaving non-Aura process %s (%s) on shared port %s untouched.",
                            pid,
                            name,
                            port,
                        )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, PermissionError, SystemError, OSError) as exc:
            logger.debug("Skipping process during port cleanup: %s", exc)

def clean_artifacts():
    """Purge stale bytecode and temporary caches."""
    logger.info("🧹 Purging runtime artifacts...")
    for p in PROJECT_ROOT.rglob("__pycache__"):
        try:
            shutil.rmtree(p)
        except OSError as exc:
            logger.debug("Unable to remove cache directory %s: %s", p, exc)
    for p in PROJECT_ROOT.rglob("*.pyc"):
        try:
            p.unlink()
        except OSError as exc:
            logger.debug("Unable to remove bytecode file %s: %s", p, exc)


def _select_preferred_launcher_python(current_executable: str | None = None) -> Path | None:
    """Prefer a stable Homebrew Python 3.12 launcher over shimmed venv paths."""
    if sys.platform != "darwin":
        return None

    current_raw = Path(current_executable or sys.executable)
    current_raw_str = str(current_raw)
    if "/.venv/" not in current_raw_str and "/.venv_aura/" not in current_raw_str:
        return None

    candidates: list[Path] = []
    explicit = os.environ.get("AURA_PREFERRED_PYTHON")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/opt/homebrew/opt/python@3.12/bin/python3.12"),
            Path("/opt/homebrew/bin/python3.12"),
        ]
    )

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            current_resolved = current_raw.resolve()
        except FileNotFoundError:
            continue
        if resolved.exists() and resolved != current_resolved:
            return candidate
    return None


def _launcher_python_executable() -> str:
    preferred = os.environ.get("AURA_PREFERRED_PYTHON", "").strip()
    if preferred and Path(preferred).exists():
        return preferred
    return sys.executable


REAPER_MANIFEST_ENV = "AURA_REAPER_MANIFEST"
LEGACY_REAPER_MANIFEST = Path(tempfile.gettempdir()) / "aura_reaper_manifest.json"
REAPER_MANIFEST_DIR = Path.home() / ".aura" / "run" / "reaper"


def _new_reaper_manifest_path() -> Path:
    runtime_id = os.environ.get("AURA_RUNTIME_ID", "").strip()
    if not runtime_id:
        runtime_id = f"{int(time.time())}-{os.getpid()}"
        os.environ["AURA_RUNTIME_ID"] = runtime_id
    return REAPER_MANIFEST_DIR / f"manifest-{runtime_id}.json"


def _ensure_reaper_manifest_env() -> Path:
    """Force every launcher/process surface for this boot to share one manifest path."""
    raw_path = os.environ.get(REAPER_MANIFEST_ENV, "").strip()
    if raw_path and Path(raw_path).expanduser() != LEGACY_REAPER_MANIFEST:
        manifest_path = Path(raw_path).expanduser()
    else:
        manifest_path = _new_reaper_manifest_path()
        os.environ[REAPER_MANIFEST_ENV] = str(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return manifest_path


def _maybe_relaunch_with_preferred_python():
    if os.environ.get("AURA_SKIP_PREFERRED_PYTHON_RELAUNCH") == "1":
        return

    preferred = _select_preferred_launcher_python()
    if not preferred:
        return

    logger.warning("🔁 Relaunching Aura with preferred interpreter: %s", preferred)
    env = os.environ.copy()
    env["AURA_SKIP_PREFERRED_PYTHON_RELAUNCH"] = "1"
    env["AURA_PREFERRED_PYTHON"] = str(preferred)
    env["AURA_LOCAL_BACKEND"] = "llama_cpp"
    os.execve(str(preferred), [str(preferred), *sys.argv], env)

# ---------------------------------------------------------------------------
# Shims & Compatibility
# ---------------------------------------------------------------------------
try:
    from core.cognitive_integration import CognitiveIntegrationLayer
    CognitiveIntegration = CognitiveIntegrationLayer # Legacy Alias shim
except ImportError:
    logger.debug("CognitiveIntegrationLayer unavailable; legacy alias not installed.")

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

async def bootstrap_aura(orchestrator: Any):
    """Initialize background services using the Resilient Boot sequence."""
    from core.bus.actor_bus import create_actor_bus
    from core.container import ServiceContainer
    from core.ops.resilient_boot import ResilientBoot
    
    # Register core services early to satisfy boot dependencies
    supervisor = get_supervisor_tree()
    ServiceContainer.register_instance("supervisor", supervisor)
    
    actor_bus = create_actor_bus() # Main bus for orchestrator
    ServiceContainer.register_instance("actor_bus", actor_bus, failure_policy="degrade_with_receipt")
    actor_bus.start()

    # Guarded stage-based ignition
    # Explicitly link internal refs to ensure property lookups match initialized instances
    orchestrator._actor_bus = actor_bus
    orchestrator._supervisor_tree = supervisor
    tracker = get_task_tracker()
    tracker.install_loop_hygiene(asyncio.get_running_loop())
    
    boot = ResilientBoot(orchestrator)
    # [STABILITY] Wait for ignition to complete before proceeding
    # This ensures all core services and state repository are ready.
    status = await boot.ignite()
    logger.info("🛡️ [BOOT] Resilient Ignition finished with status: %s", status)
    
    # Final interface check
    if hasattr(orchestrator, "kernel_interface") and orchestrator.kernel_interface:
        for _ in range(5):
            if orchestrator.kernel_interface.is_ready():
                break
            await asyncio.sleep(1.0)
    
    # Register supervisor tree in container (Redundant but safe)
    # ServiceContainer.register_instance("supervisor", supervisor)
    
    # Post-boot background tasks
    from core.utils.memory_monitor import AppleSiliconMemoryMonitor
    mem_monitor = AppleSiliconMemoryMonitor()
    try:
        ServiceContainer.register_instance("memory_monitor", mem_monitor, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("Memory monitor registration skipped: %s", exc)
    tracker.create_task(mem_monitor.start(), name="memory_monitor.start")
    
    logger.info("🛡️  Task Supervisor active (Memory monitoring enabled).")

    # Hot-Swap Bridge
    runtime_loop = asyncio.get_running_loop()

    def _on_actor_restart(name: str, new_pipe: Any):
        logger.info("🔄 [HOTSWAP] Detected restart of %s. Re-binding IPC...", name)

        def _schedule_rebind():
            actor_bus = ServiceContainer.get("actor_bus", default=None)
            if actor_bus:
                tracker.create_task(
                    actor_bus.update_actor(name, new_pipe),
                    name=f"actor_bus.update_actor.{name}",
                )

        runtime_loop.call_soon_threadsafe(_schedule_rebind)
                
    supervisor.set_restart_callback(_on_actor_restart)

    # Joy & Social Integration
    try:
        from skills.joy_social_integration import integrate_joy_social
        # We integrate without explicit config to use local development adapters by default
        # unless user has set environment variables.
        integrate_joy_social(orchestrator)
        logger.info("🌟 Joy & Social systems integrated into startup sequence.")
    except ImportError:
        logger.warning("⚠️ JoySocial skills not found — skipping integration.")
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.error("❌ Failed to integrate JoySocial: %s", exc)

    # Apply Consciousness, Response, and SafeMode Genesis Patches
    try:
        from core.apply_response_patches import apply_response_patches
        from core.consciousness.apply_patches import apply_consciousness_patches
        from core.safe_mode import apply_orchestrator_patches

        apply_consciousness_patches(orchestrator)
        apply_response_patches()
        # Activate the dynamic autonomy bridge
        apply_orchestrator_patches(orchestrator)
        logger.info("🛡️ [GENESIS] Autonomy bridge and stability patches active.")
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.error("❌ Failed to apply gap-closing patches: %s", exc)


async def _boot_runtime_orchestrator(
    *,
    ready_label: str,
    readiness_context: str | None = None,
    profile: str | None = None,
    artifact_root: str | Path | None = None,
):
    """Canonical runtime boot path shared by CLI/server/desktop surfaces."""
    from core.container import ServiceContainer
    from core.orchestrator import create_orchestrator

    orchestrator = create_orchestrator()
    await bootstrap_aura(orchestrator)

    # ── Morphogenetic self-organization runtime ───────────────────────
    # Starts bounded cell ecology, metabolism, and organ stabilizer.
    # Must boot after ServiceContainer is populated (bootstrap_aura)
    # and before orchestrator enters long-running loops.
    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_MORPHOGENESIS", True):
        try:
            from core.morphogenesis.integration import start_morphogenesis_runtime
            await start_morphogenesis_runtime()
            logger.info("🧬 Morphogenetic self-organization runtime online.")
        except _AURA_MAIN_BOUNDARY_ERRORS as morph_exc:
            record_degradation('aura_main', morph_exc)
            # Never block boot. Morphogenesis is an adaptive layer, not the boot root.
            logger.warning("Morphogenetic runtime startup skipped/degraded: %s", morph_exc)
    else:
        logger.info("🧬 Morphogenetic runtime disabled for foreground-only boot.")

    await orchestrator.start()

    if readiness_context and hasattr(orchestrator, "_ensure_inference_gate_ready"):
        await orchestrator._ensure_inference_gate_ready(context=readiness_context)

    # Register the runtime singletons that the ServiceManifest names as
    # canonical owners but that did not previously live in ServiceContainer.
    # Done before lock_registration so they show up to the manifest check.
    _register_runtime_singletons(orchestrator)

    ServiceContainer.lock_registration()
    _enforce_service_manifest(ready_label)
    try:
        ownership_root = (
            PROJECT_ROOT
            if _env_flag("AURA_WRITE_TRACKED_SERVICE_OWNERSHIP", False)
            else Path(
                artifact_root
                or os.environ.get("AURA_ARTIFACTS_DIR")
                or PROJECT_ROOT / "artifacts" / "current"
            )
        )
        ownership_path = ServiceContainer.write_service_ownership_manifest(ownership_root)
        logger.info("🧾 Runtime service ownership manifest written: %s", ownership_path)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        logger.warning("⚠️ Failed to write runtime service ownership manifest: %s", exc)
    await _enforce_boot_probes(ready_label)
    readiness_snapshot = _refresh_orchestrator_health_before_manifest(orchestrator, ready_label)
    _write_runtime_manifest(
        profile=profile or ready_label.lower(),
        ready_label=ready_label,
        artifact_root=artifact_root,
        readiness_snapshot=readiness_snapshot,
    )
    _schedule_runtime_manifest_ready_refresh(
        orchestrator=orchestrator,
        profile=profile or ready_label.lower(),
        ready_label=ready_label,
        artifact_root=artifact_root,
        initial_readiness=readiness_snapshot,
    )
    logger.info("🛡️ Registry Locked. Aura Ready (%s).", ready_label)

    # ── Wire viability + self-healing + stem cells + boot phases ───────
    # All four subsystems live in core/ and are independent of the
    # orchestrator's existing lifecycle. We start them here so they are
    # active for the same lifetime as the orchestrator.
    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_VIABILITY", True):
        try:
            from core.organism.viability import get_viability
            await get_viability().start(interval=5.0)
            ServiceContainer.register_instance("viability", get_viability(), required=False)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.warning("viability engine start failed: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_SELF_HEALING", True):
        try:
            from core.runtime.self_healing import get_healer
            healer = get_healer()
            # Watch the orchestrator main loop; the orchestrator is expected
            # to call `healer.heartbeat("orchestrator")` on every tick. If
            # the heartbeat goes stale by 2.5x its expected interval, the
            # healer asks the orchestrator to restart_async() (no-op if
            # the method isn't defined — falls back to ServiceContainer).
            healer.watch("orchestrator", expected_interval_s=15.0, container_key="orchestrator")
            healer.watch("agency_bus",   expected_interval_s=30.0, container_key="agency_bus")
            healer.watch("phi_core",     expected_interval_s=30.0, container_key="phi_core")
            healer.watch("memory_facade", expected_interval_s=60.0, container_key="memory_facade")
            await healer.start(interval=5.0)
            ServiceContainer.register_instance("self_healing", healer, required=False)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.warning("self-healing watcher start failed: %s", exc)
    else:
        logger.info("Self-healing watcher disabled for foreground-only boot.")

    try:
        from core.runtime.boot_phases import get_boot_phases
        bp = get_boot_phases()
        # Reflect the post-boot ready state into the boot panel.
        bp.update_organ("core", "ready")
        bp.update_organ("memory", "ready")
        bp.update_organ("cortex", "ready")
        bp.update_organ("voice", "waiting")
        bp.update_organ("autonomy", "ready")
        ServiceContainer.register_instance("boot_phases", bp, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("boot phases hook skipped: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_PERFORMANCE_GUARD", True):
        try:
            from core.runtime.performance_guard import get_guard
            await get_guard().start(interval=5.0)
            ServiceContainer.register_instance("performance_guard", get_guard(), required=False)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.debug("performance guard start skipped: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_SEED_AUTONOMY_GOALS", True):
        try:
            from core.goals.default_goals import seed_default_autonomy_goals

            goal_engine = ServiceContainer.get("goal_engine", default=None)
            seeded = await seed_default_autonomy_goals(goal_engine)
            if seeded:
                logger.info("🎯 Seeded %d durable autonomy goals.", len(seeded))
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            logger.warning("Default autonomy goal seeding failed: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_AUTONOMY_CONDUCTOR", True):
        try:
            from core.runtime.autonomy_conductor import start_default_conductor
            from core.runtime.overt_action_loop import get_overt_action_loop

            ServiceContainer.register_instance("overt_action_loop", get_overt_action_loop(), required=False)
            conductor = await start_default_conductor()
            ServiceContainer.register_instance("autonomy_conductor", conductor, required=False)
            logger.info("🧭 AutonomyConductor online — proof, validation, and maintenance loops scheduled.")
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            logger.warning("AutonomyConductor start failed: %s", exc)
    else:
        logger.info("AutonomyConductor disabled for foreground-only boot.")

    try:
        from core.adaptation.online_lora_governor import get_online_lora_governor

        ServiceContainer.register_instance("online_lora_governor", get_online_lora_governor(), required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation("aura_main", exc)
        logger.debug("online_lora_governor registration skipped: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_SENSORIMOTOR_GROUNDING", True):
        try:
            from core.brain.llm.sensorimotor_grounding import SensorimotorGroundingBridge

            substrate = (
                ServiceContainer.get("continuous_substrate", default=None)
                or ServiceContainer.get("liquid_state", default=None)
                or ServiceContainer.get("liquid_substrate", default=None)
                or getattr(orchestrator, "substrate", None)
            )
            if substrate is not None:
                bridge = SensorimotorGroundingBridge(substrate=substrate)
                await bridge.start()
                ServiceContainer.register_instance("sensorimotor_grounding_bridge", bridge, required=False)
                logger.info("👁️🎙️ Sensorimotor grounding bridge online — substrate receives live sensor observations.")
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            logger.warning("Sensorimotor grounding bridge start failed: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_ACTIVATION_AUDIT", True):
        try:
            from core.runtime.activation_audit import get_activation_auditor

            auditor = get_activation_auditor()
            report = await auditor.audit(orchestrator, reconcile=True)
            auditor.write_report(report, config.paths.data_dir / "runtime" / "activation_report.json")
            ServiceContainer.register_instance("activation_auditor", auditor, required=False)
            if report.missing_required:
                logger.warning(
                    "Activation audit missing required loops: %s",
                    ", ".join(status.name for status in report.missing_required),
                )
            else:
                logger.info("Activation audit passed: %.0f%% required loops active.", report.required_active_ratio * 100)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            logger.warning("Activation audit failed: %s", exc)

    # Capture stem-cell snapshots for the load-bearing organs so the
    # immune layer has something to revert to if a future mutation
    # corrupts them. Snapshots are HMAC-signed in
    # ~/.aura/data/stem_cells/.
    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_STEM_CELL_CAPTURE", True):
        try:
            from core.resilience.stem_cell import get_registry
            reg = get_registry()
            will = ServiceContainer.get("unified_will", default=None) or ServiceContainer.get("will", default=None)
            if will is not None:
                reg.register("unified_will")
                reg.capture("unified_will", will, schema_version="1")
            from core.agency.agency_orchestrator import get_orchestrator as _get_ao
            ao = _get_ao()
            reg.register("agency_orchestrator")
            reg.capture("agency_orchestrator", ao, schema_version="1")
            from core.identity.self_object import get_self
            reg.register("self_object")
            reg.capture("self_object", get_self().snapshot().continuity_hash, schema_version="1")
            ServiceContainer.register_instance("stem_cell_registry", reg, required=False)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.warning("stem-cell capture at boot failed: %s", exc)

    if not _foreground_only_runtime() and _env_flag("AURA_ENABLE_FLAGSHIP_DOCTOR", True):
        try:
            from core.runtime.flagship_doctor import get_flagship_doctor_daemon
            daemon = get_flagship_doctor_daemon(root_dir=PROJECT_ROOT)
            daemon.start(asyncio.get_running_loop())
            ServiceContainer.register_instance("flagship_doctor_daemon", daemon, required=False)

            from core.runtime.shutdown_coordinator import get_shutdown_coordinator
            get_shutdown_coordinator().register(
                daemon.stop,
                phase="task_supervisor",
                name="flagship_doctor_daemon",
                timeout=5.0
            )
            logger.info("🩺 FlagshipDoctorDaemon started and registered for shutdown.")
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.warning("FlagshipDoctorDaemon failed to start: %s", exc)

    return orchestrator


def _install_fault_forensics() -> None:
    """A death with no traceback is forbidden: faulthandler dumps every
    thread's stack to a persistent file on native faults (SIGSEGV/BUS/
    ABRT/ILL) and on SIGTERM, so the next silent exit names its killer.
    SIGUSR1 is also registered without chaining so a live-but-stuck runtime can
    be sampled without killing it."""
    try:
        import faulthandler
        import signal as _signal

        crash_dir = Path("data/error_logs/crash")
        crash_dir.mkdir(parents=True, exist_ok=True)
        crash_file = open(crash_dir / "faulthandler.log", "a")
        crash_file.write(
            f"\n===== boot pid={os.getpid()} at={time.time()} =====\n"
        )
        crash_file.flush()
        faulthandler.enable(file=crash_file, all_threads=True)
        faulthandler.register(_signal.SIGTERM, file=crash_file, all_threads=True, chain=True)
        if hasattr(_signal, "SIGUSR1"):
            faulthandler.register(_signal.SIGUSR1, file=crash_file, all_threads=True, chain=False)
        logger.info("🛡️ Fault forensics armed: data/error_logs/crash/faulthandler.log")
    except (OSError, ValueError, RuntimeError, AttributeError) as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued boot without fault forensics",
            severity="warning",
        )


class _ExternalMemorySentinelStatus:
    """Health-contract handle for the out-of-process memory sentinel."""

    def __init__(self, proc: subprocess.Popen | None, *, lethal_mb: float = 0.0, interval_s: float = 0.0):
        self.proc = proc
        self.pid = int(getattr(proc, "pid", 0) or 0) if proc is not None else 0
        self.lethal_mb = float(lethal_mb or 0.0)
        self.interval_s = float(interval_s or 0.0)
        self.started_at = time.time() if proc is not None else 0.0

    def is_armed(self) -> bool:
        if self.proc is None or self.pid <= 0:
            return False
        if self.proc.poll() is not None:
            return False
        try:
            import psutil

            return psutil.pid_exists(self.pid)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return True

    def get_status(self) -> dict[str, Any]:
        return {
            "armed": self.is_armed(),
            "pid": self.pid,
            "lethal_mb": self.lethal_mb,
            "interval_s": self.interval_s,
            "started_at": self.started_at,
        }


def _install_systemwide_memory_protection() -> None:
    """Memory protection that survives process suspension.

    The in-process MemoryWatchdog thread was alive during a 115GB host
    crash and could not act: when the main process itself is the hog,
    macOS thrashes/suspends the whole process — watchdog threads
    included. These three layers do not need the dying process to
    cooperate:

    1. RLIMIT_DATA: the kernel caps malloc'd heap; a runaway allocation
       raises MemoryError inside the offending call instead of taking
       the host.
    2. MLX Metal memory limit: GPU-wired model memory gets a hard cap
       (inherited by workers via env).
    3. External sentinel process: SIGKILLs this process tree past the
       lethal ceiling. It lives outside us; it cannot be paused with us.
    """
    import resource

    try:
        import psutil

        total_mb = psutil.virtual_memory().total / (1024 * 1024)
    except (ImportError, OSError, AttributeError):
        total_mb = 65536.0

    try:
        rlimit_gb = float(os.environ.get("AURA_RLIMIT_DATA_GB", "48") or 48)
        if rlimit_gb > 0:
            limit_bytes = int(rlimit_gb * (1024**3))
            soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
            new_soft = limit_bytes if hard == resource.RLIM_INFINITY else min(limit_bytes, hard)
            resource.setrlimit(resource.RLIMIT_DATA, (new_soft, hard))
            logger.info("🛡️ RLIMIT_DATA installed: %.0fGB heap ceiling (kernel-enforced).", rlimit_gb)
    except ValueError as exc:
        if "current limit exceeds maximum limit" in str(exc):
            logger.info(
                "RLIMIT_DATA unsupported by this kernel/Python runtime; "
                "continuing with MLX memory ceiling and external RSS sentinel."
            )
        else:
            record_degradation(
                _AURA_MAIN_DEGRADATION_KEY,
                exc,
                action="continued boot without RLIMIT_DATA heap ceiling",
                severity="warning",
            )
    except OSError as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued boot without RLIMIT_DATA heap ceiling",
            severity="warning",
        )

    # MLX memory cap travels by ENV ONLY: importing mlx.core in this
    # parent process violates the deferred-Metal-bindings protection
    # (platform_root defers exactly to protect spawn children) and is
    # the prime suspect in a silent native death mid-generation during
    # live proof round 5. Workers apply the limit on their side.
    try:
        from core.runtime.desktop_boot_safety import (
            compute_mlx_memory_limit,
            compute_process_rss_limit,
            desktop_safe_boot_enabled,
            env_flag_enabled,
        )

        default_mlx_limit_gb = max(
            8.0,
            compute_mlx_memory_limit(int(total_mb * 1024 * 1024)) / float(1024**3),
        )
        default_process_rss_limit_gb = max(
            8.0,
            compute_process_rss_limit(int(total_mb * 1024 * 1024)) / float(1024**3),
        )
        safe_desktop_memory_limits = (
            desktop_safe_boot_enabled()
            and not env_flag_enabled(os.environ.get("AURA_ALLOW_UNSAFE_MEMORY_LIMITS"))
        )
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        default_mlx_limit_gb = 28.0
        default_process_rss_limit_gb = 36.0
        safe_desktop_memory_limits = True
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="used conservative MLX worker memory ceiling after policy computation failed",
            severity="warning",
        )
    mlx_gb = str(
        os.environ.get("AURA_MLX_MEMORY_LIMIT_GB", f"{default_mlx_limit_gb:.0f}")
        or f"{default_mlx_limit_gb:.0f}"
    )
    if safe_desktop_memory_limits:
        mlx_gb = f"{default_mlx_limit_gb:.0f}"
        os.environ["AURA_MLX_MEMORY_LIMIT_GB"] = mlx_gb
    else:
        os.environ.setdefault("AURA_MLX_MEMORY_LIMIT_GB", mlx_gb)
    process_rss_gb = str(
        os.environ.get("AURA_PROCESS_RSS_LIMIT_GB", f"{default_process_rss_limit_gb:.0f}")
        or f"{default_process_rss_limit_gb:.0f}"
    )
    if safe_desktop_memory_limits:
        process_rss_gb = f"{default_process_rss_limit_gb:.0f}"
        os.environ["AURA_PROCESS_RSS_LIMIT_GB"] = process_rss_gb
    else:
        os.environ.setdefault("AURA_PROCESS_RSS_LIMIT_GB", process_rss_gb)

    sentinel_enabled = str(os.environ.get("AURA_MEMORY_SENTINEL", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if sentinel_enabled:
        try:
            from core.container import ServiceContainer

            configured_lethal = os.environ.get("AURA_MEMWATCH_LETHAL_MB", "").strip()
            lethal_mb = _bounded_memory_ceiling_mb(
                total_mb,
                configured_lethal if configured_lethal else None,
            )
            sentinel_interval_s = float(os.environ.get("AURA_MEMORY_SENTINEL_INTERVAL_S", "1.0") or 1.0)
            sentinel_log = Path("data/error_logs/memory")
            sentinel_log.mkdir(parents=True, exist_ok=True)
            sentinel_proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "tools" / "memory_sentinel.py"),
                    "--pid",
                    str(os.getpid()),
                    "--lethal-mb",
                    str(lethal_mb),
                    "--interval",
                    str(max(0.5, sentinel_interval_s)),
                ],
                stdout=open(sentinel_log / "sentinel.log", "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(Path(__file__).resolve().parent),
            )
            ServiceContainer.register_instance(
                "external_memory_sentinel",
                _ExternalMemorySentinelStatus(
                    sentinel_proc,
                    lethal_mb=lethal_mb,
                    interval_s=max(0.5, sentinel_interval_s),
                ),
                required=False,
            )
            logger.info(
                "🛡️ External memory sentinel armed: lethal=%.0fMB (kills from outside).",
                lethal_mb,
            )
        except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
            record_degradation(
                _AURA_MAIN_DEGRADATION_KEY,
                exc,
                action="continued boot without external memory sentinel",
                severity="degraded",
            )
    else:
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance(
                "external_memory_sentinel",
                _ExternalMemorySentinelStatus(None),
                required=False,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                _AURA_MAIN_DEGRADATION_KEY,
                exc,
                action="external memory sentinel disabled and status registration failed",
                severity="degraded",
            )


def _install_liveness_sentinel() -> None:
    """Spawn the external event-loop liveness sentinel (out-of-process).

    The in-process StallWatchdog hard-exit cannot fire when a Metal GPU deadlock
    holds the GIL (no Python thread runs). This external process watches the
    heartbeat file the StallWatchdog refreshes; when it goes stale past the
    ceiling it SIGKILLs the tree so the launchd supervisor restarts the runtime.
    """
    if str(os.environ.get("AURA_LIVENESS_SENTINEL", "1")).strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        import subprocess

        heartbeat = os.environ.get(
            "AURA_LIVENESS_HEARTBEAT_FILE", "data/runtime/liveness_heartbeat.json"
        )
        Path(heartbeat).parent.mkdir(parents=True, exist_ok=True)
        desktop_foreground = (
            str(os.environ.get("AURA_SAFE_BOOT_DESKTOP", "")).strip().lower()
            in {"1", "true", "yes", "on"}
            or "--headless" in sys.argv
        )
        default_stale_ceiling = "45" if desktop_foreground else "180"
        default_grace = "90" if desktop_foreground else "300"
        default_interval = "2" if desktop_foreground else "5"
        stale_ceiling = os.environ.get("AURA_LIVENESS_STALE_CEILING_S", default_stale_ceiling)
        grace = os.environ.get("AURA_LIVENESS_GRACE_S", default_grace)
        interval = os.environ.get("AURA_LIVENESS_INTERVAL_S", default_interval)
        log_dir = Path("data/error_logs/liveness")
        log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "tools" / "liveness_sentinel.py"),
                "--pid", str(os.getpid()),
                "--heartbeat", str(heartbeat),
                "--stale-ceiling", str(stale_ceiling),
                "--grace", str(grace),
                "--interval", str(interval),
            ],
            stdout=open(log_dir / "liveness_sentinel.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parent),
        )
        logger.info(
            "🛡️ External liveness sentinel armed: heartbeat=%s stale_ceiling=%ss "
            "(kills+restarts a GIL-locked/wedged loop from outside).",
            heartbeat, stale_ceiling,
        )
    except (ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued boot without external liveness sentinel",
            severity="degraded",
        )


async def _log_macos_permission_preflight(profile: str) -> None:
    """Log a one-shot macOS TCC permission summary at boot.

    Surfaces denied permissions (mic/camera/screen/accessibility/automation) at
    startup instead of letting the dependent feature fail silently later. Purely
    informational and bounded — never blocks or fails boot. Skipped on non-macOS,
    the minimal/proof profile, or when AURA_PERMISSION_PREFLIGHT=0.
    """
    if sys.platform != "darwin" or profile == "minimal":
        return
    if str(os.environ.get("AURA_PERMISSION_PREFLIGHT", "1")).strip().lower() in {
        "0", "false", "no", "off",
    }:
        return
    try:
        from core.security.permission_setup import check_all_permissions, format_report

        report = await asyncio.wait_for(check_all_permissions(), timeout=8.0)
        summary = format_report(report).replace("\n", " | ")
        if report.missing:
            logger.warning("🔐 macOS permissions need attention: %s", summary)
        else:
            logger.info("🔐 macOS permission preflight: %s", summary)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued boot without a macOS permission preflight summary",
            severity="debug",
        )


def _warn_if_active_model_missing(profile: str) -> None:
    """Log a clear, actionable warning when the cortex model isn't on disk.

    A fresh install otherwise stalls silently on the first generation while a
    multi-GB model downloads with no surfaced progress. Best-effort and bounded
    (filesystem-only); skipped on the minimal profile and offline-LLM runs.
    """
    if profile == "minimal" or os.environ.get("AURA_USE_MOCK_LLM") == "1":
        return
    try:
        from core.brain.llm.model_lifecycle import get_model_lifecycle_manager

        manager = get_model_lifecycle_manager()
        if manager.active_model_present():
            return
        missing = [s.name for s in manager.missing()]
        logger.warning(
            "🧠 Cortex model not found on disk (missing: %s). Run "
            "`python scripts/fetch_models.py` to download it, or the first "
            "generation will block while it downloads.",
            ", ".join(missing) or "active model",
        )
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued boot without a model-presence check",
            severity="debug",
        )


async def boot_aura_runtime(
    *,
    profile: str,
    artifact_root: str | Path | None = None,
    readiness_context: str | None = None,
    ready_label: str | None = None,
):
    """Public canonical boot entry for launchers and proof runners.

    CLI, desktop, server, and validation/proof surfaces must use this path so
    their evidence reflects the same live Aura boot contract.
    """
    _install_fault_forensics()
    _install_systemwide_memory_protection()
    _install_liveness_sentinel()
    await _log_macos_permission_preflight(profile)
    _warn_if_active_model_missing(profile)
    if profile == "minimal":
        os.environ["AURA_BOOT_PROFILE"] = "minimal"
        os.environ["AURA_USE_MOCK_LLM"] = "1"
        os.environ["AURA_FEATURES__CAMERA_ENABLED"] = "false"
        os.environ["AURA_FEATURES__VOICE_ENABLED"] = "false"
        os.environ["AURA_SECURITY__ALLOW_NETWORK_ACCESS"] = "false"
        os.environ["AURA_ENABLE_CAMERA"] = "0"
        os.environ["AURA_ENABLE_MIC"] = "0"
        os.environ["AURA_ENABLE_DESKTOP"] = "0"
        os.environ["AURA_DISABLE_CLOUD"] = "1"
        os.environ["AURA_MINIMAL_PROFILE"] = "1"

        try:
            from core.config import config
            config.skeletal_mode = True
            config.features.camera_enabled = False
            config.features.voice_enabled = False
            config.security.allow_network_access = False
            if hasattr(config, "soma"):
                config.soma.enabled = False
            config.features.mycelium_visualizer = False
            config.features.autonomous_impulses = False
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation(
                _AURA_MAIN_DEGRADATION_KEY,
                exc,
                action="continued minimal boot after optional config mutation failed",
                severity="warning",
            )
            logger.warning("Minimal profile config mutation skipped: %s", exc)

    resolved_ready_label = ready_label or profile.title()
    if not _RUNTIME_LOCK_CLAIMED:
        bootstrap_lock(skip_lock=False)
    _activate_proof_runtime_policy(profile, resolved_ready_label)
    # Validate and log the runtime mode (production/research/dev/simulation/safe)
    try:
        from core.runtime.mode import validate_mode_at_startup
        validate_mode_at_startup()
    except (ImportError, RuntimeError) as _mode_exc:
        record_degradation("aura_main", _mode_exc)
        logger.warning("Runtime mode validation skipped: %s", _mode_exc)
    return await _boot_runtime_orchestrator(
        ready_label=resolved_ready_label,
        readiness_context=readiness_context or f"{profile}_boot",
        profile=profile,
        artifact_root=artifact_root,
    )


def _refresh_orchestrator_health_before_manifest(orchestrator: Any, ready_label: str) -> dict[str, Any]:
    """Refresh live runtime health immediately before writing proof/desktop manifests."""

    try:
        healthy = bool(orchestrator.health_check())
        if healthy:
            logger.info("✅ Runtime health confirmed before manifest (%s).", ready_label)
            return {
                "ready": True,
                "status": "healthy",
                "critical": [],
                "important": [],
                "required_probe_blockers": [],
            }

        from core.runtime.health_contract import (
            required_probe_blockers,
            required_probe_status,
            runtime_health_report,
        )

        contract = runtime_health_report()
        failures = contract.get("failures", {}) if isinstance(contract, dict) else {}
        critical = [
            str(item.get("container_key") or item.get("name") or "")
            for item in failures.get("critical", [])
            if isinstance(item, dict)
        ]
        important = [
            str(item.get("container_key") or item.get("name") or "")
            for item in failures.get("important", [])
            if isinstance(item, dict)
        ]
        probes = required_probe_blockers(required_probe_status(contract))
        logger.warning(
            "⚠️ Runtime health still not clean before manifest (%s): "
            "critical=%s important=%s probes=%s",
            ready_label,
            critical,
            important,
            probes,
        )
        return {
            "ready": False,
            "status": str(contract.get("status", "unhealthy")) if isinstance(contract, dict) else "unhealthy",
            "critical": critical,
            "important": important,
            "required_probe_blockers": probes,
        }
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="continued canonical boot after pre-manifest health refresh failed",
            severity="critical",
        )
        logger.warning("⚠️ Runtime health refresh before manifest failed: %s", exc)
        return {
            "ready": False,
            "status": "refresh_failed",
            "critical": ["runtime_health_refresh_failed"],
            "important": [],
            "required_probe_blockers": ["runtime_health_refresh_failed"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_runtime_manifest(
    *,
    profile: str,
    ready_label: str,
    artifact_root: str | Path | None = None,
    readiness_snapshot: dict[str, Any] | None = None,
) -> None:
    try:
        from core.runtime.runtime_manifest import write_runtime_manifest

        if artifact_root is None:
            artifact_root = os.environ.get("AURA_ARTIFACTS_DIR") or PROJECT_ROOT / "artifacts" / "current"
        path = write_runtime_manifest(
            profile=profile,
            ready_label=ready_label,
            project_root=PROJECT_ROOT,
            artifact_root=Path(artifact_root),
            readiness_snapshot=readiness_snapshot,
        )
        logger.info("🧾 Runtime manifest written: %s", path)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            "aura_main",
            exc,
            action="continued canonical boot after runtime manifest emission failed",
        )
        logger.warning("Runtime manifest emission failed: %s", exc)


def _schedule_runtime_manifest_ready_refresh(
    *,
    orchestrator: Any,
    profile: str,
    ready_label: str,
    artifact_root: str | Path | None,
    initial_readiness: dict[str, Any],
) -> None:
    """Refresh the manifest once live readiness catches up after model warmup."""

    if bool(initial_readiness.get("ready")):
        return

    try:
        timeout_s = float(os.environ.get("AURA_RUNTIME_MANIFEST_READY_REFRESH_SECONDS", "240"))
        interval_s = float(os.environ.get("AURA_RUNTIME_MANIFEST_READY_REFRESH_INTERVAL_SECONDS", "2"))
    except (TypeError, ValueError):
        timeout_s = 240.0
        interval_s = 2.0

    if timeout_s <= 0:
        return

    try:
        get_task_tracker().create_task(
            _refresh_runtime_manifest_until_ready(
                orchestrator=orchestrator,
                profile=profile,
                ready_label=ready_label,
                artifact_root=artifact_root,
                timeout_s=timeout_s,
                interval_s=max(0.25, interval_s),
            ),
            name="runtime_manifest.ready_refresh",
        )
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            _AURA_MAIN_DEGRADATION_KEY,
            exc,
            action="runtime manifest ready refresh scheduling failed",
            severity="warning",
        )


async def _refresh_runtime_manifest_until_ready(
    *,
    orchestrator: Any,
    profile: str,
    ready_label: str,
    artifact_root: str | Path | None,
    timeout_s: float,
    interval_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_snapshot: dict[str, Any] | None = None
    while time.monotonic() < deadline and not is_shutdown_requested():
        await asyncio.sleep(interval_s)
        snapshot = _refresh_orchestrator_health_before_manifest(orchestrator, ready_label)
        last_snapshot = snapshot
        if bool(snapshot.get("ready")):
            _write_runtime_manifest(
                profile=profile,
                ready_label=ready_label,
                artifact_root=artifact_root,
                readiness_snapshot=snapshot,
            )
            logger.info("🧾 Runtime manifest refreshed after readiness settled: %s", ready_label)
            return

    if last_snapshot is not None:
        logger.warning(
            "Runtime manifest readiness refresh expired for %s: status=%s blockers=%s",
            ready_label,
            last_snapshot.get("status"),
            last_snapshot.get("required_probe_blockers"),
        )


def _register_runtime_singletons(orchestrator: Any) -> None:
    """Register module-level singletons + orchestrator-attached components
    with ServiceContainer so the manifest verification finds canonical owners.

    Surfaces registered:
      - task_tracker / task_supervisor (alias)
      - shutdown_coordinator
      - output_gate (orchestrator attribute -> registry)
      - aura_runtime / orchestrator (alias)
    """
    try:
        from core.container import ServiceContainer
    except ImportError as exc:
        record_degradation("aura_main", exc)
        logger.warning("ServiceContainer unavailable during service registration: %s", exc)
        return

    # ── Substrate Voice Engine ──
    try:
        if not ServiceContainer.has("substrate_voice_engine"):
            from core.voice.substrate_voice_engine import get_substrate_voice_engine
            # Eagerly initialize and register the voice engine singleton
            get_substrate_voice_engine()
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("substrate_voice_engine eager registration failed: %s", exc)


    try:
        from core.utils.task_tracker import get_task_tracker

        tracker = get_task_tracker()
        if not ServiceContainer.has("task_tracker"):
            ServiceContainer.register_instance("task_tracker", tracker, required=False)
        if not ServiceContainer.has("task_supervisor"):
            ServiceContainer.register_instance("task_supervisor", tracker, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("task_tracker registration skipped: %s", exc)

    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        coord = get_shutdown_coordinator()
        if not ServiceContainer.has("shutdown_coordinator"):
            ServiceContainer.register_instance("shutdown_coordinator", coord, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("shutdown_coordinator registration skipped: %s", exc)

    # ── NetHack Adapter ──
    try:
        from core.adapters.nethack_adapter import NetHackAdapter

        if not ServiceContainer.has("nethack_adapter"):
            adapter = NetHackAdapter()
            # We don't start it here; it will be started by the skill or
            # a dedicated starter to avoid immediate process spawn on every boot.
            # Actually, the skill expects it to be ready. Let's register a factory.
            ServiceContainer.register_instance("nethack_adapter", adapter, required=False)
            logger.info("🎮 NetHack adapter registered in ServiceContainer.")
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("nethack_adapter registration failed: %s", exc)

    output_gate = getattr(orchestrator, "output_gate", None)
    if output_gate is not None:
        try:
            if not ServiceContainer.has("output_gate"):
                ServiceContainer.register_instance("output_gate", output_gate, required=False)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation('aura_main', exc)
            logger.debug("output_gate registration skipped: %s", exc)

    try:
        if not ServiceContainer.has("orchestrator"):
            ServiceContainer.register_instance("orchestrator", orchestrator, required=False)
        if not ServiceContainer.has("aura_runtime"):
            ServiceContainer.register_instance("aura_runtime", orchestrator, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("orchestrator registration skipped: %s", exc)

    try:
        system2 = ServiceContainer.get("native_system2", default=None)
        if system2 is None:
            from core.reasoning.native_system2 import get_native_system2

            system2 = get_native_system2()
            ServiceContainer.register_instance("native_system2", system2, required=False)
        if not ServiceContainer.has("system2_search"):
            ServiceContainer.register_instance("system2_search", system2, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("native_system2 boot singleton unavailable: %s", exc)

    try:
        if _foreground_only_runtime() or not _env_flag("AURA_REGISTER_REIMPLEMENTATION_LAB", True):
            lab = None
        else:
            lab = ServiceContainer.get("reimplementation_lab", default=None)
            if lab is None:
                from core.config import config
                from core.llm.code_generator import LLMCodeGenerator
                from core.self_improvement.reimplementation_lab import ReimplementationLab

                lab = ReimplementationLab(
                    project_root=str(config.paths.base_dir),
                    generator=LLMCodeGenerator(prefer_tier="primary"),
                )
                ServiceContainer.register_instance("reimplementation_lab", lab, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("reimplementation_lab boot singleton unavailable: %s", exc)

    try:
        store = ServiceContainer.get("markdown_workspace", default=None)
        if store is None:
            from core.workspace.markdown_workspace import MarkdownWorkspace

            store = MarkdownWorkspace()
            ServiceContainer.register_instance("markdown_workspace", store, required=False)
        workspace = ServiceContainer.get("aura_workspace", default=None)
        if workspace is None:
            from core.workspace.aura_workspace import AuraWorkspace

            workspace = AuraWorkspace(store=store)
            ServiceContainer.register_instance("aura_workspace", workspace, required=False)
        if not ServiceContainer.has("agent_workspace"):
            ServiceContainer.register_instance("agent_workspace", workspace, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("agent workspace boot singleton unavailable: %s", exc)

    try:
        if _foreground_only_runtime() or not _env_flag("AURA_REGISTER_ARCHITECTURE_GOVERNOR", True):
            return
        governor = ServiceContainer.get("architecture_governor", default=None)
        if governor is None:
            from core.architect.config import ASAConfig
            from core.architect.governor import AutonomousArchitectureGovernor

            governor = AutonomousArchitectureGovernor(ASAConfig.from_env(PROJECT_ROOT))
            ServiceContainer.register_instance("architecture_governor", governor, required=False)
        if not ServiceContainer.has("autonomous_architecture_governor"):
            ServiceContainer.register_instance("autonomous_architecture_governor", governor, required=False)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.warning("architecture_governor boot singleton unavailable: %s", exc)


async def _enforce_boot_probes(ready_label: str) -> None:
    """Run behavioral boot probes. In strict mode any failure aborts boot.

    The probe set covers the audit's required surfaces: memory write/read,
    state mutate/read, governance approve/deny, output gate, event bus, and
    actor supervisor. Surface-level probes (output_gate, event_bus,
    actor_supervisor) verify the contract is wired but do not require live
    backends.
    """
    try:
        from core.runtime.boot_probes import run_boot_probes
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        logger.debug("boot_probes module unavailable: %s", exc)
        return
    strict_mode = os.environ.get("AURA_STRICT_RUNTIME") == "1"
    try:
        report = await run_boot_probes(strict=strict_mode)
        for r in report.results:
            if not r.ok:
                logger.warning(
                    "Boot probe %s failed during %s boot: %s",
                    r.name, ready_label, r.detail,
                )
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation('aura_main', exc)
        if strict_mode:
            raise
        logger.warning(
            "Boot probes raised in non-strict mode (%s); continuing degraded: %s",
            ready_label, exc,
        )


def _enforce_service_manifest(ready_label: str) -> None:
    """Verify ServiceManifest invariants once the registry is locked.

    In strict runtime, a critical violation aborts boot. Otherwise the
    violation is logged so operators can see the drift without forcing
    a desktop crash.
    """
    try:
        from core.container import ServiceContainer
        from core.runtime.service_manifest import (
            SERVICE_MANIFEST,
            critical_violations,
            verify_manifest,
        )
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:  # pragma: no cover - defensive import
        logger.debug("ServiceManifest unavailable during boot: %s", exc)
        return

    snapshot: dict = {}
    for role in SERVICE_MANIFEST.values():
        for candidate in (role.canonical_owner, *role.aliases):
            if ServiceContainer.has(candidate):
                instance = ServiceContainer.get(candidate, default=None)
                if instance is not None:
                    snapshot[candidate] = instance

    violations = verify_manifest(snapshot)
    crit = critical_violations(violations)
    if violations:
        for v in violations:
            logger.warning(
                "ServiceManifest %s violation [%s]: %s",
                v.severity,
                v.role,
                v.reason,
            )
    strict_mode = os.environ.get("AURA_STRICT_RUNTIME", "0") == "1"
    if crit and strict_mode:
        raise RuntimeError(
            f"AURA_STRICT_RUNTIME: ServiceManifest critical violations during {ready_label} boot: "
            + "; ".join(f"{v.role}: {v.reason}" for v in crit)
        )

async def run_console(profile: str = "cli"):
    """Interactive CLI Mode"""
    orchestrator = await boot_aura_runtime(profile=profile, ready_label="CLI")

    from core.main import conversation_loop
    await conversation_loop(orchestrator=orchestrator)


async def run_philosophy_stream(port: int = 8000):
    """Stream the live qualia-gap proof surface as JSON lines."""
    import json

    from core.container import ServiceContainer

    orchestrator = await boot_aura_runtime(
        profile="philosophy",
        ready_label="Philosophy",
        readiness_context="philosophy_stream",
    )

    get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
    logger.info("🧾 Philosophy stream active. Press Ctrl-C to stop.")

    interval_s = _env_float("AURA_PHILOSOPHY_STREAM_INTERVAL", 1.0, minimum=0.1, maximum=60.0)
    while not is_shutdown_requested():
        payload = {"timestamp": time.time(), "mode": "philosophy"}
        try:
            substrate = (
                ServiceContainer.get("continuous_substrate", default=None)
                or ServiceContainer.get("liquid_state", default=None)
                or ServiceContainer.get("liquid_substrate", default=None)
            )
            if substrate is not None:
                summary = substrate.get_state_summary() if hasattr(substrate, "get_state_summary") else {}
                if asyncio.iscoroutine(summary):
                    summary = await summary
                payload["substrate"] = summary
                if hasattr(substrate, "get_state_vector"):
                    vec = substrate.get_state_vector()
                elif hasattr(substrate, "x"):
                    vec = substrate.x
                else:
                    vec = []
                payload["trajectory_head"] = [round(float(x), 5) for x in list(vec)[:16]]
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            payload["substrate_error"] = str(exc)

        try:
            phi = ServiceContainer.get("phi_core", default=None)
            if phi and hasattr(phi, "get_live_phi"):
                payload["phi"] = float(phi.get_live_phi(include_surrogate=True))
            elif phi and hasattr(phi, "current_phi"):
                payload["phi"] = float(phi.current_phi)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            payload["phi_error"] = str(exc)

        try:
            affect = ServiceContainer.get("affect_engine", default=None) or ServiceContainer.get("affect_facade", default=None)
            if affect and hasattr(affect, "get_state_sync"):
                payload["affect"] = affect.get_state_sync()
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            payload["affect_error"] = str(exc)

        try:
            from core.will import get_will

            payload["will_receipts"] = get_will().get_recent_decisions(n=5)
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            payload["will_error"] = str(exc)

        try:
            overt = ServiceContainer.get("overt_action_loop", default=None)
            if overt is not None and hasattr(overt, "status"):
                payload["overt_action_loop"] = overt.status()
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)
            payload["overt_action_error"] = str(exc)

        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        await asyncio.sleep(interval_s)
    logger.info("🧾 Philosophy stream stopped after shutdown request.")

async def run_server_async(host: str, port: int):
    """API Server Mode (Unified Loop)"""
    import uvicorn

    from interface.server import app as fastapi_app
    logger.info("🚀 Starting API Server on %s:%s", host, port)
    
    server_config = uvicorn.Config(
        fastapi_app, host=host, port=port, log_level="info"
    )
    server = uvicorn.Server(server_config)
    await server.serve()


async def _stop_orchestrator_once(
    orchestrator: Any,
    *,
    reason: str,
    timeout_s: float = 30.0,
    request_global_shutdown: bool = True,
) -> None:
    """Stop the canonical runtime owner exactly once before process exit."""

    if orchestrator is None or getattr(orchestrator, "_aura_stop_invoked", False):
        return
    orchestrator._aura_stop_invoked = True
    if request_global_shutdown:
        request_shutdown(reason)
    stop = getattr(orchestrator, "stop", None)
    if not callable(stop):
        return
    try:
        result = stop()
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            await asyncio.wait_for(result, timeout=timeout_s)
    except TimeoutError as exc:
        record_degradation(
            "aura_main",
            exc,
            action="continued process shutdown after orchestrator stop exceeded bounded timeout",
            severity="degraded",
        )
        logger.error("Orchestrator stop timed out after %.1fs", timeout_s)
    except _AURA_MAIN_BOUNDARY_ERRORS as exc:
        record_degradation(
            "aura_main",
            exc,
            action="continued process shutdown after orchestrator stop failed",
            severity="degraded",
        )
        logger.error("Orchestrator stop failed: %s", exc, exc_info=True)


async def _wait_for_server_http(url: str, timeout_s: float = 60.0) -> bool:
    """Wait for internal API server to return 200 OK and report ready status."""
    from core.runtime.network_gateway import get_network_gateway

    start = time.time()

    logger.info("📡 Waiting for API Server health check: %s", url)
    count = 0
    while time.time() - start < timeout_s:
        count += 1
        try:
            response = await asyncio.to_thread(
                get_network_gateway().request,
                "GET",
                url,
                timeout=5.0,
                source="maintenance_tooling:server_health_wait",
                read_only=True,
                suppress_degradation=True,
            )
            if response.get("status_code") == 200:
                data = json.loads((response.get("content") or b"{}").decode("utf-8"))
                status = str(data.get("status", "")).lower()
                ready = bool(data.get("ready"))
                logger.info("📡 API Health status received: '%s'", status)
                if ready or status in ("online", "operational", "healthy", "ok", "ready"):
                    logger.info("✅ API Server is ONLINE and HEALTHY after %ds.", int(time.time() - start))
                    return True
                logger.warning("📡 API Server status is '%s', not yet 'online'. Full data: %s", status, data)
            elif count % 10 == 0:
                logger.info("📡 API Server not yet listening (Attempt %d)...", count)
        except _AURA_MAIN_BOUNDARY_ERRORS as e:
            record_degradation('aura_main', e)
            logger.error("📡 API Health check probe FAILURE: %s", e)

        await asyncio.sleep(1.0)
        
    logger.error("❌ API Server health check TIMEOUT after %.1fs", timeout_s)
    return False

def _native_launcher_owns_gui() -> bool:
    return (
        os.environ.get("AURA_EXTERNAL_GUI_OWNER", "").strip() == "1"
        or os.environ.get("AURA_LAUNCHED_FROM_APP", "").strip() == "1"
    )


async def run_desktop(port: int, *, launch_gui: bool | None = None, profile: str = "desktop"):
    """GUI Mode (Managed Actor Process)"""
    from core.container import ServiceContainer
    from core.graceful_shutdown import GracefulShutdown
    from core.supervisor.tree import ActorSpec
    from interface.gui_actor import gui_actor_entry
    
    supervisor = get_supervisor_tree()
    tracker = get_task_tracker()
    
    if launch_gui is None:
        launch_gui = not _native_launcher_owns_gui()

    if sys.platform == "darwin":
        try:
            import AppKit

            # Keep the kernel process dockless; the launcher or GUI actor owns
            # the visible desktop surface.
            app = AppKit.NSApplication.sharedApplication()
            app.setActivationPolicy_(2)  # NSApplicationActivationPolicyProhibited
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation("aura_main", exc)

    # 1. Start API Server (v21: Server now runs in Kernel)
    # [STABILITY] Start API immediately so port 8000 binds while brain thaws.

    host = "127.0.0.1" if config.security.internal_only_mode else "0.0.0.0"

    def _serve_api_sync():
        """Synchronous wrapper for uvicorn to run in a thread."""
        try:
            # Re-import inside thread to avoid loop issues
            import uvicorn

            from interface.server import app as _app
            server_config = uvicorn.Config(_app, host=host, port=port, log_level="info", loop="asyncio")
            server_config.handle_signals = False
            server = uvicorn.Server(server_config)
            stop_wait = threading.Event()

            def _api_shutdown_watcher():
                while not server.should_exit:
                    if is_shutdown_requested():
                        logger.info("📡 API Server received runtime shutdown request.")
                        server.should_exit = True
                        return
                    stop_wait.wait(0.25)

            threading.Thread(
                target=_api_shutdown_watcher,
                name="AuraAPIShutdownWatcher",
                daemon=True,
            ).start()
            logger.info("🚀 API Server (Kernel Thread) starting on port %s...", port)
            server.run()
            stop_wait.set()
            logger.info("📡 API Server thread has exited.")
        except _AURA_MAIN_BOUNDARY_ERRORS as e:
            record_degradation('aura_main', e)
            logger.critical("🛑 API THREAD CRITICAL FAILURE: %s", e, exc_info=True)

    async def _run_api_server():
        logger.info("📡 API Server task starting (offloading to thread for Apple Silicon stability)...")
        await asyncio.to_thread(_serve_api_sync)

    async def _wait_for_task_exit(task: asyncio.Task | None, *, name: str, timeout_s: float) -> None:
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(task, timeout=timeout_s)
        except TimeoutError as exc:
            record_degradation(
                "aura_main",
                exc,
                action=f"cancelled {name} after bounded desktop shutdown wait timed out",
                severity="degraded",
            )
            logger.warning("%s did not exit within %.1fs; cancelling.", name, timeout_s)
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError, RuntimeError) as cancel_exc:
                record_degradation(
                    "aura_main",
                    cancel_exc,
                    action=f"continued desktop shutdown after {name} cancellation wait ended",
                    severity="warning",
                )
        except asyncio.CancelledError:
            raise
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation(
                "aura_main",
                exc,
                action=f"continued desktop shutdown after {name} wait failed",
                severity="warning",
            )

    async def _desktop_final_shutdown(
        *,
        orchestrator: Any | None,
        api_task: asyncio.Task | None,
        reason: str,
    ) -> None:
        try:
            supervisor._is_running = False
            supervisor._shutting_down = True
        except AttributeError as exc:
            record_degradation("aura_main", exc)

        await _stop_orchestrator_once(
            orchestrator,
            reason=reason,
            timeout_s=20.0,
            request_global_shutdown=False,
        )
        request_shutdown(reason)
        await _wait_for_task_exit(api_task, name="api_server", timeout_s=8.0)
        try:
            await asyncio.wait_for(GracefulShutdown.trigger_shutdown(reason), timeout=20.0)
        except TimeoutError as exc:
            record_degradation(
                "aura_main",
                exc,
                action="continued desktop shutdown after graceful shutdown timed out",
                severity="degraded",
            )
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation(
                "aura_main",
                exc,
                action="continued desktop shutdown after graceful shutdown failed",
                severity="degraded",
            )

        try:
            await asyncio.wait_for(get_task_tracker().shutdown(timeout=3.0), timeout=5.0)
        except TimeoutError as exc:
            record_degradation(
                "aura_main",
                exc,
                action="continued desktop shutdown after task tracker drain timed out",
                severity="degraded",
            )
        except _AURA_MAIN_BOUNDARY_ERRORS as exc:
            record_degradation(
                "aura_main",
                exc,
                action="continued desktop shutdown after task tracker drain failed",
                severity="warning",
            )

    async def _main_loop():
        loop = asyncio.get_running_loop()
        installed_signal_handlers: list[signal.Signals] = []
        orchestrator = None
        api_task: asyncio.Task | None = None

        def _request_desktop_shutdown(sig: signal.Signals) -> None:
            nonlocal shutdown_reason
            shutdown_reason = f"desktop_signal:{sig.name}"
            try:
                supervisor._is_running = False
                supervisor._shutting_down = True
            except AttributeError as exc:
                record_degradation("aura_main", exc)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_desktop_shutdown, sig)
                installed_signal_handlers.append(sig)
            except (RuntimeError, AttributeError, NotImplementedError, ValueError) as exc:
                logger.debug("Desktop signal handler registration skipped for %s: %s", sig, exc)

        shutdown_reason = "desktop_exit"
        try:
        
            # 1. Initialize Orchestrator and wait for boot
            logger.info("🧠 Orchestrator boot beginning...")
            orchestrator = await boot_aura_runtime(
                profile=profile,
                ready_label="Desktop",
                readiness_context="server_boot",
            )
            tracker.create_task(orchestrator.run(), name="OrchestratorMainLoop")

            # 2. Start API Server (v21: Server now runs in Kernel)
            # [STABILITY] Start API after brain is ready to ensure correct ServiceContainer lookups.
            logger.info("🎬 [DEBUG] Pre-starting API server mission...")
            api_task = tracker.create_task(_run_api_server(), name="api_server")
            logger.info("🎬 [DEBUG] API server task created successfully.")

            # Wait for API server to be TRULY ready (HTTP 200)
            # This prevents the GUI from launching too early and hitting "Connection Refused".
            health_url = f"http://127.0.0.1:{port}/api/health/boot"
            logger.info("⏳ Waiting for API health check on port %s...", port)
            try:
                desktop_health_wait_s = float(os.environ.get("AURA_DESKTOP_HEALTH_WAIT_SECONDS", "90"))
            except ValueError:
                desktop_health_wait_s = 90.0
            if await _wait_for_server_http(health_url, desktop_health_wait_s):
                logger.info("✅ API Server is HEALTHY. Proceeding to GUI launch.")
            else:
                logger.warning(
                    "⚠️ API Server did not report full readiness after %.0fs; launching GUI with readiness heartbeat gating.",
                    desktop_health_wait_s,
                )
            
            # 3. Start GUI Actor (WebView Only). When the native launcher owns
            # the visible GUI, the runtime stays headless here so one click does
            # not create two Dock-visible Python/WebView processes.
            if not launch_gui:
                pipe = None
                logger.info("🎨 Desktop runtime launched without child GUI; external launcher owns the window.")
            elif sys.platform == "darwin":
                logger.info("🎨 Launching GUI via SUBPROCESS for macOS stability...")

                async def _gui_reaper_loop():
                    """Re-implements supervision for the subprocess-based macOS GUI."""
                    max_restarts = 5
                    restart_count = 0
                    while restart_count < max_restarts:
                        with local_internal_governed_scope(
                            "environment_action:gui_actor_reaper",
                            domain="environment_action",
                        ):
                            proc = await get_subprocess_gateway().spawn_async(
                                [_launcher_python_executable(), "interface/gui_actor.py", str(port)],
                                cwd=str(PROJECT_ROOT),
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                start_new_session=True,
                                source="environment_action:gui_actor_reaper",
                            )
                        logger.info("🎨 GUI Process Started (PID: %s)", proc.pid)

                        async def _stream_logger(stream, level):
                            content = []
                            while line := await stream.readline():
                                decoded = line.decode('utf-8', errors='replace').rstrip()
                                if decoded:
                                    if level == "ERROR":
                                        logger.error("[GUI] %s", decoded)
                                    else:
                                        logger.debug("[GUI] %s", decoded)
                                    content.append(decoded)
                            return "\n".join(content)

                        out_task = tracker.create_task(
                            _stream_logger(proc.stdout, "DEBUG"),
                            name="gui.stdout_stream",
                        )
                        err_task = tracker.create_task(
                            _stream_logger(proc.stderr, "ERROR"),
                            name="gui.stderr_stream",
                        )

                        # Watch for exit
                        while proc.returncode is None:
                            # Check for system-wide shutdown
                            if not getattr(supervisor, "_is_running", True):
                                try:
                                    proc.terminate()
                                except ProcessLookupError:
                                    logger.debug("GUI process already exited before termination.")
                                return

                            try:
                                # Wait with timeout to allow checking shutdown flag
                                await asyncio.wait_for(proc.wait(), timeout=2.0)
                            except TimeoutError:
                                continue

                        # Ensure stream reading completes
                        await out_task
                        stderr_output = await err_task
                        
                        if proc.returncode == 0:
                            # User closed the window cleanly — treat this as "quit
                            # Aura", not "restart the GUI in the background".
                            # Otherwise the orchestrator stays alive pinned to
                            # MLX workers while the user believes they've quit,
                            # which is how "multiple versions in the background"
                            # happens.
                            logger.info("🎨 GUI closed by user — initiating full shutdown.")
                            shutdown_reason = "gui_closed"
                            try:
                                supervisor._is_running = False
                            except AttributeError as exc:
                                record_degradation("aura_main", exc)
                                logger.debug("GUI supervisor did not expose running flag: %s", exc)
                            return

                        if is_shutdown_requested() or proc.returncode in {-signal.SIGTERM, -signal.SIGINT}:
                            logger.info(
                                "🎨 GUI process ended during runtime shutdown (code=%s); not restarting.",
                                proc.returncode,
                            )
                            return

                        restart_count += 1
                        logger.critical("🛑 GUI Process crashed (code: %s). Reason:\n%s", proc.returncode, stderr_output)
                        logger.warning("🎨 Restarting GUI in 5s... (Attempt %s/%s)", restart_count, max_restarts)
                        await asyncio.sleep(5.0)

                tracker.create_task(_gui_reaper_loop(), name="gui_reaper")
                pipe = None # Subprocess doesn't use the actor pipe
            else:
                # Linux/Others can still use the supervised actor
                spec = ActorSpec(
                    name="desktop_gui",
                    entry_point=gui_actor_entry,
                    args=(port,),
                    restart_policy="always"
                )
                supervisor.add_actor(spec)
                pipe = supervisor.start_actor("desktop_gui")
            
            # 4. Register GUI in ActorBus
            actor_bus = ServiceContainer.get("actor_bus", default=None)
            if actor_bus and launch_gui:
                actor_bus.add_actor("desktop_gui", pipe, is_child=True)

            if launch_gui:
                logger.info("🎨 Desktop GUI Actor launched and supervised (WebView-only mode).")

            # Wait until the supervisor sees shutdown. The explicit finalizer
            # below owns teardown so SIGTERM cannot leave API/model/GUI workers
            # alive behind the desktop process.
            await supervisor.wait_forever()
        except asyncio.CancelledError:
            shutdown_reason = "desktop_cancelled"
            raise
        finally:
            for sig in installed_signal_handlers:
                try:
                    loop.remove_signal_handler(sig)
                except (RuntimeError, AttributeError, NotImplementedError, ValueError) as exc:
                    logger.debug("Desktop signal handler cleanup skipped for %s: %s", sig, exc)
            await _desktop_final_shutdown(
                orchestrator=orchestrator,
                api_task=api_task,
                reason=shutdown_reason,
            )

    await _main_loop()

def _watchdog_child_args(args: argparse.Namespace | None = None) -> list[str]:
    """Build restart args that preserve the mode the watchdog was asked to supervise."""
    if args is None:
        return ["--desktop"]

    child_args: list[str] = []
    if getattr(args, "gui_window", False):
        child_args.append("--gui-window")
    elif getattr(args, "headless", False):
        child_args.append("--headless")
    elif getattr(args, "server", False):
        child_args.append("--server")
    elif getattr(args, "cli", False):
        child_args.append("--cli")
    else:
        child_args.append("--desktop")

    if getattr(args, "skeletal", False):
        child_args.append("--skeletal")

    host = str(getattr(args, "host", "") or "").strip()
    if host:
        child_args.extend(["--host", host])
    port = getattr(args, "port", None)
    if port is not None:
        child_args.extend(["--port", str(port)])
    return child_args


async def run_watchdog(args: argparse.Namespace | None = None):
    """Stability Watchdog Loop (Async)."""
    # [STABILITY] Ensure only one Watchdog is active.
    # This prevents the "two instances" issue where multiple supervisors
    # compete for the orchestrator lock and GPU memory.
    acquire_instance_lock(lock_name="watchdog")
    
    logger.info("🛡️ Watchdog supervisor active (supervision-only mode).")
    child_args = _watchdog_child_args(args)
    logger.info("🛡️ Watchdog restart command preserves mode args: %s", " ".join(child_args))

    try:
        restart_count = 0
        while restart_count < 10:
            logger.info("🛡️  Watchdog: Launching Aura (Attempt %s)", restart_count+1)
            start_time = time.time()
            
            # Use the canonical gateway for non-blocking child supervision.
            with local_internal_governed_scope(
                "environment_action:watchdog_supervisor",
                domain="environment_action",
            ):
                proc = await get_subprocess_gateway().spawn_async(
                    [_launcher_python_executable(), __file__, *child_args],
                    source="environment_action:watchdog_supervisor",
                )
            await proc.wait()
            
            # Perplexity Audit Fix: Detect deterministic config errors (Exit 1)
            if proc.returncode == 1:
                logger.error("🛡️  Watchdog: Fatal configuration or security error (Code 1). Aborting restart loop.")
                break

            if proc.returncode == 0:
                logger.info("Clean shutdown detected. Watchdog exiting.")
                break

            # If ran for > 10 mins, reset counter
            if time.time() - start_time > 600:
                restart_count = 0
                
            restart_count += 1
            # Exponential backoff: 5, 10, 20, 40, 60...
            delay = min(60, 5 * (2 ** max(0, restart_count - 1)))
            logger.warning("Crash detected (Code: %s). Restarting in %ss...", proc.returncode, delay)
            await asyncio.sleep(delay)
    finally:
        release_instance_lock()

def _reap_orphaned_aura_processes() -> int:
    """Kill any stale Aura main processes owned by this user before taking the
    singleton lock.

    Why: the singleton lock file is cleared on clean exit, but if a previous
    launch hard-crashed (SIGKILL, power cut, WebView hang that left the
    supervisor orphaned) the orchestrator process can live on as a headless
    background job consuming 3–6 GB of RAM and pinning MLX worker subprocesses.
    Users then relaunch, see the new window, close it, and end up with two (or
    three, or five) full Aura stacks running simultaneously — which is exactly
    what the "multiple versions eating up resources" report describes.

    We match processes conservatively: only Python interpreters whose argv
    contains aura_main.py and whose PID is NOT ours or our parent's, and only
    processes owned by the current user. TERM first, KILL after a short grace
    window.
    """
    if sys.platform not in ("darwin", "linux"):
        return 0
    me = os.getpid()
    parent = os.getppid()
    killed = 0
    try:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        proc = get_subprocess_gateway().run(
            ["ps", "-axo", "pid=,user=,command="],
            timeout=5,
            source="maintenance_tooling:process_reaper",
            offline_tooling=True,
        )
        out = proc.stdout
    except (subprocess.SubprocessError, OSError) as exc:
        record_degradation("aura_main", exc)
        logger.warning("Unable to inspect process table for stale Aura processes: %s", exc)
        return 0
    current_user = os.environ.get("USER") or ""
    stale_pids: list[int] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        user = parts[1]
        cmd = parts[2]
        if pid in (me, parent):
            continue
        if current_user and user != current_user:
            continue
        # Target main orchestrator, GUI actors, and MLX workers owned by this user
        is_stale_aura = (
            "aura_main.py" in cmd
            or "gui_actor.py" in cmd
            or "mlx_worker" in cmd
            or "MLXWorker" in cmd
        )
        if not is_stale_aura:
            continue
        # Skip this launcher/reaper context
        if "reaper" in cmd.lower():
            continue
        stale_pids.append(pid)
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil:
        for pid in stale_pids:
            try:
                parent_proc = psutil.Process(pid)
                # Terminate children recursively first to prevent orphans
                for child in parent_proc.children(recursive=True):
                    try:
                        child.terminate()
                    except psutil.Error:
                        continue
                parent_proc.terminate()
                killed += 1
            except psutil.Error:
                continue
    else:
        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                continue
            except PermissionError:
                continue

    if stale_pids:
        time.sleep(1.5)
        if psutil:
            for pid in stale_pids:
                try:
                    parent_proc = psutil.Process(pid)
                    for child in parent_proc.children(recursive=True):
                        try:
                            if child.is_running():
                                child.kill()
                        except psutil.Error:
                            continue
                    if parent_proc.is_running():
                        parent_proc.kill()
                except psutil.Error:
                    continue
        else:
            for pid in stale_pids:
                try:
                    os.kill(pid, 0)  # still alive?
                except OSError:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    logger.debug("Stale Aura process %s exited before SIGKILL: %s", pid, exc)
        logger.warning(
            "🧹 Reaped %d orphaned Aura process(es) before boot: %s",
            killed, stale_pids,
        )
    return killed


def bootstrap_lock(skip_lock: bool = False):
    """Bridge to the shared singleton utility."""
    global _RUNTIME_LOCK_CLAIMED
    if _RUNTIME_LOCK_CLAIMED and not skip_lock:
        return
    # Clean up orphaned stacks from prior hard-crashes before grabbing the lock.
    _reap_orphaned_aura_processes()
    acquire_instance_lock(lock_name="orchestrator", skip_lock=skip_lock)
    if not skip_lock:
        _RUNTIME_LOCK_CLAIMED = True


def _unlink_orchestrator_lock(lock_file: Path) -> None:
    lock_file.unlink(missing_ok=True)
    instance_lock_metadata_path("orchestrator").unlink(missing_ok=True)


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return str(left) == str(right)


def _lock_pid_matches_aura_runtime(pid: int, metadata: dict[str, Any]) -> tuple[bool, str]:
    """Verify a lock PID is the runtime that wrote it before sending signals."""
    try:
        import psutil

        proc = psutil.Process(pid)
        cmdline = [str(part) for part in (proc.cmdline() or [])]
        command = " ".join(cmdline).lower()
        cwd = proc.cwd()
        actual_create_time = float(proc.create_time())
    except ImportError:
        return True, "psutil_unavailable_legacy_signal_check"
    except psutil.NoSuchProcess:
        return False, "pid_not_running"
    except (psutil.AccessDenied, psutil.ZombieProcess, OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, f"pid_identity_unavailable:{type(exc).__name__}"

    expected_pid = metadata.get("pid") if metadata else None
    if expected_pid is not None:
        try:
            if int(expected_pid) != int(pid):
                return False, "metadata_pid_mismatch"
        except (TypeError, ValueError):
            return False, "metadata_pid_invalid"

    expected_create_time = metadata.get("create_time") if metadata else None
    if expected_create_time is not None:
        try:
            if abs(actual_create_time - float(expected_create_time)) > 0.05:
                return False, "pid_reused_or_stale"
        except (TypeError, ValueError):
            return False, "metadata_create_time_invalid"

    expected_cwd = metadata.get("cwd") if metadata else ""
    if expected_cwd and not _same_path(str(expected_cwd), cwd):
        return False, "metadata_cwd_mismatch"

    project_match = _same_path(cwd, PROJECT_ROOT)
    explicit_aura_command = any(
        marker in command
        for marker in (
            "aura_main.py",
            "run_dnu_agi_proof_battery.py",
            "run_external_live_validation.py",
            "run_agency_emergence_battery.py",
            "run_continual_learning_battery.py",
            "run_novel_environment_battery.py",
            "run_unified_aura_scenario.py",
            "run_longevity_soak.py",
            "certify_boot.py",
        )
    )
    if expected_create_time is not None and project_match:
        return True, "metadata_identity_verified"
    if project_match and explicit_aura_command:
        return True, "legacy_identity_verified"
    return False, "pid_not_owned_by_aura_runtime"


def stop_aura():
    """Reads PID from lock file and sends SIGTERM. Also unloads launchd agent."""
    lock_file = Path.home() / ".aura" / "locks" / "orchestrator.lock"
    
    # 1. Unload Launchd Agent (Prevents auto-revival on macOS)
    if sys.platform == "darwin":
        plist_path = Path.home() / "Library/LaunchAgents/com.aura.sovereign.plist"
        if plist_path.exists():
            logger.info("Unloading launchd daemon to prevent auto-revival...")
            try:
                from core.runtime.subprocess_gateway import get_subprocess_gateway

                get_subprocess_gateway().run(
                    ["launchctl", "unload", str(plist_path)],
                    capture_output=True,
                    timeout=5,
                    source="maintenance_tooling:stop_aura",
                    offline_tooling=True,
                )
            except subprocess.TimeoutExpired:
                logger.warning("launchctl unload timed out.")
            except _AURA_MAIN_BOUNDARY_ERRORS as e:
                record_degradation('aura_main', e)
                logger.warning("launchctl unload failed: %s", e)
            
    if not lock_file.exists():
        print("Aura does not appear to be running (no lock file found).")
        return

    try:
        with open(lock_file) as f:
            lock_text = f.read()
            pid = parse_instance_lock_pid(lock_text)
            if pid is None:
                print("Lock file found but no PID recorded.")
                return
        metadata = read_instance_lock_metadata("orchestrator")
        
        verified, verify_reason = _lock_pid_matches_aura_runtime(pid, metadata)
        if not verified:
            print(
                f"⚠️  Lock file PID {pid} is not a verified Aura runtime "
                f"({verify_reason}). Cleaning stale lock."
            )
            _unlink_orchestrator_lock(lock_file)
            return

        try:
            import psutil
        except ImportError:
            psutil = None

        print(f"Stopping Aura (PID: {pid})...")
        if psutil:
            try:
                p = psutil.Process(pid)
                # Signal the verified parent first. Child actors are owned by
                # the runtime shutdown contract; terminating them here races
                # final state commits and forces replay/direct-snapshot paths.
                p.send_signal(signal.SIGTERM)
            except psutil.Error:
                print("Process already dead or inaccessible.")
                _unlink_orchestrator_lock(lock_file)
                return
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                print("Process already dead or inaccessible.")
                _unlink_orchestrator_lock(lock_file)
                return

        # Wait for cleanup. The desktop/32B lane needs enough time to stop
        # the API server, persist snapshots, unload MLX workers, and reap
        # actors. Five seconds made healthy shutdowns look stubborn and
        # forced SIGKILL while teardown was still progressing.
        stop_grace_s = max(5.0, float(os.environ.get("AURA_STOP_GRACE_SECONDS", "30")))
        stopped_cleanly = False
        if psutil:
            try:
                p.wait(timeout=stop_grace_s)
                stopped_cleanly = True
            except psutil.TimeoutExpired:
                stopped_cleanly = False
            except psutil.Error:
                stopped_cleanly = True
        else:
            deadline = time.monotonic() + stop_grace_s
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0) # Check if alive
                except (ProcessLookupError, PermissionError):
                    stopped_cleanly = True
                    break
                time.sleep(0.5)
        if not stopped_cleanly:
            print("Aura is stubborn. Sending SIGKILL...")
            if psutil:
                try:
                    p = psutil.Process(pid)
                    for child in p.children(recursive=True):
                        try:
                            if child.is_running():
                                child.kill()
                        except psutil.Error:
                            pass
                    if p.is_running():
                        p.kill()
                except psutil.Error:
                    pass
            else:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError) as exc:
                    logger.debug("Process %s unavailable for SIGKILL during stop: %s", pid, exc)
            
        # Force remove lock if still there
        if lock_file.exists():
            try:
                _unlink_orchestrator_lock(lock_file)
            except OSError as exc:
                record_degradation("aura_main", exc)
                print(f"Failed to remove Aura lock file: {exc}")
            
        print("✅ Aura stopped successfully.")
    except _AURA_MAIN_BOUNDARY_ERRORS as e:
        record_degradation('aura_main', e)
        print(f"Failed to stop Aura: {e}")

# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    operator_commands = {
        "doctor", "conformance", "backup", "restore", "migrate",
        "verify-state", "verify-memory", "rebuild-index", "chaos", "plugin"
    }
    if len(sys.argv) > 1 and sys.argv[1] in operator_commands:
        from core.runtime import operator_cli
        sys.exit(operator_cli.main(sys.argv[1:]))

    _maybe_relaunch_with_preferred_python()
    reaper_manifest_path = _ensure_reaper_manifest_env()

    parser = argparse.ArgumentParser(description="Aura Unified Entry Point")
    parser.add_argument("--cli", action="store_true", help="Interactive Console Mode")
    parser.add_argument("--server", action="store_true", help="API Server Mode")
    parser.add_argument("--headless", action="store_true", help="Headless local mode (API server only, no desktop GUI)")
    parser.add_argument("--desktop", action="store_true", help="Desktop GUI Mode")
    parser.add_argument("--gui-window", action="store_true", help="Open a desktop GUI window attached to an existing Aura server")
    parser.add_argument("--watchdog", action="store_true", help="Watchdog / Keep-alive Mode")
    parser.add_argument("--philosophy", action="store_true", help="Stream live substrate/phi/affect/Will proof surface as JSONL")
    parser.add_argument("--stop", action="store_true", help="Stop any running Aura instance")
    parser.add_argument("--reboot", action="store_true", help="Force cleanup and restart (Standardize)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Server/GUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Server")
    parser.add_argument("--skeletal", action="store_true", help="Skeletal Mode: Bypass heavy subsystems")
    parser.add_argument("--profile", type=str, default=None, help="Boot profile (e.g. minimal)")
    
    args = parser.parse_args()

    # Publish the serving port for in-process surfaces (voice wake-word
    # dispatch) that route through the canonical /api/chat loopback lane.
    os.environ["AURA_SERVER_PORT"] = str(args.port)

    # Desktop/headless live sessions run alongside the GUI, browser probes, and
    # other macOS apps. Default them into safe boot so Cortex warmup is admitted
    # by live RAM headroom instead of being scheduled optimistically during boot.
    if (args.desktop or args.headless) and "AURA_SAFE_BOOT_DESKTOP" not in os.environ:
        os.environ["AURA_SAFE_BOOT_DESKTOP"] = "1"
    
    # Standardize: Reboot behavior
    if args.stop:
        stop_aura()
        sys.exit(0)

    if args.skeletal:
        logger.info("💀 SKELETAL MODE ACTIVATED")
        config.skeletal_mode = True
        # Force-disable heavy components in config
        if hasattr(config, "soma"):
            config.soma.enabled = False
        config.features.mycelium_visualizer = False
        config.features.autonomous_impulses = False

    if args.reboot:
        logger.info("🔄 REBOOT SEQUENCE ACTIVATED")
        clean_artifacts()
        stop_aura()
        time.sleep(1.0)
        # Default to desktop if no other mode specified
        if not (args.cli or args.server or args.desktop):
            args.desktop = True

    if args.headless:
        logger.info("🖥️ HEADLESS MODE ACTIVATED")
        args.server = True
        args.desktop = False
        args.gui_window = False
        os.environ["AURA_HEADLESS"] = "1"
        os.environ.setdefault("AURA_EAGER_LOCAL_SENSORY_BOOT", "0")
        os.environ.setdefault("AURA_ENABLE_PROACTIVE_VISION", "0")
        # Headless demo mode should stay local even when public API mode is enabled.
        args.host = "127.0.0.1"
    elif args.desktop:
        os.environ.setdefault("AURA_EAGER_LOCAL_SENSORY_BOOT", "0")

    if not args.gui_window:
        check_environment()
    
    # uvloop activation is opt-in on macOS because native media stacks have
    # repeatedly crashed under the libuv event loop during desktop boot.
    from core.runtime.boot_safety import uvloop_allowed

    if not args.gui_window and uvloop_allowed():
        try:
            import uvloop
            # We'll set the policy inside the setup phase if needed,
            # but asyncio.run handles loop creation itself in 3.11+.
            # Setting policy here is safe as long as no loop exists.
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("⚡ uvloop activated for maximum concurrency performance.")
        except ImportError:
            logger.debug("uvloop not installed. Using standard asyncio event loop.")
        except _AURA_MAIN_BOUNDARY_ERRORS as e:
            record_degradation('aura_main', e)
            logger.warning("Could not set uvloop policy: %s", e)
    else:
        logger.info(
            "🛡️ uvloop disabled for this runtime profile. "
            "Set AURA_ENABLE_UVLOOP=1 to force-enable it."
        )
    
    # Do not acquire lock for watchdog, since the watchdog itself runs the child process
    if not args.gui_window:
        bootstrap_lock(skip_lock=args.watchdog)

    # Only the lock owner is allowed to reclaim ports or spawn the reaper.
    # This prevents a second boot attempt from disrupting a healthy live Aura
    # instance before the singleton fence has a chance to reject it.
    if not args.cli and not args.gui_window and not args.watchdog:
        logger.info("🧹 Pre-clearing known ports...")
        kill_port(args.port)
        kill_port(10003, pattern="aura")

    # SIGKILL Reaper Initialization
    if not args.gui_window and not args.watchdog:
        try:
            from core.reaper import reaper_loop
            reaper_proc = multiprocessing.Process(
                target=reaper_loop, 
                args=(os.getpid(), reaper_manifest_path),
                daemon=True,
                name="AuraReaper"
            )
            reaper_proc.start()
            logger.info("🛡️  REAPER ACTIVE (Survives SIGKILL). Monitoring Kernel PID: %s", os.getpid())
        except _AURA_MAIN_BOUNDARY_ERRORS as e:
            logger.error("⚠️ Reaper initialization skipped or failed: %s", e)

    # Perplexity Audit Fix: Use asyncio.run for cleaner entry points
    try:
        if args.philosophy:
            asyncio.run(run_philosophy_stream(args.port))
        elif args.server:
            # Dynamic host selection if default was used
            host = args.host
            if (
                host == "127.0.0.1"
                and not args.headless
                and not getattr(config.security, "internal_only_mode", False)
            ):
                host = "0.0.0.0"
            async def _run_server_with_bootstrap():
                orchestrator = await boot_aura_runtime(
                    profile=args.profile or "server",
                    ready_label="Server",
                    readiness_context="server_boot",
                )
                from core.utils.task_tracker import get_task_tracker

                get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
                try:
                    await run_server_async(host, args.port)
                finally:
                    from core.graceful_shutdown import GracefulShutdown

                    await _stop_orchestrator_once(orchestrator, reason="server_exit")
                    try:
                        await asyncio.wait_for(
                            GracefulShutdown.trigger_shutdown("server_exit"),
                            timeout=20.0,
                        )
                    except TimeoutError as exc:
                        record_degradation(
                            "aura_main",
                            exc,
                            action="continued server exit after graceful shutdown trigger timed out",
                            severity="degraded",
                        )
                    try:
                        await asyncio.wait_for(
                            GracefulShutdown.wait_for_shutdown(),
                            timeout=5.0,
                        )
                    except TimeoutError as exc:
                        record_degradation(
                            "aura_main",
                            exc,
                            action="continued server exit after graceful shutdown wait timed out",
                            severity="warning",
                        )
            asyncio.run(_run_server_with_bootstrap())
        elif args.desktop:
            # For desktop, we'll need a way to bootstrap the loop if uvicorn starts it
            # But Desktop mode in aura_main runs uvicorn in a thread.
            # We should probably bootstrap the main thread for the GUI if it needs it.
            asyncio.run(
                run_desktop(
                    args.port,
                    launch_gui=None,
                    profile=args.profile or "desktop",
                )
            )
        elif args.gui_window:
            from interface.gui_actor import gui_actor_entry

            logger.info("🪟 Opening Aura desktop window on port %s...", args.port)
            gui_actor_entry(args.port)
        elif args.watchdog:
            asyncio.run(run_watchdog(args))
        elif args.cli:
            
            asyncio.run(run_console(profile=args.profile or "cli"))
        else:
            # Default fallback: Desktop if double-clicked, else CLI if terminal
            if sys.stdin and sys.stdin.isatty():
                asyncio.run(run_console(profile=args.profile or "cli"))
            else:
                logger.info("Initializing in Desktop/Autonomy mode...")
                asyncio.run(run_desktop(args.port, profile=args.profile or "desktop"))
    except KeyboardInterrupt:
        request_shutdown("keyboard_interrupt")
        logger.info("Shutdown requested by user.")
    except _AURA_MAIN_BOUNDARY_ERRORS as e:
        record_degradation('aura_main', e)
        logger.critical("FATAL BOOT ERROR: %s", e, exc_info=True)
        sys.exit(1)
    _finalize_root_runtime_process_exit(args, exit_code=0)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
