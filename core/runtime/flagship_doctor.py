"""Aura Flagship Doctor.

A one-command operational doctor for flagship readiness. This module avoids
importing heavyweight Aura subsystems at module import time. It checks the repo
layout, runtime version, expected guard modules, known ports, optional local log
evidence, and available quality gates.

It is intentionally conservative: a PASS means "basic operational readiness
checks are green", not "Aura is proven conscious" or "the whole product is
perfect".
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway


@dataclass
class DoctorFinding:
    code: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    root: str
    created_at: float
    overall: str
    findings: list[DoctorFinding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.flagship.doctor.v1",
            "root": self.root,
            "created_at": self.created_at,
            "overall": self.overall,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=repr)


REQUIRED_FILES = [
    "aura_main.py",
    "core/runtime/task_ownership.py",
    "core/runtime/persistence_ownership.py",
    "core/runtime/flagship_readiness.py",
    "core/morphogenesis/runtime.py",
    "core/morphogenesis/hooks.py",
    "core/morphogenesis/registry.py",
]

EXPECTED_SCRIPTS = [
    "scripts/aura_task_ownership_codemod.py",
    "scripts/aura_persistence_audit.py",
    "scripts/aura_collect_flagship_evidence.py",
]

EXPECTED_PORTS = [8000, 9090, 10003]


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return False


def _run(cmd: list[str], root: Path, timeout: float = 45.0) -> dict[str, Any]:
    started = time.time()
    try:
        proc = get_subprocess_gateway().run(
            cmd,
            cwd=str(root),
            capture_output=True,
            timeout=timeout,
            source="maintenance_tooling:flagship_doctor",
            offline_tooling=True,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
            "duration_s": round(time.time() - started, 3),
        }
    except (subprocess.SubprocessError, OSError) as exc:
        record_degradation('flagship_doctor', exc)
        return {
            "cmd": cmd,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.time() - started, 3),
        }


def _log_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (root / "logs", Path.home() / ".aura" / "logs"):
        if base.exists():
            try:
                candidates.extend(sorted(base.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:10])
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("Flagship doctor log candidate scan skipped for %s: %s", base, exc)
    return candidates


def _tail(path: Path, max_chars: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return ""


def check_python() -> DoctorFinding:
    ok = sys.version_info >= (3, 12)
    return DoctorFinding(
        code="python_version",
        status="pass" if ok else "fail",
        message=f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        suggestion="Use Python 3.12+." if not ok else "",
    )


def check_layout(root: Path) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    for rel in REQUIRED_FILES:
        exists = (root / rel).exists()
        findings.append(
            DoctorFinding(
                code="required_file",
                status="pass" if exists else "fail",
                message=f"{rel} {'exists' if exists else 'missing'}",
                detail={"path": rel},
                suggestion=f"Restore or apply patch that provides {rel}." if not exists else "",
            )
        )
    for rel in EXPECTED_SCRIPTS:
        exists = (root / rel).exists()
        findings.append(
            DoctorFinding(
                code="expected_script",
                status="pass" if exists else "warn",
                message=f"{rel} {'exists' if exists else 'missing'}",
                detail={"path": rel},
                suggestion=f"Apply closure patches to install {rel}." if not exists else "",
            )
        )
    return findings


def check_aura_main(root: Path) -> list[DoctorFinding]:
    path = root / "aura_main.py"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    checks = [
        ("asyncio_patch_import", "core.utils.asyncio_patch" in text, "Import core.utils.asyncio_patch near the top of aura_main.py."),
        ("morphogenesis_boot", "start_morphogenesis_runtime" in text, "Start morphogenesis after core services are registered."),
        ("python_312_guard", "Python 3.12+" in text or "sys.version_info < (3, 12)" in text, "Normalize runtime guard to Python 3.12+."),
        ("security_guard", "validate_security_config" in text and "AURA_API_TOKEN" in text, "Keep fail-closed API security validation."),
    ]
    out: list[DoctorFinding] = []
    for code, ok, suggestion in checks:
        out.append(
            DoctorFinding(
                code=code,
                status="pass" if ok else "warn",
                message=f"{code}: {'present' if ok else 'not detected'}",
                suggestion="" if ok else suggestion,
            )
        )
    return out


def check_ports() -> list[DoctorFinding]:
    out: list[DoctorFinding] = []
    for port in EXPECTED_PORTS:
        open_ = _port_open(port)
        # Open ports are informational, not failure: Aura may not be running.
        out.append(
            DoctorFinding(
                code="port_probe",
                status="info",
                message=f"127.0.0.1:{port} {'open' if open_ else 'closed'}",
                detail={"port": port, "open": open_},
            )
        )
    return out


def check_logs(root: Path) -> list[DoctorFinding]:
    logs = _log_candidates(root)
    if not logs:
        return [
            DoctorFinding(
                code="logs",
                status="warn",
                message="No local Aura log files found.",
                suggestion="Run Aura once, then collect flagship evidence.",
            )
        ]
    joined = "\n".join(_tail(p) for p in logs[:5])
    probes = [
        ("log_morphogenesis_started", "MorphogeneticRuntime started" in joined),
        ("log_hooks_wired", "Morphogenesis hooks" in joined or "Morphogenesis hooks wired" in joined),
        ("log_task_supervisor", "Task Supervisor active" in joined or "TaskTracker" in joined),
        ("log_consciousness_online", "Consciousness System ONLINE" in joined),
    ]
    out: list[DoctorFinding] = [
        DoctorFinding("logs_found", "pass", f"Found {len(logs)} log file(s).", {"logs": [str(p) for p in logs[:5]]})
    ]
    for code, ok in probes:
        out.append(
            DoctorFinding(
                code=code,
                status="pass" if ok else "warn",
                message=f"{code}: {'detected' if ok else 'not detected in recent logs'}",
                suggestion="Run a fresh boot and collect evidence." if not ok else "",
            )
        )
    return out


def check_optional_gates(root: Path) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    commands = [
        ("flagship_readiness", [sys.executable, "-m", "core.runtime.flagship_readiness", "--json", "."]),
        ("task_ownership_report", [sys.executable, "scripts/aura_task_ownership_codemod.py", ".", "--json"]),
        ("persistence_audit", [sys.executable, "scripts/aura_persistence_audit.py", ".", "--json"]),
    ]
    for code, cmd in commands:
        module_or_script = cmd[1]
        runnable = module_or_script == "-m" or (root / module_or_script).exists()
        if not runnable:
            findings.append(DoctorFinding(code, "warn", f"{module_or_script} not available."))
            continue
        result = _run(cmd, root)
        rc = result.get("returncode")
        status = "pass" if rc == 0 else "warn"
        findings.append(DoctorFinding(code, status, f"{code} return code: {rc}", result))
    return findings


def run_doctor(root: str | Path, *, include_gates: bool = True) -> DoctorReport:
    root = Path(root).resolve()
    findings: list[DoctorFinding] = []
    findings.append(check_python())
    findings.extend(check_layout(root))
    findings.extend(check_aura_main(root))
    findings.extend(check_ports())
    findings.extend(check_logs(root))
    if include_gates:
        findings.extend(check_optional_gates(root))

    if any(f.status == "fail" for f in findings):
        overall = "fail"
    elif any(f.status == "warn" for f in findings):
        overall = "warn"
    else:
        overall = "pass"

    return DoctorReport(root=str(root), created_at=time.time(), overall=overall, findings=findings)


import gc
import logging
import sqlite3
import threading

from core.runtime.shutdown_coordinator import is_shutdown_requested

logger = logging.getLogger("Aura.FlagshipDoctor")


class FlagshipDoctorDaemon:
    """Active background doctor daemon for event-loop latency tracking and database/memory self-healing."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        check_interval: float = 1.0,
        lag_threshold: float = 5.0,
        ram_threshold: float = 90.0,
    ) -> None:
        self.root = Path(root_dir or ".").resolve()
        self.check_interval = check_interval
        self.lag_threshold = lag_threshold
        self.ram_threshold = ram_threshold
        self._last_heartbeat = time.time()
        try:
            self.active_lag_threshold = max(
                self.lag_threshold,
                float(os.getenv("AURA_FLAGSHIP_DOCTOR_ACTIVE_LAG_THRESHOLD_S", "30.0")),
            )
        except (TypeError, ValueError):
            self.active_lag_threshold = max(self.lag_threshold, 30.0)
        try:
            self.min_heal_interval = max(
                1.0,
                float(os.getenv("AURA_FLAGSHIP_DOCTOR_HEAL_COOLDOWN_S", "30.0")),
            )
        except (TypeError, ValueError):
            self.min_heal_interval = 30.0
        self._last_heal_at = 0.0
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._loop: Any = None
        self._heartbeat_task: Any = None

    def is_alive(self) -> bool:
        """Return daemon liveness only; use ``is_ready`` for runtime health."""
        return bool(self._running and not is_shutdown_requested())

    def is_ready(self) -> bool:
        """Return true only when the daemon and canonical runtime probes pass."""
        return bool(self.get_status().get("healthy", False))

    def get_status(self) -> dict[str, Any]:
        """Return a fail-closed daemon readiness snapshot.

        The event-loop heartbeat is necessary but insufficient. It can only
        contribute to a healthy verdict when the canonical runtime contract also
        proves kernel, inference, memory, scheduler, and tool-governance probes.
        """
        now = time.time()
        heartbeat_age_s = max(0.0, now - self._last_heartbeat) if self._last_heartbeat else None
        lag_threshold, lag_context = self._lag_threshold_for_context()
        task = self._heartbeat_task
        try:
            task_done = bool(task.done()) if task is not None else True
        except (RuntimeError, AttributeError, TypeError, ValueError):
            task_done = True
        heartbeat_fresh = bool(
            self._running
            and self._loop is not None
            and task is not None
            and not task_done
            and heartbeat_age_s is not None
            and heartbeat_age_s <= lag_threshold
        )

        blockers: list[str] = []
        if not self._running:
            blockers.append("flagship_doctor_not_running")
        if self._loop is None or task is None or task_done:
            blockers.append("event_loop_heartbeat_unavailable")
        elif not heartbeat_fresh:
            blockers.append("event_loop_heartbeat_stale")

        try:
            from core.runtime import health_contract

            runtime_report = health_contract.runtime_health_report()
            required_probes = runtime_report.get("required_probes", {})
            runtime_contract_operational = bool(runtime_report.get("operational", False))
            runtime_contract_healthy = bool(runtime_report.get("healthy", False))
            runtime_probe_healthy = health_contract.required_probe_groups_pass(required_probes)
            probe_blockers = health_contract.required_probe_blockers(required_probes)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "flagship_doctor",
                exc,
                severity="critical",
                action="failed closed: runtime health contract unavailable to flagship doctor",
            )
            runtime_report = {
                "status": "unknown",
                "healthy": False,
                "operational": False,
                "required_probes": {"all_passed": False},
            }
            required_probes = {"all_passed": False}
            runtime_contract_operational = False
            runtime_contract_healthy = False
            runtime_probe_healthy = False
            probe_blockers = ["runtime_health_probe_error", "runtime_required_probes"]

        if not runtime_contract_operational:
            blockers.append("runtime_contract")
        if not runtime_contract_healthy:
            blockers.append("runtime_contract_healthy")
        blockers.extend(probe_blockers)
        blockers = list(dict.fromkeys(blockers))

        healthy = bool(
            heartbeat_fresh
            and runtime_contract_healthy
            and runtime_probe_healthy
            and not blockers
        )
        return {
            "status": "healthy" if healthy else "unhealthy",
            "healthy": healthy,
            "daemon_running": self._running,
            "heartbeat_fresh": heartbeat_fresh,
            "heartbeat_age_s": round(heartbeat_age_s, 3) if heartbeat_age_s is not None else None,
            "lag_threshold_s": round(float(lag_threshold), 3),
            "lag_context": lag_context,
            "runtime_contract_healthy": runtime_contract_healthy,
            "runtime_contract_operational": runtime_contract_operational,
            "runtime_probe_healthy": runtime_probe_healthy,
            "required_probes": required_probes,
            "blockers": blockers,
            "runtime_health": runtime_report,
        }

    def start(self, loop: Any = None) -> None:
        """Start the background monitoring thread and event-loop heartbeat updater."""
        if self._running or is_shutdown_requested():
            return
        
        import asyncio
        try:
            self._loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("FlagshipDoctorDaemon: No running event loop found during start. Heartbeat updater deferred.")
        
        self._running = True
        self._last_heartbeat = time.time()
        
        # Schedule the heartbeat task on the event loop
        if self._loop and self._loop.is_running():
            self._heartbeat_task = self._loop.create_task(self._heartbeat_updater())
            
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="AuraFlagshipDoctorDaemon"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            try:
                task.cancel()
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("FlagshipDoctorDaemon heartbeat cancellation skipped: %s", exc)

    async def _heartbeat_updater(self) -> None:
        """Async task that constantly updates the heartbeat timestamp on the event loop."""
        import asyncio
        while self._running and not is_shutdown_requested():
            self._last_heartbeat = time.time()
            try:
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    def _active_runtime_reason(self) -> str | None:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active():
                return "proof_run_active"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.runtime.flagship_doctor: %s", type(_exc).__name__, _exc)

        try:
            from core.runtime import foreground_guard

            reason = foreground_guard.foreground_activity_reason()
            if reason:
                return str(reason)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.runtime.flagship_doctor: %s", type(_exc).__name__, _exc)

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            status_getter = getattr(gate, "get_conversation_status", None)
            if callable(status_getter):
                status = status_getter()
                if bool(getattr(status, "active", False)):
                    return "foreground_generation"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.runtime.flagship_doctor: %s", type(_exc).__name__, _exc)

        return None

    def _lag_threshold_for_context(self) -> tuple[float, str]:
        reason = self._active_runtime_reason()
        if reason:
            return self.active_lag_threshold, reason
        return self.lag_threshold, "idle"

    def _should_self_heal(self, lag: float, ram_percent: float, *, now: float | None = None) -> tuple[bool, str, bool]:
        now = float(now or time.time())
        lag_threshold, lag_context = self._lag_threshold_for_context()
        ram_pressure = ram_percent > 0.0 and ram_percent >= self.ram_threshold
        lag_pressure = lag > lag_threshold
        if not lag_pressure and not ram_pressure:
            return False, lag_context, ram_pressure

        if lag_context != "idle" and not ram_pressure:
            logger.debug(
                "FlagshipDoctorDaemon observed active-runtime lag without memory "
                "pressure; treating as occupied runtime, not self-healing trigger "
                "(lag=%.2fs context=%s threshold=%.2fs RAM=%.1f%%).",
                lag,
                lag_context,
                lag_threshold,
                ram_percent,
            )
            return False, lag_context, ram_pressure

        if now - self._last_heal_at < self.min_heal_interval:
            logger.debug(
                "FlagshipDoctorDaemon self-heal cooldown active "
                "(lag=%.2fs context=%s threshold=%.2fs RAM=%.1f%%).",
                lag,
                lag_context,
                lag_threshold,
                ram_percent,
            )
            return False, lag_context, ram_pressure

        return True, lag_context, ram_pressure

    def _monitor_loop(self) -> None:
        """Standard thread loop running in the background to detect event-loop stalls or high memory."""
        logger.info("FlagshipDoctorDaemon background thread started.")
        
        while self._running and not is_shutdown_requested():
            time.sleep(self.check_interval)
            if not self._running or is_shutdown_requested():
                break
                
            # 1. Event Loop Lag check
            lag = time.time() - self._last_heartbeat
            
            # 2. RAM Pressure check
            ram_percent = 0.0
            try:
                import psutil
                ram_percent = psutil.virtual_memory().percent
            except ImportError as _exc:
                logger.debug("Suppressed %s in core.runtime.flagship_doctor: %s", type(_exc).__name__, _exc)
                
            # Trigger self-healing if limits are violated
            should_heal, lag_context, ram_pressure = self._should_self_heal(
                lag,
                ram_percent,
            )
            if should_heal:
                logger.warning(
                    "⚠️ [HEALTH DEGRADED] FlagshipDoctorDaemon triggered self-healing. "
                    "Lag: %.2fs (context=%s), RAM: %.1f%%",
                    lag,
                    lag_context,
                    ram_percent
                )
                try:
                    self._last_heal_at = time.time()
                    self._execute_self_healing(
                        lag,
                        ram_percent,
                        lag_context=lag_context,
                        ram_pressure=ram_pressure,
                    )
                except (RuntimeError, OSError, AttributeError, ValueError, TypeError, ImportError, sqlite3.Error) as e:
                    logger.error("FlagshipDoctorDaemon self-healing failed: %s", e)

    def _execute_self_healing(
        self,
        lag: float,
        ram_percent: float,
        *,
        lag_context: str = "idle",
        ram_pressure: bool = False,
    ) -> None:
        """Executes garbage collection and compacts SQLite databases under pressure."""
        # 1. Run CPU Garbage Collection
        logger.info("♻️ Reclaiming memory via gc.collect()...")
        gc.collect()
        
        # 2. Compact SQLite Databases (specifically test_projects.db)
        db_paths = [
            self.root / "tests" / "test_projects.db",
            Path.home() / ".aura" / "live-source" / "tests" / "test_projects.db",
        ]
        
        compacted_count = 0
        if ram_pressure or lag_context == "idle":
            for db_path in db_paths:
                if db_path.exists():
                    try:
                        logger.info("🗄️ Compacting SQLite database: %s", db_path)
                        conn = sqlite3.connect(str(db_path), timeout=5.0)
                        conn.execute("VACUUM;")
                        conn.close()
                        compacted_count += 1
                    except (sqlite3.Error, OSError, RuntimeError, ValueError) as e:
                        logger.error("Failed to compact DB %s: %s", db_path, e)
        else:
            logger.info(
                "🗄️ Skipping database compaction for active-runtime lag context: %s",
                lag_context,
            )
                    
        # 3. Trigger global database maintenance if available
        try:
            from core.persistence.db_maintenance import get_db_maintenance
            if ram_pressure or lag_context == "idle":
                maint = get_db_maintenance()
                logger.info("🗄️ Triggering global DatabaseMaintenance pass...")
                maint.run_maintenance(force=True)
                compacted_count += 1
        except ImportError as _exc:
            logger.debug("Suppressed %s in core.runtime.flagship_doctor: %s", type(_exc).__name__, _exc)
        except (RuntimeError, AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Global database maintenance run failed: %s", e)
            
        # 4. Record systemic degradation telemetry
        try:
            from core.runtime.errors import record_degradation
            record_degradation(
                "flagship_doctor",
                RuntimeError(f"Self-healing active: lag={lag:.2f}s, RAM={ram_percent:.1f}%"),
                severity="warning",
                action=f"reclaimed RAM with gc.collect() and compacted {compacted_count} databases"
            )
        except (ImportError, RuntimeError, AttributeError, ValueError, TypeError, OSError) as e:
            logger.error("Failed to record degradation telemetry: %s", e)


_daemon_instance: FlagshipDoctorDaemon | None = None


def get_flagship_doctor_daemon(root_dir: str | Path | None = None) -> FlagshipDoctorDaemon:
    global _daemon_instance
    if _daemon_instance is None:
        _daemon_instance = FlagshipDoctorDaemon(root_dir=root_dir)
    return _daemon_instance


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aura flagship operational doctor")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-gates", action="store_true", help="Skip running slower external gates.")
    parser.add_argument("--out", default="", help="Optional report JSON path.")
    args = parser.parse_args(argv)

    report = run_doctor(args.root, include_gates=not args.no_gates)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out, report.to_json(), encoding="utf-8")
    if args.json:
        sys.stdout.write(report.to_json() + "\n")
    else:
        sys.stdout.write(f"Aura flagship doctor: {report.overall.upper()}\n")
        for finding in report.findings:
            sys.stdout.write(f"[{finding.status.upper()}] {finding.code}: {finding.message}\n")
            if finding.suggestion:
                sys.stdout.write(f"  -> {finding.suggestion}\n")

    return 1 if report.overall == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
