"""StallWatchdog: Async Event Loop Monitoring + Active Recovery
Part of Aura's Neural Neuro-Surgeon (Phase 29).

Design notes:
- The watchdog runs in its own daemon thread so it survives even if the
  asyncio loop is wedged.
- A heartbeat is scheduled via call_soon_threadsafe every second. If the
  loop fails to run that callback within `threshold` seconds, we record a
  stall.
- On a long stall we now do more than log: we (a) dump task state, (b)
  cancel asyncio tasks that look hung, and (c) signal subsystems to
  recycle. This is what turns "we noticed the freeze" into "we ended
  the freeze."
"""

import asyncio
import io
import logging
import os
import sys
import threading
import time
import traceback
from importlib import import_module
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.Resilience.Watchdog")

_STALL_WATCHDOG_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_watchdog_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    extra: dict | None = None,
) -> None:
    record_degradation(
        "stall_watchdog",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )

# How long an asyncio task can be pending (not done) before the watchdog
# considers it suspect during a stall and cancels it. Conservative — only
# fires after a confirmed stall, not as routine cleanup.
_TASK_HUNG_SECONDS = 90.0

# Minimum stall length to trigger active recovery. Below this, we just log.
_ACTIVE_RECOVERY_THRESHOLD = 30.0

# Absolute ceiling on CONTINUOUS event-loop unresponsiveness, immune to all
# stall suppression. A healthy runtime runs the watchdog heartbeat every
# second even during long generations (those run off the loop), so the loop
# only goes silent this long when it is genuinely wedged. Observed live: a
# steered 32B generation triggered a Metal GPU command-buffer hang that
# deadlocked every thread (all parked on a Metal semaphore) — the event loop
# died, in-process "active recovery" could not run (it schedules onto the dead
# loop), and even SIGTERM could not shut down. In-process recovery of a shared
# GPU-context deadlock is impossible, so the out-of-band watchdog thread — the
# only code still running — force-exits the process. The launchd KeepAlive
# supervisor (tools/install_supervisor.sh) restarts within ~15s with continuity
# from the periodic state-vault snapshots. 0 disables the ceiling.
_LOOP_WEDGE_HARD_EXIT_S = 150.0
# Distinct from memory_watchdog's categorized exit (70). Non-zero ⇒ the
# supervisor's KeepAlive/SuccessfulExit=false restarts the runtime.
_LOOP_WEDGE_EXIT_CODE = 71


class StallWatchdog(threading.Thread):
    """Monitor thread that tracks event loop responsiveness."""

    def __init__(self, loop: asyncio.AbstractEventLoop, threshold: float = 5.0):
        super().__init__(daemon=True, name="AuraStallWatchdog")
        self.loop = loop
        self.threshold = threshold
        self._last_heartbeat = time.time()
        self._running = False
        self._stop_event = threading.Event()
        self._task_birth: dict[int, float] = {}
        self._consecutive_long_stalls: int = 0
        self._started_at: float = time.time()
        # Source of truth for "the loop actually ran a callback", updated ONLY
        # by _heartbeat (never by stall suppression). The suppression paths
        # reset _last_heartbeat to silence expected warmup/foreground stalls,
        # which would otherwise mask a genuinely wedged loop forever — so the
        # absolute hard-exit ceiling keys off this independent timestamp.
        self._last_loop_run: float = time.time()
        self._diagnostic_only_notice_logged: bool = False
        self._last_boot_suppression_log_at: float = 0.0
        self._last_foreground_suppression_log_at: float = 0.0
        # Out-of-process liveness beacon. This daemon thread writes _last_loop_run
        # to a heartbeat file each tick; the external liveness_sentinel watches
        # it and kills+restarts the tree when it goes stale. This covers the case
        # the in-process hard-exit below CANNOT: a Metal GPU deadlock that holds
        # the GIL so no Python thread (this one included) can run — the file then
        # simply stops updating and the out-of-process sentinel acts.
        self._heartbeat_file: Path | None = self._resolve_heartbeat_file()
        self._last_heartbeat_file_write: float = 0.0

    @staticmethod
    def _resolve_heartbeat_file() -> Path | None:
        raw = os.getenv("AURA_LIVENESS_HEARTBEAT_FILE", "data/runtime/liveness_heartbeat.json")
        if not str(raw).strip():
            return None
        try:
            path = Path(raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return None

    def _write_liveness_heartbeat(self) -> None:
        """Append-free atomic write of the loop-liveness beacon (best-effort)."""
        if self._heartbeat_file is None:
            return
        now = time.time()
        # Throttle to ~1s; the run loop already ticks at 1s but guard anyway.
        if now - self._last_heartbeat_file_write < 1.0:
            return
        self._last_heartbeat_file_write = now
        try:
            payload = (
                '{"pid": %d, "last_loop_run": %.3f, "written_at": %.3f}'
                % (os.getpid(), self._last_loop_run, now)
            )
            tmp = self._heartbeat_file.with_suffix(".tmp")
            tmp.write_text(payload)
            os.replace(tmp, self._heartbeat_file)
        except OSError:
            pass  # The beacon must never crash the watchdog.

    @staticmethod
    def _suppression_log_interval_s() -> float:
        try:
            return max(5.0, float(os.getenv("AURA_WATCHDOG_SUPPRESSION_LOG_INTERVAL_S", "60") or 60))
        except (TypeError, ValueError):
            return 60.0

    def _log_suppression(self, attr_name: str, message: str, *args: object) -> None:
        now = time.time()
        last = float(getattr(self, attr_name, 0.0) or 0.0)
        if now - last >= self._suppression_log_interval_s():
            logger.info(message, *args)
            setattr(self, attr_name, now)
        else:
            logger.debug(message, *args)

    def run(self):
        logger.info("🛡️ StallWatchdog: Monitoring loop (Threshold: %.1fs)", self.threshold)
        self._running = True

        while not self._stop_event.is_set():
            # Schedule a heartbeat on the loop
            try:
                if self.loop.is_closed():
                    logger.debug("StallWatchdog: event loop closed, exiting.")
                    break
                self.loop.call_soon_threadsafe(self._heartbeat)
            except RuntimeError:
                # Event loop closed during shutdown — exit silently
                break
            except (AttributeError, TypeError, ValueError) as e:
                _record_watchdog_degradation(
                    e,
                    action="skipped watchdog heartbeat schedule and kept monitoring thread alive",
                    severity="warning",
                    extra={"stage": "heartbeat_schedule"},
                )
                logger.debug("Watchdog heartbeat schedule issue: %s", e)

            time.sleep(1.0)  # Check every second

            # Refresh the out-of-process liveness beacon every tick. When the
            # loop wedges, _last_loop_run stops advancing; when the GIL is held
            # by a Metal deadlock, this write stops entirely — either way the
            # external liveness_sentinel sees staleness and restarts the tree.
            self._write_liveness_heartbeat()

            # Absolute loop-liveness ceiling — checked FIRST and immune to all
            # stall suppression. _last_loop_run is advanced only when the loop
            # actually executes the heartbeat callback, so this measures true
            # loop death (e.g. a Metal GPU deadlock) that suppression would
            # otherwise hide. If the loop is wedged beyond any in-process
            # recovery, hard-exit so the supervisor restarts the runtime.
            loop_silence = time.time() - self._last_loop_run
            if self._should_force_exit(loop_silence):
                self._force_exit_for_restart(loop_silence)

            # Check for stall
            elapsed = time.time() - self._last_heartbeat
            if elapsed > self.threshold:
                if self._should_suppress_stall(elapsed):
                    self._last_heartbeat = time.time()
                    self._consecutive_long_stalls = 0
                    continue
                self._report_stall(elapsed)
                if elapsed >= _ACTIVE_RECOVERY_THRESHOLD:
                    self._consecutive_long_stalls += 1
                    self._attempt_active_recovery(elapsed)
                else:
                    self._consecutive_long_stalls = 0
                # Reset so the next stall measurement is fresh.
                self._last_heartbeat = time.time()
            else:
                self._consecutive_long_stalls = 0

    def stop(self):
        self._stop_event.set()

    def _heartbeat(self):
        now = time.time()
        self._last_heartbeat = now
        # Independent liveness proof for the hard-exit ceiling: this only runs
        # when the loop is genuinely alive, and nothing else writes it.
        self._last_loop_run = now
        # Track task ages so a future stall can pick out which ones look hung.
        # This runs on the loop thread — cheap and safe.
        try:
            now = time.time()
            tasks = asyncio.all_tasks(self.loop)
            seen = set()
            for task in tasks:
                tid = id(task)
                seen.add(tid)
                if tid not in self._task_birth:
                    self._task_birth[tid] = now
            # Drop dead bookkeeping
            for tid in list(self._task_birth.keys()):
                if tid not in seen:
                    self._task_birth.pop(tid, None)
        except _STALL_WATCHDOG_ERRORS as exc:
            _record_watchdog_degradation(
                exc,
                action="kept watchdog alive after task-age bookkeeping failed",
                severity="warning",
                extra={"stage": "task_age_bookkeeping"},
            )
            logger.debug("Task age bookkeeping failed: %s", exc)

    @staticmethod
    def _hard_exit_ceiling_s() -> float:
        try:
            return float(
                os.getenv("AURA_WATCHDOG_HARD_EXIT_S", str(_LOOP_WEDGE_HARD_EXIT_S))
                or _LOOP_WEDGE_HARD_EXIT_S
            )
        except (TypeError, ValueError):
            return _LOOP_WEDGE_HARD_EXIT_S

    @staticmethod
    def _hard_exit_code() -> int:
        try:
            return int(
                os.getenv("AURA_WATCHDOG_HARD_EXIT_CODE", str(_LOOP_WEDGE_EXIT_CODE))
                or _LOOP_WEDGE_EXIT_CODE
            )
        except (TypeError, ValueError):
            return _LOOP_WEDGE_EXIT_CODE

    def _should_force_exit(self, loop_silence: float) -> bool:
        """True when the loop is wedged beyond any in-process recovery.

        Immune to stall suppression by design: this is the last-resort ceiling
        for a genuinely dead loop (e.g. a Metal GPU deadlock). The only thing
        we honor is an intentional shutdown, when the loop stops on purpose.
        """
        ceiling = self._hard_exit_ceiling_s()
        if ceiling <= 0 or loop_silence < ceiling:
            return False
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            if is_shutdown_requested():
                return False
        except (ImportError, RuntimeError):
            pass
        return True

    def _force_exit_for_restart(self, loop_silence: float) -> None:
        """Hard-exit a wedged process so the supervisor restarts it.

        Runs on the out-of-band watchdog thread — the only code still alive
        when the loop is deadlocked. Dumps thread stacks via low-level
        faulthandler to a raw fd (never through app locks/gateways the wedge
        may be holding), then exits immediately with a non-zero code. No
        logging.shutdown()/handler flush: those can block under a wedge (the
        lesson from the 115GB crash where the lethal path never reached exit).
        """
        logger.critical(
            "🛑 [WATCHDOG] Event loop unresponsive %.0fs ≥ hard ceiling — wedged beyond "
            "in-process recovery; forcing process exit %d for supervisor restart.",
            loop_silence,
            self._hard_exit_code(),
        )
        try:
            import faulthandler

            crash_dir = Path("data/error_logs/crash")
            crash_dir.mkdir(parents=True, exist_ok=True)
            with open(crash_dir / "loop_wedge_stacks.log", "a") as fh:
                fh.write(
                    f"\n===== LOOP WEDGE pid={os.getpid()} "
                    f"silence={loop_silence:.1f}s at={time.time()} =====\n"
                )
                fh.flush()
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("Loop-wedge stack dump failed: %s", exc)
        os._exit(self._hard_exit_code())

    def _should_suppress_stall(self, elapsed: float) -> bool:
        """Suppress expected launch/shutdown stalls without hiding live hangs."""
        try:
            from core.runtime.shutdown_coordinator import is_shutdown_requested

            if is_shutdown_requested():
                logger.debug("StallWatchdog: suppressing %.1fs stall during shutdown.", elapsed)
                return True
        except (ImportError, RuntimeError) as exc:
            logger.debug("StallWatchdog shutdown probe skipped: %s", exc)

        try:
            boot_grace = float(os.getenv("AURA_WATCHDOG_BOOT_GRACE_S", "120") or 120)
        except (TypeError, ValueError):
            boot_grace = 120.0
        if boot_grace > 0 and (time.time() - self._started_at) < boot_grace:
            self._log_suppression(
                "_last_boot_suppression_log_at",
                "StallWatchdog: suppressing %.1fs launch stall during boot grace.",
                elapsed,
            )
            return True
        try:
            foreground_grace = float(os.getenv("AURA_WATCHDOG_FOREGROUND_GRACE_S", "75") or 75)
        except (TypeError, ValueError):
            foreground_grace = 75.0
        if foreground_grace > 0 and elapsed <= foreground_grace:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                lane = gate.get_conversation_status() if gate and hasattr(gate, "get_conversation_status") else {}
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                logger.debug("StallWatchdog foreground lane probe unavailable: %s", exc)
                lane = {}
            lane_state = str(lane.get("state") or "").lower()
            foreground_active = bool(
                lane.get("foreground_owned")
                or int(lane.get("active_generations", 0) or 0) > 0
                or lane.get("warmup_in_flight")
                or lane_state in {"spawning", "handshaking", "warming", "recovering"}
            )
            if foreground_active:
                self._log_suppression(
                    "_last_foreground_suppression_log_at",
                    "StallWatchdog: suppressing %.1fs foreground inference stall "
                    "(state=%s active=%s warmup=%s owner=%s).",
                    elapsed,
                    lane.get("state"),
                    lane.get("active_generations", 0),
                    lane.get("warmup_in_flight"),
                    lane.get("foreground_owner", ""),
                )
                return True
        return False

    def _report_stall(self, elapsed: float):
        logger.error("🚨 [WATCHDOG] EVENT LOOP STALL DETECTED! (Elapsed: %.1fs)", elapsed)

        # Dump tracebacks of all threads
        dump_dir = Path("data/error_logs/stalls")
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_file = dump_dir / f"stall_{int(time.time())}.txt"

        buffer = io.StringIO()
        buffer.write(f"STALL DETECTED: {elapsed:.1f}s\n")
        buffer.write("=" * 40 + "\n")
        for thread_id, frame in sys._current_frames().items():
            buffer.write(f"\nThread ID: {thread_id}\n")
            traceback.print_stack(frame, file=buffer)
        try:
            with local_internal_governed_scope(
                "resilience.stall_watchdog.traceback_dump",
                domain="file_write",
            ):
                get_file_write_gateway().write_text(
                    dump_file,
                    buffer.getvalue(),
                    source="resilience.stall_watchdog.traceback_dump",
                )
        except _STALL_WATCHDOG_ERRORS as exc:
            _record_watchdog_degradation(
                exc,
                action="continued stall handling after traceback dump write failed",
                severity="warning",
                extra={"stage": "traceback_dump", "elapsed_s": elapsed},
            )
        finally:
            buffer.close()

        logger.info("💉 [IMMUNE] Stall traceback dumped to: %s", dump_file)

        # Proactively trigger Neuro-Surgeon analysis
        try:
            from core.resilience.diagnostic_hub import get_diagnostic_hub
            get_diagnostic_hub()
            # Future: trigger auto-repair or circuit break
        except (ImportError, AttributeError, RuntimeError) as _e:
            record_degradation(
                "stall_watchdog",
                _e,
                severity="debug",
                action="continued after optional stall diagnostic hub was unavailable",
            )
            logger.debug('Ignored Exception in stall_watchdog.py: %s', _e)

    def _attempt_active_recovery(self, elapsed: float) -> None:
        """Don't just log a stall — try to break it.

        We schedule a recovery coroutine onto the (possibly wedged) loop.
        If the loop is truly frozen, the coroutine will queue and run when
        the loop wakes up — and it will then prevent the next stall by
        cancelling the hung tasks. If the loop is partially responsive, the
        coroutine runs immediately.
        """
        def _schedule_recovery() -> None:
            try:
                create_tracked_task(
                    self._recover_on_loop(elapsed),
                    name="stall_watchdog.active_recovery",
                )
            except _STALL_WATCHDOG_ERRORS as exc:
                _record_watchdog_degradation(
                    exc,
                    action="continued watchdog reporting after active recovery task ownership failed",
                    severity="warning",
                    extra={"stage": "active_recovery_task", "elapsed_s": elapsed},
                )

        try:
            self.loop.call_soon_threadsafe(_schedule_recovery)
        except RuntimeError:
            return
        except (AttributeError, TypeError, ValueError) as exc:
            _record_watchdog_degradation(
                exc,
                action="continued watchdog reporting after active recovery scheduling failed",
                severity="warning",
                extra={"stage": "active_recovery_schedule", "elapsed_s": elapsed},
            )
            logger.debug("Stall recovery scheduling failed: %s", exc)

    async def _recover_on_loop(self, elapsed: float) -> None:
        """Cancel hung tasks and ask known subsystems to recycle.

        Runs on the asyncio loop. Conservative — we only cancel tasks that
        have been alive for far longer than this stall, and we never touch
        the kernel main loop / watchdog / orchestrator coordinator tasks.
        """
        protected_substrings = (
            "AuraKernel",
            "OrchestratorMainLoop",
            "AuraStallWatchdog",
            "ConsciousnessLoopMonitor",
            "Server.Chat",
            "uvicorn",
            "server.ws",
        )
        cutoff = time.time() - max(_TASK_HUNG_SECONDS, elapsed * 1.5)
        cancelled = 0
        try:
            tasks = asyncio.all_tasks(self.loop)
        except RuntimeError:
            return

        aggressive_cancel = os.getenv("AURA_WATCHDOG_CANCEL_HUNG_TASKS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if aggressive_cancel:
            for task in tasks:
                if task.done():
                    continue
                name = getattr(task, "get_name", lambda: "")() or repr(task)
                if any(p in name for p in protected_substrings):
                    continue
                birth = self._task_birth.get(id(task))
                if birth is None or birth >= cutoff:
                    continue
                try:
                    task.cancel()
                    cancelled += 1
                except RuntimeError as exc:
                    _record_watchdog_degradation(
                        exc,
                        action="continued stall recovery after hung-task cancellation failed",
                        severity="warning",
                        extra={"stage": "cancel_hung_task", "task_name": name},
                    )
                    logger.debug("Stall recovery: failed to cancel %s: %s", name, exc)
        else:
            message = (
                "💉 [IMMUNE] Stall recovery is diagnostic-only; not bulk-cancelling asyncio tasks. "
                "Set AURA_WATCHDOG_CANCEL_HUNG_TASKS=1 to enable aggressive cancellation."
            )
            if self._diagnostic_only_notice_logged:
                logger.debug(message)
            else:
                logger.info(message)
                self._diagnostic_only_notice_logged = True

        if cancelled:
            logger.warning(
                "💉 [IMMUNE] Stall recovery cancelled %d hung asyncio tasks (stall=%.0fs).",
                cancelled,
                elapsed,
            )

        # Ask the brainstem and cortex MLX clients to self-check; the
        # stale-handshake path in mlx_client._ensure_worker_alive will
        # recycle anyone that's been wedged.
        try:
            mlx_client_module = import_module("core.brain.llm.mlx_client")
            live_mlx_clients = getattr(mlx_client_module, "_LIVE_MLX_CLIENTS", None)
        except (ImportError, AttributeError, RuntimeError):
            live_mlx_clients = None

        if live_mlx_clients:
            for client in list(live_mlx_clients):
                try:
                    if hasattr(client, "_lane_state") and client._lane_state == "handshaking":
                        # Schedule a no-op alive probe so the stale-handshake
                        # branch fires on next entry.
                        self.loop.call_soon(client._mark_progress)
                except (OSError, ConnectionError, TimeoutError) as exc:
                    _record_watchdog_degradation(
                        exc,
                        action="continued stall recovery after MLX liveness poke failed",
                        severity="warning",
                        extra={"stage": "mlx_liveness_poke"},
                    )
                    logger.debug("Stall recovery MLX poke failed: %s", exc)

        # If we've taken many long stalls in a row, ask the orchestrator's
        # state vault to flush so we don't lose continuity.
        if self._consecutive_long_stalls >= 3:
            try:
                from core.container import ServiceContainer
                state_repo = ServiceContainer.get("state_repository", default=None)
                if state_repo and hasattr(state_repo, "request_flush"):
                    state_repo.request_flush()
                    logger.info("💉 [IMMUNE] Requested state vault flush after %d consecutive stalls.", self._consecutive_long_stalls)
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_watchdog_degradation(
                    exc,
                    action="continued stall recovery after state flush request failed",
                    severity="warning",
                    extra={"stage": "state_flush_request"},
                )
                logger.debug("Stall recovery state-flush request failed: %s", exc)

def start_watchdog(loop: asyncio.AbstractEventLoop | None = None, threshold: float = 5.0):
    """Convenience helper to start the watchdog."""
    try:
        target_loop = loop or asyncio.get_running_loop()
    except RuntimeError:
        target_loop = asyncio.new_event_loop()
    dog = StallWatchdog(target_loop, threshold=threshold)
    dog.start()
    return dog
