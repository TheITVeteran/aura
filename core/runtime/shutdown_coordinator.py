"""Canonical ShutdownCoordinator.

Codifies the shutdown phase ordering required by the runtime invariants
audit:

    output flush -> memory commit -> state vault -> actors
    -> model runtime -> bus -> task supervisor

Each phase is a list of registered handlers. Handlers in the same phase
run concurrently; phases run sequentially. A handler that raises is logged
and treated as a failed phase, but does not abort the remaining phases —
the goal is to flush as much as possible during shutdown rather than abort
early on the first error.

Strict mode (AURA_STRICT_RUNTIME=1) elevates phase failures so they are
visible in tests and conformance harnesses, while still completing the
remaining phases.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from core.runtime.atomic_writer import atomic_write_text
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.ShutdownCoordinator")

ShutdownHandler = Callable[[], None | Awaitable[None]]


# Canonical phases. Order matters.
SHUTDOWN_PHASES: tuple[str, ...] = (
    "output_flush",
    "memory_commit",
    "state_vault",
    "actors",
    "model_runtime",
    "event_bus",
    "task_supervisor",
)


@dataclass
class _RegisteredHandler:
    name: str
    handler: ShutdownHandler
    phase: str
    timeout: float = 15.0


@dataclass
class ShutdownReport:
    completed_phases: list[str] = field(default_factory=list)
    failed_phases: list[str] = field(default_factory=list)
    handler_failures: dict[str, str] = field(default_factory=dict)
    phase_durations_seconds: dict[str, float] = field(default_factory=dict)
    started_at_unix: float | None = None
    completed_at_unix: float | None = None
    duration_seconds: float | None = None
    current_phase: str | None = None
    repeated_call_count: int = 0

    @property
    def clean(self) -> bool:
        return not self.failed_phases and not self.handler_failures

    def clone(self) -> ShutdownReport:
        """Return a detached report so callers cannot mutate coordinator state."""

        return ShutdownReport(
            completed_phases=list(self.completed_phases),
            failed_phases=list(self.failed_phases),
            handler_failures=dict(self.handler_failures),
            phase_durations_seconds=dict(self.phase_durations_seconds),
            started_at_unix=self.started_at_unix,
            completed_at_unix=self.completed_at_unix,
            duration_seconds=self.duration_seconds,
            current_phase=self.current_phase,
            repeated_call_count=self.repeated_call_count,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "clean": self.clean,
            "completed_phases": list(self.completed_phases),
            "failed_phases": list(self.failed_phases),
            "handler_failures": dict(self.handler_failures),
            "phase_durations_seconds": dict(self.phase_durations_seconds),
            "started_at_unix": self.started_at_unix,
            "completed_at_unix": self.completed_at_unix,
            "duration_seconds": self.duration_seconds,
            "current_phase": self.current_phase,
            "repeated_call_count": self.repeated_call_count,
        }


class ShutdownCoordinator:
    """Single owner of teardown ordering."""

    # Coordinator phase -> verified-lifecycle state entered when that phase
    # begins. The FSM formalizes the coarse teardown arc (drain -> stop
    # services -> flush) on top of the fine-grained phase loop. Repeated
    # shutdown callers share one execution and replay its final report.
    _LIFECYCLE_BOUNDARIES: dict[str, str] = {
        "actors": "STOPPING_SERVICES",
        "task_supervisor": "FLUSHING_STATE",
    }

    def __init__(self, phases: tuple[str, ...] = SHUTDOWN_PHASES):
        self._phases = phases
        self._handlers: dict[str, list[_RegisteredHandler]] = {p: [] for p in phases}
        self._lock = threading.RLock()
        self._running = False
        self._shutdown_task: asyncio.Task[ShutdownReport] | None = None
        self._task_loop: asyncio.AbstractEventLoop | None = None
        self._completion = threading.Event()
        self._report: ShutdownReport | None = None
        self._working_report: ShutdownReport | None = None
        self._repeated_call_count = 0
        self._lifecycle = self._build_lifecycle()

    @staticmethod
    def _build_lifecycle() -> Any:
        try:
            from core.resilience.verified_state_machine import (
                create_shutdown_lifecycle_machine,
            )
            return create_shutdown_lifecycle_machine()
        except (ImportError, RuntimeError, ValueError) as exc:
            # Lifecycle bookkeeping must never block teardown.
            logger.debug("Shutdown lifecycle machine unavailable: %s", exc)
            return None

    def lifecycle_state(self) -> str:
        """Current verified-lifecycle state (for diagnostics)."""
        return self._lifecycle.current if self._lifecycle is not None else "UNTRACKED"

    def _lifecycle_transition(self, to_state: str) -> None:
        """Advance the lifecycle machine; F17 is recorded on illegal moves,
        and teardown continues regardless — bookkeeping never blocks it."""
        if self._lifecycle is None:
            return
        try:
            from core.resilience.verified_state_machine import IllegalTransitionError
            try:
                self._lifecycle.transition(to_state)
            except IllegalTransitionError:
                pass  # already recorded as F17 by the machine
        except ImportError as exc:
            logger.debug("Lifecycle transition skipped: %s", exc)

    # --- Registration ---------------------------------------------------

    def register(
        self,
        handler: ShutdownHandler,
        *,
        phase: str,
        name: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        if phase not in self._handlers:
            raise ValueError(
                f"unknown shutdown phase '{phase}'; expected one of {self._phases}"
            )
        if not callable(handler):
            raise TypeError("shutdown handler must be callable")
        record = _RegisteredHandler(
            name=str(name or getattr(handler, "__name__", "anonymous")),
            handler=handler,
            phase=phase,
            timeout=timeout,
        )
        with self._lock:
            if self._running or self._report is not None:
                raise RuntimeError("cannot register shutdown handlers after teardown has started")
            self._handlers[phase].append(record)

    def clear(self) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("cannot clear shutdown handlers while teardown is running")
            for phase in self._handlers:
                self._handlers[phase] = []

    def phases(self) -> tuple[str, ...]:
        return self._phases

    def handler_names(self, phase: str) -> list[str]:
        with self._lock:
            return [h.name for h in self._handlers.get(phase, [])]

    def get_status(self) -> dict[str, object]:
        """Return a stable operator snapshot of latch and teardown progress."""

        with self._lock:
            report = self._report or self._working_report
            report_payload = report.clone().as_dict() if report is not None else None
            registered_handlers = {
                phase: [record.name for record in records]
                for phase, records in self._handlers.items()
            }
            return {
                "running": self._running,
                "lifecycle_state": self.lifecycle_state(),
                "request": shutdown_request_snapshot(),
                "report": report_payload,
                "registered_handlers": registered_handlers,
            }

    # --- Execution ------------------------------------------------------

    async def shutdown(self, *, timeout_per_phase: float | None = None) -> ShutdownReport:
        request_shutdown("coordinator")
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._report is not None:
                self._repeated_call_count += 1
                replay = self._report.clone()
                replay.repeated_call_count = self._repeated_call_count
                return replay

            task: asyncio.Task[ShutdownReport] | None = self._shutdown_task
            task_loop = self._task_loop
            if task is None:
                started_at = time.time()
                self._working_report = ShutdownReport(started_at_unix=started_at)
                self._running = True
                self._completion.clear()
                task = cast(
                    asyncio.Task[ShutdownReport],
                    get_task_tracker().create_task(
                        self._execute_shutdown(timeout_per_phase=timeout_per_phase),
                        name="shutdown_coordinator.execute",
                        allow_during_shutdown=True,
                    ),
                )
                self._shutdown_task = task
                self._task_loop = loop
                task_loop = loop
            else:
                self._repeated_call_count += 1

        if task_loop is loop:
            report = await asyncio.shield(task)
            with self._lock:
                replay = report.clone()
                replay.repeated_call_count = self._repeated_call_count
                return replay

        # A coordinator may be reached from a second loop during process exit.
        # The teardown owner remains the first loop; other loops wait on a
        # thread-safe completion event instead of attempting duplicate cleanup.
        await asyncio.to_thread(self._completion.wait)
        with self._lock:
            if self._report is None:
                raise RuntimeError("shutdown coordinator completed without a report")
            replay = self._report.clone()
            replay.repeated_call_count = self._repeated_call_count
            return replay

    async def _execute_shutdown(
        self,
        *,
        timeout_per_phase: float | None,
    ) -> ShutdownReport:
        with self._lock:
            report = self._working_report
        if report is None:  # pragma: no cover - protected by shutdown()
            report = ShutdownReport(started_at_unix=time.time())

        started_monotonic = time.monotonic()
        self._lifecycle_transition("DRAINING")
        try:
            for phase in self._phases:
                phase_started = time.monotonic()
                report.current_phase = phase
                boundary = self._LIFECYCLE_BOUNDARIES.get(phase)
                if boundary:
                    self._lifecycle_transition(boundary)
                with self._lock:
                    handlers = list(self._handlers.get(phase, []))
                if not handlers:
                    report.completed_phases.append(phase)
                    report.phase_durations_seconds[phase] = round(
                        time.monotonic() - phase_started, 6
                    )
                    continue
                phase_failed = False
                coros: list[asyncio.Future[Any]] = []
                for record in handlers:
                    coros.append(
                        get_task_tracker().track(
                            self._invoke(record),
                            name=f"shutdown:{phase}:{record.name}",
                            allow_during_shutdown=True,
                        )
                    )
                effective_timeout = (
                    float(timeout_per_phase)
                    if timeout_per_phase is not None
                    else max((h.timeout for h in handlers), default=15.0)
                )
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*coros, return_exceptions=True),
                        timeout=effective_timeout,
                    )
                except TimeoutError:
                    for fut in coros:
                        if not fut.done():
                            fut.cancel()
                    report.failed_phases.append(phase)
                    report.handler_failures[phase] = "phase timed out"
                    logger.error("Shutdown phase '%s' timed out", phase)
                    report.phase_durations_seconds[phase] = round(
                        time.monotonic() - phase_started, 6
                    )
                    continue
                for record, result in zip(handlers, results, strict=True):
                    if isinstance(result, BaseException):
                        phase_failed = True
                        msg = repr(result)
                        report.handler_failures[f"{phase}:{record.name}"] = msg
                        logger.error(
                            "Shutdown handler '%s' in phase '%s' failed: %s",
                            record.name,
                            phase,
                            msg,
                        )
                if phase_failed:
                    report.failed_phases.append(phase)
                else:
                    report.completed_phases.append(phase)
                report.phase_durations_seconds[phase] = round(
                    time.monotonic() - phase_started, 6
                )
        except asyncio.CancelledError as exc:
            # Teardown is process-critical. Record cancellation as a failure but
            # still publish a final report so concurrent waiters never hang.
            report.failed_phases.append("coordinator")
            report.handler_failures["coordinator"] = repr(exc)
            logger.error("Shutdown coordinator task was cancelled before all phases completed")
        except Exception as exc:  # noqa: BLE001 - final process teardown boundary
            report.failed_phases.append("coordinator")
            report.handler_failures["coordinator"] = repr(exc)
            logger.error("Shutdown coordinator failed internally: %s", exc, exc_info=True)
        finally:
            self._lifecycle_transition("TERMINATED")
            report.current_phase = None
            report.completed_at_unix = time.time()
            report.duration_seconds = round(time.monotonic() - started_monotonic, 6)
            with self._lock:
                self._running = False
                self._report = report.clone()
                self._working_report = None
                self._completion.set()

        if not report.clean and os.environ.get("AURA_STRICT_RUNTIME") == "1":
            logger.error(
                "ShutdownCoordinator: strict mode shutdown failures: phases=%s handlers=%s",
                report.failed_phases,
                report.handler_failures,
            )
        else:
            logger.info(
                "ShutdownCoordinator: shutdown complete (clean=%s phases=%s)",
                report.clean,
                report.completed_phases,
            )
        return report

    async def _invoke(self, record: _RegisteredHandler) -> None:
        try:
            result = record.handler()
            if inspect.isawaitable(result):
                await asyncio.wait_for(result, timeout=record.timeout)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, TimeoutError, AttributeError):
            raise


# Singleton accessor ---------------------------------------------------------

_shutdown_coordinator: ShutdownCoordinator | None = None
_singleton_lock = threading.RLock()
_shutdown_requested = threading.Event()
_shutdown_state_lock = threading.RLock()
_shutdown_first_reason = ""
_shutdown_last_reason = ""
_shutdown_first_requested_at_unix: float | None = None
_shutdown_first_requested_at_monotonic: float | None = None
_shutdown_request_count = 0


def shutdown_request_snapshot() -> dict[str, object]:
    """Return process-wide latch metadata for health and stop diagnostics."""

    with _shutdown_state_lock:
        requested = _shutdown_requested.is_set()
        elapsed = None
        if requested and _shutdown_first_requested_at_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - _shutdown_first_requested_at_monotonic)
        return {
            "requested": requested,
            "first_reason": _shutdown_first_reason,
            "last_reason": _shutdown_last_reason,
            "first_requested_at_unix": _shutdown_first_requested_at_unix,
            "elapsed_seconds": round(elapsed, 6) if elapsed is not None else None,
            "request_count": _shutdown_request_count,
        }


def _write_grace_flag(*, reason: str, created_at_unix: float) -> None:
    from pathlib import Path

    grace_file = Path.home() / ".aura" / "run" / "grace_exit.flag"
    grace_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        grace_file,
        json.dumps(
            {
                "schema": "aura.shutdown_grace.v1",
                "pid": os.getpid(),
                "reason": reason,
                "created_at_unix": created_at_unix,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def request_shutdown(reason: str = "") -> dict[str, object]:
    """Latch process shutdown before logging or filesystem side effects.

    The event is the mechanical no-new-work fence. Metadata and the grace flag
    are useful evidence, but neither is allowed to delay the fence becoming
    visible to task, subprocess, or model admission paths.
    """

    global _shutdown_first_reason
    global _shutdown_last_reason
    global _shutdown_first_requested_at_unix
    global _shutdown_first_requested_at_monotonic
    global _shutdown_request_count

    normalized_reason = str(reason or "")
    now_unix = time.time()
    now_monotonic = time.monotonic()
    with _shutdown_state_lock:
        first_request = not _shutdown_requested.is_set()
        if first_request:
            _shutdown_requested.set()
            _shutdown_first_reason = normalized_reason
            _shutdown_first_requested_at_unix = now_unix
            _shutdown_first_requested_at_monotonic = now_monotonic
            _shutdown_request_count = 1
        else:
            _shutdown_request_count += 1
        _shutdown_last_reason = normalized_reason
        snapshot = shutdown_request_snapshot()

    if first_request:
        logger.info("Shutdown requested%s.", f": {normalized_reason}" if normalized_reason else "")
        try:
            _write_grace_flag(reason=normalized_reason, created_at_unix=now_unix)
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            logger.debug(
                "Suppressed %s in core.runtime.shutdown_coordinator: %s",
                type(exc).__name__,
                exc,
            )
    return snapshot


def is_shutdown_requested() -> bool:
    return _shutdown_requested.is_set()


def clear_shutdown_request() -> None:
    """Test helper / warm-reboot helper."""
    global _shutdown_first_reason
    global _shutdown_last_reason
    global _shutdown_first_requested_at_unix
    global _shutdown_first_requested_at_monotonic
    global _shutdown_request_count

    with _shutdown_state_lock:
        _shutdown_requested.clear()
        _shutdown_first_reason = ""
        _shutdown_last_reason = ""
        _shutdown_first_requested_at_unix = None
        _shutdown_first_requested_at_monotonic = None
        _shutdown_request_count = 0


def get_shutdown_coordinator() -> ShutdownCoordinator:
    global _shutdown_coordinator
    with _singleton_lock:
        if _shutdown_coordinator is None:
            _shutdown_coordinator = ShutdownCoordinator()
        return _shutdown_coordinator


def reset_shutdown_coordinator() -> None:
    """Test helper. Drops the singleton so a fresh instance is created."""
    global _shutdown_coordinator
    with _singleton_lock:
        _shutdown_coordinator = None
    clear_shutdown_request()
