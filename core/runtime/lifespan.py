"""Compatibility lifecycle facade over Aura's canonical runtime owners.

Historically this module maintained a second hard-coded service start/stop
list.  That path could lazily resolve services while the process was already
quiescing and it was invisible to the process-wide shutdown coordinator.
Keep the public API for older callers, but make it a strict adapter: startup
hooks are explicit, shutdown hooks belong to the canonical phase graph, and
every stop request uses the monotonic process latch.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.executors import run_blocking_io
from core.runtime.lockdep import checked_async_lock, checked_lock
from core.runtime.shutdown_coordinator import (
    ShutdownReport,
    get_shutdown_coordinator,
    is_shutdown_requested,
    request_shutdown,
)

logger = logging.getLogger("Aura.Lifespan")


class LifespanManager:
    """Legacy-compatible facade with one canonical shutdown owner."""

    def __init__(self) -> None:
        self.startup_tasks: list[Callable[[], Any]] = []
        self.shutdown_tasks: list[Callable[[], Any]] = []
        self._running = False
        self._startup_lock = checked_async_lock("lifespan.startup")
        self._state_lock = checked_lock("lifespan.state", reentrant=True)
        self._last_shutdown_report: ShutdownReport | None = None

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    async def _invoke_startup_hook(self, hook: Callable[[], Any]) -> None:
        if inspect.iscoroutinefunction(hook):
            result = hook()
        else:
            result = await run_blocking_io(
                hook,
                timeout_s=15.0,
                label=f"lifespan-startup:{getattr(hook, '__qualname__', 'hook')}",
            )
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=15.0)

    async def startup(self) -> None:
        """Run explicitly registered startup hooks without resolving services."""

        if is_shutdown_requested():
            raise RuntimeError("runtime_shutdown")
        async with self._startup_lock:
            if self.running:
                return
            if is_shutdown_requested():
                raise RuntimeError("runtime_shutdown")
            logger.info("Aura lifespan startup sequence initiated")
            try:
                for hook in tuple(self.startup_tasks):
                    if is_shutdown_requested():
                        raise RuntimeError("runtime_shutdown")
                    await self._invoke_startup_hook(hook)
            except asyncio.CancelledError as exc:
                record_degradation(
                    "lifespan",
                    exc,
                    severity="degraded",
                    action="latched canonical shutdown after startup cancellation",
                )
                await self.emergency_shutdown(reason="lifespan_startup_cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - startup ownership boundary
                record_degradation(
                    "lifespan",
                    exc,
                    severity="degraded",
                    action="latched canonical shutdown after startup hook failure",
                )
                await self.emergency_shutdown(reason="lifespan_startup_failed")
                raise
            with self._state_lock:
                self._running = True

    async def shutdown(self, *, reason: str = "lifespan_shutdown") -> ShutdownReport:
        """Latch quiescence and replay the one canonical teardown execution."""

        request_shutdown(reason)
        report = await get_shutdown_coordinator().shutdown()
        with self._state_lock:
            self._running = False
            self._last_shutdown_report = report.clone()
        logger.info(
            "Aura canonical shutdown complete (clean=%s repeated=%d)",
            report.clean,
            report.repeated_call_count,
        )
        return report

    async def on_stop_async(self) -> None:
        """ServiceContainer hook; canonical shutdown remains the sole owner."""

        await self.shutdown(reason="lifespan_container_stop")

    def register_startup(self, callback: Callable[[], Any]) -> None:
        if not callable(callback):
            raise TypeError("startup callback must be callable")
        with self._state_lock:
            if self._running:
                raise RuntimeError("cannot register startup hooks after startup")
            self.startup_tasks.append(callback)

    def register_shutdown(
        self,
        callback: Callable[[], Any],
        *,
        phase: str = "actors",
        name: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not callable(callback):
            raise TypeError("shutdown callback must be callable")
        get_shutdown_coordinator().register(
            callback,
            phase=phase,
            name=name or getattr(callback, "__qualname__", "lifespan_hook"),
            timeout=timeout,
        )
        with self._state_lock:
            self.shutdown_tasks.append(callback)

    async def emergency_shutdown(
        self,
        *,
        reason: str = "lifespan_emergency_shutdown",
    ) -> ShutdownReport:
        logger.critical("Emergency shutdown requested through lifespan facade")
        return await self.shutdown(reason=reason)

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            report = (
                self._last_shutdown_report.clone().as_dict()
                if self._last_shutdown_report is not None
                else None
            )
            return {
                "running": self._running,
                "shutdown_requested": is_shutdown_requested(),
                "startup_hook_count": len(self.startup_tasks),
                "shutdown_hook_count": len(self.shutdown_tasks),
                "last_shutdown_report": report,
            }


_lifespan: LifespanManager | None = None
_lifespan_lock = checked_lock("lifespan.singleton")


def get_lifespan_manager() -> LifespanManager:
    global _lifespan
    if _lifespan is None:
        with _lifespan_lock:
            if _lifespan is None:
                _lifespan = LifespanManager()
    return _lifespan
