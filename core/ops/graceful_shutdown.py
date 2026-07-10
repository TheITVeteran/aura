from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, ClassVar, cast

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import get_shutdown_coordinator, request_shutdown
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Shutdown")

ShutdownHook = Callable[[], Any | Awaitable[Any]]


class GracefulShutdown:
    """Root signal owner and compatibility hook bridge for canonical teardown."""

    _hooks: ClassVar[list[ShutdownHook]] = []
    _is_shutting_down: ClassVar[bool] = False
    _shutdown_event: ClassVar[asyncio.Event | None] = None
    _shutdown_owner_task: ClassVar[asyncio.Task[Any] | None] = None

    @classmethod
    def register(cls, hook: ShutdownHook) -> None:
        """Register a compatibility cleanup hook in LIFO order."""

        if hook not in cls._hooks:
            cls._hooks.append(hook)
            logger.debug(
                "Registered shutdown hook: %s",
                getattr(hook, "__name__", str(hook)),
            )

    @classmethod
    def setup_signals(cls) -> None:
        """Bind OS signals without allowing inner services to replace them."""

        loop = asyncio.get_running_loop()
        if cls._shutdown_event is None:
            cls._shutdown_event = asyncio.Event()

        def _request_signal_shutdown(sig: signal.Signals) -> None:
            # Latch synchronously in the signal callback. Scheduling the cleanup
            # coroutine first leaves a window where recovery can create work.
            request_shutdown(f"signal:{sig.name}")
            shutdown_coro = cls.trigger_shutdown(sig)
            try:
                get_task_tracker().create_task(
                    shutdown_coro,
                    name=f"graceful_shutdown.{sig.name}",
                    allow_during_shutdown=True,
                )
            except TypeError:
                # Compatibility for minimal tracker implementations used by
                # embedded hosts; the production tracker accepts the flag.
                get_task_tracker().create_task(
                    shutdown_coro,
                    name=f"graceful_shutdown.{sig.name}",
                )

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    partial(_request_signal_shutdown, sig),
                )
            except NotImplementedError as exc:
                logger.debug("Signal handler registration unsupported for %s: %s", sig, exc)
            except (RuntimeError, AttributeError, TypeError) as exc:
                logger.error("Signal handler registration failed for %s: %s", sig, exc)
                raise

    @classmethod
    async def trigger_shutdown(cls, sig: signal.Signals | str | None = None) -> None:
        """Run compatibility hooks, canonical phases, and container teardown once."""

        if cls._shutdown_event is None:
            cls._shutdown_event = asyncio.Event()
        current_task = asyncio.current_task()
        if cls._is_shutting_down:
            if current_task is cls._shutdown_owner_task:
                return
            await cls._shutdown_event.wait()
            return

        cls._is_shutting_down = True
        cls._shutdown_owner_task = current_task
        reason = f"signal:{getattr(sig, 'name', sig)}" if sig else "graceful_shutdown"
        request_shutdown(reason)

        prefix = f"Received signal {sig}: " if sig else ""
        logger.warning("Shutdown: %sinitiating graceful teardown", prefix)
        try:
            compatibility_deadline = time.monotonic() + 8.0
            while cls._hooks:
                hook = cls._hooks.pop()
                remaining = compatibility_deadline - time.monotonic()
                if remaining <= 0:
                    skipped = 1 + len(cls._hooks)
                    cls._hooks.clear()
                    exc = TimeoutError(
                        f"compatibility shutdown hook budget exhausted; skipped={skipped}"
                    )
                    record_degradation("graceful_shutdown", exc)
                    logger.error("%s", exc)
                    break
                try:
                    if inspect.iscoroutinefunction(hook):
                        result = hook()
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(cast(Callable[[], Any], hook)),
                            timeout=min(8.0, remaining),
                        )
                    if inspect.isawaitable(result):
                        remaining = max(0.05, compatibility_deadline - time.monotonic())
                        await asyncio.wait_for(result, timeout=min(8.0, remaining))
                    logger.info("Shutdown compatibility hook completed: %s", hook)
                except Exception as exc:  # noqa: BLE001 - final teardown boundary
                    record_degradation("graceful_shutdown", exc)
                    logger.error("Shutdown compatibility hook failed: %s", exc)

            try:
                report = await get_shutdown_coordinator().shutdown(timeout_per_phase=8.0)
                if not report.clean:
                    logger.warning(
                        "Shutdown coordinator completed with failures: phases=%s handlers=%s",
                        report.failed_phases,
                        sorted(report.handler_failures),
                    )
            except Exception as exc:  # noqa: BLE001 - final teardown boundary
                record_degradation("graceful_shutdown", exc)
                logger.error("Canonical shutdown coordinator failed: %s", exc)

            try:
                from core.container import get_container

                await get_container().shutdown()
            except Exception as exc:  # noqa: BLE001 - final teardown boundary
                record_degradation("graceful_shutdown", exc)
                logger.error("Container shutdown failed: %s", exc)

            logger.info("All Aura core services terminated")
        finally:
            cls._shutdown_owner_task = None
            cls._shutdown_event.set()

    @classmethod
    async def wait_for_shutdown(cls) -> None:
        """Block until the one canonical shutdown execution has completed."""

        if cls._shutdown_event is None:
            cls.setup_signals()
        shutdown_event = cls._shutdown_event
        if shutdown_event is None:  # pragma: no cover - setup_signals initializes it
            raise RuntimeError("shutdown event initialization failed")
        await shutdown_event.wait()


def register_shutdown_hook(hook: ShutdownHook) -> None:
    GracefulShutdown.register(hook)
