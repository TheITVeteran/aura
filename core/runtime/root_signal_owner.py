"""Root-owned OS signal handling for monotonic Aura shutdown.

The root launcher must retain SIGINT/SIGTERM ownership from pre-boot through
the last durable shutdown receipt. Inner services may observe the shutdown
latch, but they must not replace the handler or turn a repeated signal into an
ungraceful process exit while persistence is still running.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Callable
from types import FrameType

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import request_shutdown

logger = logging.getLogger("Aura.RootSignalOwner")

SignalObserver = Callable[[signal.Signals, dict[str, object]], None]


class RootShutdownSignalOwner:
    """Own SIGINT/SIGTERM for one root runtime lifecycle."""

    def __init__(self, *, scope: str, observer: SignalObserver | None = None) -> None:
        normalized_scope = str(scope or "runtime_signal").strip()
        self._scope = normalized_scope or "runtime_signal"
        self._observer = observer
        self._event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._installed: list[signal.Signals] = []
        self._bootstrap_installed = False
        self._first_reason = ""
        self._retain_for_process_exit = False

    @property
    def event(self) -> asyncio.Event:
        return self._event

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    @property
    def first_reason(self) -> str:
        return self._first_reason

    def set_observer(self, observer: SignalObserver | None) -> None:
        self._observer = observer

    def retain_for_process_exit(self) -> None:
        """Keep synchronous ownership after the event loop closes."""

        self._retain_for_process_exit = True

    def _bootstrap_handler(self, signum: int, _frame: FrameType | None) -> None:
        self._handle_signal(signal.Signals(signum))

    def install_bootstrap(self) -> int:
        """Install synchronous handlers before the event loop exists."""

        if self._bootstrap_installed:
            return len(self._installed)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._bootstrap_handler)
                self._installed.append(sig)
            except (OSError, RuntimeError, ValueError) as exc:
                record_degradation(
                    "root_signal_owner",
                    exc,
                    action=f"could not install bootstrap handler for {sig.name}",
                    severity="degraded",
                )
                logger.error("Bootstrap signal handler registration failed for %s: %s", sig.name, exc)
        self._bootstrap_installed = True
        return len(self._installed)

    def finish_async_ownership(self) -> None:
        """Close local ownership or hand it back to the synchronous root."""

        if not self._retain_for_process_exit:
            self.close()
            return

        loop = self._loop
        self._loop = None
        if loop is not None:
            for sig in self._installed:
                try:
                    loop.remove_signal_handler(sig)
                except (RuntimeError, AttributeError, NotImplementedError, ValueError) as exc:
                    logger.debug("Root signal handler handoff skipped for %s: %s", sig.name, exc)
        for sig in self._installed:
            try:
                signal.signal(sig, self._bootstrap_handler)
            except (OSError, RuntimeError, ValueError) as exc:
                record_degradation(
                    "root_signal_owner",
                    exc,
                    action=f"could not retain root handler for {sig.name} after event-loop close",
                    severity="degraded",
                )
                logger.error("Root signal handler handoff failed for %s: %s", sig.name, exc)
        self._bootstrap_installed = True

    def install(self) -> int:
        """Install handlers on the current event loop and return their count."""

        if self._loop is not None:
            return len(self._installed)
        loop = asyncio.get_running_loop()
        self._loop = loop
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
                if sig not in self._installed:
                    self._installed.append(sig)
            except (RuntimeError, AttributeError, NotImplementedError, ValueError) as exc:
                record_degradation(
                    "root_signal_owner",
                    exc,
                    action=f"could not install root handler for {sig.name}",
                    severity="degraded",
                )
                logger.error("Root signal handler registration failed for %s: %s", sig.name, exc)
        return len(self._installed)

    def close(self) -> None:
        """Remove handlers only after the root finalizer has completed."""

        loop = self._loop
        self._loop = None
        if loop is not None:
            for sig in self._installed:
                try:
                    loop.remove_signal_handler(sig)
                except (RuntimeError, AttributeError, NotImplementedError, ValueError) as exc:
                    logger.debug("Root signal handler cleanup skipped for %s: %s", sig.name, exc)
        elif self._bootstrap_installed:
            for sig in self._installed:
                try:
                    signal.signal(sig, signal.SIG_DFL)
                except (OSError, RuntimeError, ValueError) as exc:
                    logger.debug("Bootstrap signal handler cleanup skipped for %s: %s", sig.name, exc)
        self._installed.clear()
        self._bootstrap_installed = False

    async def wait(self) -> None:
        await self._event.wait()

    def _handle_signal(self, sig: signal.Signals) -> None:
        reason = f"{self._scope}:{sig.name}"
        if not self._first_reason:
            self._first_reason = reason
        snapshot = request_shutdown(reason)
        self._event.set()
        logger.info(
            "Root shutdown signal observed: scope=%s signal=%s request_count=%s first_reason=%s",
            self._scope,
            sig.name,
            snapshot.get("request_count"),
            snapshot.get("first_reason"),
        )
        if self._observer is None:
            return
        try:
            self._observer(sig, snapshot)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "root_signal_owner",
                exc,
                action="continued shutdown after root signal observer failed",
                severity="warning",
            )
            logger.warning("Root signal observer failed for %s: %s", sig.name, exc)
