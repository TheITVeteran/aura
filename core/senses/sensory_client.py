from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking

logger = logging.getLogger("core.senses.sensory_client")

class SensoryLocalClient:
    """
    Supervisor for the isolated Sensory Worker process.
    Manages Vision and Audio libraries (cv2, mss, sounddevice) in a sidecar PID.
    """
    def __init__(self) -> None:
        self._process: Any | None = None
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._running = False
        self._lock: asyncio.Lock | None = None
        self._start_lock: asyncio.Lock | None = None

    def _ensure_async_locks(self) -> tuple[asyncio.Lock, asyncio.Lock]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        return self._lock, self._start_lock

    async def start(self) -> bool:
        """Start the isolated sensory worker."""
        if is_shutdown_requested():
            logger.info("Sensory worker start refused: runtime shutdown requested")
            return False
        from .sensory_worker import sensory_worker_loop
        _, start_lock = self._ensure_async_locks()

        async with start_lock:
            if is_shutdown_requested():
                return False
            if self.is_alive():
                logger.debug("👀 Sensory Client: Worker already alive.")
                return True

            ctx_name = "spawn" if sys.platform == "darwin" else "forkserver"
            ctx: Any = mp.get_context(ctx_name)
            self._replace_queues(ctx)
            process = ctx.Process(
                target=sensory_worker_loop,
                args=(self._req_q, self._res_q),
                name="AuraSensoryWorker",
                daemon=True
            )
            self._process = process
            previous_sidecar_flag = os.environ.get("AURA_MEDIA_SIDECAR_PROCESS")
            os.environ["AURA_MEDIA_SIDECAR_PROCESS"] = "1"
            try:
                if is_shutdown_requested():
                    process = self._process
                    self._process = None
                    self._close_queues()
                    close = getattr(process, "close", None)
                    if callable(close):
                        close()
                    return False
                process.start()
            except RuntimeError:
                if is_shutdown_requested():
                    self._process = None
                    self._close_queues()
                    close = getattr(process, "close", None)
                    if callable(close):
                        close()
                    return False
                raise
            finally:
                if previous_sidecar_flag is None:
                    os.environ.pop("AURA_MEDIA_SIDECAR_PROCESS", None)
                else:
                    os.environ["AURA_MEDIA_SIDECAR_PROCESS"] = previous_sidecar_flag
            if is_shutdown_requested():
                logger.info("Sensory worker crossed shutdown during spawn; stopping it")
                await self.stop()
                return False
            self._running = True
            logger.info("👀 Sensory Client: Worker started via %s (PID: %d)", ctx_name, process.pid)

            if not await self._send_command("ping", timeout=2.0, auto_restart=False):
                logger.error("🛑 Sensory Client: Worker failed initial ping.")
                await self.stop()
                return False

            success = await self._send_command("init_vision", auto_restart=False)
            if success:
                logger.info("   ✅ Vision isolated successfully")

            success = await self._send_command("init_audio", auto_restart=False)
            if success:
                logger.info("   ✅ Audio isolated successfully")

            return True

    def _drain_queues(self) -> None:
        if self._req_q is None or self._res_q is None:
            return
        while not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                break
        while not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except (RuntimeError, AttributeError, TypeError, ValueError):
                break

    @staticmethod
    def _safe_close_queue(queue_obj: Any) -> None:
        if queue_obj is None:
            return

        def _close_and_join() -> None:
            close = getattr(queue_obj, "close", None)
            if callable(close):
                close()
            join_thread = getattr(queue_obj, "join_thread", None)
            if callable(join_thread):
                join_thread()

        try:
            run_sync_shutdown_callable_blocking(
                _close_and_join,
                timeout_s=1.0,
                name="sensory-queue-close",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
            logger.debug("Suppressed queue close error in sensory client: %s", exc)
        else:
            try:
                from core.runtime.runtime_hygiene import get_runtime_hygiene

                get_runtime_hygiene().unregister_shutdown_resource(queue_obj)
            except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                pass

    def _close_queues(self) -> None:
        self._safe_close_queue(self._req_q)
        self._safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    def _replace_queues(self, ctx: Any | None = None) -> None:
        self._close_queues()
        factory = ctx.Queue if ctx is not None and hasattr(ctx, "Queue") else mp.Queue
        self._req_q = factory()
        self._res_q = factory()
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            hygiene = get_runtime_hygiene()
            hygiene.register_shutdown_resource(
                self._req_q,
                kind="multiprocessing_queue",
                name="sensory.request_queue",
                source="core.senses.sensory_client",
                timeout_s=1.0,
            )
            hygiene.register_shutdown_resource(
                self._res_q,
                kind="multiprocessing_queue",
                name="sensory.response_queue",
                source="core.senses.sensory_client",
                timeout_s=1.0,
            )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            self._close_queues()
            raise

    async def _send_command(  # noqa: ASYNC109 - timeout bounds blocking sidecar IPC.
        self,
        cmd: str,
        data: Any = None,
        *,
        timeout: float = 5.0,  # noqa: ASYNC109 - bounds blocking sidecar IPC.
        auto_restart: bool = True,
    ) -> bool:
        if is_shutdown_requested():
            logger.info("Sensory command %s refused during runtime shutdown", cmd)
            return False
        if not self.is_alive():
            if not auto_restart:
                logger.warning("👀 Sensory Client: Worker unavailable for command %s", cmd)
                return False
            logger.warning("♻️ Sensory Client: Worker offline before command %s. Restarting.", cmd)
            started = await self.start()
            if not started:
                return False

        command_lock, _ = self._ensure_async_locks()

        async with command_lock:
            if self._req_q is None or self._res_q is None:
                logger.warning("👀 Sensory Client: Worker queues unavailable for command %s", cmd)
                return False

            # [STRUCTURAL UNIFICATION] Report sensory tasks to registry
            from core.supervisor.registry import TaskStatus, get_task_registry
            registry = get_task_registry()
            task_id = registry.register_task("sensory_gate", f"Sensory: {cmd}", {"data": str(data)})
            
            try:
                self._req_q.put({"command": cmd, "data": data})
                registry.update_task(task_id, status=TaskStatus.RUNNING)
                
                # Wait for response in a thread to non-block
                res = await asyncio.to_thread(self._res_q.get, timeout=timeout)
                
                if res.get("status") == "ok":
                    registry.update_task(task_id, status=TaskStatus.COMPLETED)
                    return True
                else:
                    registry.update_task(task_id, status=TaskStatus.FAILED, error=res.get("msg"))
                    return False
            except (OSError, ConnectionError, TimeoutError, queue.Empty) as e:
                record_degradation('sensory_client', e)
                logger.error("🛑 Sensory Client Command [%s] failed: %s", cmd, e)
                registry.update_task(task_id, status=TaskStatus.FAILED, error=str(e))
                return False

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def stop(self) -> None:
        self._running = False
        process, self._process = self._process, None
        if process:
            try:
                if self._req_q is not None:
                    self._req_q.put({"command": "exit"})
            except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                record_degradation('sensory_client', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            # Issue 26: Use asyncio.to_thread for blocking process join
            await asyncio.to_thread(process.join, timeout=2.0)
            if process.is_alive():
                process.terminate()
                await asyncio.to_thread(process.join, timeout=1.0)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, timeout=1.0)
            logger.info("👀 Sensory Client: Worker stopped")
            self._drain_queues()
        self._close_queues()

    close = stop
    cleanup = stop
    on_stop = stop

_instance: SensoryLocalClient | None = None
_client_lock = threading.Lock()

def get_sensory_client() -> SensoryLocalClient:
    global _instance
    with _client_lock:
        if _instance is None:
            _instance = SensoryLocalClient()
    return _instance
