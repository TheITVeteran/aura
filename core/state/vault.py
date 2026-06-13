from core.runtime.errors import record_degradation
import asyncio
import logging
import time
import os
import json
import signal
import traceback
from multiprocessing import Process, Pipe
from types import SimpleNamespace
from typing import Dict, Any, Optional

from .state_repository import StateRepository, get_state_shm_size_bytes
from .aura_state import AuraState
from ..bus.shared_mem_bus import SharedMemoryTransport
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Actor.StateVault")


def _should_force_vault_process_exit() -> bool:
    """Return true for live spawned actor children after cleanup completes."""
    return not bool(
        os.getenv("PYTEST_CURRENT_TEST")
        or os.getenv("AURA_DISABLE_VAULT_HARD_EXIT")
    )


def _finalize_vault_process_exit(exit_code: int = 0) -> None:
    """Prevent inherited helper threads/resources from keeping actor children alive."""
    if not _should_force_vault_process_exit():
        return
    logger.debug("StateVaultActor process cleanup complete; exiting child process.")
    os._exit(exit_code)


class StateVaultActor:
    """
    Standalone process that manages the canonical AuraState.
    Protects the 'Self' from kernel stalls or actor crashes.
    """
    
    def __init__(self, db_path: str = "data/aura_state.db"):
        self.db_path = db_path
        # Vault must be the OWNER of the state repository
        self.repo = StateRepository(db_path=db_path, is_vault_owner=True)
        self.shm_transport = SharedMemoryTransport(name="aura_state_shm", size=get_state_shm_size_bytes())
        self._is_running = False
        self._bus: Optional[Any] = None # Will be linked to the pipe
        self._heartbeat_interval = 3.0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()
        self._stop_event: Optional[asyncio.Event] = None

    def _track_task(self, coro: Any, *, name: Optional[str] = None) -> asyncio.Task:
        task = get_task_tracker().create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_background_tasks(self):
        tasks = [task for task in self._background_tasks if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def run(self, pipe):
        """Main actor loop."""
        from ..bus.local_pipe_bus import LocalPipeBus
        # Use the full LocalPipeBus power with concurrent handlers
        # This prevents head-of-line blocking (e.g. slow commit vs fast ping)
        self._bus = LocalPipeBus(is_child=True, connection=pipe, start_reader=True)
        self._stop_event = asyncio.Event()
        try:
            # Register handlers
            self._bus.register_handler("commit", self._process_commit_bus)
            self._bus.register_handler("get_state", self._process_get_state_bus)
            self._bus.register_handler("ping", self._process_ping_bus)
            self._bus.register_handler("stop", self._process_stop_bus)

            self._bus.start()
            logger.info("Starting State Vault Actor with concurrent bus handlers...")
            self._is_running = True
            self._heartbeat_task = self._track_task(
                self._heartbeat_loop(),
                name="state_vault.heartbeat",
            )

            # 1. Initialize Repo
            await self.repo.initialize()
            self.shm_transport = self.repo._shm
            logger.info("State Vault Actor ONLINE.")

            # Keep process alive while bus is running. The stop handler sets
            # the event so shutdown does not wait for the next sleep tick.
            while self._is_running and self._bus._is_running:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                except TimeoutError:
                    continue
                break
        finally:
            self._is_running = False
            await self._cancel_background_tasks()
            if self._bus is not None:
                await self._bus.stop()
            repo_shm = getattr(self.repo, "_shm", None)
            try:
                await self.repo.close()
            except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                record_degradation(
                    "vault",
                    exc,
                    action="continued StateVaultActor shutdown after repository close failed",
                )
                logger.warning("StateVaultActor repository close failed during shutdown: %s", exc)
            if self.shm_transport is not None and self.shm_transport is not repo_shm:
                try:
                    self.shm_transport.close()
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation(
                        "vault",
                        exc,
                        action="continued StateVaultActor shutdown after shared memory close failed",
                    )
                    logger.debug("StateVaultActor shared memory close issue: %s", exc)

    async def _heartbeat_loop(self):
        """Emit liveness pulses without racing the parent transport reader."""
        while self._is_running:
            try:
                if self._bus:
                    await self._bus.send(
                        "heartbeat",
                        {"pid": os.getpid(), "ts": time.time(), "status": "healthy"},
                    )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('vault', e)
                logger.debug("StateVault heartbeat failed: %s", e)
            await asyncio.sleep(self._heartbeat_interval)

    async def _process_ping_bus(self, payload: Any, trace_id: Optional[str]):
        """Respond to health pings immediately."""
        return {"type": "pong", "ts": time.time()}

    async def _process_stop_bus(self, payload: Any, trace_id: Optional[str]):
        self._is_running = False
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        return {"ok": True, "stopping": True}

    async def _process_commit_bus(self, payload: Any, trace_id: Optional[str]):
        """Bridge between bus handler and existing commit logic."""
        return await self._process_commit_inner(payload, trace_id)

    async def _process_get_state_bus(self, payload: Any, trace_id: Optional[str]):
        """Bridge for get_state."""
        if not self.repo._current:
            return None
            
        res = {
            "version": self.repo._current.version,
            "shm_name": self.shm_transport.name
        }
        
        if payload and payload.get("full"):
            # Provide the full state dictionary for cold-boot sync
            res["state"] = self.repo._circular_safe_asdict(self.repo._current)
            
        return res

    def _preserve_cold_store_for_hot_payload(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Restore cold continuity when a proxy sends a bounded hot-state payload."""
        if "cold" in state_data or self.repo._current is None:
            return state_data

        merged = dict(state_data)
        try:
            merged["cold"] = self.repo._circular_safe_asdict(self.repo._current.cold)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "vault",
                exc,
                action="continued hot state commit with default cold store after preservation failed",
            )
            logger.warning("Cold-store preservation failed for hot state commit: %s", exc)
        return merged

    async def _deserialize_commit_state(self, state_data: Dict[str, Any]) -> AuraState:
        """Deserialize bus commit payloads without dropping durable cold state."""
        if not isinstance(state_data, dict):
            raise ValueError("state_vault commit payload must include a state dictionary")
        normalized = self._preserve_cold_store_for_hot_payload(state_data)
        return await asyncio.to_thread(lambda: self.repo._deserialize(json.dumps(normalized)))

    async def _process_commit_inner(self, payload: Dict[str, Any], trace_id: Optional[str]):
        """Atomically commit a state mutation (Core Logic)."""
        try:
            new_state_data = payload.get("state")
            cause = payload.get("cause", "remote_update")
            
            # Offload heavy serialization/deserialization to thread
            new_state = await self._deserialize_commit_state(new_state_data)

            committed_state = await self.repo.commit(new_state, cause, trace_id)

            # Debounced SHM Update
            now = time.time()
            if not hasattr(self, "_last_shm_update") or (now - self._last_shm_update > 0.1):
                self._last_shm_update = now
                self._track_task(
                    self._update_shared_memory_async(committed_state),
                    name="state_vault.sync_shared_memory",
                )

            return {"version": committed_state.version, "state_id": committed_state.state_id}
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('vault', e)
            logger.error("Commit failed: %s", e)
            raise

    async def _update_shared_memory_async(self, state: AuraState):
        """Async wrapper for non-blocking SHM sync."""
        try:
            serialized_state = await asyncio.to_thread(self.repo._serialize, state)
            from core.governance_context import governed_scope

            sync_decision = SimpleNamespace(
                receipt_id=f"state_vault_shm_sync:{getattr(state, 'version', 'unknown')}",
                domain="state_mutation",
                source="state_vault.sync_shared_memory",
                constraints={"path": "shared_memory", "state_version": getattr(state, "version", None)},
            )
            async with governed_scope(sync_decision):
                mode = await self.repo._sync_to_shm(state, serialized_state)
            logger.debug("SHM Updated: Version %s (%s)", state.version, mode)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('vault', e)
            logger.error("SHM Update Failed: %s", e)

    def _update_shared_memory(self, state: AuraState):
        """Legacy synchronous path (deprecated)."""
        state_dict = self.repo._circular_safe_asdict(state)
        self.shm_transport.write(state_dict)

def vault_process_entry(db_path: str, pipe):
    """Entry point for the vault process."""
    exit_code = 0
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
        logger.debug("Suppressed %s in core.state.vault: %s", type(_exc).__name__, _exc)
    # Force bounded logging to stderr so it shows up in main logs even if setup fails.
    # DEBUG here is unsafe: aiosqlite logs full SQL parameter payloads, including
    # large state snapshots, which can stall proof and live runtimes.
    import sys
    raw_level = os.getenv("AURA_VAULT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format='[VAULT-PROC] %(levelname)s: %(message)s',
        force=True,
    )
    logging.getLogger().setLevel(level)
    for inherited_debug_logger in ("Aura.Core", "core", "Bus.SharedMem"):
        logging.getLogger(inherited_debug_logger).setLevel(max(level, logging.INFO))
    for noisy_logger in ("aiosqlite", "aiosqlite.core"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.info("Vault process entry started. DB Path: %s", db_path)
    try:
        logger.debug("StateVaultActor instantiating...")
        actor = StateVaultActor(db_path=db_path) 
        logger.debug("StateVaultActor instantiated. Running asyncio loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(actor.run(pipe))
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending:
                logger.debug(
                    "StateVaultActor cancelling %d pending loop task(s) before asyncgen shutdown.",
                    len(pending),
                )
                for task in pending:
                    task.cancel()
                results = loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                unexpected = [
                    result
                    for result in results
                    if isinstance(result, BaseException)
                    and not isinstance(result, asyncio.CancelledError)
                ]
                if unexpected:
                    logger.warning(
                        "StateVaultActor shutdown task cancellation produced %d unexpected error(s).",
                        len(unexpected),
                    )
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        logger.debug("StateVaultActor asyncio loop exited gracefully.")
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        exit_code = 1
        record_degradation('vault', e)
        logger.critical("Vault process CRASHED: %s", e)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        _finalize_vault_process_exit(exit_code)
