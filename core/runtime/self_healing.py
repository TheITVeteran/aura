"""core/runtime/self_healing.py

Self-Healing Loop
==================
A small, hot-path loop that watches for cognitive-module hangs and
restarts the offending module without losing receipts. It is
complementary to StabilityGuardian (broad health checks) and the
OrganSupervisor (subprocess restart): self_healing operates *inside*
the same process, on async tasks that have stopped progressing.

Detection signals:

  * a registered "heartbeat" callable hasn't been called in N seconds
  * an asyncio.Task referenced in the registry is in a "hanging" state
    (still pending after a grace window beyond its declared budget)

Repair actions:

  * cancel the hanging task
  * call the module's ``restart_async()`` if available; otherwise
    re-instantiate via ServiceContainer
  * record an action receipt + a phenomenal envelope (severity = 0.5)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.SelfHealing")

_DIR = Path.home() / ".aura" / "data" / "self_healing"
_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER = _DIR / "events.jsonl"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _deep_repair_block_reason(origin: str = "self_healing_deep_repair") -> str:
    """Return a reason deep repair must not run in this runtime mode."""

    if not _env_flag("AURA_ENABLE_DEEP_REPAIR", False):
        return "deep_repair_disabled"
    try:
        from core.runtime.background_policy import background_loop_start_reason

        return background_loop_start_reason(origin)
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "self_healing",
            exc,
            action="blocked deep repair because background policy was unavailable",
            receipt_required=True,
        )
        return "background_policy_unavailable"


@dataclass
class WatchEntry:
    name: str
    last_heartbeat_at: float = field(default_factory=time.time)
    expected_interval_s: float = 30.0
    restart_async: Callable[[], Awaitable[None]] | None = None
    container_key: str | None = None
    restarts: int = 0


class SelfHealing:
    def __init__(self) -> None:
        self._watches: dict[str, WatchEntry] = {}
        self._deep_repairs: dict[str, asyncio.Task] = {}
        self._module_path_cache: dict[str, str | None] = {}
        self._task: asyncio.Task | None = None
        self._running = False
        try:
            configured_timeout = float(os.environ.get("AURA_SELF_HEALING_LEDGER_TIMEOUT_S", "5.0"))
        except (TypeError, ValueError):
            configured_timeout = 5.0
        self._ledger_write_timeout_s = min(30.0, max(1.0, configured_timeout))

    def watch(
        self,
        name: str,
        *,
        expected_interval_s: float = 30.0,
        restart_async: Callable[[], Awaitable[None]] | None = None,
        container_key: str | None = None,
    ) -> None:
        self._watches[name] = WatchEntry(
            name=name,
            expected_interval_s=expected_interval_s,
            restart_async=restart_async,
            container_key=container_key,
        )

    def heartbeat(self, name: str) -> None:
        w = self._watches.get(name)
        if w is None:
            return
        w.last_heartbeat_at = time.time()

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop():
            while self._running:
                try:
                    await self._tick()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    if not self._running or is_shutdown_requested():
                        break
                    logger.warning("SelfHealing loop spuriously cancelled. Ignoring.")
                    continue
                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation('self_healing', e)
                    logger.error("SelfHealing loop error: %s", e)
                    await asyncio.sleep(1.0)

        self._task = create_tracked_task(_loop(), name="SelfHealing")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None

    async def _tick(self) -> None:
        now = time.time()
        for w in list(self._watches.values()):
            age = now - w.last_heartbeat_at
            if age <= w.expected_interval_s * 2.5:
                continue
            if self._foreground_runtime_busy():
                w.last_heartbeat_at = now
                await self._append_record_async(
                    {
                        "when": now,
                        "name": w.name,
                        "stale_for_s": age,
                        "result": "deferred_foreground_busy",
                    }
                )
                continue
            await self._heal(w, age)

    def _foreground_runtime_busy(self) -> bool:
        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            if foreground_activity_reason():
                return True
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("self_healing", exc)
            return True

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            status_getter = getattr(gate, "get_conversation_status", None)
            if callable(status_getter):
                status = status_getter() or {}
                if isinstance(status, dict):
                    return bool(
                        status.get("active")
                        or status.get("foreground_owned")
                        or int(status.get("active_generations", 0) or 0) > 0
                        or status.get("kernel_lock_held")
                        or str(status.get("state", "")).lower()
                        in {"spawning", "handshaking", "warming", "recovering"}
                    )
                if bool(getattr(status, "active", False)):
                    return True
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("self_healing", exc)
            return True

        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None) or ServiceContainer.get("aura_runtime", default=None)
            if orch is None:
                return False
            status = getattr(orch, "status", None)
            if not bool(getattr(status, "is_processing", False)):
                return False
            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("self_healing", exc)
            return False

    async def _heal(self, w: WatchEntry, age: float) -> None:
        record = {
            "when": time.time(),
            "name": w.name,
            "stale_for_s": age,
            "restart_count": w.restarts,
        }
        try:
            if w.restarts >= 3:
                module_path = await asyncio.to_thread(self._module_path_for_watch, w)
                block_reason = _deep_repair_block_reason("self_healing_watchdog_deep_repair")
                if module_path and not block_reason:
                    logger.warning("Deep repair triggered for %s (%s)", w.name, module_path)
                    scheduled = self.schedule_deep_repair(
                        module_path,
                        reason="watchdog_restart_exhausted",
                        watch_name=w.name,
                        metadata={"stale_for_s": age, "restart_count": w.restarts},
                    )
                    record.update(scheduled)
                    w.restarts = 0
                    w.last_heartbeat_at = time.time()
                else:
                    record["result"] = (
                        block_reason
                        if module_path
                        else "deep_repair_failed_no_module_path"
                    )
                    w.restarts = 0
                    w.last_heartbeat_at = time.time()
                    await self._append_record_async(record)
                    return

            if record.get("result") not in (
                "deep_repair_scheduled",
                "deep_repair_already_running",
                "deep_repair_failed_no_module_path",
                "deep_repair_failed_no_lab",
            ):
                if w.restart_async is not None:
                    await w.restart_async()
                elif w.container_key:
                    from core.container import ServiceContainer
                    instance = ServiceContainer.get(w.container_key, default=None)
                    if instance is not None and hasattr(instance, "restart_async"):
                        await instance.restart_async()
                w.restarts += 1
                w.last_heartbeat_at = time.time()
                record["result"] = "restarted"
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_healing', exc)
            record["result"] = f"restart_failed:{exc}"
        await self._append_record_async(record)

    def _module_path_for_watch(self, w: WatchEntry) -> str | None:
        if not w.container_key:
            return None
        cached = self._module_path_cache.get(w.container_key)
        if w.container_key in self._module_path_cache:
            return cached
            
        fallbacks = {
            "orchestrator": "core/orchestrator/main.py",
            "mind_tick": "core/mind_tick.py",
            "scheduler": "core/scheduler.py",
            "morphogenetic_runtime": "core/morphogenesis/runtime.py",
            "motor_cortex": "core/somatic/motor_cortex.py"
        }
        fallback = fallbacks.get(w.container_key)
        
        try:
            from core.config import config
            from core.container import ServiceContainer

            instance = ServiceContainer.get(w.container_key, default=None)
            if instance is None:
                self._module_path_cache[w.container_key] = fallback
                return fallback
                
            # Unpack proxies if present
            if hasattr(instance, "__wrapped__"):
                instance = instance.__wrapped__
            elif hasattr(instance, "_instance"):
                instance = instance._instance or instance

            source_file = inspect.getsourcefile(type(instance)) or inspect.getfile(type(instance))
            if source_file:
                source_path = Path(source_file)
                try:
                    base_dir = Path(config.paths.base_dir)
                    resolved = str(source_path.relative_to(base_dir))
                    self._module_path_cache[w.container_key] = resolved
                    return resolved
                except ValueError as _exc:
                    logger.debug("Suppressed %s in core.runtime.self_healing: %s", type(_exc).__name__, _exc)

            module_name = type(instance).__module__
            candidate = module_name.replace(".", "/") + ".py"
            if (config.paths.base_dir / candidate).exists():
                self._module_path_cache[w.container_key] = candidate
                return candidate
                
            self._module_path_cache[w.container_key] = fallback
            return fallback
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_healing', exc)
            logger.debug("Could not resolve watched module path for %s: %s", w.name, exc)
            self._module_path_cache[w.container_key] = fallback
            return fallback

    def schedule_deep_repair(
        self,
        module_path: str,
        *,
        reason: str,
        watch_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Schedule a ReimplementationLab repair without blocking the watchdog."""

        key = str(module_path)
        block_reason = _deep_repair_block_reason("self_healing_schedule_deep_repair")
        if block_reason:
            return {
                "result": block_reason,
                "module_path": key,
                "reason": reason,
            }
        existing = self._deep_repairs.get(key)
        if existing is not None and not existing.done():
            return {
                "result": "deep_repair_already_running",
                "module_path": key,
                "reason": reason,
            }

        async def _runner() -> None:
            await self.request_deep_repair(
                key,
                reason=reason,
                watch_name=watch_name,
                metadata=metadata,
                max_attempts=max_attempts,
            )

        try:
            task = create_tracked_task(_runner(), name=f"SelfHealing.deep_repair.{key}")
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "self_healing",
                exc,
                action="failed closed when deep repair task could not be scheduled through task ownership",
                receipt_required=True,
            )
            return {
                "result": "deep_repair_schedule_failed",
                "module_path": key,
                "reason": reason,
            }
        self._deep_repairs[key] = task
        task.add_done_callback(lambda _task: self._deep_repairs.pop(key, None))
        return {
            "result": "deep_repair_scheduled",
            "module_path": key,
            "reason": reason,
        }

    async def request_deep_repair(
        self,
        module_path: str,
        *,
        reason: str,
        watch_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Run ReimplementationLab as a deep repair strategy.

        This is intentionally separate from restart healing so patch-based
        repair systems can call the same hardened path when local diffs fail.
        """

        record: dict[str, Any] = {
            "when": time.time(),
            "name": watch_name or "deep_repair",
            "module_path": str(module_path),
            "reason": reason,
            "metadata": metadata or {},
        }
        block_reason = _deep_repair_block_reason("self_healing_request_deep_repair")
        if block_reason:
            record["result"] = block_reason
            await self._append_record_async(record)
            return record
        try:
            from core.container import ServiceContainer

            lab = ServiceContainer.get("reimplementation_lab", default=None)
            if lab is None:
                record["result"] = "deep_repair_failed_no_lab"
                return record

            lab_metadata = {
                "trigger": "self_healing",
                "reason": reason,
                "watch_name": watch_name,
                **(metadata or {}),
            }
            result = await lab.run_reconstruction(
                str(module_path),
                max_attempts=max_attempts,
                metadata=lab_metadata,
            )
            result_dict = result.to_dict() if hasattr(result, "to_dict") else {"success": False}
            record["result"] = "deep_repair_succeeded" if result_dict.get("success") else "deep_repair_rejected"
            record["lab_result"] = result_dict
            return record
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_healing', exc)
            record["result"] = f"deep_repair_failed:{exc}"
            return record
        finally:
            await self._append_record_async(record)

    async def _append_record_async(self, record: dict[str, Any]) -> None:
        """Persist a healing receipt without blocking the main asyncio loop."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._append_record, record),
                timeout=self._ledger_write_timeout_s,
            )
        except TimeoutError:
            logger.warning("SelfHealing ledger write timed out; preserving live loop responsiveness.")
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("SelfHealing ledger write failed asynchronously: %s", exc)

    def _append_record(self, record: dict[str, Any]) -> None:
        try:
            with local_internal_governed_scope(
                "runtime.self_healing.ledger",
                domain="file_write",
            ):
                get_file_write_gateway().append_text(
                    _LEDGER,
                    json.dumps(record, default=str) + "\n",
                    source="runtime.self_healing.ledger",
                )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("SelfHealing ledger append failed: %s", exc)


_HEALER: SelfHealing | None = None


def get_healer() -> SelfHealing:
    global _HEALER
    if _HEALER is None:
        _HEALER = SelfHealing()
    return _HEALER


__all__ = ["SelfHealing", "WatchEntry", "get_healer"]
