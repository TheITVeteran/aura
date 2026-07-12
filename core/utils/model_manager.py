# core/utils/model_manager.py
"""
ModelManager: single point-of-truth for loading/unloading heavy model objects.
- serializes heavy loads with an asyncio.Semaphore
- tracks loaded models in LRU order (OrderedDict)
- evicts least-recently-used model when memory pressure or configured cap exceeded
- exposes async load_model / unload_model
"""

import asyncio
import inspect
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import psutil

from core.runtime.errors import record_degradation

logger = logging.getLogger("aura.model_manager")


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded into the local manager."""


class ModelManager:
    def __init__(
        self,
        load_fn: Callable[[str, dict[str, Any]], Any],
        max_models: int = 2,
        semaphore_value: int = 1,
    ) -> None:
        """
        load_fn(name, opts) -> model_object
        """
        if max_models <= 0:
            raise ValueError("max_models must be positive")
        if semaphore_value <= 0:
            raise ValueError("semaphore_value must be positive")
        self._load_fn = load_fn
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._meta: dict[str, dict[str, Any]] = {}
        self._semaphore = asyncio.Semaphore(semaphore_value)
        self._max_models = max_models
        self._lock = asyncio.Lock()
        self._last_used: dict[str, float] = {}

    def _pop_model_locked(
        self,
        name: str,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Internal helper to remove model from state tracking. MUST hold _lock."""
        if name not in self._models:
            return None, None
        obj = self._models.pop(name)
        meta = self._meta.pop(name, {})
        self._last_used.pop(name, None)
        return obj, meta

    async def _cleanup_model(self, obj: Any, name: str) -> None:
        """Internal helper to actually close/unload a model object. NO lock needed."""
        try:
            if hasattr(obj, "close"):
                maybe = obj.close()
                if inspect.isawaitable(maybe):
                    await maybe
            elif hasattr(obj, "unload"):
                maybe = obj.unload()
                if inspect.isawaitable(maybe):
                    await maybe
        except (RuntimeError, AttributeError, TypeError):
            logger.exception("ModelManager: exception while unloading %s", name)

    @staticmethod
    async def _release_lane_lease(meta: dict[str, Any] | None, *, reason: str) -> None:
        lease = dict(meta or {}).get("lane_lease")
        if lease is None:
            return
        try:
            await lease.release(reason=reason)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "model_manager",
                exc,
                action="model object unloaded but in-process lane lease release failed",
            )

    async def load_model(self, name: str, opts: dict[str, Any] | None = None) -> Any:
        opts = dict(opts or {})
        async with self._lock:
            if name in self._models:
                # move to end (most-recently used)
                self._models.move_to_end(name)
                self._last_used[name] = time.time()
                logger.debug("ModelManager: model %s already loaded (touch)", name)
                return self._models[name]

        # serialize heavy model loads
        async with self._semaphore:
            cleanup_obj = None
            cleanup_meta = None
            evicted_name = None
            pressure_error: ModelLoadError | None = None
            
            async with self._lock:
                # double-check after obtaining lock
                if name in self._models:
                    self._models.move_to_end(name)
                    self._last_used[name] = time.time()
                    return self._models[name]

                # if we've hit capacity, evict LRU
                if len(self._models) >= self._max_models:
                    evicted_name = next(iter(self._models.keys()))
                    logger.info("ModelManager: capacity full (%d). Evicting LRU model: %s", self._max_models, evicted_name)
                    cleanup_obj, cleanup_meta = self._pop_model_locked(evicted_name)

                # check memory pressure before loading
                vm = psutil.virtual_memory()
                if vm.percent > 85.0:
                    pressure_error = ModelLoadError(
                        f"Refusing to load model {name} — host memory at {vm.percent:.1f}%"
                    )

            # Clean up evicted model WITHOUT holding self._lock to avoid deadlock
            if cleanup_obj is not None and evicted_name is not None:
                await self._cleanup_model(cleanup_obj, evicted_name)
                await self._release_lane_lease(
                    cleanup_meta,
                    reason=f"model_manager_lru_evict:{evicted_name}",
                )
            if pressure_error is not None:
                raise pressure_error

            # perform actual load (synchronous or async support)
            logger.info("ModelManager: loading model %s", name)
            from core.runtime.model_lane_control import (
                acquire_in_process_model_lane,
                run_owned_model_thread_call,
            )

            model_path = str(opts.get("model_path") or name)
            purpose = str(opts.get("purpose") or "serve")
            try:
                lane_lease = await acquire_in_process_model_lane(
                    owner_id=f"model-manager:{id(self)}:{name}",
                    model_path=model_path,
                    purpose=purpose,
                    request_gb=opts.get("declared_gb"),
                    priority=int(opts.get("priority", 50)),
                    preemptible=False,
                    metadata={"manager": "core.utils.model_manager", "model_name": name},
                )
            except asyncio.CancelledError:
                raise
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "model_manager",
                    exc,
                    action="refused model load because lane admission failed",
                )
                raise ModelLoadError(f"Failed to admit model {name}") from exc
            try:
                maybe_coro = await run_owned_model_thread_call(
                    lambda: self._load_fn(name, opts),
                    operation_name=f"model-manager-load-{name}",
                )
                if inspect.isawaitable(maybe_coro):
                    model_obj = await maybe_coro
                else:
                    model_obj = maybe_coro
            except asyncio.CancelledError:
                await lane_lease.release(reason="model_manager_load_cancelled")
                raise
            except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                await lane_lease.release(reason="model_manager_load_failed")
                record_degradation('model_manager', e)
                logger.error("ModelManager: failed to load %s: %s", name, e)
                raise ModelLoadError(f"Failed to load model {name}") from e

            try:
                async with self._lock:
                    self._models[name] = model_obj
                    self._meta[name] = {
                        "loaded_at": time.time(),
                        "opts": opts,
                        "lane_lease": lane_lease,
                    }
                    self._last_used[name] = time.time()
                    logger.info("ModelManager: loaded model %s", name)
                    return model_obj
            except asyncio.CancelledError:
                await self._cleanup_model(model_obj, name)
                await lane_lease.release(reason="model_manager_publish_cancelled")
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                await self._cleanup_model(model_obj, name)
                await lane_lease.release(reason="model_manager_publish_failed")
                raise ModelLoadError(f"Failed to publish model {name}") from exc

    async def unload_model(self, name: str) -> bool:
        async with self._lock:
            obj, meta = self._pop_model_locked(name)
        
        if obj is None:
            logger.debug("ModelManager: unload requested for unknown model %s", name)
            return False
            
        await self._cleanup_model(obj, name)
        await self._release_lane_lease(meta, reason=f"model_manager_unload:{name}")
        return True

    async def evict_if_needed(self) -> None:
        """Evict LRU while memory pressure or over capacity."""
        while self._models:
            cleanup_obj = None
            cleanup_meta = None
            evicted_name = None
            
            async with self._lock:
                vm = psutil.virtual_memory()
                if len(self._models) > 0 and (vm.percent > 80.0 or len(self._models) > self._max_models):
                    evicted_name = next(iter(self._models.keys()))
                    logger.warning("ModelManager: evicting %s due to memory/capacity", evicted_name)
                    cleanup_obj, cleanup_meta = self._pop_model_locked(evicted_name)
                else:
                    break
            
            if cleanup_obj is not None and evicted_name is not None:
                await self._cleanup_model(cleanup_obj, evicted_name)
                await self._release_lane_lease(
                    cleanup_meta,
                    reason=f"model_manager_pressure_evict:{evicted_name}",
                )

    def list_loaded(self) -> list[str]:
        return list(self._models.keys())

    async def unload_all(self) -> None:
        async with self._lock:
            names = list(self._models.keys())
        for n in names:
            await self.unload_model(n)
