"""Canonical task ownership helpers for Aura.

Use this instead of raw asyncio.create_task in production code.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Runtime.TaskOwnership")


def _raw_asyncio_create_task(awaitable: Awaitable[Any], *, name: str | None, context: Any = None) -> asyncio.Task:
    create_task = getattr(asyncio.create_task, "__wrapped__", asyncio.create_task)
    try:
        if context is not None:
            return create_task(awaitable, name=name, context=context)
    except TypeError:
        pass
    if name is not None:
        return create_task(awaitable, name=name)
    return create_task(awaitable)


def _get_tracker() -> Any:
    try:
        from core.utils.task_tracker import get_task_tracker
        return get_task_tracker()
    except (ImportError, AttributeError, RuntimeError):
        return None


def close_awaitable(awaitable: Any) -> None:
    if inspect.iscoroutine(awaitable):
        awaitable.close()
        return
    cancel = getattr(awaitable, "cancel", None)
    if callable(cancel):
        with suppress(Exception):
            cancel()


def _create_owned_asyncio_task(awaitable: Awaitable[Any], *, name: str | None) -> asyncio.Task:
    """Create a fallback task while preserving strict task-owner semantics."""

    try:
        from core.utils.task_tracker import _SKIP_FACTORY_TRACK
    except (ImportError, AttributeError, RuntimeError):
        return _raw_asyncio_create_task(awaitable, name=name)

    child_context = contextvars.copy_context()
    child_context.run(_SKIP_FACTORY_TRACK.set, False)
    token = _SKIP_FACTORY_TRACK.set(True)
    try:
        try:
            return _raw_asyncio_create_task(awaitable, name=name, context=child_context)
        except TypeError:
            return _raw_asyncio_create_task(awaitable, name=name)
    finally:
        _SKIP_FACTORY_TRACK.reset(token)


def create_owned_asyncio_task(awaitable: Awaitable[Any], *, name: str | None = None) -> asyncio.Task:
    """Create a raw asyncio task inside the canonical task-ownership sink."""
    return _create_owned_asyncio_task(awaitable, name=name)


def create_tracked_task(
    awaitable: Awaitable[Any],
    *,
    name: str | None = None,
    owner: str | None = None,
    bounded: bool = False,
    on_done: Callable[[asyncio.Task], Any] | None = None,
    cancel_on_fail: bool = True,
) -> asyncio.Task:
    tracker = _get_tracker()
    task: asyncio.Task | None = None
    task_kwargs: dict[str, Any] = {"name": name}
    if owner is not None:
        task_kwargs["owner"] = owner
    try:
        if tracker is not None:
            if bounded and hasattr(tracker, "bounded_track"):
                task = tracker.bounded_track(awaitable, **task_kwargs)
            elif hasattr(tracker, "create_task"):
                task = tracker.create_task(awaitable, **task_kwargs)
            elif hasattr(tracker, "track_task"):
                task = tracker.track_task(awaitable, **task_kwargs)
            elif hasattr(tracker, "track"):
                task = tracker.track(awaitable, **task_kwargs)

        if task is None:
            task = _create_owned_asyncio_task(awaitable, name=name)
            if tracker is not None:
                try:
                    if hasattr(tracker, "observe"):
                        observe_kwargs: dict[str, Any] = {
                            "name": name,
                            "source": "task_ownership_fallback",
                        }
                        if owner is not None:
                            observe_kwargs["owner"] = owner
                        tracker.observe(task, **observe_kwargs)
                    elif hasattr(tracker, "track_task"):
                        tracker.track_task(task, name=name)
                except (RuntimeError, AttributeError, TypeError) as exc:
                    record_degradation('task_ownership', exc)
                    logger.debug("Failed to observe fallback task %s: %s", name or task, exc)

        if on_done is not None:
            task.add_done_callback(on_done)
        return task
    except (RuntimeError, AttributeError, TypeError):
        if cancel_on_fail:
            close_awaitable(awaitable)
        raise


def fire_and_forget(
    awaitable: Awaitable[Any],
    *,
    name: str | None = None,
    owner: str | None = None,
    bounded: bool = False,
    log_exceptions: bool = True,
) -> asyncio.Task | None:
    def _log_done(task: asyncio.Task) -> None:
        if not log_exceptions:
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return
        if exc is not None:
            logger.warning("Background task %s failed: %s", name or task.get_name(), exc, exc_info=exc)

    try:
        return create_tracked_task(
            awaitable,
            name=name,
            owner=owner,
            bounded=bounded,
            on_done=_log_done,
        )
    except RuntimeError:
        close_awaitable(awaitable)
        return None
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation('task_ownership', exc)
        close_awaitable(awaitable)
        logger.debug("fire_and_forget scheduling failed for %s: %s", name, exc)
        return None
