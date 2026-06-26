import asyncio
import contextvars
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.task_ownership import close_awaitable, create_owned_asyncio_task

logger = logging.getLogger(__name__)

_SKIP_FACTORY_TRACK = contextvars.ContextVar("aura_skip_factory_track", default=False)


def _runtime_shutdown_requested() -> bool:
    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        return bool(is_shutdown_requested())
    except (ImportError, AttributeError, RuntimeError):
        return False


def mark_task_protected(task: asyncio.Task[Any], *, owner: str = "task_tracker") -> asyncio.Task[Any]:
    """Mark a task as shutdown-critical without exempting it from cancellation."""
    try:
        task._aura_protected = True
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(owner, exc)
        logger.debug("Task protection annotation failed for %s: %s", owner, exc)
    return task


@dataclass
class TaskRecord:
    task_id: int
    name: str
    tracker: str
    supervision: str
    source: str
    created_at: float
    coroutine: str = "unknown"
    done: bool = False
    cancelled: bool = False
    failed: bool = False
    finished_at: float | None = None
    exception: str | None = None
    last_heartbeat: float = field(default_factory=time.monotonic)

    def age_s(self, now: float | None = None) -> float:
        current_time = now if now is not None else time.monotonic()
        return max(0.0, current_time - self.created_at)

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        duration = None
        if self.finished_at is not None:
            duration = max(0.0, self.finished_at - self.created_at)
        return {
            "task_id": self.task_id,
            "name": self.name,
            "tracker": self.tracker,
            "supervision": self.supervision,
            "source": self.source,
            "coroutine": self.coroutine,
            "age_s": self.age_s(now),
            "done": self.done,
            "cancelled": self.cancelled,
            "failed": self.failed,
            "finished_at": self.finished_at,
            "duration_s": duration,
            "exception": self.exception,
            "last_heartbeat": self.last_heartbeat,
        }


class TaskTracker:
    """Track and manage background asyncio tasks to ensure graceful shutdown.

    Prevents "Task was destroyed but it is pending!" errors and provides
    lifecycle telemetry for tasks created both through the tracker and through
    raw asyncio task creation APIs.
    """

    def __init__(self, name: str = "Global", max_concurrent: int = 20):
        self.name = name
        self.tasks: set[asyncio.Task] = set()
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None  # Lazy init
        self._high_water = 0
        self._total_tracked = 0
        self._total_observed = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._failed_total = 0
        self._shutdown_suppressed_total = 0
        self._records: dict[int, TaskRecord] = {}
        self._recently_completed: deque[dict[str, Any]] = deque(maxlen=128)
        self._installed_loop_factories: dict[int, Any] = {}
        self._max_records_in_memory = 256  # Bounded history of completed tasks

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init semaphore (must be in event loop context)."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    def track(
        self,
        coro_or_task,
        name: str | None = None,
        *,
        allow_during_shutdown: bool = False,
    ) -> asyncio.Task:
        """Track a new task or coroutine (no concurrency limit)."""
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            if _runtime_shutdown_requested() and not allow_during_shutdown:
                return self._suppress_shutdown_start(coro_or_task, name=name, source="track")
            try:
                task = create_owned_asyncio_task(coro_or_task, name=name)
            except RuntimeError:
                close_awaitable(coro_or_task)
                raise
        self._total_tracked += 1
        self._attach(task, name=name, supervision="explicit", source="track")
        return task

    # Alias for compatibility with components calling track_task or create_task
    track_task = track
    create_task = track

    def observe(self, task: asyncio.Task, name: str | None = None, source: str = "loop_factory") -> asyncio.Task:
        """Observe a task created outside the tracker so it still gets cleaned up and audited."""
        self._attach(task, name=name, supervision="implicit", source=source)
        return task

    def bounded_track(
        self,
        coro,
        name: str | None = None,
        *,
        allow_during_shutdown: bool = False,
    ) -> asyncio.Task:
        """Track a task WITH concurrency limiting via semaphore.

        Use this for short-lived tasks (maintenance, learning, reflection).
        Long-running loops should use track() directly.
        """
        if _runtime_shutdown_requested() and not allow_during_shutdown:
            return self._suppress_shutdown_start(coro, name=name, source="bounded_track")

        async def _bounded():
            sem = self._get_semaphore()
            async with sem:
                if asyncio.iscoroutine(coro):
                    return await coro
                if inspect.iscoroutinefunction(coro):
                    return await coro()
                return await coro

        bounded_coro = _bounded()
        try:
            task = create_owned_asyncio_task(bounded_coro, name=name)
        except RuntimeError:
            close_awaitable(coro)
            close_awaitable(bounded_coro)
            raise
        self._total_tracked += 1
        self._attach(task, name=name, supervision="explicit", source="bounded_track")
        return task

    def _suppress_shutdown_start(self, awaitable: Any, *, name: str | None, source: str) -> asyncio.Task:
        """Close late runtime work after shutdown starts and return a completed owned task.

        Shutdown is not a valid time for ordinary subsystems to spawn new
        inference, recovery, telemetry, or repair work. Returning a tiny
        completed task preserves call-site compatibility while preventing the
        original coroutine from running after executors and event loops begin
        teardown.
        """
        close_awaitable(awaitable)
        self._shutdown_suppressed_total += 1

        async def _shutdown_suppressed() -> None:
            return None

        suppressed_coro = _shutdown_suppressed()
        try:
            task = create_owned_asyncio_task(
                suppressed_coro,
                name=name or f"{self.name}.shutdown_suppressed",
            )
        except RuntimeError:
            close_awaitable(suppressed_coro)
            raise
        try:
            task._aura_shutdown_suppressed = True
            task._aura_shutdown_suppressed_source = source
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("task_tracker", exc)
            logger.debug("TaskTracker[%s]: failed to annotate suppressed task: %s", self.name, exc)
        self._total_tracked += 1
        self._attach(task, name=name, supervision="explicit", source=f"{source}:shutdown_suppressed")
        logger.debug(
            "TaskTracker[%s]: suppressed late task start during runtime shutdown (name=%s source=%s).",
            self.name,
            name or "",
            source,
        )
        return task

    def install_loop_hygiene(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install a task factory so raw asyncio.create_task/loop.create_task calls are still observed."""
        target_loop = loop or asyncio.get_running_loop()
        loop_id = id(target_loop)
        if loop_id in self._installed_loop_factories:
            return

        previous_factory = target_loop.get_task_factory()
        tracker = self

        def _factory(factory_loop, coro, **kwargs):
            if previous_factory is not None:
                try:
                    task = previous_factory(factory_loop, coro, **kwargs)
                except TypeError:
                    kwargs.pop("context", None)
                    try:
                        task = previous_factory(factory_loop, coro, **kwargs)
                    except TypeError:
                        kwargs.pop("name", None)
                        task = previous_factory(factory_loop, coro, **kwargs)
            else:
                task = asyncio.Task(coro, loop=factory_loop, **kwargs)
            if not _SKIP_FACTORY_TRACK.get():
                try:
                    tracker.observe(task, source="loop_factory")
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to observe loop task: %s", tracker.name, exc)
            return task

        target_loop.set_task_factory(_factory)
        self._installed_loop_factories[loop_id] = (target_loop, previous_factory)

    def restore_loop_hygiene(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Restore a loop's original task factory."""
        if loop is not None:
            info = self._installed_loop_factories.pop(id(loop), None)
            if info is not None:
                target_loop, previous_factory = info
                try:
                    target_loop.set_task_factory(previous_factory)
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('task_tracker', exc)
                    logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            return

        for loop_id, info in list(self._installed_loop_factories.items()):
            target_loop, previous_factory = info
            try:
                target_loop.set_task_factory(previous_factory)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('task_tracker', exc)
                logger.debug("TaskTracker[%s]: failed to restore loop factory: %s", self.name, exc)
            finally:
                self._installed_loop_factories.pop(loop_id, None)

    def get_stale_tasks(self, min_age_s: float = 900.0, *, include_supervised: bool = False) -> list[dict[str, Any]]:
        """Return a sample of long-lived tasks that may need inspection."""
        now = time.monotonic()
        stale: list[dict[str, Any]] = []
        for task in list(self.tasks):
            if task.done():
                continue
            record = self._records.get(id(task))
            if record is None:
                continue
            if record.age_s(now) < min_age_s:
                continue
            if not include_supervised and record.supervision == "explicit":
                continue
            stale.append(record.to_dict(now))
        stale.sort(key=lambda item: item["age_s"], reverse=True)
        return stale

    def heartbeat(self, task: asyncio.Task | None = None) -> None:
        """Register a heartbeat for the given task, or the current task if None."""
        target_task = task or asyncio.current_task()
        if not target_task:
            return
            
        record = self._records.get(id(target_task))
        if record:
            record.last_heartbeat = time.monotonic()

    def _mark_supervised(self, task: asyncio.Task) -> None:
        try:
            task._aura_supervised = True
            task._aura_task_tracker = self.name
            task._aura_task_supervision = "explicit"
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('task_tracker', e)
            logger.debug("TaskTracker[%s]: failed to mark task supervised: %s", self.name, e)

    def _attach(
        self,
        task: asyncio.Task,
        *,
        name: str | None,
        supervision: str,
        source: str,
    ) -> None:
        # The tracked set is typed asyncio.Task and shutdown relies on
        # .cancel()/awaitability. A non-Task slipping in (observed: a test
        # double returned by a monkeypatched asyncio.create_task, with
        # done() pinned False) poisons the global tracker permanently --
        # every later shutdown raised AttributeError and erred ten
        # unrelated teardowns in the chunked suite. Refuse loudly.
        if not isinstance(task, asyncio.Task):
            record_degradation(
                "task_tracker",
                TypeError(f"refused non-Task attach: {type(task).__name__}"),
                severity="warning",
                action="ignored non-Task object handed to tracker",
                extra={"source": source, "name": str(name or "")},
            )
            return
        task_name = name or task.get_name()
        task_id = id(task)
        record = self._records.get(task_id)
        if record is None:
            record = TaskRecord(
                task_id=task_id,
                name=task_name,
                tracker=self.name,
                supervision=supervision,
                source=source,
                created_at=time.monotonic(),
                coroutine=self._describe_task(task),
            )
            self._records[task_id] = record
            self.tasks.add(task)
            task.add_done_callback(self._on_task_done)
            self._total_observed += 1
        else:
            if name:
                record.name = task_name
            if record.source == "loop_factory" and source != "loop_factory":
                record.source = source
            if supervision == "explicit":
                record.supervision = "explicit"

        try:
            task._aura_task_tracker = self.name
            task._aura_task_supervision = record.supervision
            task._aura_task_source = record.source
            task._aura_task_created_at = record.created_at
            if record.supervision == "explicit":
                self._mark_supervised(task)
            elif not hasattr(task, "_aura_supervised"):
                task._aura_supervised = False
        except (RuntimeError, AttributeError, TypeError) as exc:
            record_degradation('task_tracker', exc)
            logger.debug("TaskTracker[%s]: failed to annotate task: %s", self.name, exc)

        if task.done():
            self._on_task_done(task)
        else:
            self._high_water = max(self._high_water, len(self.tasks))

    def _describe_task(self, task: asyncio.Task) -> str:
        try:
            coro = task.get_coro()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return "unknown"
        qualname = getattr(coro, "__qualname__", None)
        if qualname:
            return qualname
        return repr(coro)

    def _on_task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        record = self._records.get(id(task))
        if record is None or record.done:
            return

        record.done = True
        record.finished_at = time.monotonic()
        self._completed_total += 1

        if task.cancelled():
            record.cancelled = True
            self._cancelled_total += 1
        else:
            try:
                exc = task.exception()
            except asyncio.CancelledError:
                record.cancelled = True
                self._cancelled_total += 1
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('task_tracker', exc)
                record.failed = True
                record.exception = f"{type(exc).__name__}: {exc}"
                self._failed_total += 1
            else:
                if exc is not None:
                    record.failed = True
                    record.exception = f"{type(exc).__name__}: {exc}"
                    self._failed_total += 1
                    logger.warning(
                        "TaskTracker[%s]: task %s failed: %s",
                        self.name,
                        record.name,
                        record.exception,
                    )

        self._recently_completed.append(record.to_dict())
        
        # CRITICAL FIX: Clean up old records to prevent unbounded memory growth
        # This was causing 114GB memory leak - keeping ALL completed task records forever
        if len(self._records) > self._max_records_in_memory:
            # Find and remove oldest completed records
            completed_records = [
                (task_id, rec) for task_id, rec in self._records.items() 
                if rec.done and rec.finished_at is not None
            ]
            if completed_records:
                # Sort by finish time, remove oldest 25% of completed records
                completed_records.sort(key=lambda x: x[1].finished_at or 0)
                remove_count = max(1, len(completed_records) // 4)
                for task_id, _ in completed_records[:remove_count]:
                    del self._records[task_id]

    @property
    def active_count(self) -> int:
        """Number of currently active (not done) tasks."""
        return len(self.tasks)

    async def shutdown(self, timeout: float = 5.0):  # noqa: ASYNC109 - forwarded to asyncio.wait.
        """Cancel and wait for all tracked tasks.

        Tasks marked ``_aura_protected`` are cancelled after ordinary tracked
        work so shutdown can drain short-lived background jobs first. Protection
        never means "leave this task alive"; a clean runtime shutdown must not
        strand scheduler, substrate, or watchdog loops behind the caller.
        """
        pending = {
            task
            for task in self.tasks
            if not task.done() and task is not asyncio.current_task()
        }
        if not pending:
            return

        ordinary = {task for task in pending if not getattr(task, "_aura_protected", False)}
        protected = pending - ordinary
        remaining: list[asyncio.Task[Any]] = []

        async def _cancel_group(group: set[asyncio.Task[Any]], label: str) -> None:
            if not group:
                return
            logger.info(
                "TaskTracker[%s]: cancelling %s %s task(s) during shutdown.",
                self.name,
                len(group),
                label,
            )

            for task in group:
                task.cancel()

            try:
                await asyncio.wait(group, timeout=timeout)
            except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError) as e:
                record_degradation('task_tracker', e)
                logger.error("Error during TaskTracker shutdown: %s", e)
            remaining.extend(task for task in group if not task.done())

        await _cancel_group(ordinary, "ordinary")
        await _cancel_group(protected, "protected")

        if remaining:
            logger.warning("%d tasks still pending after timeout. Forcing abandonment.", len(remaining))
        for task in remaining:
            self.tasks.discard(task)

    def cleanup_old_records(self, max_age_s: float = 300.0):
        """Explicitly clean up task records older than max_age_s.
        
        Called periodically to prevent unbounded memory growth from completed tasks.
        """
        now = time.monotonic()
        removed = 0
        for task_id in list(self._records.keys()):
            record = self._records[task_id]
            if record.done and record.finished_at is not None:
                age = now - record.finished_at
                if age > max_age_s:
                    del self._records[task_id]
                    removed += 1
        if removed > 0:
            logger.debug("TaskTracker[%s]: cleaned up %d old records", self.name, removed)
        return removed

    def get_stats(self) -> dict:
        explicit_active = 0
        implicit_active = 0
        for task in list(self.tasks):
            record = self._records.get(id(task))
            if record is None:
                continue
            if record.supervision == "explicit":
                explicit_active += 1
            else:
                implicit_active += 1
        stale_tasks = self.get_stale_tasks(min_age_s=300.0)
        return {
            "active": self.active_count,
            "high_water": self._high_water,
            "total_tracked": self._total_tracked,
            "total_observed": self._total_observed,
            "explicit_active": explicit_active,
            "implicit_active": implicit_active,
            "completed_total": self._completed_total,
            "cancelled_total": self._cancelled_total,
            "failed_total": self._failed_total,
            "shutdown_suppressed_total": self._shutdown_suppressed_total,
            "max_concurrent": self._max_concurrent,
            "stale_tasks": stale_tasks[:5],
            "recently_completed": list(self._recently_completed)[-5:],
        }


_task_tracker = TaskTracker(name="Global")


def get_task_tracker() -> TaskTracker:
    """Return the canonical process-wide task tracker.

    Both compatibility paths in this module must resolve to the same object.
    Otherwise tasks can be supervised by one tracker while shutdown, health
    checks, or imports of ``task_tracker`` query a different tracker.
    """
    return _task_tracker


# Backward compatibility for modules that import ``task_tracker`` directly.
task_tracker = _task_tracker


def fire_and_track(coro, name: str | None = None) -> asyncio.Task:
    """Convenience function to create and track a task in one go."""
    tracker = get_task_tracker()
    return tracker.track(coro, name=name)
