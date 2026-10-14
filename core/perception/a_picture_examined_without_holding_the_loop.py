"""A visual guess uses one owned worker without holding the control loop.

Offline 2026-10-13, checks for unrelated play structures kept the loop from
seeing a moving body for two seconds. Pure picture checks can wait elsewhere.
A timeout leaves their actual worker owned, and its late answer is discarded.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import math
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.runtime.executors import submit_blocking_io
from core.runtime.lockdep import checked_thread_semaphore
from core.verify import invariant

__all__ = ["UNMEASURED", "PictureProbe"]


class _Unavailable(Enum):
    UNMEASURED = "the picture has not been examined"


# A detector's measured None is a miss; unavailable work has no such answer.
UNMEASURED = _Unavailable.UNMEASURED
_SLOTS = checked_thread_semaphore("perception.optional_picture_probe_operations", 2)


@dataclass
class _Operation:
    raw: Future
    wrapped: asyncio.Future
    generation: int
    waiting: bool = True
    discard: bool = False


def _examined(detector: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Keep the detector's inputs and answer apart from its caller's objects."""
    return copy.deepcopy(detector(*args, **kwargs))


def _detached_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple:
    return copy.deepcopy((args, kwargs))


def _completed(raw: Future, slots: Any) -> None:
    """The actual worker returns its permit, including a cancelled queued call."""
    try:
        if not raw.cancelled():
            raw.exception()
    finally:
        slots.release()


def _received(wrapped: asyncio.Future) -> None:
    """Receive an eventual error after the awaiting owner has left."""
    if not wrapped.cancelled():
        wrapped.exception()


class PictureProbe:
    """One pure picture check at a time, with a finite caller wait.

    The detector gets detached arguments and must use only those arguments.
    Browser capture and physical inputs stay with the calling event loop.
    """

    def __init__(self, *, wait_s: float = .5) -> None:
        if isinstance(wait_s, bool) or not isinstance(wait_s, (float, int)) or not math.isfinite(wait_s) or wait_s <= 0:
            raise ValueError("a picture probe wait must be finite and positive")
        self._wait_s = float(wait_s)
        self._operation: _Operation | None = None
        self._generation = 0
        self._closed = False
        self._status = "not started"

    @property
    def status(self) -> str:
        return self._status

    @property
    def busy(self) -> bool:
        return self._operation is not None and not self._operation.raw.done()

    def _submit(self, detector: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Operation | None:
        slots = _SLOTS
        if not slots.acquire(blocking=False):
            self._status = "probe capacity unavailable"
            return None
        try:
            owned_args, owned_kwargs = _detached_inputs(args, kwargs)
            context = contextvars.copy_context()
            raw = submit_blocking_io(context.run, _examined, detector, owned_args, owned_kwargs,
                                     label="optional_picture_probe")
        except BaseException:
            slots.release()
            raise
        raw.add_done_callback(lambda done: _completed(done, slots))
        wrapped = asyncio.wrap_future(raw)
        wrapped.add_done_callback(_received)
        operation = _Operation(raw, wrapped, self._generation)
        self._operation = operation
        self._status = "probe pending"
        return operation

    async def examine(self, detector: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Return a measured answer, or UNMEASURED when no answer is available."""
        if self._closed:
            return UNMEASURED
        previous = self._operation
        if previous is not None:
            if previous.waiting or not previous.raw.done():
                self._status = "probe pending"
                return UNMEASURED
            self._operation = None
        operation = self._submit(detector, args, kwargs)
        if operation is None:
            return UNMEASURED
        try:
            await asyncio.wait_for(asyncio.shield(operation.wrapped), self._wait_s)
        except TimeoutError:
            operation.discard = True
            if not self._closed:
                self._status = "probe unmeasured"
            return UNMEASURED
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if self._closed and task is not None and not task.cancelling():
                return UNMEASURED
            self.retire()
            raise
        except BaseException:
            self.retire()
            raise
        finally:
            operation.waiting = False
        if (self._closed or self._operation is not operation
                or operation.generation != self._generation or operation.discard):
            return UNMEASURED
        self._operation = None
        self._status = "examined"
        return operation.raw.result()

    def retire(self) -> None:
        """Close publication while a running worker keeps its permit and input."""
        self._closed = True
        self._generation += 1
        if self._operation is not None:
            self._operation.discard = True
            self._operation.raw.cancel()
        self._status = "closed"

    async def close(self) -> None:
        """Leave without waiting for an uncancellable worker to finish."""
        self.retire()


@invariant('perception.picture_probes_own_detached_inputs', scope='perception',
           owner='core/perception/a_picture_examined_without_holding_the_loop.py', observational=False)
def _picture_probe_inputs_invariant() -> tuple:
    words, rules = ['first'], {'rule': ['original']}
    args, kwargs = _detached_inputs((words,), {'rules': rules})
    words.append('later')
    rules['rule'].append('changed')
    assert args == (['first'],) and kwargs == {'rules': {'rule': ['original']}}
    answer = _examined(lambda *_args, **_kwargs: args[0], args, kwargs)
    args[0].append('worker change')
    assert answer == ['first']
    return ()
