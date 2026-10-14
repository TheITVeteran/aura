"""Optional numerical work yields instead of queuing behind another host worker."""
from __future__ import annotations

import fcntl
import math
import os
from contextlib import contextmanager
from pathlib import Path

from core.runtime.resource_observation import get_resource_observer
from core.runtime.state_ownership import (
    RuntimeProfile,
    live_state_root,
    runtime_profile,
    state_root,
)
from core.verify import invariant


class OptionalComputeBusy(RuntimeError):  # noqa: N818 - this is expected backpressure
    """The current optional operation was deferred, not measured or queued."""


def optional_compute_path() -> Path:
    root = live_state_root() if runtime_profile() == RuntimeProfile.LIVE else state_root()
    return root / 'run/optional_compute.lock'


def optional_headroom_reason(*, required_bytes: int = 2 * 1024**3, observer=None) -> str | None:
    """Unknown pressure does not establish capacity for optional work."""
    try:
        memory = (observer or get_resource_observer()).memory(include_process_tree=False)
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return 'memory headroom unobserved'
    if (not memory.available or not math.isfinite(memory.percent)
            or type(memory.available_bytes) is not int or memory.available_bytes < 0):
        return 'memory headroom unobserved'
    if memory.percent >= 85. or memory.available_bytes < required_bytes:
        return 'memory headroom reserved for interactive work'
    return None


@contextmanager
def optional_compute(*, path: Path | None = None, required_bytes: int = 2 * 1024**3, observer=None):
    """One immediate attempt; the OS releases the permit if its process dies."""
    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError('optional compute needs a nonnegative memory reserve')
    target = path or optional_compute_path()
    if not target.parent.is_dir():
        raise OptionalComputeBusy('runtime permit directory is not initialized')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(target, flags, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as why:
            raise OptionalComputeBusy('another optional operation owns the host permit') from why
        reason = optional_headroom_reason(required_bytes=required_bytes, observer=observer)
        if reason is not None:
            raise OptionalComputeBusy(reason)
        yield
    finally:
        os.close(fd)


@invariant('runtime.optional_compute_preserves_unobserved_headroom', scope='runtime',
           owner='core/runtime/optional_compute.py', observational=False)
def _optional_headroom_invariant() -> tuple:
    from types import SimpleNamespace

    class Observer:
        def memory(self, **_kwargs):
            return SimpleNamespace(available=False, percent=0., available_bytes=1024**4)

    assert optional_headroom_reason(observer=Observer()) == 'memory headroom unobserved'
    return ()
