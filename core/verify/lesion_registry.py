"""core/verify/lesion_registry.py

The system-wide protocol for "turn this faculty off and see".

:mod:`core.verify.causal_influence` can only reach a verdict about a channel if
somebody hands it paired trials, and the pairs only exist if the faculty can be
neutralized on demand. One engine in this codebase already worked that way —
``AffectiveValenceEngine.lesion()`` returns a flat neutral affect so its
contribution is fully ablatable. Everything else claiming causal influence
claimed it without offering any way to check.

This is that capability generalized. A faculty registers how to make itself
neutral; the registry hands the harness a uniform way to do it, so measurement
does not need to know anything about the subsystem it is measuring.

Neutral is not broken. A lesioned faculty must return its no-information
output — a flat affect, an identity transform, an empty context block, alpha
zero — and must not raise, hang, or degrade the run into an error path. A
lesion that crashes the turn measures the crash, not the faculty.

Layering: registration is inverted. Faculties import this; this imports no
faculty. See DEPS.
"""

from __future__ import annotations

import logging
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

logger = logging.getLogger("Verify.LesionRegistry")

__all__ = [
    "LesionHandle",
    "LesionRegistry",
    "get_lesion_registry",
    "register_lesion",
    "lesionable",
    "lesioned",
    "reset_lesion_registry_for_test",
]


@dataclass(frozen=True)
class LesionHandle:
    """How to neutralize one channel, and who is answerable for it."""

    channel: str
    lesion: Callable[[], None]
    restore: Callable[[], None]
    owner: str
    #: What this channel's output becomes when lesioned. Recorded because a
    #: reader of a verdict needs to know what the counterfactual actually was:
    #: "steering alpha forced to 0" and "the whole block omitted from the
    #: prompt" are different questions with different answers.
    neutral_description: str
    #: True when the channel reaches the output as numbers (sampler settings,
    #: steering coefficients, gains) rather than as text in a prompt. The
    #: distinction is the point of the audit: a faculty whose only actuator is
    #: a sentence in the system prompt is doing prompt engineering, however
    #: much machinery sits behind the sentence.
    direct_actuation: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "owner": self.owner,
            "neutral": self.neutral_description,
            "direct_actuation": self.direct_actuation,
        }


class LesionUnavailable(RuntimeError):
    """Raised when a channel is asked to lesion and has not registered how."""


class LesionRegistry:
    """Every faculty that can be turned off, and how."""

    def __init__(self) -> None:
        self._handles: dict[str, LesionHandle] = {}
        self._active: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, handle: LesionHandle, *, replace: bool = False) -> None:
        with self._lock:
            existing = self._handles.get(handle.channel)
            if existing is not None and not replace:
                if existing.owner == handle.owner:
                    # Re-import or a second instance of the same owner. Idempotent.
                    return
                raise ValueError(
                    f"lesion channel {handle.channel!r} already registered by "
                    f"{existing.owner!r}; pass replace=True to take it over"
                )
            self._handles[handle.channel] = handle

    def unregister(self, channel: str) -> None:
        with self._lock:
            self._handles.pop(channel, None)
            self._active.pop(channel, None)

    def get(self, channel: str) -> LesionHandle | None:
        with self._lock:
            return self._handles.get(channel)

    def channels(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handles))

    def is_registered(self, channel: str) -> bool:
        with self._lock:
            return channel in self._handles

    def is_lesioned(self, channel: str) -> bool:
        with self._lock:
            return self._active.get(channel, 0) > 0

    def active_lesions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(name for name, depth in self._active.items() if depth > 0))

    @contextmanager
    def lesion(self, channel: str) -> Iterator[LesionHandle]:
        """Neutralize ``channel`` for the duration of the block.

        Reentrant by depth: nested lesions of the same channel restore once, on
        the outermost exit. Restoration runs even when the body raises — a
        measurement harness that leaves a faculty lesioned after a failed trial
        silently lobotomizes the live runtime.
        """

        with self._lock:
            handle = self._handles.get(channel)
            if handle is None:
                raise LesionUnavailable(
                    f"channel {channel!r} has no registered lesion; it cannot be "
                    "measured, and nothing may claim it is causally influential"
                )
            depth = self._active.get(channel, 0)
            self._active[channel] = depth + 1
            should_lesion = depth == 0

        try:
            if should_lesion:
                handle.lesion()
            yield handle
        finally:
            with self._lock:
                remaining = self._active.get(channel, 1) - 1
                if remaining <= 0:
                    self._active.pop(channel, None)
                else:
                    self._active[channel] = remaining
                should_restore = remaining <= 0
            if should_restore:
                try:
                    handle.restore()
                except Exception:
                    # A restore that fails leaves the runtime lesioned, which is
                    # far worse than a failed measurement. Say so at CRITICAL and
                    # let the caller's degradation path see the raise.
                    logger.critical(
                        "🚨 [LESION] restore FAILED for channel %s (owner %s): the "
                        "faculty is still neutralized",
                        channel,
                        handle.owner,
                    )
                    raise

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            handles = dict(self._handles)
            active = tuple(sorted(n for n, d in self._active.items() if d > 0))
        return {
            "registered": {name: h.as_dict() for name, h in sorted(handles.items())},
            "registered_count": len(handles),
            "direct_actuation_count": sum(1 for h in handles.values() if h.direct_actuation),
            "active_lesions": active,
        }


_REGISTRY = LesionRegistry()


def get_lesion_registry() -> LesionRegistry:
    return _REGISTRY


def reset_lesion_registry_for_test() -> None:
    global _REGISTRY
    _REGISTRY = LesionRegistry()


def register_lesion(
    channel: str,
    *,
    lesion: Callable[[], None],
    restore: Callable[[], None],
    owner: str,
    neutral: str,
    direct_actuation: bool,
    replace: bool = False,
) -> LesionHandle:
    """Register how to neutralize ``channel``. The functional entry point."""

    handle = LesionHandle(
        channel=str(channel),
        lesion=lesion,
        restore=restore,
        owner=str(owner),
        neutral_description=str(neutral),
        direct_actuation=bool(direct_actuation),
    )
    _REGISTRY.register(handle, replace=replace)
    return handle


def lesionable(
    channel: str,
    *,
    owner: str,
    neutral: str,
    direct_actuation: bool,
    lesion_method: str = "lesion",
    restore_method: str = "restore",
) -> Callable[[type], type]:
    """Class decorator: register a singleton-ish faculty as lesionable.

    The decorated class must expose ``lesion()`` and ``restore()``. The registry
    binds to the most recently constructed instance, which is what a container
    singleton gives you::

        @lesionable(
            "affect.generation_controls",
            owner="core/being/affective_valence.py",
            neutral="flat affect: neutral valence, no control_effects",
            direct_actuation=True,
        )
        class AffectiveValenceEngine: ...
    """

    def decorate(cls: type) -> type:
        holder: dict[str, Any] = {"instance": None}
        original_init = cls.__init__

        def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            holder["instance"] = self

        cls.__init__ = tracking_init  # type: ignore[method-assign]

        def do(method_name: str) -> Callable[[], None]:
            def run() -> None:
                instance = holder.get("instance")
                if instance is None:
                    raise LesionUnavailable(
                        f"channel {channel!r} is registered against {cls.__name__} "
                        "but no instance has been constructed"
                    )
                getattr(instance, method_name)()

            return run

        register_lesion(
            channel,
            lesion=do(lesion_method),
            restore=do(restore_method),
            owner=owner,
            neutral=neutral,
            direct_actuation=direct_actuation,
            replace=True,
        )
        return cls

    return decorate


@contextmanager
def lesioned(*channels: str) -> Iterator[tuple[LesionHandle, ...]]:
    """Neutralize several channels at once, restoring all of them on exit."""

    registry = get_lesion_registry()
    with ExitStack() as stack:
        handles = tuple(stack.enter_context(registry.lesion(c)) for c in channels)
        yield handles
