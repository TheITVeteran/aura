"""Having a faculty and being able to use it are different facts.

A person who knows how to search the web does not get confused when the wifi
drops. They do not conclude they have forgotten how to search. They separate
two things without effort:

    the capability      "I know how to look things up"      — still true
    its precondition    "there is a network"                — false right now

and they combine them into a third thing they can say out loud: "I can't look
that up at the moment, there's no internet."

Aura had only the first axis. The capability registry knows which skills are
registered and enabled, and nothing in it knows whether the world outside the
process is currently in a state where those skills can do anything. So a
missing network looked identical to a missing skill, and she would either
claim a capability that could not possibly work, or deny one she has.

This module is the second axis. It probes the preconditions themselves —
cheaply, with a short cache, never blocking a turn — and reports them as
facts. Composition happens in capability_condition, so the derived verdict is
computed, not asserted: capability AND preconditions => usable. Remove the
network and the conclusion changes on its own, without a prompt mentioning
networks anywhere.

Preconditions are declared per capability rather than inferred from names, so
adding a skill means declaring what the world must provide for it — which is
the honest place for that knowledge to live.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("Aura.CapabilityPreconditions")

__all__ = [
    "PreconditionState",
    "declared_preconditions",
    "failing_preconditions",
    "precondition_state",
    "reset_precondition_cache",
]

#: How long a probe result stands. Long enough that a burst of turns costs one
#: probe; short enough that unplugging the network is noticed within a turn or
#: two, which is the whole point.
_CACHE_TTL_SECONDS = 12.0

#: A probe must never be the reason a reply is slow.
_PROBE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class PreconditionState:
    name: str
    satisfied: bool
    #: The plain fact, for her to say however she says things.
    fact: str
    #: True when the probe itself could not run. Unknown is not failure: a
    #: probe that cannot answer must not be reported as a missing world.
    unknown: bool = False


def _probe_network() -> PreconditionState:
    """Is anything reachable off this machine right now?

    A TCP connect, not an HTTP fetch: it answers the actual question (is
    there a route out) without touching a service that might itself be down,
    and it fails fast.
    """
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_SECONDS):
                return PreconditionState("network", True, "there is a network connection")
        except OSError:
            continue
        except Exception as exc:  # noqa: BLE001 - a probe may never raise upward
            logger.debug("Network probe error: %s", exc)
            return PreconditionState(
                "network", True, "network state could not be determined", unknown=True
            )
    return PreconditionState("network", False, "there is no network connection right now")


def _probe_desktop_session() -> PreconditionState:
    """Is there a windowing session to drive?

    Over SSH or in a headless daemon there is no desktop to click, and that
    is a fact about the world rather than about her.
    """
    if os.environ.get("AURA_HEADLESS") == "1":
        return PreconditionState("desktop_session", False, "this runtime is headless")
    try:
        if os.uname().sysname == "Darwin":
            # A login session that owns the window server has these set; a
            # bare daemon does not.
            has_session = bool(os.environ.get("SSH_CONNECTION")) is False
            return PreconditionState(
                "desktop_session",
                has_session,
                "there is a desktop session"
                if has_session
                else "this session has no desktop attached",
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Desktop session probe error: %s", exc)
        return PreconditionState(
            "desktop_session", True, "desktop session state is unknown", unknown=True
        )
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return PreconditionState("desktop_session", True, "there is a desktop session")
    return PreconditionState(
        "desktop_session", False, "there is no desktop session attached"
    )


def _probe_writable_home() -> PreconditionState:
    try:
        home = os.path.expanduser("~")
        writable = os.access(home, os.W_OK)
        return PreconditionState(
            "writable_storage",
            writable,
            "local storage is writable" if writable else "local storage is not writable",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Storage probe error: %s", exc)
        return PreconditionState(
            "writable_storage", True, "storage state is unknown", unknown=True
        )


_PROBES: dict[str, Callable[[], PreconditionState]] = {
    "network": _probe_network,
    "desktop_session": _probe_desktop_session,
    "writable_storage": _probe_writable_home,
}

#: What the world must provide for each capability to be usable at all.
#:
#: Declared, not guessed from the name: a new skill states what it needs, and
#: the reasoning above it keeps working without being taught about that skill.
_CAPABILITY_PRECONDITIONS: dict[str, tuple[str, ...]] = {
    "web_search": ("network",),
    "sovereign_browser": ("network", "desktop_session"),
    "browser_action": ("network", "desktop_session"),
    "email_adapter": ("network",),
    "reddit_adapter": ("network",),
    "computer_use": ("desktop_session",),
    "desktop_task": ("desktop_session",),
    "build_app": ("writable_storage",),
    "file_operation": ("writable_storage",),
}

_CACHE: dict[str, tuple[float, PreconditionState]] = {}
_CACHE_LOCK = threading.Lock()


def reset_precondition_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def precondition_state(name: str) -> PreconditionState:
    """Current state of one precondition, cached briefly."""
    key = str(name or "").strip()
    probe = _PROBES.get(key)
    if probe is None:
        return PreconditionState(key, True, "no precondition declared", unknown=True)

    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    state = probe()
    with _CACHE_LOCK:
        _CACHE[key] = (time.monotonic(), state)
    return state


def declared_preconditions(capability: Any) -> tuple[str, ...]:
    return _CAPABILITY_PRECONDITIONS.get(str(capability or "").strip(), ())


def failing_preconditions(capability: Any) -> tuple[PreconditionState, ...]:
    """Preconditions this capability needs that the world is not providing.

    An UNKNOWN probe never appears here. Reporting "there's no network"
    because a socket call raised something unexpected would be the same
    confident-lie failure as reporting a missing skill because a registry
    read failed.
    """
    failures: list[PreconditionState] = []
    for name in declared_preconditions(capability):
        state = precondition_state(name)
        if not state.satisfied and not state.unknown:
            failures.append(state)
    return tuple(failures)
