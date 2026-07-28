"""The system watchdog has to be running, and has to watch what heartbeats it.

Measured live on the desktop runtime: `mind_tick` registered itself, the
orchestrator heartbeated every tick, and "System Watchdog started" never
appeared in the log at all. The boot path that calls ``start()`` had not run, so
every component was registered into a dict nobody checked — and the
orchestrator, the single most important component, landed in the
"unknown component" branch that logged a warning and DROPPED the heartbeat.

A monitor that is never started is indistinguishable from a monitor that finds
nothing wrong. These tests pin both halves: it starts itself, and it adopts
whatever heartbeats it instead of leaving it unwatched.
"""

from __future__ import annotations

import time

from infrastructure.watchdog import SystemWatchdog


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_registering_a_component_starts_the_monitor():
    watchdog = SystemWatchdog(check_interval=0.05)
    assert watchdog._thread is None, "fixture must begin unstarted"
    try:
        watchdog.register_component("mind_tick", timeout=30.0)
        assert watchdog._thread is not None and watchdog._thread.is_alive(), (
            "registering is a request to be WATCHED; a dict nobody checks is not that"
        )
    finally:
        watchdog.stop()


def test_an_unregistered_heartbeat_is_adopted_not_dropped():
    watchdog = SystemWatchdog(check_interval=0.05)
    try:
        watchdog.heartbeat("orchestrator")
        assert "orchestrator" in watchdog._heartbeats, (
            "the orchestrator's heartbeat was dropped, so a wedged orchestrator "
            "could never be detected"
        )
        assert watchdog._timeouts.get("orchestrator") == watchdog._default_timeout
        assert watchdog._thread is not None and watchdog._thread.is_alive()
    finally:
        watchdog.stop()


def test_an_adopted_component_is_really_monitored_and_stalls_are_detected():
    stalls: list[str] = []
    watchdog = SystemWatchdog(check_interval=0.05, default_timeout=0.2)
    try:
        watchdog.heartbeat("orchestrator")
        watchdog._callbacks["orchestrator"] = lambda: stalls.append("orchestrator")

        assert _wait_until(lambda: "orchestrator" in watchdog._stalled), (
            "an adopted component that stops heartbeating must be reported stalled"
        )
        assert _wait_until(lambda: stalls == ["orchestrator"]), (
            "the stall callback must fire, not just the internal flag"
        )

        # Recovery: a fresh heartbeat clears the stall rather than latching it.
        watchdog.heartbeat("orchestrator")
        assert _wait_until(lambda: "orchestrator" not in watchdog._stalled), (
            "a component that resumes heartbeating must stop being reported stalled"
        )
    finally:
        watchdog.stop()


def test_start_is_idempotent_so_repeat_heartbeats_do_not_spawn_threads():
    watchdog = SystemWatchdog(check_interval=0.05)
    try:
        watchdog.heartbeat("orchestrator")
        first = watchdog._thread
        for _ in range(50):
            watchdog.heartbeat("orchestrator")
            watchdog.register_component("mind_tick", timeout=30.0)
        assert watchdog._thread is first, "the monitor thread must not be respawned"
    finally:
        watchdog.stop()
