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


def test_an_unregistered_heartbeat_is_recorded_not_dropped():
    """The heartbeat itself must survive, so liveness is at least observable.

    `orchestrator` — heartbeating on every tick — used to hit an "unknown
    component" branch that logged a warning and DISCARDED the heartbeat, so the
    component was invisible and the one signal saying so looked like log noise.
    """
    watchdog = SystemWatchdog(check_interval=0.05)
    try:
        watchdog.heartbeat("orchestrator")
        assert "orchestrator" in watchdog._heartbeats, (
            "the orchestrator's heartbeat was dropped, so its liveness was invisible"
        )
        # Observable, but no cadence was declared, so nothing is enforced.
        assert "orchestrator" in watchdog._unenforced
        assert "orchestrator" not in watchdog._timeouts
        assert watchdog._thread is not None and watchdog._thread.is_alive()
    finally:
        watchdog.stop()


def test_a_registered_component_is_really_monitored_and_stalls_are_detected():
    """The half that was actually missing: enforcement that runs."""
    stalls: list[str] = []
    watchdog = SystemWatchdog(check_interval=0.05)
    try:
        watchdog.register_component(
            "mind_tick", timeout=0.15, on_stall=lambda: stalls.append("mind_tick")
        )

        assert _wait_until(lambda: "mind_tick" in watchdog._stalled), (
            "a registered component that stops heartbeating must be reported stalled"
        )
        assert _wait_until(lambda: stalls == ["mind_tick"]), (
            "the stall callback must fire, not just the internal flag"
        )

        # Recovery: a fresh heartbeat clears the stall rather than latching it.
        watchdog.heartbeat("mind_tick")
        assert _wait_until(lambda: "mind_tick" not in watchdog._stalled), (
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


def test_an_event_driven_component_is_observed_but_not_falsely_stalled():
    """A heartbeat without a declared cadence must not get an invented one.

    `orchestrator` heartbeats per processed message, not on a timer. Adopting it
    at a default 60s timeout produced "SYSTEM STALL DETECTED: Component
    'orchestrator' has not responded for 60.4s!" on an idle, healthy runtime —
    and false stall reports are worse than silence, because they train everyone
    to ignore the real ones. It is observable in status; enforcement is what
    register_component() opts into.
    """
    watchdog = SystemWatchdog(check_interval=0.05)
    try:
        watchdog.heartbeat("orchestrator")
        assert "orchestrator" in watchdog._heartbeats, "liveness must stay observable"
        assert "orchestrator" in watchdog._unenforced

        # Quiet for many multiples of any plausible invented timeout.
        time.sleep(0.6)
        assert "orchestrator" not in watchdog._stalled, (
            "a component with no declared cadence must not be reported stalled"
        )

        # Declaring a cadence opts it in, and enforcement then works.
        watchdog.register_component("orchestrator", timeout=0.1)
        assert "orchestrator" not in watchdog._unenforced
        assert _wait_until(lambda: "orchestrator" in watchdog._stalled), (
            "once a cadence is declared, a missed heartbeat must be reported"
        )
    finally:
        watchdog.stop()


def test_the_critical_stall_path_does_not_call_a_method_that_never_existed():
    """The remedy was a phantom: SnapshotManager has no `rollback`.

    Every critical stall called it, raised AttributeError into the handler, and
    logged "Watchdog rollback failed" — which reads as a remedy that was tried
    and failed rather than one that was never there. It is not replaced with an
    automatic thaw(): restoring a snapshot is a governed, consequential action
    and a watchdog tick is the wrong authority for it.
    """
    import inspect

    from core.resilience.snapshot_manager import SnapshotManager

    assert not hasattr(SnapshotManager, "rollback"), (
        "if a real rollback() is ever added, revisit the watchdog remedy deliberately"
    )
    assert hasattr(SnapshotManager, "freeze") and hasattr(SnapshotManager, "thaw")

    import infrastructure.watchdog as watchdog_module

    source = inspect.getsource(watchdog_module)
    executable = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "sm.rollback(" not in executable and "SnapshotManager()" not in executable, (
        "the critical-stall path must not call a phantom recovery API"
    )
    assert "record_degradation" in executable, (
        "a critical stall must be recorded, not silently swallowed"
    )
