"""tests/test_stall_watchdog_hard_exit.py
=========================================
The StallWatchdog must force-exit a genuinely wedged event loop so the
launchd supervisor can restart the runtime.

Live forensics (2026-06-13) captured a Metal GPU command-buffer hang during a
steered 32B generation: every thread parked on a Metal semaphore, the event
loop died, in-process "active recovery" could not run (it schedules onto the
dead loop), and even SIGTERM could not shut down. The process hung *alive*, so
the supervisor's KeepAlive (restart-on-exit) never fired. The only code still
running was the out-of-band watchdog thread — which previously never escalated
to a process exit, and whose foreground-stall suppression reset the stall clock
every second while a (wedged) generation was "in flight", masking the wedge
forever.

These tests pin the recovery contract: an absolute loop-liveness ceiling,
immune to suppression, that hard-exits with a non-zero (restart-worthy) code.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from core.resilience import stall_watchdog
from core.resilience.stall_watchdog import (
    _LOOP_WEDGE_EXIT_CODE,
    _LOOP_WEDGE_HARD_EXIT_S,
    StallWatchdog,
)


def _make_watchdog() -> StallWatchdog:
    # The hard-exit decision/action never touch the loop object.
    return StallWatchdog(loop=SimpleNamespace(), threshold=5.0)


def test_force_exit_fires_past_ceiling(monkeypatch):
    monkeypatch.delenv("AURA_WATCHDOG_HARD_EXIT_S", raising=False)
    dog = _make_watchdog()
    assert dog._should_force_exit(_LOOP_WEDGE_HARD_EXIT_S + 1.0) is True


def test_no_force_exit_within_ceiling(monkeypatch):
    monkeypatch.delenv("AURA_WATCHDOG_HARD_EXIT_S", raising=False)
    dog = _make_watchdog()
    assert dog._should_force_exit(_LOOP_WEDGE_HARD_EXIT_S - 10.0) is False


def test_ceiling_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AURA_WATCHDOG_HARD_EXIT_S", "0")
    dog = _make_watchdog()
    assert dog._should_force_exit(10_000.0) is False


def test_no_force_exit_during_shutdown(monkeypatch):
    monkeypatch.delenv("AURA_WATCHDOG_HARD_EXIT_S", raising=False)
    dog = _make_watchdog()

    import core.runtime.shutdown_coordinator as sc

    monkeypatch.setattr(sc, "is_shutdown_requested", lambda: True)
    assert dog._should_force_exit(_LOOP_WEDGE_HARD_EXIT_S + 100.0) is False


def test_heartbeat_advances_loop_run_timestamp(monkeypatch):
    dog = _make_watchdog()
    dog._last_loop_run = 0.0
    # _heartbeat touches asyncio.all_tasks(self.loop); stub it out.
    monkeypatch.setattr(stall_watchdog.asyncio, "all_tasks", lambda _loop: set())
    dog._heartbeat()
    assert dog._last_loop_run > 0.0


def test_suppression_reset_does_not_mask_wedge():
    """The actual bug: foreground suppression reset _last_heartbeat each second
    while a wedged generation was 'in flight', so the stall clock never grew.
    The hard-exit ceiling must key off _last_loop_run, which suppression never
    touches — so a wedged loop is still caught.
    """
    dog = _make_watchdog()
    # Simulate many seconds of the run-loop suppressing stalls: it resets
    # _last_heartbeat but must NOT touch _last_loop_run.
    import time as _time

    now = _time.time()
    dog._last_heartbeat = now  # suppression keeps this fresh
    dog._last_loop_run = now - (_LOOP_WEDGE_HARD_EXIT_S + 5.0)  # loop truly dead
    loop_silence = _time.time() - dog._last_loop_run
    assert loop_silence >= _LOOP_WEDGE_HARD_EXIT_S
    assert dog._should_force_exit(loop_silence) is True


def test_force_exit_calls_os_exit_with_nonzero_code(monkeypatch):
    monkeypatch.delenv("AURA_WATCHDOG_HARD_EXIT_CODE", raising=False)
    dog = _make_watchdog()

    # Skip the real thread-stack dump; we are pinning the exit contract here.
    monkeypatch.setattr(
        stall_watchdog.StallWatchdog,
        "_force_exit_for_restart",
        lambda self, silence: os._exit(self._hard_exit_code()),
        raising=True,
    )

    captured = {}

    def _fake_exit(code):
        captured["code"] = code
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _fake_exit)

    with pytest.raises(SystemExit):
        dog._force_exit_for_restart(_LOOP_WEDGE_HARD_EXIT_S + 1.0)

    assert captured["code"] == _LOOP_WEDGE_EXIT_CODE
    assert captured["code"] != 0  # supervisor restarts only on non-zero exit


def test_exit_code_env_override(monkeypatch):
    monkeypatch.setenv("AURA_WATCHDOG_HARD_EXIT_CODE", "73")
    assert StallWatchdog._hard_exit_code() == 73


# --- End-to-end: a really-wedged loop force-exits with the restart code -------

_WEDGE_SCRIPT = """
import asyncio, os, sys, time
os.environ["AURA_WATCHDOG_HARD_EXIT_S"] = "3"
os.environ["AURA_WATCHDOG_BOOT_GRACE_S"] = "0"
from core.resilience.stall_watchdog import StallWatchdog

async def main():
    dog = StallWatchdog(asyncio.get_running_loop(), threshold=1.0)
    dog.start()
    await asyncio.sleep(1.0)        # let the watchdog establish a heartbeat
    time.sleep(60)                  # BLOCK the loop thread — simulate the wedge
    print("LOOP_NOT_WEDGED")        # must never run; watchdog exits at ~3s
    sys.stdout.flush()

asyncio.run(main())
"""


def test_wedged_loop_actually_force_exits_with_restart_code():
    """Real thread, real blocked loop, real os._exit — proves the recovery
    mechanism end-to-end: a wedged loop exits with the supervisor-restart code,
    not 0 and not a hang."""
    import pathlib
    import subprocess
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", _WEDGE_SCRIPT],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert "LOOP_NOT_WEDGED" not in proc.stdout
    assert proc.returncode == _LOOP_WEDGE_EXIT_CODE, (
        f"expected force-exit {_LOOP_WEDGE_EXIT_CODE}, got {proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr[-500:]!r}"
    )
