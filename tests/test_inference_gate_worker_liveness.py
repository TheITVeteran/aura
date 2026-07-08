"""Regression contract for the cascade-cleanup worker-liveness check.

Seen live (July 8 boot): the foreground-exhaustion cleanup path polled the
cortex worker handle without a None guard — when the worker was never spawned
or already reaped, `None.poll()` raised AttributeError mid-recovery and
surfaced as a CRITICAL inference_gate degradation in the unified runtime
pressure contract. The check is now a total function over every handle shape
the gate encounters.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import _worker_process_is_running

pytestmark = pytest.mark.unit


class MPStyleProcess:
    """multiprocessing.Process shape: has is_alive()."""

    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class PopenStyleProcess:
    """subprocess.Popen shape: has poll() → None while running."""

    def __init__(self, returncode):
        self._returncode = returncode

    def poll(self):
        return self._returncode


class RaisingProcess:
    """A handle whose liveness probe itself fails (mid-reap races)."""

    def is_alive(self) -> bool:
        raise OSError("process table race")


def test_none_handle_is_not_running():
    assert _worker_process_is_running(None) is False


def test_mp_process_alive_states():
    assert _worker_process_is_running(MPStyleProcess(alive=True)) is True
    assert _worker_process_is_running(MPStyleProcess(alive=False)) is False


def test_popen_running_and_exited():
    assert _worker_process_is_running(PopenStyleProcess(returncode=None)) is True
    assert _worker_process_is_running(PopenStyleProcess(returncode=0)) is False


def test_probe_failure_reads_as_not_running():
    assert _worker_process_is_running(RaisingProcess()) is False


def test_unknown_handle_shape_reads_as_not_running():
    assert _worker_process_is_running(object()) is False
