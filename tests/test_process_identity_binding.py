"""A kill must land on the process the decision was made about.

``select_script_process_tree`` has carried the rule in its docstring since
it was written — "callers must still revalidate PID creation time
immediately before sending a signal so PID reuse cannot target an unrelated
process" — with nothing to revalidate it with. These tests cover the
binding that closes it.
"""
from __future__ import annotations

import subprocess
import types

import pytest

from core.runtime.process_identity import (
    ProcessIdentity,
    assert_owned,
    capture_identity,
    identity_still_current,
)


# Creation time is a HOST fact. conftest installs a SimulatedResourceObserver
# process-wide so ordinary tests do not depend on the developer machine, and
# that simulation legitimately knows nothing about a subprocess this file just
# spawned. Tests that read real process state opt in, exactly as conftest
# documents.
pytestmark = pytest.mark.host_observation


@pytest.fixture
def live_process():
    proc = subprocess.Popen(["/bin/sleep", "60"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_binding_captures_creation_time(live_process):
    identity = capture_identity(live_process, label="probe")
    assert identity is not None
    assert identity.pid == live_process.pid
    assert identity.bound, "creation time must be readable on this host"
    assert "probe" in identity.describe()


def test_the_same_process_stays_current(live_process):
    identity = capture_identity(live_process)
    assert identity_still_current(identity, live_process)


def test_a_dead_process_is_not_current(live_process):
    identity = capture_identity(live_process)
    live_process.kill()
    live_process.wait(timeout=5)
    assert not identity_still_current(identity, live_process)


def test_a_reused_pid_is_rejected(live_process):
    """The case a plain PID comparison cannot see.

    Same PID, different creation time — a new process in a recycled slot.
    A PID-only check says 'yes, that's mine' and kills a stranger.
    """
    identity = capture_identity(live_process)
    assert identity is not None
    reused = ProcessIdentity(
        pid=identity.pid,
        create_time=(identity.create_time or 0.0) - 3600.0,
        label="impostor",
    )
    assert not identity_still_current(reused, live_process)
    # And the PID-only check that this replaces would have passed:
    assert reused.pid == live_process.pid


def test_a_different_pid_is_rejected(live_process):
    identity = capture_identity(live_process)
    assert not identity_still_current(identity, types.SimpleNamespace(pid=identity.pid + 1))


def test_a_handle_with_no_pid_cannot_be_bound():
    assert capture_identity(types.SimpleNamespace()) is None
    assert capture_identity(None) is None
    assert capture_identity(types.SimpleNamespace(pid=0)) is None
    assert capture_identity(types.SimpleNamespace(pid="not-a-pid")) is None


def test_assert_owned_refuses_an_unbound_decision(live_process):
    assert not assert_owned(None, live_process, action="kill", subsystem="test")


def test_assert_owned_permits_the_bound_process(live_process):
    identity = capture_identity(live_process)
    assert assert_owned(identity, live_process, action="kill", subsystem="test")


def test_assert_owned_refuses_once_the_process_is_gone(live_process):
    identity = capture_identity(live_process)
    live_process.kill()
    live_process.wait(timeout=5)
    assert not assert_owned(identity, live_process, action="kill", subsystem="test")


def test_a_weak_binding_is_visible_as_weak():
    """An unreadable creation time must not masquerade as a real binding."""
    weak = ProcessIdentity(pid=999_999, create_time=None)
    assert not weak.bound
    assert "unknown" in weak.describe()
