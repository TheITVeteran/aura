"""A refused start must SAY it was refused.

Live incident 2026-07-25: a background headless runtime (launched by an agent
session from a worktree) held the global `orchestrator` instance lock. The
desktop app started, its runtime was correctly refused with EX_TEMPFAIL, and the
boot monitor then sat on "Aura is waking up… waiting for boot health" forever —
because the refusal existed only as a print() on a stdout nobody reads.

The lock itself is right (a second runtime would load a second copy of the
resident model and exhaust the host). What was wrong is that a positively
KNOWN failure was rendered as ordinary progress — the same defect class the
CP126 review keeps finding.
"""
from __future__ import annotations

import json
import os

import pytest

from core.utils import singleton


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".aura" / "run").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".aura" / "locks").mkdir(parents=True, exist_ok=True)
    yield


def _write_holder_meta(lock_name: str, pid: int, **over):
    meta = {
        "schema": "aura.instance_lock.v1",
        "lock_name": lock_name,
        "pid": pid,
        "cmdline": ["aura_main.py", "--headless", "--port", "8001"],
        "cwd": "/Users/bryan/.aura/live-source/.claude/worktrees/fable-improvement-pass",
        "create_time": 1785019140.7,
    }
    meta.update(over)
    singleton.instance_lock_metadata_path(lock_name).write_text(json.dumps(meta), encoding="utf-8")


def test_blocked_start_publishes_a_reason_and_a_remedy():
    _write_holder_meta("orchestrator", os.getpid())
    singleton._publish_boot_blocked("orchestrator", os.getpid())

    notice = singleton.read_boot_blocked()
    assert notice, "a refused start must publish a notice the launcher can read"
    assert notice["holder_pid"] == os.getpid()
    assert "already holds" in notice["reason"]
    assert f"kill {os.getpid()}" in notice["remedy"]


def test_background_worktree_instance_is_named_as_such():
    _write_holder_meta("orchestrator", os.getpid())
    singleton._publish_boot_blocked("orchestrator", os.getpid())
    notice = singleton.read_boot_blocked()
    # The user should be told this is NOT their desktop app.
    assert notice["holder_is_background_instance"] is True
    assert "background/headless" in notice["remedy"]


def test_a_desktop_holder_is_not_labelled_background():
    _write_holder_meta(
        "orchestrator", os.getpid(),
        cmdline=["aura_main.py", "--desktop"], cwd="/Users/bryan/.aura/live-source",
    )
    singleton._publish_boot_blocked("orchestrator", os.getpid())
    notice = singleton.read_boot_blocked()
    assert notice["holder_is_background_instance"] is False
    assert "quit the other Aura window" in notice["remedy"]


def test_notice_for_a_dead_holder_is_not_a_live_blocker():
    # A notice naming a process that has since exited must not keep scaring the
    # launcher — otherwise the fix (stopping the other instance) looks ineffective.
    _write_holder_meta("orchestrator", 999999)
    singleton._publish_boot_blocked("orchestrator", 999999)
    assert singleton.boot_blocked_path().exists()
    assert singleton.read_boot_blocked() == {}


def test_successful_acquisition_clears_a_stale_notice():
    _write_holder_meta("orchestrator", os.getpid())
    singleton._publish_boot_blocked("orchestrator", os.getpid())
    assert singleton.boot_blocked_path().exists()

    singleton.clear_boot_blocked()
    assert not singleton.boot_blocked_path().exists()
    assert singleton.read_boot_blocked() == {}


def test_acquiring_a_free_lock_clears_the_notice_and_does_not_block(tmp_path):
    _write_holder_meta("probe_lock", 999999)
    singleton._publish_boot_blocked("probe_lock", 999999)
    try:
        singleton.acquire_instance_lock("probe_lock")
        assert not singleton.boot_blocked_path().exists(), (
            "a start that succeeded must retire any previous refusal notice"
        )
    finally:
        singleton.release_instance_lock()


# ── the live-state pollution guard ────────────────────────────────────────


def test_tests_cannot_drop_a_grace_flag_into_the_real_runtime_dir(monkeypatch):
    """Observed live: a leaked grace_exit.flag with reason='unit_test'."""
    import pwd
    from pathlib import Path

    from core.runtime import shutdown_coordinator

    # The REAL home from the passwd entry — NOT expanduser(), which would just
    # echo back the tmp HOME this module's autouse fixture installed.
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    # Point HOME back at it, simulating a test that forgot to isolate.
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.delenv("AURA_ALLOW_LIVE_RUNTIME_WRITES", raising=False)
    live_flag = real_home / ".aura" / "run" / "grace_exit.flag"
    before = live_flag.read_bytes() if live_flag.exists() else None

    shutdown_coordinator._write_grace_flag(reason="unit_test", created_at_unix=1.0)

    after = live_flag.read_bytes() if live_flag.exists() else None
    assert after == before, "a test process must not write into the live runtime dir"
