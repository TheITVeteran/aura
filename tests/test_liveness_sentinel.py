"""tests/test_liveness_sentinel.py
===================================
The external liveness sentinel must kill a wedged runtime tree from OUTSIDE the
process, since a Metal GPU deadlock can hold the GIL so no in-process Python
thread (including the StallWatchdog hard-exit) can fire. It watches a heartbeat
file the event loop refreshes; staleness (loop-run OR writer timestamp) past the
ceiling = a dead loop → SIGKILL the tree → supervisor restarts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tools.liveness_sentinel import (
    read_heartbeat,
    read_heartbeat_state,
    read_runtime_service_progress,
    should_kill,
)


def test_read_heartbeat_parses_and_degrades(tmp_path):
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"pid": 1, "last_loop_run": 111.0, "written_at": 222.0}))
    last_loop_run, written_at, mtime = read_heartbeat(hb)
    assert last_loop_run == 111.0
    assert written_at == 222.0
    assert mtime is not None
    # Missing file → all None (sentinel treats as not-yet-alive, grace covers it).
    assert read_heartbeat(tmp_path / "nope.json") == (None, None, None)
    # Garbage file → unparseable, so values are None (unknown) but mtime present.
    hb.write_text("{not json")
    llr, wa, mt = read_heartbeat(hb)
    assert llr is None and wa is None and mt is not None


def test_read_runtime_service_progress_parses_optional_field(tmp_path):
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"last_runtime_service_progress": 333.0}))

    assert read_runtime_service_progress(hb) == 333.0
    assert read_runtime_service_progress(tmp_path / "missing.json") is None


def test_fresh_heartbeat_never_kills():
    now = time.time()
    assert should_kill(
        now=now, last_loop_run=now - 2, written_at=now - 1, file_mtime=now - 1,
        started_at=0, grace_s=0, stale_ceiling_s=180, consecutive_stale=10,
    ) is False


def test_retired_heartbeat_state_is_readable(tmp_path):
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"loop_state": "retired", "last_loop_run": time.time() - 999}))

    assert read_heartbeat_state(hb) == "retired"


def test_stale_kills_only_after_two_consecutive():
    now = time.time()
    common = dict(
        now=now, last_loop_run=now - 400, written_at=now - 400, file_mtime=now - 400,
        started_at=0, grace_s=0, stale_ceiling_s=180,
    )
    assert should_kill(**common, consecutive_stale=1) is False
    assert should_kill(**common, consecutive_stale=2) is True


def test_loop_wedge_detected_even_when_daemon_still_writes():
    # The KEY semantic: last_loop_run is the loop-liveness signal. A stale
    # last_loop_run means the event loop is wedged EVEN IF the watchdog daemon
    # is still rewriting the file (fresh written_at/mtime). file_mtime must not
    # mask the wedge.
    now = time.time()
    assert should_kill(
        now=now, last_loop_run=now - 400, written_at=now - 1, file_mtime=now - 1,
        started_at=0, grace_s=0, stale_ceiling_s=180, consecutive_stale=2,
    ) is True
    # Conversely, a fresh last_loop_run = loop alive = never kill.
    assert should_kill(
        now=now, last_loop_run=now - 1, written_at=now - 400, file_mtime=now - 400,
        started_at=0, grace_s=0, stale_ceiling_s=180, consecutive_stale=10,
    ) is False


def test_recent_runtime_service_progress_does_not_mask_stale_loop():
    now = time.time()
    assert should_kill(
        now=now,
        last_loop_run=now - 400,
        written_at=now - 1,
        file_mtime=now - 1,
        started_at=0,
        grace_s=0,
        stale_ceiling_s=180,
        consecutive_stale=2,
        last_runtime_service_progress=now - 2,
        service_progress_grace_s=240,
    ) is True


def test_recent_runtime_service_progress_suppresses_startup_before_loop_heartbeat():
    now = time.time()
    assert should_kill(
        now=now,
        last_loop_run=0,
        written_at=0,
        file_mtime=0,
        started_at=0,
        grace_s=0,
        stale_ceiling_s=180,
        consecutive_stale=2,
        last_runtime_service_progress=now - 2,
        service_progress_grace_s=240,
    ) is False


def test_stale_runtime_service_progress_does_not_mask_wedge():
    now = time.time()
    assert should_kill(
        now=now,
        last_loop_run=now - 400,
        written_at=now - 1,
        file_mtime=now - 1,
        started_at=0,
        grace_s=0,
        stale_ceiling_s=180,
        consecutive_stale=2,
        last_runtime_service_progress=now - 400,
        service_progress_grace_s=240,
    ) is True


def test_stallwatchdog_writes_liveness_beacon(tmp_path, monkeypatch):
    """The StallWatchdog must refresh the heartbeat file the sentinel reads —
    with a fresh last_loop_run while the loop is alive."""
    import asyncio

    hb = tmp_path / "beacon.json"
    monkeypatch.setenv("AURA_LIVENESS_HEARTBEAT_FILE", str(hb))
    from core.resilience.stall_watchdog import StallWatchdog

    async def _drive():
        loop = asyncio.get_running_loop()
        dog = StallWatchdog(loop, threshold=5.0)
        dog.start()
        try:
            await asyncio.sleep(3.0)  # let the loop run + the daemon write
            assert hb.exists(), "beacon file was not written"
            llr, written_at, _ = read_heartbeat(hb)
            assert llr and llr > 0.0
            # last_loop_run is fresh because the loop is alive and running callbacks.
            assert time.time() - llr < 5.0
            assert written_at and written_at > 0.0
            assert read_runtime_service_progress(hb)
        finally:
            dog.stop()

    asyncio.run(_drive())


def test_end_to_end_sentinel_kills_wedged_process(tmp_path):
    """A child writes one heartbeat then 'wedges' (stops updating); the sentinel
    must SIGKILL it. Proves the out-of-process kill path end-to-end."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    hb = tmp_path / "hb.json"

    # Victim: write a fresh heartbeat, then sleep forever (simulated wedge —
    # the heartbeat goes stale because it stops updating).
    victim_src = (
        "import json,time,os\n"
        f"hb={str(hb)!r}\n"
        "open(hb,'w').write(json.dumps({'pid':os.getpid(),'last_loop_run':time.time(),'written_at':time.time()}))\n"
        "time.sleep(120)\n"
    )
    victim = subprocess.Popen([sys.executable, "-c", victim_src])
    try:
        time.sleep(1.0)  # let it write the heartbeat
        sentinel = subprocess.Popen(
            [
                sys.executable, str(repo / "tools" / "liveness_sentinel.py"),
                "--pid", str(victim.pid),
                "--heartbeat", str(hb),
                "--stale-ceiling", "3",
                "--grace", "2",
                "--interval", "1",
                "--tombstone-dir", str(tmp_path),
            ],
            cwd=str(repo),
        )
        try:
            # Stale after ~3s + 2 samples + grace 2s → killed well within 25s.
            rc = victim.wait(timeout=25)
            assert rc is not None  # victim was killed
            # Sentinel should write a tombstone and exit 0.
            sentinel.wait(timeout=10)
            tombstones = list(tmp_path.glob("liveness_tombstone_*.json"))
            assert tombstones, "sentinel did not record a kill tombstone"
        finally:
            sentinel.poll() is None and sentinel.kill()
    finally:
        if victim.poll() is None:
            victim.kill()
