"""#50 phase 2: the out-of-process wedge-recovery supervisor.

Proves end-to-end with a REAL subprocess that a wedged runtime (process alive,
loop-liveness beacon frozen — exactly a Metal-GPU deadlock) is detected
out-of-process, SIGKILLed, and restarted; that a clean exit is NOT restarted;
and that the should_kill refactor is behavior-preserving.
"""

import os
import sys
import time

import pytest

from core.runtime.subprocess_gateway import get_subprocess_gateway
from tools.liveness_sentinel import is_stale_sample, should_kill
from tools.wedge_recovery_supervisor import WedgeRecoverySupervisor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A victim that writes the liveness beacon for `beat_s`, then either hangs
# (process alive, beacon frozen = a Metal deadlock) or exits cleanly.
_VICTIM = (
    "import json,os,sys,time\n"
    "beacon,beat_s,mode=sys.argv[1],float(sys.argv[2]),sys.argv[3]\n"
    "start=time.time()\n"
    "while time.time()-start<beat_s:\n"
    "    now=time.time()\n"
    "    open(beacon,'w').write(json.dumps({'pid':os.getpid(),'last_loop_run':now,'written_at':now,'loop_state':'alive'}))\n"
    "    time.sleep(0.2)\n"
    "if mode=='hang':\n"
    "    time.sleep(10000)\n"
    "sys.exit(0)\n"
)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _make_spawn(victim_path, beacon_path, beat_s, mode, spawned):
    def _spawn():
        proc = get_subprocess_gateway().spawn(
            [sys.executable, str(victim_path), str(beacon_path), str(beat_s), mode],
            start_new_session=True,
            offline_tooling=True,
            source="proof_tooling:wedge_recovery_supervisor_test",
        )
        spawned.append(proc)
        return proc
    return _spawn


@pytest.fixture
def victim_script(tmp_path):
    p = tmp_path / "_wedge_victim.py"
    p.write_text(_VICTIM, encoding="utf-8")
    return p


def test_supervisor_detects_wedge_and_restarts(tmp_path, victim_script):
    beacon = tmp_path / "hb.json"
    spawned: list = []
    restarts_seen: list[int] = []

    sup = WedgeRecoverySupervisor(
        spawn=_make_spawn(victim_script, beacon, beat_s=1.5, mode="hang", spawned=spawned),
        heartbeat_path=beacon,
        stale_ceiling_s=1.5,
        grace_s=1.0,
        poll_interval_s=0.4,
        max_restarts=3,
        on_restart=lambda n: restarts_seen.append(n),
    )
    try:
        # Stop as soon as one wedge has been recovered.
        outcome = sup.run(done=lambda: len(restarts_seen) >= 1, max_polls=60)
        assert outcome.completed is True
        assert outcome.restarts >= 1, "a frozen-beacon wedge must trigger a restart"
        # The original (wedged) victim was SIGKILLed; a replacement was spawned.
        assert len(spawned) >= 2
        time.sleep(0.3)
        assert not _pid_alive(spawned[0].pid), "wedged victim must be killed out-of-process"
    finally:
        for proc in spawned:
            try:
                proc.kill()
            except OSError:
                pass


def test_supervisor_does_not_restart_clean_exit(tmp_path, victim_script):
    beacon = tmp_path / "hb.json"
    spawned: list = []
    sup = WedgeRecoverySupervisor(
        spawn=_make_spawn(victim_script, beacon, beat_s=0.5, mode="exit", spawned=spawned),
        heartbeat_path=beacon,
        stale_ceiling_s=2.0,
        grace_s=0.2,
        poll_interval_s=0.3,
        max_restarts=2,
    )
    try:
        outcome = sup.run(max_polls=60)
        assert outcome.completed is True
        assert outcome.reason == "clean_exit"
        assert outcome.restarts == 0
        assert len(spawned) == 1
    finally:
        for proc in spawned:
            try:
                proc.kill()
            except OSError:
                pass


def test_should_kill_refactor_preserves_behavior():
    now = 1000.0
    common = dict(
        now=now, last_loop_run=900.0, written_at=900.0, file_mtime=900.0,
        started_at=0.0, grace_s=10.0, stale_ceiling_s=50.0,
    )
    # Beacon 100s old, ceiling 50s, past grace → per-sample stale.
    assert is_stale_sample(**common) is True
    # One stale sample is not enough; two are.
    assert should_kill(**common, consecutive_stale=1) is False
    assert should_kill(**common, consecutive_stale=2) is True
    # Within boot grace → never stale / never kill.
    assert is_stale_sample(**{**common, "started_at": now - 5.0}) is False
    assert should_kill(**{**common, "started_at": now - 5.0}, consecutive_stale=5) is False
    # Fresh beacon → not stale.
    assert is_stale_sample(**{**common, "last_loop_run": now - 1.0, "written_at": now - 1.0}) is False
